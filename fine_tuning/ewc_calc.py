import argparse
import torch
import pickle

from pathlib import Path
from tqdm import tqdm

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils import data_handling



def calculate_fisher_information(model_dir,
                                 dataset,
                                 processor,
                                 device,
                                 lang_code,
                                 num_samples=1000):
    # calculate Fisher Information Matrix (diagonal approximation) for EWC.
    
    # load model
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    
    # set lang_code for generation
    model.generation_config.language = lang_code
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    
    # initialize fisher information storage
    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data).to(device)
    
    # feature extractor and tokenizer
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    
    # process samples
    sample_count = 0
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Calculating Fisher Information", total=num_samples):
            if sample_count >= num_samples:
                break
                
            # extract audio
            audio = sample["audio"]
            arr, sr = audio["array"], audio["sampling_rate"]
            
            # get reference text
            reference_text = sample["sentence"]
            
            # prepare inputs
            features = feature_extractor(
                arr,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(device)
            
            # tokenize reference text for teacher forcing
            labels = tokenizer(
                reference_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            
            # enable gradient computation for this forward pass
            model.zero_grad()
            
            # forward pass with teacher forcing
            with torch.enable_grad():
                outputs = model(
                    input_features=features,
                    labels=labels,
                    return_dict=True
                )
                
                # compute loss (negative log-likelihood)
                loss = outputs.loss
                loss.backward()
            
            # accumulate squared gradients (fisher information)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.detach() ** 2
            
            sample_count += 1
    
    # average over all samples
    for name in fisher_dict:
        fisher_dict[name] /= sample_count
    
    return fisher_dict


def main():
    parser = argparse.ArgumentParser()
    
    # model paths
    parser.add_argument(
        "--model_size",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
    )
    
    # test dataset (german)
    parser.add_argument("--dataset_dir", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--lang_code", default="de",
                        help="Language code inside Common Voice (default: de)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_limit", type=int, default=500)

    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="sentence")
    
    # output
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load german set
    dataset = data_handling.load_local_cv_dataset(args.dataset_dir, args.split, args.sample_limit, 42, args.audio_column, args.text_column)

    # load processor
    model_id = f"openai/whisper-{args.model_size}"
    print(f"Loading processor from {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)

    # calculate fisher information
    print(f"Calculating Fisher Information using {args.sample_limit} samples...")
    fisher_dict = calculate_fisher_information(
        model_id,
        dataset,
        processor,
        device,
        args.lang_code,
        num_samples=args.sample_limit
    )

    # save fisher information
    out_path = args.output_dir / f"fisher_{args.model_size}_{args.split}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(fisher_dict, f)
    print(f"Fisher information saved to {out_path}")
    
    # print statistics
    total_params = sum(f.numel() for f in fisher_dict.values())
    print(f"Total parameters with fisher information: {total_params:,}")
    
    # show top 10 parameters by average fisher value
    print("\nTop 10 parameters by average fisher information:")
    param_stats = [(name, f.mean().item()) for name, f in fisher_dict.items()]
    param_stats.sort(key=lambda x: x[1], reverse=True)
    for i, (name, avg_fisher) in enumerate(param_stats[:10]):
        print(f"{i+1}. {name}: {avg_fisher:.6f}")


if __name__ == "__main__":
    main()