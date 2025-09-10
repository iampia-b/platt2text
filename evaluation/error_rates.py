import argparse
import torch
import pandas as pd
import evaluate
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, pipeline

from core.custom_whisper import CustomWhisperProcessor


def load_test_dataset(args):
    if args.dataset_path:
        # local dataset
        dataset = load_from_disk(args.dataset_path)
        if args.split:
            dataset = dataset[args.split]
    else:
        # huggingface dataset
        dataset = load_dataset(
            args.hf_dataset,
            args.hf_config,
            split=args.hf_split,
            trust_remote_code=True
        )
    
    # limit samples if requested
    if args.sample_limit:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))
    
    # ensure audio is at 16khz
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=16000))
    
    return dataset


def evaluate_model(model_path, dataset, args):
    device = args.device
    
    # load model and processor
    print(f"loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = CustomWhisperProcessor.from_pretrained(model_path)
    
    # set language 
    if args.force_language:
        processor.tokenizer.set_prefix_tokens(language=args.force_language, task="transcribe")
    
    # create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )
    
    # evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []
    
    # process each sample
    for sample in tqdm(dataset, desc="evaluating"):
        audio = sample[args.audio_column]["array"]
        reference = sample[args.text_column]
        
        # generate prediction
        result = pipe(audio, generate_kwargs={"language": args.force_language})
        prediction = result["text"]
        
        predictions.append(prediction)
        references.append(reference)
    
    # calculate metrics
    wer = wer_metric.compute(predictions=predictions, references=references) * 100
    cer = cer_metric.compute(predictions=predictions, references=references) * 100
    
    return {
        "model_path": model_path,
        "wer": wer,
        "cer": cer,
        "num_samples": len(dataset),
        "forced_language": args.force_language,
    }, predictions, references


def main():
    parser = argparse.ArgumentParser()
    
    # model paths
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--model_paths", nargs="+", type=Path)
    
    # dataset
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="text")
    
    # huggingface dataset options
    parser.add_argument("--hf_dataset", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--hf_config", default="de")
    parser.add_argument("--hf_split", default="test")
    
    # evaluation settings
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force_language", help="force decode language (e.g., 'de', 'nds')")
    parser.add_argument("--sample_limit", type=int, help="limit number of samples")
    parser.add_argument("--fp16", action="store_true", help="use fp16 for inference")
    
    # output
    parser.add_argument("--output_dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # load test dataset
    print("loading test dataset...")
    dataset = load_test_dataset(args)
    print(f"dataset size: {len(dataset)} samples")
    
    # find models to evaluate
    if args.model_paths:
        model_paths = args.model_paths
    else:
        # find all models in directory
        model_paths = []
        for path in args.model_dir.rglob("*/best"):
            if path.is_dir() and (path / "config.json").exists():
                model_paths.append(path)
        for path in args.model_dir.rglob("*/final"):
            if path.is_dir() and (path / "config.json").exists():
                model_paths.append(path)
    
    if not model_paths:
        print(f"no models found in {args.model_dir}")
        return
    
    print(f"found {len(model_paths)} models to evaluate")
    
    # evaluate each model
    all_results = []
    for model_path in model_paths:
        print(f"\nevaluating {model_path.name}...")
        
        try:
            metrics, predictions, references = evaluate_model(model_path, dataset, args)
            all_results.append(metrics)
            
            # save per-sample predictions
            sample_df = pd.DataFrame({
                "reference": references,
                "prediction": predictions,
                "model": model_path.name
            })
            sample_file = args.output_dir / f"predictions_{model_path.parent.name}_{model_path.name}.csv"
            sample_df.to_csv(sample_file, index=False)
            
            print(f"  wer: {metrics['wer']:.2f}%")
            print(f"  cer: {metrics['cer']:.2f}%")
            
        except Exception as e:
            print(f"  failed to evaluate: {e}")
            continue
    
    # save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("wer")
        
        summary_file = args.output_dir / "evaluation_summary.csv"
        results_df.to_csv(summary_file, index=False)
        
        print("evaluation summary:")
        print(results_df.to_string(index=False))
        
        # best model
        best = results_df.iloc[0]
        print(f"\nbest model: {best['model_path']}")
        print(f"  wer: {best['wer']:.2f}%")
        print(f"  cer: {best['cer']:.2f}%")


if __name__ == "__main__":
    main()