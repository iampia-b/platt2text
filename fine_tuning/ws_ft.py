import torch

from pathlib import Path
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from datasets import load_from_disk, Audio
from transformers import (
    set_seed,
)
from tqdm import tqdm

from utils import data_handling
from utils import ft_utils
from fine_tuning.data_collator import WhisperDataCollator


def detect_language_openai_method(
    model: WhisperForConditionalGeneration,
    audio_features: torch.Tensor,
    processor: WhisperProcessor,
    device: torch.device
):
    # ensure batch dimension
    single = audio_features.ndim == 2
    if single:
        audio_features = audio_features.unsqueeze(0)
    
    audio_features = audio_features.to(device)
    n_audio = audio_features.shape[0]
    
    # get the start-of-transcript token
    sot_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    # create decoder input - ONLY the SOT token
    x = torch.tensor([[sot_token]] * n_audio, device=device)
    
    # forward pass through decoder
    with torch.no_grad():
        outputs = model(
            input_features=audio_features,
            decoder_input_ids=x
        )
        logits = outputs.logits[:, 0, :]  # get logits at language position
    
    # get all language tokens
    vocab = processor.tokenizer.get_vocab()
    all_language_tokens = [tid for token, tid in vocab.items()
                          if (len(token) == 6 and token.startswith("<|") and 
                              token.endswith("|>") and token[2:4].isalpha() and 
                              token[2:4].islower())]
    
    # create mask and suppress non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
    mask[all_language_tokens] = False
    logits[:, mask] = -float('inf')
    
    # compute softmax over entire vocabulary
    full_probs = torch.softmax(logits, dim=-1)
    
    # extract language probabilities
    language_token_probs = full_probs[:, all_language_tokens]
    
    if single:
        language_token_probs = language_token_probs[0]
    
    return language_token_probs.cpu(), all_language_tokens


def compute_corpus_wise_embedding(
    model: WhisperForConditionalGeneration,
    dataset,
    processor: WhisperProcessor,
    audio_column: str,
    device: torch.device,
    custom_weights: dict = None
):
    # compute corpus-wise weighted language embedding from dataset samples

    print(f"Computing corpus-wise embedding from {len(dataset)} samples...")
    
    # collect language probabilities from multiple samples
    all_probs = []
    lang_ids = None
    
    if custom_weights is not None:
        # use provided custom weights
        print("Using custom weights for language embedding...")

        # 1) map codes -> tokenizer ids (only keep known Whisper language tokens)
        code_ids = []
        weight_vals = []
        for code, w in custom_weights.items():
            tok = f"<|{str(code).lower()}|>"
            tok_id = processor.tokenizer.convert_tokens_to_ids(tok)
            if tok_id is None or tok_id < 0:
                print(f"  [warn] skipping unknown language code '{code}' (token {tok} not found).")
                continue
            code_ids.append(int(tok_id))
            weight_vals.append(float(w))

        if not code_ids:
            raise ValueError("No valid language codes found in custom_weights.")
        total = sum(weight_vals)
        if total <= 0.0:
            raise ValueError("Sum of custom_weights must be > 0.")

        # 2) normalize to probabilities; set lang_ids and avg_probs
        weight_probs = [w / total for w in weight_vals]
        lang_ids = torch.tensor(code_ids, device=device, dtype=torch.long)
        lang_ids = lang_ids.tolist()

        # match dtype/device used later when multiplying embeddings
        emb_weight = model.model.decoder.embed_tokens.weight
        avg_probs = torch.tensor(weight_probs, device=device, dtype=emb_weight.dtype)
    else:
        for i in tqdm(range(len(dataset)), desc="Analyzing samples"):
            sample = dataset[i]
            
            # process audio
            audio = sample[audio_column]
            inputs = processor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"], 
                return_tensors="pt"
            )
            
            # get language probabilities
            probs, current_lang_ids = detect_language_openai_method(
                model, inputs.input_features, processor, device
            )
            probs = probs.squeeze(0) if probs.dim() > 1 else probs
            all_probs.append(probs)
            if lang_ids is None:
                lang_ids = current_lang_ids
    
        # stack and average probabilities
        all_probs_tensor = torch.stack(all_probs)
        avg_probs = all_probs_tensor.mean(dim=0)
    
    # get language embeddings
    embedding_layer = model.model.decoder.embed_tokens
    lang_embeddings = embedding_layer.weight[lang_ids].to(device)
    
    # compute weighted sum
    weighted_embedding = torch.sum(
        avg_probs.to(device).unsqueeze(1) * lang_embeddings, 
        dim=0
    )
    
    # print top detected languages
    id_to_token = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    top_indices = torch.argsort(avg_probs, descending=True)[:10]
    print("\nTop 10 detected languages (corpus average):")
    for idx in top_indices:
        token = id_to_token[lang_ids[idx]]
        lang_code = token[2:4]
        prob = avg_probs[idx].item()
        print(f"  {lang_code}: {prob:.4f}")
    
    return weighted_embedding, avg_probs, lang_ids

def main():
    args = ft_utils.get_args()

    model_id = f"openai/whisper-{args.model_size}"

    # reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # load dataset
    print(f"loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))

    # loading model for WS calculation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)
    model.eval()  # Set to eval for language detection
    
    # compute corpus-wise weighted embedding
    weighted_embedding, avg_probs, lang_ids = compute_corpus_wise_embedding(
        model=model,
        dataset=dataset["train"],
        processor=processor,
        audio_column=args.audio_column,
        device=device,
        #custom_weights={"de": 0.33, "nl": 0.33, "en": 0.33}
    )
    
    print(f"\nWeighted embedding computed. Shape: {weighted_embedding.shape}")

    # create a fresh model instance to avoid cached state issues
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # load dataset
    print(f"loading dataset from {args.dataset_dir}")
    dataset = data_handling.load_local_dataset(args.dataset_dir, args.audio_column, args.sampling_rate)

    # load model & processor
    model, processor = ft_utils.load_custom_whisper(args.model_size, args.lang_code, weighted_embedding)

    # prepare dataset
    dataset = data_handling.prepare_dataset(processor, dataset, args.audio_column, args.text_column)

    # data collator
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    trainer = ft_utils.build_trainer(model, processor, dataset, data_collator, args)

    # save baseline model
    baseline_path = Path(args.output_dir).parent / "baseline"
    trainer.save_model(baseline_path)
    processor.save_pretrained(baseline_path)

    # save processor before training
    processor.save_pretrained(args.output_dir)

    # training
    print(f"\nstarting training with custom language: {args.lang_code}")
    if args.lang_alias:
        print(f"  alias: {args.lang_alias}")

    trainer.train()

    # save best model
    ft_utils.save_best_model(trainer, processor, args.output_dir)


if __name__ == "__main__":
    main()
