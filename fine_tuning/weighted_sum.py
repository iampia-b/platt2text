import os
import argparse
import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from datasets import load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from core.custom_whisper import CustomWhisperProcessor, update_model_for_custom_language


def get_args():
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--audio_column", default="audio")
    p.add_argument("--text_column", default="text")
    p.add_argument("--sampling_rate", type=int, default=16_000)

    # model    
    p.add_argument("--lang_code", required=True,
                   help="token code without <| |>, e.g. 'nds'")
    p.add_argument("--lang_alias", default=None,
                   help="optional friendly alias, e.g. 'low_german'")
    p.add_argument("--model_size", default="large-v3",
                   choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_zero_shot", action="store_true",
                   help="save initialized model before training")
    p.add_argument("--zero_shot_dir", default=None)

    # weighted sum initialization
    p.add_argument("--n_samples_for_embedding", type=int, default=2000)
    p.add_argument("--detect_batch_size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--initialization_scale", type=float, default=1.0)

    # training 
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return p.parse_args()


def get_language_token_ids(processor, model):
    # extracting all language token ids from model
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "lang_to_id", None):
        # use generation config if available
        ids = sorted(set(int(v) for v in gc.lang_to_id.values()))
        return ids

    # fallback: find language tokens in vocab
    vocab = processor.tokenizer.get_vocab()
    specials = {
        "<|startoftranscript|>", "<|endoftext|>", "<|notimestamps|>",
        "<|transcribe|>", "<|translate|>"
    }
    lang_ids = []
    for tok, tid in vocab.items():
        if tok.startswith("<|") and tok.endswith("|>") and tok not in specials:
            lang_ids.append(tid)
    return sorted(set(lang_ids))


@torch.no_grad()
def detect_language_openai_method(model, input_features, processor, language_token_ids, device):
    """
    whisper's language detection: 
    - run decoder for single step with SOT token,
    - mask logits to language tokens and softmax for per-language probabilities
    """
    if input_features.ndim == 2:
        input_features = input_features.unsqueeze(0)
    input_features = input_features.to(device)

    sot_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    dec_in = torch.full(
        (input_features.size(0), 1),
        fill_value=sot_token,
        device=device,
        dtype=torch.long
    )

    outputs = model(input_features=input_features, decoder_input_ids=dec_in)
    logits = outputs.logits[:, 0, :]  # (batch, vocab)

    # mask to language tokens only
    mask = torch.full_like(logits, fill_value=-float("inf"))
    mask[:, language_token_ids] = logits[:, language_token_ids]
    probs_full = torch.softmax(mask, dim=-1)
    lang_probs = probs_full[:, language_token_ids]  # (batch, n_langs)
    return lang_probs


def compute_corpus_weighted_embedding(
    model, dataset, processor, audio_column, language_token_ids, 
    device, n_samples, detect_batch_size, seed):
    # compute weighted-sum embedding from corpus language detection
    print(f"\ncomputing corpus language embedding from {n_samples} samples...")
    n = min(n_samples, len(dataset))

    # accumulate language probabilities
    running_sum = None
    n_seen = 0

    def to_input_features(batch_audio):
        feats = processor.feature_extractor(batch_audio, sampling_rate=processor.feature_extractor.sampling_rate)
        return torch.tensor(feats.input_features, dtype=torch.float32)

    batch_audio = []

    sampled = dataset.shuffle(seed=seed).select(range(n))
    for k in tqdm(range(len(sampled)), desc="analyzing samples"):
        a = sampled[k][audio_column]
        batch_audio.append(a["array"])
        
        # process batch
        if len(batch_audio) == detect_batch_size or k == n - 1:
            feats = to_input_features(batch_audio).to(device)
            probs = detect_language_openai_method(
                model, feats, processor, language_token_ids, device
            )
            
            # normalize and accumulate
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            batch_sum = probs.sum(dim=0)
            running_sum = batch_sum if running_sum is None else (running_sum + batch_sum)
            n_seen += probs.shape[0]
            batch_audio = []

    avg_probs = running_sum / n_seen

    # weighted sum of language embeddings
    emb = model.model.decoder.embed_tokens.weight
    lang_emb = emb[torch.tensor(language_token_ids, dtype=torch.long, device=device)]
    weighted = (avg_probs.to(device).unsqueeze(0) @ lang_emb).squeeze(0)

    # normalize to mean norm of language embeddings
    target_norm = lang_emb.norm(dim=1).mean()
    wn = weighted.norm() + 1e-8
    weighted = weighted * (target_norm / wn)

    # report top languages
    id_to_token = {tid: processor.tokenizer.convert_ids_to_tokens(tid) for tid in language_token_ids}
    topk = torch.topk(avg_probs, k=min(10, avg_probs.numel()))
    print("\ntop languages detected in corpus:")
    for rank, (val, j) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), start=1):
        print(f"  {rank:2d}. {id_to_token[language_token_ids[j]]:>12} : {val:.4f}")

    return weighted, avg_probs, language_token_ids


def initialize_language_with_weighted_embedding(model, processor, lang_code, lang_alias, weighted_embedding, scale):
    model, processor = update_model_for_custom_language(
        model=model, processor=processor, lang_code=lang_code, lang_alias=lang_alias
    )

    token = f"<|{lang_code}|>"
    token_id = processor.tokenizer.convert_tokens_to_ids(token)

    with torch.no_grad():
        model.model.decoder.embed_tokens.weight[token_id] = weighted_embedding.to(model.device) * scale

    print(f"initialized {token} (id={token_id}) with weighted embedding")
    
    # update generation config
    model.generation_config.language = lang_alias or lang_code
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    return model, processor, token_id


@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # drop decoder_start if present
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor, audio_col, text_col):
    audio = batch[audio_col]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(
        batch[text_col], add_special_tokens=True
    ).input_ids
    return batch


def compute_metrics(pred, processor, wer_metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": 100.0 * wer}


def main():
    args = get_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = f"openai/whisper-{args.model_size}"

    print(f"\nloading base model: {model_id}")
    processor = CustomWhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()  # eval mode for language detection

    # load dataset
    print(f"loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    ds = ds.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))

    # compute weighted embedding from corpus
    lang_token_ids = get_language_token_ids(processor, model)
    weighted_embedding, avg_probs, lang_ids = compute_corpus_weighted_embedding(
        model=model,
        dataset=ds["train"],
        processor=processor,
        audio_column=args.audio_column,
        language_token_ids=lang_token_ids,
        device=device,
        n_samples=args.n_samples_for_embedding,
        detect_batch_size=args.detect_batch_size,
        seed=args.seed,
    )

    # initialize new language token
    model, processor, token_id = initialize_language_with_weighted_embedding(
        model=model,
        processor=processor,
        lang_code=args.lang_code,
        lang_alias=args.lang_alias,
        weighted_embedding=weighted_embedding,
        scale=args.initialization_scale,
    )

    # setup for training
    language_to_use = args.lang_alias or args.lang_code
    processor.tokenizer.set_prefix_tokens(language=language_to_use, task="transcribe")

    # save zero-shot model if requested
    if args.save_zero_shot:
        zero_shot_dir = args.zero_shot_dir or os.path.join(args.output_dir, "zero_shot")
        print(f"\nsaving zero-shot model to: {zero_shot_dir}")
        os.makedirs(zero_shot_dir, exist_ok=True)
        processor.save_pretrained(zero_shot_dir)
        model.save_pretrained(zero_shot_dir)
        torch.save(
            {
                "weighted_embedding": weighted_embedding.cpu(),
                "avg_language_probs": avg_probs.cpu(),
                "language_token_ids": lang_ids,
                "initialization_scale": args.initialization_scale,
                "token_id": token_id,
            },
            os.path.join(zero_shot_dir, "ws_init_meta.pt"),
        )

    # prepare for training
    model.train()
    
    print("\npreparing dataset...")
    columns = ds["train"].column_names
    ds = ds.map(
        lambda batch: prepare_dataset(batch, processor, args.audio_column, args.text_column),
        remove_columns=columns,
        num_proc=1
    )

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        tokenizer=processor.tokenizer,
        callbacks=[early_stopping],
    )

    print(f"starting weighted-sum training for: {language_to_use}")
    print(f"output dir: {args.output_dir}")

    result = trainer.train()

    # save best model
    if trainer.state.best_model_checkpoint:
        best_dir = os.path.join(args.output_dir, "best")
        print(f"\nsaving best model to {best_dir}")
        os.makedirs(best_dir, exist_ok=True)
        
        # reload best checkpoint and save cleanly
        best_model = WhisperForConditionalGeneration.from_pretrained(
            trainer.state.best_model_checkpoint
        ).to(device)
        best_model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        
        # save training metadata
        torch.save(
            {
                "weighted_embedding": weighted_embedding.cpu(),
                "avg_language_probs": avg_probs.cpu(),
                "language_token_ids": lang_ids,
                "initialization_scale": args.initialization_scale,
                "token_id": token_id,
                "best_wer": trainer.state.best_metric,
            },
            os.path.join(best_dir, "ws_train_meta.pt"),
        )

    # save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"saving final model to {final_dir}")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print("\ntraining complete!")


if __name__ == "__main__":
    main()