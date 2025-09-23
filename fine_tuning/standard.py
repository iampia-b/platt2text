"""
standard fine-tuning for whisper with custom language tokens
adapted from s. gandhi, hugging face blog: "fine-tune whisper for multilingual asr with transformers"
"""

import argparse
import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from datasets import load_from_disk, Audio
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

from core.new_custom_whisper import setup_model_with_custom_language

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--audio_column", default="audio")
    p.add_argument("--text_column", default="text")
    p.add_argument(
        "--lang_code",
        required=True,
    )
    p.add_argument(
        "--lang_alias", default=None,
    )
    p.add_argument(
        "--model_size",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
    )
    p.add_argument("--sampling_rate", type=int, default=16_000)
    p.add_argument("--output_dir", required=True)

    # training params
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()

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
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor, audio_col, text_col):
    # load and resample audio data
    audio = batch[audio_col]

    # compute log-mel input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    ids = processor.tokenizer(batch[text_col], add_special_tokens=False).input_ids
    ids = ids + [processor.tokenizer.eos_token_id]    
    batch["labels"] = ids
    return batch

def compute_metrics(pred, processor, wer_metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    args = get_args()
    model_id = f"openai/whisper-{args.model_size}"

    # reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # load dataset
    print(f"loading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sampling_rate))

    # load model & processor
    print(f"\nLoading {model_id}...")
    model, processor, token_id = setup_model_with_custom_language(
        model_id, args.lang_code
    )
    print(f"added <|{args.lang_code}|> (id={token_id})")

    model.config.use_cache = False

    # display info
    print(f"\nInitial setup:")
    print(f"    Vocab size: {len(processor.tokenizer)}")
    print(f"    Decoder embeddings: {model.model.decoder.embed_tokens.weight.shape}")
    print(f"    Token <|{args.lang_code}|> ID: {token_id}")
    
    # checking generation config
    print(f"    Lang_to_id entries: \n{len(model.generation_config.lang_to_id)}")
    if f"<|{args.lang_code}|>" in model.generation_config.lang_to_id:
        print(f"    <|{args.lang_code}|> -> {model.generation_config.lang_to_id[f'<|{args.lang_code}|>']}")
        print(f"    model.generation_config.forced_decoder_ids = {model.generation_config.forced_decoder_ids}")
        print(f"    model.config.forced_decoder_ids = {model.config.forced_decoder_ids}")

    # prepare dataset
    print("preparing dataset...")
    columns = dataset["train"].column_names
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor, args.audio_column, args.text_column),
        remove_columns=columns,
        num_proc=1,
    )

    input_feat = dataset['train'][0]['input_features']
    print("\nProcessed dataset check:")
    print(f"  Keys: {dataset['train'][0].keys()}")
    print(f"  Input features shape: ({len(input_feat)}, {len(input_feat[0])})")
    print(f"  Labels length: {len(dataset['train'][0]['labels'])}")
    print(f"  First 10 label IDs: {dataset['train'][0]['labels'][:10]}")

    # data collator
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # metrics
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
        save_total_limit=2,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    # trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        processing_class=processor.tokenizer,
    )

    # save processor before training
    processor.save_pretrained(args.output_dir)

    # training
    print(f"\nstarting training with custom language: {args.lang_code}")
    if args.lang_alias:
        print(f"  alias: {args.lang_alias}")

    trainer.train()

    # save best model
    if trainer.state.best_model_checkpoint:
        best_path = os.path.join(args.output_dir, "best")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"\nbest model saved to: {best_path}")

    print(f"\ntraining complete!")
    print(f"model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
