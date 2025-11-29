import argparse
import evaluate
import torch
import os

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from core.custom_whisper import setup_model_with_custom_language

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

    args, rest = p.parse_known_args()
    
    return (args, rest) if rest else args

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

def load_custom_whisper(model_size, lang_code, init_vector=None):
    model_id = f"openai/whisper-{model_size}"
    print(f"\nLoading {model_id}...")
    model, processor, token_id = setup_model_with_custom_language(
        model_id, lang_code, init_vector
    )
    print(f"added <|{lang_code}|> (id={token_id})")

    model.config.use_cache = False

    # display info
    print(f"\nInitial setup:")
    print(f"    Vocab size: {len(processor.tokenizer)}")
    print(f"    Decoder embeddings: {model.model.decoder.embed_tokens.weight.shape}")
    print(f"    Token <|{lang_code}|> ID: {token_id}")
    
    # checking generation config
    print(f"    Lang_to_id entries: \n{len(model.generation_config.lang_to_id)}")
    if f"<|{lang_code}|>" in model.generation_config.lang_to_id:
        print(f"    <|{lang_code}|> -> {model.generation_config.lang_to_id[f'<|{lang_code}|>']}")
        print(f"    model.generation_config.forced_decoder_ids = {model.generation_config.forced_decoder_ids}")
    
    return model, processor

def build_trainer(model, processor, dataset, data_collator, args):

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

    return trainer

def save_best_model(trainer, processor, output_dir):
    if trainer.state.best_model_checkpoint:
        best_path = os.path.join(output_dir, "best")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"\nbest model saved to: {best_path}")

    print(f"\ntraining complete!")
    print(f"model saved to: {output_dir}")