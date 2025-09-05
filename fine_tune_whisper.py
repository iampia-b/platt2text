import argparse, os, torch, evaluate
from datasets import load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)

from custom_whisper import CustomWhisperProcessor, update_model_for_custom_language


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",   required=True)
    p.add_argument("--audio_column",  default="audio_filepath")
    p.add_argument("--text_column",   default="text")
    p.add_argument("--lang_code",     required=True,
                   help="Token code without <| |>, e.g. 'nde' or 'de'")
    p.add_argument("--lang_alias",    default=None,
                   help="Optional friendly alias, e.g. 'low_german'")
    p.add_argument("--model_size",    default="small", choices=[
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
    p.add_argument("--sampling_rate", type=int, default=16_000)
    p.add_argument("--output_dir",    required=True)
    
    # training params
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--eval_steps",  type=int, default=200)
    p.add_argument("--save_steps",  type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()

args = get_args()
MODEL_ID = f"openai/whisper-{args.model_size}"

# reproducibility
set_seed(args.seed)
torch.manual_seed(args.seed)

# dataset
dataset = load_from_disk(args.dataset_dir)
AUDIO_COL = args.audio_column
TEXT_COL = args.text_column

dataset = dataset.cast_column(AUDIO_COL, Audio(sampling_rate=args.sampling_rate))

# model & CustomWhisperProcessor
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
processor = CustomWhisperProcessor.from_pretrained(MODEL_ID)

# adding custom language
model, processor = update_model_for_custom_language(
    model, processor, 
    lang_code=args.lang_code, 
    lang_alias=args.lang_alias
)

# setting language for tokenization
language_to_use = args.lang_alias
processor.tokenizer.set_prefix_tokens(language=language_to_use, task="transcribe")

# preview tokenization (for debugging)
preview_sample = dataset["train"][0]
enc = processor.tokenizer(preview_sample[TEXT_COL], add_special_tokens=True)
#print("\n   token IDs:", enc.input_ids)
#print("    verbatim decode:", processor.tokenizer.decode(enc.input_ids, skip_special_tokens=False))
#print("    clean decode   :", processor.tokenizer.decode(enc.input_ids, skip_special_tokens=True))

# language token ID for data collator
lang_token = f"<|{args.lang_code}|>"
LANG_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids(lang_token)
print(f"\n  Language token '{lang_token}' has ID: {LANG_TOKEN_ID}")


def prepare(batch):
    # load and resample audio data
    audio = batch[AUDIO_COL]

    # compute log-Mel input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(
        batch[TEXT_COL], add_special_tokens=True
    ).input_ids
    return batch

columns = dataset["train"].column_names
print("    mapping dataset to input_features/labels ...")
dataset = dataset.map(prepare, remove_columns=columns, num_proc=1)

# data collator 
from dataclasses import dataclass
from typing import Any, List, Dict, Union

@dataclass
class WhisperCollator:
    processor: Any
    decoder_start_token_id: int
    
    def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inputs = [{"input_features": f["input_features"]} for f in feats]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in feats]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

collator = WhisperCollator(processor, model.config.decoder_start_token_id)

# metrics & training 
wer_metric = evaluate.load("wer")

def compute(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

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
    save_total_limit=1,
    gradient_checkpointing=True,
    seed=args.seed,
)
'''
For early stopping:
    stop = EarlyStoppingCallback(args.early_stopping_patience)
    if early stopping: add 'callbacks=[stop], in trainer
'''

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
    compute_metrics=compute,
    processing_class=processor.tokenizer, 
)


# saving processor before training to preserve custom config
processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    print(f"\ntraining with custom language:")
    print(f"  - Code: {args.lang_code}")
    print(f"  - Alias: {args.lang_alias or 'None'}")
    print(f"  - Token ID: {LANG_TOKEN_ID}")
    
    # debugging
    if hasattr(model.generation_config, 'lang_to_id'):
        print(f"  - In generation config: {args.lang_code in model.generation_config.lang_to_id}")
    
    result = trainer.train()

    if trainer.state.best_model_checkpoint:
        best_path = os.path.join(args.output_dir, "best")
        trainer.save_model(best_path)
        # save processor with best model
        processor.save_pretrained(best_path)
        
    print(f"\n-> Training complete!")
    print(f"-> Model saved to: {args.output_dir}")
    if args.lang_alias:
        print(f"  processor.tokenizer.set_prefix_tokens(language='{args.lang_alias}', task='transcribe')")
    else:
        print(f"  processor.tokenizer.set_prefix_tokens(language='{args.lang_code}', task='transcribe')")