import argparse
import torch
import evaluate
import pandas as pd
import evaluation.plot_eval as plot_eval

from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration
from pathlib import Path
from tqdm import tqdm

import evaluation.eval_training as train_eval
import core.new_custom_whisper as custom

def find_model_paths(baseline_model, model_dir):

    # find models to evaluate
    model_paths = [baseline_model]
    for path in model_dir.rglob("**/best"):
        if path.is_dir() and (path / "config.json").exists():
            model_paths.append(path)
    for path in model_dir.rglob("**/final"):
        if path.is_dir() and (path / "config.json").exists():
            model_paths.append(path)
    
    if not model_paths:
        print(f"no models found in {model_dir}")
        return
    
    print(f"found {len(model_paths)} models to evaluate")
    return model_paths

def load_test_dataset(dataset_path, split, sample_limit, audio_column):
    
    print("loading test dataset...")

    # local dataset
    dataset = load_from_disk(dataset_path)
    if split:
        dataset = dataset[split]
    
    # limit samples if requested
    if sample_limit:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    
    # ensure audio is at 16khz
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16000))
    
    print(f"dataset size: {len(dataset)} samples")
    return dataset

def save_model_results(references, predictions, model_name, output_dir, metrics):
    # save per-sample predictions
    sample_df = pd.DataFrame({
        "reference": references,
        "prediction": predictions,
        "model": model_name.name
    })
    sample_file = output_dir / f"predictions_{model_name.parent.name}_{model_name.name}.csv"
    sample_df.to_csv(sample_file, index=False)
            
    print(f"  wer: {metrics['wer']:.2f}%")
    print(f"  cer: {metrics['cer']:.2f}%")

def evaluate_model(model_path, dataset, audio_column, text_column, lang_code = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and processor
    print(f"loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = custom.CustomWhisperProcessor.from_pretrained(model_path)
    
    # create pipeline
    pipe = processor.build_transcription_pipeline(model, lang_code)
    
    # evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []
    
    # process each sample
    for sample in tqdm(dataset, desc="evaluating"):
        audio = sample[audio_column]["array"]
        reference = sample[text_column]
        
        # generate prediction
        result = pipe(audio)
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
        "eval_language": lang_code,
    }, predictions, references

def save_results_summary(results, output_dir):
    results_df = pd.DataFrame(results)
    sort_cols = [c for c in ["train_language", "learning_rate", "wer"] if c in results_df.columns]
    results_df = results_df.sort_values(sort_cols)
        
    summary_file = output_dir / "evaluation_summary.csv"
    results_df.to_csv(summary_file, index=False)
        
    print("evaluation summary:")
    print(results_df.to_string(index=False))
        
    # best model
    best = results_df.iloc[0]
    print(f"\nbest model: {best['model_path']}")
    print(f"  wer: {best['wer']:.2f}%")
    print(f"  cer: {best['cer']:.2f}%")

    plot_eval.plot_lr_lang_comparison_boxplot(results_df, Path(summary_file).parent / 'wer_lr_lang_comparison_boxplot.png', "wer")
    plot_eval.plot_lr_lang_comparison_boxplot(results_df, Path(summary_file).parent / 'cer_lr_lang_comparison_boxplot.png', "cer")


def main():
    parser = argparse.ArgumentParser()
    
    # model paths
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--langs", nargs='+', default=["de","ar","nl"])
    parser.add_argument("--zero-shot", type=bool, default=False, help="whether to do zero-shot evaluation")
    
    # dataset
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_limit", type=int, default=None)

    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="text")
    
    # output
    parser.add_argument("--output_dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # load test dataset
    dataset = load_test_dataset(args.dataset_path, args.split, args.sample_limit, args.audio_column)
    
    # find models
    model_paths = find_model_paths(args.baseline_model, args.model_dir)
    
    # evaluate each model
    results = []
    if args.zero_shot:
        for lang in args.langs:
            print(f"\nevaluating zero-shot for language: {lang}...")
            try:
                metrics, predictions, references = evaluate_model(args.baseline_model, dataset, args.audio_column, args.text_column, lang)
                
                metrics["eval_language"] = lang
                metrics["train_language"] = "zero-shot"
                metrics["learning_rate"] = 0.0
                metrics["seed"] = None

                results.append(metrics)
                
                save_model_results(references, predictions, Path((args.baseline_model+"_"+lang)), args.output_dir, metrics)
                
            except Exception as e:
                print(f"  failed to evaluate: {e}")
                continue
    else:
        for idx, model_path in enumerate(model_paths):
            model_name = model_path if isinstance(model_path, Path) else Path(model_path)
            print(f"\nevaluating {model_name.name}...")
            
            try:
                if idx == 0:
                    metrics, predictions, references = evaluate_model(model_path, dataset, args.audio_column, args.text_column, "de") # baseline always in german
                    #metrics, predictions, references = evaluate_model(model_path, dataset, args.audio_column, args.text_column, "nds") # for ws eval -> baseline always in low german
                else:
                    metrics, predictions, references = evaluate_model(model_path, dataset, args.audio_column, args.text_column)
                
                lang_guess, lr_float, seed = train_eval.parse_lang_lr_seed(model_name)

                # prefer explicit eval lang over guessed
                metrics["eval_language"] = metrics.get("eval_language") or None
                metrics["train_language"] = lang_guess
                metrics["learning_rate"] = lr_float

                metrics["seed"] = seed

                results.append(metrics)
                
                save_model_results(references, predictions, model_name, args.output_dir, metrics)
                
            except Exception as e:
                print(f"  failed to evaluate: {e}")
                continue
    
    # save summary results
    if results:
        save_results_summary(results, args.output_dir)


if __name__ == "__main__":
    main()