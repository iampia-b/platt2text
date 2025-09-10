import argparse
import torch
import pandas as pd
import evaluate
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, pipeline

from core.custom_whisper import CustomWhisperProcessor


def evaluate_on_german(model_path, test_dataset, args):
    device = args.device
    
    # load model
    print(f"  loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = CustomWhisperProcessor.from_pretrained(model_path)
    
    # force german decoding
    processor.tokenizer.set_prefix_tokens(language="de", task="transcribe")
    
    # create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )
    
    # metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []
    
    # evaluate
    for sample in tqdm(test_dataset, desc="testing on german", leave=False):
        audio = sample["audio"]["array"]
        reference = sample["text"]
        
        # force german decoding
        result = pipe(audio, generate_kwargs={"language": "de"})
        prediction = result["text"]
        
        predictions.append(prediction)
        references.append(reference)
    
    # calculate metrics
    wer = wer_metric.compute(predictions=predictions, references=references) * 100
    cer = cer_metric.compute(predictions=predictions, references=references) * 100
    
    return {
        "wer": wer,
        "cer": cer,
        "predictions": predictions,
        "references": references
    }


def main():
    parser = argparse.ArgumentParser()
    
    # model paths
    parser.add_argument("--model_dir", type=Path, required=True,
                       help="directory with trained models")
    parser.add_argument("--baseline_model", type=Path,
                       help="german baseline model for comparison")
    
    # test dataset (german)
    parser.add_argument("--dataset", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--config", default="de")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_limit", type=int, default=500,
                       help="number of test samples")
    
    # settings
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # load german test set
    print("loading german test dataset...")
    test_dataset = load_dataset(args.dataset, args.config, split=args.split)
    if args.sample_limit:
        test_dataset = test_dataset.select(range(min(args.sample_limit, len(test_dataset))))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"test set size: {len(test_dataset)} samples")
    
    # evaluate baseline
    baseline_wer = None
    if args.baseline_model and args.baseline_model.exists():
        print("\nevaluating baseline german model...")
        baseline_results = evaluate_on_german(args.baseline_model, test_dataset, args)
        baseline_wer = baseline_results["wer"]
        print(f"baseline german wer: {baseline_wer:.2f}%")
    
    # find all models to test
    model_paths = []
    for path in args.model_dir.rglob("*/best"):
        if path.is_dir() and (path / "config.json").exists():
            model_paths.append(path)
    
    print(f"\nfound {len(model_paths)} models to evaluate for forgetting")
    
    # evaluate each model
    results = []
    for model_path in model_paths:
        model_name = model_path.parent.name
        print(f"\nevaluating {model_name}...")
        
        # check if this is a low german model
        is_low_german = "nds" in model_name or "low_german" in model_name
        
        try:
            metrics = evaluate_on_german(model_path, test_dataset, args)
            
            result = {
                "model": model_name,
                "model_path": str(model_path),
                "is_low_german": is_low_german,
                "german_wer": metrics["wer"],
                "german_cer": metrics["cer"],
            }
            
            # calculate forgetting if baseline available
            if baseline_wer is not None:
                result["forgetting"] = metrics["wer"] - baseline_wer
            
            results.append(result)
            
            # save predictions
            pred_df = pd.DataFrame({
                "reference": metrics["references"],
                "prediction": metrics["predictions"]
            })
            pred_file = args.output_dir / f"german_predictions_{model_name}.csv"
            pred_df.to_csv(pred_file, index=False)
            
            print(f"  german wer: {metrics['wer']:.2f}%")
            if baseline_wer:
                forgetting = metrics["wer"] - baseline_wer
                print(f"  forgetting: {forgetting:+.2f}% vs baseline")
                
        except Exception as e:
            print(f"  failed: {e}")
            continue
    
    # save and display results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("german_wer")
        df.to_csv(args.output_dir / "forgetting_analysis.csv", index=False)
        
        print("catastrophic forgetting analysis")
        
        # separate by model type
        if "is_low_german" in df.columns:
            low_german_models = df[df["is_low_german"]]
            german_models = df[~df["is_low_german"]]
            
            if not low_german_models.empty:
                print("\nlow german models on german test:")
                print(low_german_models[["model", "german_wer", "forgetting"]].to_string(index=False))
                avg_wer = low_german_models["german_wer"].mean()
                print(f"\naverage wer: {avg_wer:.2f}%")
                
                if "forgetting" in low_german_models.columns:
                    avg_forgetting = low_german_models["forgetting"].mean()
                    print(f"average forgetting: {avg_forgetting:+.2f}%")
            
            if not german_models.empty:
                print("\ngerman models on german test (sanity check):")
                print(german_models[["model", "german_wer"]].to_string(index=False))
        else:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()