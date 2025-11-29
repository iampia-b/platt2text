import os
import argparse
from pathlib import Path
from datasets import load_dataset, Audio

import evaluation.eval_error_rates as eval_model
import evaluation.eval_training as eval_train


def load_local_cv_dataset(root_dir, split, sample_limit= 500, seed = 42, audio_column = "audio", text_column = "sentence"):

    # load TSV metadata as a dataset
    split_tsv = split+".tsv"
    tsv_path = os.path.join(root_dir, split_tsv)

    ds_dict = load_dataset(
        "csv",
        data_files={"data": tsv_path},
        delimiter="\t",
    )
    ds = ds_dict["data"] 

    def _add_fields(batch):

        abs_paths = [os.path.join(root_dir, "clips", rel) for rel in batch["path"]]
        transcripts = batch["sentence"]

        batch[audio_column] = abs_paths
        batch[text_column] = transcripts
        return batch

    ds = ds.map(_add_fields, batched=True)
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    ds = ds.shuffle(seed=seed)
    if sample_limit is not None:
        n = min(sample_limit, len(ds))
        ds = ds.select(range(n))

    keep_cols = [c for c in [audio_column, text_column] if c in ds.column_names]
    ds = ds.select_columns(keep_cols)

    print(f"{split_tsv} loaded from {root_dir}")
    print(f"final size: {len(ds)} samples")
    return ds


def main():
    parser = argparse.ArgumentParser()
    
    # model paths
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--baseline_model", required=True)
    
    # test dataset (german)
    parser.add_argument("--dataset_path", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_limit", type=int, default=900)

    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--text_column", default="sentence")
    
    # output
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # load german test set
    test_dataset = load_local_cv_dataset(args.dataset_path, args.split, args.sample_limit, 42, args.audio_column, args.text_column)

    # find models
    model_paths = eval_model.find_model_paths(args.baseline_model, args.model_dir)
    
    # evaluate each model
    results = []

    for model_path in model_paths:
        model_name = model_path if isinstance(model_path, Path) else Path(model_path)
        print(f"\nevaluating {model_name.name}...")
        
        try:
            metrics, predictions, references = eval_model.evaluate_model(model_path, test_dataset, args.audio_column, args.text_column, "de")
            lang_guess, lr_float, lr_str = eval_train.parse_lang_lr_seed(model_name)

            # prefer explicit eval lang over guessed
            metrics["eval_language"] = metrics.get("eval_language") or None
            metrics["train_language"] = lang_guess
            metrics["learning_rate"] = lr_float

            results.append(metrics)
            
            eval_model.save_model_results(references, predictions, model_name, args.output_dir, metrics)
            
        except Exception as e:
            print(f"  failed to evaluate: {e}")
            continue
    
    # save summary results
    if results:
        eval_model.save_results_summary(results, args.output_dir)


if __name__ == "__main__":
    main()