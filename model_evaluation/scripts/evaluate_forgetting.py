import torch
import argparse
import logging
from pathlib import Path

from ..corelib.datasets import DatasetConfig, load_test_dataset, calculate_dataset_stats
from ..corelib.experiment import parse_experiment_name, discover_experiments
from ..analysis.performance import evaluate_experiment_folder
from ..utils.report import save_results, save_json, save_csv


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate catastrophic forgetting on Standard German'
    )
    
    # path
    parser.add_argument('--exp_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    
    # dataset - default: German test set from Common Voice
    parser.add_argument('--dataset_mode')
    parser.add_argument('--dataset_path', type=Path)
    
    # HF dataset options (default: German)
    parser.add_argument('--hf_dataset',
                       default='mozilla-foundation/common_voice_17_0')
    parser.add_argument('--hf_config', default='de')
    parser.add_argument('--hf_split', default='test')
    
    # evaluation
    parser.add_argument('--device',
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_language', default='de')
    parser.add_argument('--sample_limit', type=int)
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading test dataset for forgetting evaluation...")
    dataset_config = DatasetConfig(
        mode=args.dataset_mode,
        path=args.dataset_path,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        sample_limit=args.sample_limit
    )
    test_dataset = load_test_dataset(dataset_config)

    dataset_stats = calculate_dataset_stats(test_dataset)
    save_json(dataset_stats, args.output_dir / 'german_test_info.json')
    logger.info(f"German test set: {dataset_stats['num_samples']} samples, "
                f"{dataset_stats['total_duration_hours']:.1f} hours")
    
    experiments = discover_experiments(args.exp_dir)
    logger.info(f"Found {len(experiments)} experiments to evaluate")
    
    all_results = []
    
    for exp_folder in experiments:
        logger.info(f"Evaluating {exp_folder.name} on German...")
        
        try:
            exp_info = parse_experiment_name(exp_folder.name)
        except ValueError as e:
            logger.warning(f"Skipping {exp_folder.name}: {e}")
            continue
        
        result = evaluate_experiment_folder(
            exp_folder, test_dataset,
            args.device, args.force_language
        )
        
        if result is None:
            logger.warning(f"Failed to evaluate {exp_folder.name}")
            continue
        
        metrics, per_sample = result
        
        if metrics is None or per_sample is None:
            logger.warning(f"Invalid results for {exp_folder.name}")
            continue

        # combining results
        combined = {
            'model': exp_info.model,
            'language': exp_info.language,
            'learning_rate': exp_info.learning_rate,
            'seed': exp_info.seed,
            'folder': exp_info.folder,
            'forced_decode_language': args.force_language,
            **metrics
        }
        all_results.append(combined)
        
        save_csv(per_sample,
                args.output_dir / f"de_per_sample_{exp_folder.name}.csv")
        
        logger.info(f"  German WER: {metrics['wer']:.2f}%, CER: {metrics['cer']:.2f}%")
    
    if all_results:
        df, summary = save_results(all_results, args.output_dir, "forgetting")
        
        print("\n" + "-" * 80)
        print("CATASTROPHIC FORGETTING (German Test Set)")
        print("-" * 80)
        
        # forgetting by training language
        print("\nForgetting:")
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            avg_wer = lang_df['wer'].mean()
            std_wer = lang_df['wer'].std()
            print(f"\nModels trained on {lang}:")
            print(f"  German WER: {avg_wer:.2f}% (±{std_wer:.2f}%)")
            
            # comparing with baseline if available
            if lang == 'german':
                print("  (Baseline)")
            else:
                german_baseline = df[df['language'] == 'german']['wer'].mean()
                forgetting = avg_wer - german_baseline
                print(f"  Forgetting: +{forgetting:.2f}% WER vs German baseline")
        
        if summary is not None:
            print("\nDetailed summary:")
            print(summary)
    else:
        logger.warning("No results collected!")


if __name__ == '__main__':

    main()