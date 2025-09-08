import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from ..corelib.experiment import parse_experiment_name, discover_experiments
from ..analysis.dynamics import extract_training_metrics, calculate_efficiency_metrics
from ..utils.report import save_csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze Whisper training dynamics')
    
    parser.add_argument('--exp_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = discover_experiments(args.exp_dir)
    logger.info(f"Found {len(experiments)} experiments to analyze")
    
    all_metrics = []
    
    for exp_folder in experiments:
        logger.info(f"Processing {exp_folder.name}...")
        
        try:
            exp_info = parse_experiment_name(exp_folder.name)
        except ValueError as e:
            logger.warning(f"Skipping {exp_folder.name}: {e}")
            continue
        
        metrics = extract_training_metrics(exp_folder)
        efficiency = calculate_efficiency_metrics(metrics)
        
        combined = {
            'model': exp_info.model,
            'language': exp_info.language,
            'learning_rate': exp_info.learning_rate,
            'seed': exp_info.seed,
            'folder': exp_info.folder,
            **metrics,
            **efficiency
        }
        all_metrics.append(combined)
    
    if not all_metrics:
        logger.warning("No metrics extracted!")
        return
    
    df = pd.DataFrame(all_metrics)
    
    save_csv(all_metrics, args.output_dir / 'training_metrics.csv')
    
    print("\n" + "-" * 80)
    print("TRAINING")
    print("-" * 80)
    
    # summary by configuration
    if {'model', 'language', 'learning_rate'}.issubset(df.columns):
        numeric_cols = ['best_wer', 'training_time_hours', 'convergence_step',
                       'wer_improvement', 'stability_score', 'efficiency_score']
        
        # filter to existing columns
        agg_cols = {col: ['mean', 'std', 'min', 'max'] 
                   for col in numeric_cols if col in df.columns}
        
        if agg_cols:
            summary = df.groupby(['model', 'language', 'learning_rate']).agg(
                agg_cols
            ).round(3)
            
            summary.to_csv(args.output_dir / 'training_summary.csv')
            print("\nSummary by configuration:")
            print(summary)
    
    # best configurations by efficiency
    print("\n" + "-" * 80)
    print("EFFICIENCY")
    print("-" * 80)
    
    if 'efficiency_score' in df.columns:
        df_clean = df.dropna(subset=['efficiency_score'])
        if len(df_clean) > 0:
            best_efficient = df_clean.nlargest(5, 'efficiency_score')
            print("\nMost efficient configurations (WER improvement/hour):")
            print(best_efficient[['model', 'language', 'learning_rate', 
                                 'efficiency_score', 'best_wer']].to_string(index=False))
    
    # most stable configurations
    if 'stability_score' in df.columns:
        df_clean = df.dropna(subset=['stability_score'])
        if len(df_clean) > 0:
            most_stable = df_clean.nsmallest(5, 'stability_score')
            print("\nMost stable configurations (low WER variance):")
            print(most_stable[['model', 'language', 'learning_rate',
                              'stability_score', 'best_wer']].to_string(index=False))
    
    # learning rate results
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    
    if 'best_wer' in df.columns:
        # best LR for each model-language combination
        for (model, language), group in df.groupby(['model', 'language']):
            lr_performance = group.groupby('learning_rate')['best_wer'].agg(['mean', 'std'])
            best_lr = lr_performance['mean'].idxmin()
            best_wer = lr_performance.loc[best_lr, 'mean']
            best_std = lr_performance.loc[best_lr, 'std']
            
            print(f"\n{model} - {language}:")
            print(f"  Best LR: {best_lr} (WER: {best_wer:.2f}% ±{best_std:.2f}%)")
            
            # convergence speed
            if 'convergence_ratio' in group.columns:
                conv_ratio = group[group['learning_rate'] == best_lr]['convergence_ratio'].mean()
                if not np.isnan(conv_ratio):
                    print(f"  Convergence: {conv_ratio:.2%} of training")
    
    logger.info(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()