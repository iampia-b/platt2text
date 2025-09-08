import json
import pandas as pd


def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def save_results(
    results,
    output_dir,
    prefix = "results"
    ):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"{prefix}_detailed.csv", index=False)
    
    if 'model' in df.columns and 'language' in df.columns and 'learning_rate' in df.columns:
        summary = df.groupby(['model', 'language', 'learning_rate']).agg({
            col: ['mean', 'std', 'min', 'max']
            for col in ['wer', 'cer'] if col in df.columns
        }).round(2)
        
        summary.to_csv(output_dir / f"{prefix}_summary.csv")
        
        return df, summary
    
    return df, None


def print_results_table(df, title = "RESULTS"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if 'wer' in df.columns:
        # finding best configurations
        best_wer = df.loc[df['wer'].idxmin()]
        print(f"\nBest WER: {best_wer['wer']:.2f}%")
        print(f"  Config: {best_wer.get('model', 'N/A')} "
              f"{best_wer.get('language', 'N/A')} "
              f"lr={best_wer.get('learning_rate', 'N/A')}")
    
    if 'cer' in df.columns:
        best_cer = df.loc[df['cer'].idxmin()]
        print(f"\nBest CER: {best_cer['cer']:.2f}%")
        print(f"  Config: {best_cer.get('model', 'N/A')} "
              f"{best_cer.get('language', 'N/A')} "
              f"lr={best_cer.get('learning_rate', 'N/A')}")