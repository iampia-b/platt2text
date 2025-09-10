import argparse
import json
import pandas as pd
from pathlib import Path


def parse_trainer_state(state_file):
    # extract metrics from trainer_state.json
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    metrics = {
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
    }
    
    # extract training history
    log_history = state.get("log_history", [])
    
    # find best wer and convergence step
    eval_wers = []
    train_losses = []
    
    for entry in log_history:
        if "eval_wer" in entry:
            eval_wers.append({
                "step": entry["step"],
                "wer": entry["eval_wer"],
                "loss": entry.get("eval_loss")
            })
        if "loss" in entry and "eval_wer" not in entry:
            train_losses.append({
                "step": entry["step"],
                "loss": entry["loss"],
                "learning_rate": entry.get("learning_rate")
            })
    
    if eval_wers:
        wer_df = pd.DataFrame(eval_wers)
        metrics["best_wer"] = wer_df["wer"].min()
        metrics["best_wer_step"] = wer_df.loc[wer_df["wer"].idxmin(), "step"]
        metrics["final_wer"] = eval_wers[-1]["wer"]
        metrics["wer_improvement"] = eval_wers[0]["wer"] - metrics["best_wer"]
        
        # calculate convergence (step where wer stops improving significantly)
        wer_values = wer_df["wer"].values
        if len(wer_values) > 3:
            # find where improvement plateaus (< 1% improvement over 3 evals)
            for i in range(3, len(wer_values)):
                recent_improvement = wer_values[i-3] - wer_values[i]
                if recent_improvement < 1.0:  # less than 1% wer improvement
                    metrics["convergence_step"] = wer_df.iloc[i]["step"]
                    break
    
    if train_losses:
        loss_df = pd.DataFrame(train_losses)
        metrics["final_train_loss"] = train_losses[-1]["loss"]
        metrics["min_train_loss"] = loss_df["loss"].min()
    
    return metrics


def analyze_experiment(exp_dir):
    # analyze a single experiment directory
    exp_path = Path(exp_dir)
    
    results = {
        "experiment": exp_path.name,
        "path": str(exp_path)
    }
    
    # find trainer state files
    for subdir in ["best", "final", ""]:
        state_file = exp_path / subdir / "trainer_state.json"
        if state_file.exists():
            metrics = parse_trainer_state(state_file)
            results.update(metrics)
            break
    
    # check for weighted-sum metadata
    ws_meta = exp_path / "best" / "ws_train_meta.pt"
    if ws_meta.exists():
        results["method"] = "weighted_sum"
    else:
        results["method"] = "standard"
    
    # extract experiment parameters from name if structured
    name = exp_path.name
    if "_" in name:
        parts = name.split("_")
        for part in parts:
            if "seed" in part:
                try:
                    results["seed"] = int(part.replace("seed", ""))
                except:
                    pass
            elif part in ["german", "low-german", "nds", "de"]:
                results["language"] = part
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True,
                       help="directory containing experiment folders")
    parser.add_argument("--output_dir", type=Path, required=True)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # find all experiment directories
    exp_dirs = []
    for path in args.exp_dir.iterdir():
        if path.is_dir():
            # check if it contains model files
            if any((path / f).exists() for f in ["config.json", "best/config.json", "final/config.json"]):
                exp_dirs.append(path)
    
    print(f"found {len(exp_dirs)} experiments to analyze")
    
    # analyze each experiment
    all_results = []
    for exp_dir in exp_dirs:
        print(f"analyzing {exp_dir.name}...")
        try:
            results = analyze_experiment(exp_dir)
            all_results.append(results)
        except Exception as e:
            print(f"  failed: {e}")
            continue
    
    if not all_results:
        print("no results to analyze")
        return
    
    # create dataframe
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_dir / "training_analysis.csv", index=False)
    
    # display summary
    print("training analysis summary")
    
    # overall best model
    if "best_wer" in df.columns:
        best_idx = df["best_wer"].idxmin()
        best = df.loc[best_idx]
        print(f"\nbest model overall:")
        print(f"  experiment: {best['experiment']}")
        print(f"  wer: {best['best_wer']:.2f}%")
        if "convergence_step" in best:
            print(f"  convergence: step {best['convergence_step']}")
    
    # comparison by method
    if "method" in df.columns and "best_wer" in df.columns:
        print("\nby training method:")
        method_summary = df.groupby("method")["best_wer"].agg(["mean", "std", "min"])
        print(method_summary)
    
    # comparison by language
    if "language" in df.columns and "best_wer" in df.columns:
        print("\nby language:")
        lang_summary = df.groupby("language")["best_wer"].agg(["mean", "std", "min"])
        print(lang_summary)
    
    # convergence analysis
    if "convergence_step" in df.columns:
        print("\nconvergence analysis:")
        conv_df = df.dropna(subset=["convergence_step"])
        if not conv_df.empty:
            avg_conv = conv_df["convergence_step"].mean()
            print(f"  average convergence: step {avg_conv:.0f}")
            
            if "method" in conv_df.columns:
                conv_by_method = conv_df.groupby("method")["convergence_step"].mean()
                print("\n  by method:")
                for method, step in conv_by_method.items():
                    print(f"    {method}: step {step:.0f}")
    
    # wer improvement analysis
    if "wer_improvement" in df.columns:
        print("\nwer improvement from initial:")
        imp_df = df.dropna(subset=["wer_improvement"])
        if not imp_df.empty:
            avg_imp = imp_df["wer_improvement"].mean()
            print(f"  average improvement: {avg_imp:.2f}%")
            
            if "method" in imp_df.columns:
                imp_by_method = imp_df.groupby("method")["wer_improvement"].mean()
                print("\n  by method:")
                for method, imp in imp_by_method.items():
                    print(f"    {method}: {imp:.2f}%")
    
    # seed variance analysis
    if "seed" in df.columns and "best_wer" in df.columns:
        print("\nseed variance analysis:")
        # group by everything except seed
        grouping_cols = [col for col in ["experiment", "method", "language"] if col in df.columns]
        if grouping_cols:
            for name, group in df.groupby(grouping_cols[0] if len(grouping_cols) == 1 else grouping_cols):
                if len(group) > 1:  # multiple seeds
                    wer_std = group["best_wer"].std()
                    wer_mean = group["best_wer"].mean()
                    print(f"  {name}: {wer_mean:.2f}% ± {wer_std:.2f}%")


if __name__ == "__main__":
    main()