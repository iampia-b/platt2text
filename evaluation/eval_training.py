import argparse
import json
import re

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import evaluation.plot_eval as plot_eval

LR_RE = re.compile(r"^\d+e-?\d+$")  # 3e-5, 1e-6, 5e-05

def parse_lang_lr_seed(run_dir):
    
    if run_dir.name == "best":
        parts = run_dir.parent.name.split("_")
    else:
        parts = run_dir.name.split("_")
    lang = parts[0] if parts else None

    lr = None
    for p in parts:
        if LR_RE.fullmatch(p):
            try:
                lr = float(p)
            except Exception:
                lr = None
            break

    seed = parts[-2] if len(parts) >= 3 and parts[-2].isdigit() else None
    return lang, lr, seed


def find_state_file(run_dir):

    for sub in ("best", "final"):
        p = run_dir / sub / "trainer_state.json"
        if p.exists():
            return p

    best_ckpt, best_step = None, -1
    for ckpt in run_dir.glob("checkpoint-*"):
        if not ckpt.is_dir():
            continue
        try:
            step = int(ckpt.name.split("-")[1])
        except Exception:
            continue
        cand = ckpt / "trainer_state.json"
        if cand.exists() and step > best_step:
            best_step, best_ckpt = step, cand
    return best_ckpt


def parse_trainer_state_to_steps_df(state_file):

    with open(state_file, "r") as f:
        state = json.load(f)

    # per-step rows
    by_step: Dict[int, Dict[str, Any]] = {}
    for entry in state.get("log_history", []) or []:
        step = entry.get("step")
        if step is None:
            continue
        step = int(step)
        row = by_step.setdefault(step, {"step": step})

        # evaluation logs
        if "eval_wer" in entry:
            row["wer"] = entry.get("eval_wer")
            row["eval_loss"] = entry.get("eval_loss")

        # training logs
        if "loss" in entry and "eval_wer" not in entry:
            row["train_loss"] = entry.get("loss")
            row["log_learning_rate"] = entry.get("learning_rate")

    if not by_step:
        df = pd.DataFrame([{
            "step": None,
            "wer": None,
            "eval_loss": None,
            "train_loss": None,
            "log_learning_rate": None,
        }])
    else:
        df = pd.DataFrame(sorted(by_step.values(), key=lambda d: (d["step"] is None, d["step"])))

    # attach trainer-level fields
    df["best_metric"] = state.get("best_metric")
    df["best_model_checkpoint"] = state.get("best_model_checkpoint")
    df["global_step"] = state.get("global_step")
    df["epoch"] = state.get("epoch")

    # compute run summaries from eval rows
    eval_mask = df["wer"].notna()
    if eval_mask.any():
        eval_df = df.loc[eval_mask].sort_values("step")
        best_idx = eval_df["wer"].idxmin()
        best_row = eval_df.loc[best_idx]
        df["best_wer"] = float(best_row["wer"])
        df["best_wer_step"] = int(best_row["step"])
        df["final_wer"] = float(eval_df.iloc[-1]["wer"])
        first_wer = float(eval_df.iloc[0]["wer"])
        df["wer_improvement"] = first_wer - float(best_row["wer"])

        # convergence: absolute improvement < 1.0 across 3 evals
        conv_step = None
        if len(eval_df) > 3:
            vals = eval_df["wer"].to_list()
            steps = eval_df["step"].to_list()
            for i in range(3, len(vals)):
                if (vals[i - 3] - vals[i]) < 1.0:
                    conv_step = int(steps[i])
                    break
        df["convergence_step"] = conv_step
    else:
        df["best_wer"] = None
        df["best_wer_step"] = None
        df["final_wer"] = None
        df["wer_improvement"] = None
        df["convergence_step"] = None

    # summaries
    if "train_loss" in df and df["train_loss"].notna().any():
        train_df = df.loc[df["train_loss"].notna()].sort_values("step")
        df["final_train_loss"] = float(train_df.iloc[-1]["train_loss"])
        df["min_train_loss"] = float(train_df["train_loss"].min())
    else:
        df["final_train_loss"] = None
        df["min_train_loss"] = None

    return df


def analyze_run_dir(run_dir):

    language, lr, seed = parse_lang_lr_seed(run_dir)
    method = "weighted_sum" if (run_dir / "ws_train_meta.pt").exists() else "standard"

    state_file = find_state_file(run_dir)
    if state_file is None:
        return pd.DataFrame([{
            "step": None, "wer": None, "eval_loss": None,
            "train_loss": None, "log_learning_rate": None,
            "best_metric": None, "best_model_checkpoint": None,
            "global_step": None, "epoch": None,
            "best_wer": None, "best_wer_step": None,
            "final_wer": None, "wer_improvement": None,
            "convergence_step": None, "final_train_loss": None,
            "min_train_loss": None,
            "experiment": run_dir.name, "path": str(run_dir),
            "language": language, "learning_rate": lr,
            "seed": seed, "method": method
        }])

    df = parse_trainer_state_to_steps_df(state_file)
    df["experiment"] = run_dir.name
    df["path"] = str(run_dir)
    df["language"] = language
    df["learning_rate"] = lr
    df["seed"] = seed
    df["method"] = method
    return df


def main():
    parser = argparse.ArgumentParser(description="Create step-level and run-level training result tables.")
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Directory containing experiment folders (each with checkpoints).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where to write the CSVs.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # find trainer_state.json
    best_states = list(args.model_dir.rglob("best/trainer_state.json"))
    run_dirs = sorted({p.parent.parent for p in best_states})
    if not run_dirs:
        any_states = list(args.model_dir.rglob("*/trainer_state.json"))
        run_dirs = sorted({p.parent.parent for p in any_states})

    print(f"found {len(run_dirs)} experiments to analyze")

    per_run_frames: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        print(f"analyzing {run_dir.name}...")
        try:
            per_run_frames.append(analyze_run_dir(run_dir))
        except Exception as e:
            print(f"  failed: {e}")

    if not per_run_frames:
        print("no results to analyze")
        return

    # steps table
    steps_df = pd.concat(per_run_frames, ignore_index=True)
    if "learning_rate" in steps_df:
        steps_df["learning_rate"] = pd.to_numeric(steps_df["learning_rate"], errors="coerce").round(8)
    if "step" in steps_df:
        steps_df["step"] = pd.to_numeric(steps_df["step"], errors="coerce")

    steps_csv_path = args.output_dir / "training_results_steps.csv"
    steps_df.to_csv(steps_csv_path, index=False)
    print(f"Saved steps dataset with {len(steps_df):,} rows -> {steps_csv_path}")

    # runs table
    runs_df = (
        steps_df.sort_values(["experiment", "step"])
        .groupby("experiment", as_index=False)
        .agg(
            path=("path", "first"),
            language=("language", "first"),
            learning_rate=("learning_rate", "first"),
            seed=("seed", "first"),
            method=("method", "first"),
            best_metric=("best_metric", "first"),
            best_model_checkpoint=("best_model_checkpoint", "first"),
            global_step=("global_step", "first"),
            epoch=("epoch", "first"),
            best_wer=("best_wer", "first"),
            best_wer_step=("best_wer_step", "first"),
            final_wer=("final_wer", "first"),
            wer_improvement=("wer_improvement", "first"),
            convergence_step=("convergence_step", "first"),
            final_train_loss=("final_train_loss", "first"),
            min_train_loss=("min_train_loss", "first"),
            n_eval_points=("wer", lambda s: s.notna().sum()),
            n_train_points=("train_loss", lambda s: s.notna().sum()),
            min_step=("step", "min"),
            max_step=("step", "max"),
        )
    )

    runs_csv_path = args.output_dir / "training_results_runs.csv"
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"Saved runs dataset with {len(runs_df):,} rows -> {runs_csv_path}")

    plots_output_path = args.output_dir / "plots"
    plots_output_path.mkdir(parents=True, exist_ok=True)
    plot_eval.create_all_plots_from_steps_csv(
        steps_csv_path,
        plots_output_path
    )

if __name__ == "__main__":
    main()
