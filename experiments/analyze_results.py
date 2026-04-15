"""
Analyze batch experiment outputs and generate summary tables/plots.

Usage:
  python experiments/analyze_results.py --run-dir batch_results/noise_stress_orca_goap
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing columns: {missing}")


def analyze(run_dir: str) -> None:
    summary_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(summary_path)
    _ensure_columns(
        df,
        [
            "algorithm",
            "num_dynamic_obstacles",
            "sensor_noise",
            "execution_noise",
            "coverage",
            "collisions",
            "dynamic_collisions",
            "performance_score",
            "success",
        ],
    )

    metrics = ["coverage", "collisions", "dynamic_collisions", "performance_score", "success"]

    by_algo = df.groupby("algorithm")[metrics].mean().round(4).reset_index()
    by_dyn = df.groupby("num_dynamic_obstacles")[metrics].mean().round(4).reset_index()
    by_sensor = df.groupby("sensor_noise")[metrics].mean().round(4).reset_index()
    by_exec = df.groupby("execution_noise")[metrics].mean().round(4).reset_index()

    by_algo.to_csv(os.path.join(out_dir, "mean_by_algorithm.csv"), index=False)
    by_dyn.to_csv(os.path.join(out_dir, "mean_by_dynamic_obstacles.csv"), index=False)
    by_sensor.to_csv(os.path.join(out_dir, "mean_by_sensor_noise.csv"), index=False)
    by_exec.to_csv(os.path.join(out_dir, "mean_by_execution_noise.csv"), index=False)

    plt.figure(figsize=(8, 4))
    for algo, g in df.groupby("algorithm"):
        x = sorted(g["num_dynamic_obstacles"].unique())
        y = [g[g["num_dynamic_obstacles"] == xi]["coverage"].mean() for xi in x]
        plt.plot(x, y, marker="o", label=algo)
    plt.xlabel("num_dynamic_obstacles")
    plt.ylabel("mean coverage")
    plt.title("Coverage vs Dynamic Obstacles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage_vs_dynamic_obstacles.png"), dpi=160)
    plt.close()

    pivot = (
        df.groupby(["sensor_noise", "execution_noise"])["dynamic_collisions"]
        .mean()
        .unstack("execution_noise")
        .sort_index()
    )
    plt.figure(figsize=(6, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("execution_noise")
    plt.ylabel("sensor_noise")
    plt.title("Mean Dynamic Collisions Heatmap")
    plt.colorbar(label="dynamic_collisions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dynamic_collisions_heatmap.png"), dpi=160)
    plt.close()

    print(f"Analysis complete: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment summary.csv")
    parser.add_argument("--run-dir", required=True, help="Run directory under batch_results")
    args = parser.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
