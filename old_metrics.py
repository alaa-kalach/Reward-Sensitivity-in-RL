"""
metrics.py
Computes the two cross-experiment sensitivity analyses from summary.csv.

With 3 seeds x 3 reward functions x 3 algorithms = 27 runs total.
All sensitivity computations first average final_performance across seeds
for each (algorithm, reward_fn) condition, then compute variance across
conditions. This isolates reward/algorithm effects from seed noise.

  Reward Sensitivity (per algorithm):
    For each algorithm, variance in seed-averaged final_performance across
    its 3 reward conditions. High variance -> algorithm is sensitive to reward design.

  Algorithm Sensitivity (per reward function):
    For each reward type, variance in seed-averaged final_performance across
    its 3 algorithms. High variance -> reward type strongly differentiates algorithms.

  Training Stability (per condition):
    Variance in final_performance across the 3 seeds for each (algo, reward) pair.
    High variance -> that condition is unstable across random initializations.
"""

import csv
import os
import numpy as np
from collections import defaultdict
from typing import Optional


LOGS_DIR = "logs"
SUMMARY_FILE = os.path.join(LOGS_DIR, "summary.csv")


# -----------------------------------------------------------------------
# Summary reader
# -----------------------------------------------------------------------

def load_summary(summary_path: str = SUMMARY_FILE) -> list[dict]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found at '{summary_path}'. "
            "Run all 9 experiments first."
        )

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Cast numeric fields
    for row in rows:
        row["final_performance"]  = _safe_float(row.get("final_performance"))
        row["stability_variance"] = _safe_float(row.get("stability_variance"))
        row["eval_success_rate"]  = _safe_float(row.get("eval_success_rate"))
        row["learning_speed"]     = (
            int(row["learning_speed"])
            if row.get("learning_speed") not in (None, "", "failed")
            else "failed"
        )

    return rows


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------
# Seed averaging — collapses 27 rows into 9 condition means
# -----------------------------------------------------------------------

def average_across_seeds(rows: list[dict]) -> dict:
    groups = defaultdict(list)

    for row in rows:
        perf = row["final_performance"]
        if perf is not None:
            key = (row["algorithm"], row["reward_fn"])
            groups[key].append(perf)

    return {
        key: round(float(np.mean(vals)), 4)
        for key, vals in groups.items()
    }


# -----------------------------------------------------------------------
# Training stability — variance across seeds per condition
# -----------------------------------------------------------------------

def training_stability(rows: list[dict]) -> dict:
    groups = defaultdict(list)

    for row in rows:
        perf = row["final_performance"]
        if perf is not None:
            key = (row["algorithm"], row["reward_fn"])
            groups[key].append((int(row["seed"]), perf))

    result = {}
    for key, seed_perf_pairs in groups.items():
        seed_perf_pairs.sort()  # sort by seed for consistent display
        vals     = [p for _, p in seed_perf_pairs]
        variance = float(np.var(vals)) if len(vals) > 1 else 0.0
        result[key] = {
            "per_seed":   {s: p for s, p in seed_perf_pairs},
            "mean":       round(float(np.mean(vals)), 4),
            "variance":   round(variance, 4),
        }

    return result


# -----------------------------------------------------------------------
# Core sensitivity computations (operate on seed-averaged values)
# -----------------------------------------------------------------------

def reward_sensitivity(rows: list[dict]) -> dict:
    condition_means = average_across_seeds(rows)

    # Group seed-averaged means by algorithm
    grouped = defaultdict(dict)
    for (algo, reward_fn), mean_perf in condition_means.items():
        grouped[algo][reward_fn] = mean_perf

    result = {}
    for algo, reward_map in grouped.items():
        values   = list(reward_map.values())
        variance = float(np.var(values)) if len(values) > 1 else 0.0
        result[algo] = {
            "seed_averaged_perf_per_reward": reward_map,
            "variance": round(variance, 4),
            "interpretation": (
                f"{algo} is {'HIGH' if variance > 500 else 'LOW'} sensitivity "
                f"to reward design  (var={variance:.2f})"
            ),
        }

    return result


def algorithm_sensitivity(rows: list[dict]) -> dict:
    condition_means = average_across_seeds(rows)

    # Group seed-averaged means by reward function
    grouped = defaultdict(dict)
    for (algo, reward_fn), mean_perf in condition_means.items():
        grouped[reward_fn][algo] = mean_perf

    result = {}
    for reward_fn, algo_map in grouped.items():
        values   = list(algo_map.values())
        variance = float(np.var(values)) if len(values) > 1 else 0.0
        result[reward_fn] = {
            "seed_averaged_perf_per_algo": algo_map,
            "variance": round(variance, 4),
            "interpretation": (
                f"{reward_fn} reward causes {'HIGH' if variance > 500 else 'LOW'} "
                f"variance across algorithms  (var={variance:.2f})"
            ),
        }

    return result


# -----------------------------------------------------------------------
# All four metrics, per run
# -----------------------------------------------------------------------

def summarize_all_metrics(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append({
            "run_id":             row["run_id"],
            "algorithm":          row["algorithm"],
            "reward_fn":          row["reward_fn"],
            # Metric 1
            "learning_speed":     row["learning_speed"],
            # Metric 2
            "final_performance":  row["final_performance"],
            # Metric 3
            "eval_success_rate":  row["eval_success_rate"],
            # Metric 4
            "stability_variance": row["stability_variance"],
        })
    return out

# -----------------------------------------------------------------------
# Main report printer
# -----------------------------------------------------------------------


def compute_sensitivity_analysis(summary_path: str = SUMMARY_FILE) -> None:
    rows = load_summary(summary_path)

    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS REPORT")
    print(f"  {len(rows)} runs total  (3 seeds x 3 reward fns x 3 algorithms)")
    print("=" * 65)

    # --- [1] Per-run metrics table ---
    print("\n  [1] Per-Run Metrics (all 27 runs)")
    print(f"  {'Run ID':<34} {'Learn':>7} {'FinalPerf':>10} {'SuccRate%':>10} {'StabVar':>9}")
    print("  " + "-" * 74)
    for m in summarize_all_metrics(rows):
        ls     = m["learning_speed"]
        ls_str = str(ls) if ls != "failed" else "FAILED"
        print(
            f"  {m['run_id']:<34} "
            f"{ls_str:>7} "
            f"{str(m['final_performance']):>10} "
            f"{str(m['eval_success_rate']):>10} "
            f"{str(m['stability_variance']):>9}"
        )

    # --- [2] Training stability across seeds ---
    print("\n  [2] Training Stability — variance across 3 seeds per condition")
    print("      (High variance = outcome is sensitive to random initialization)")
    print()
    stab = training_stability(rows)
    for (algo, reward_fn), data in sorted(stab.items()):
        seed_str = "  ".join(f"seed{s}={p}" for s, p in data["per_seed"].items())
        print(f"    {algo} | {reward_fn}")
        print(f"          {seed_str}")
        print(f"          mean={data['mean']}  variance={data['variance']}")
        print()

    # --- [3] Reward sensitivity ---
    print("  [3] Reward Sensitivity — variance per algorithm across reward functions")
    print("      (Values are seed-averaged. High variance = sensitive to reward design)")
    print()
    rs = reward_sensitivity(rows)
    for algo, data in rs.items():
        print(f"    {algo}  ->  variance = {data['variance']}")
        for rf, val in data["seed_averaged_perf_per_reward"].items():
            print(f"          {rf:<20} seed-avg final_perf = {val}")
        print(f"          {data['interpretation']}")
        print()

    # --- [4] Algorithm sensitivity ---
    print("  [4] Algorithm Sensitivity — variance per reward function across algorithms")
    print("      (Values are seed-averaged. High variance = reward differentiates algos)")
    print()
    als = algorithm_sensitivity(rows)
    for rf, data in als.items():
        print(f"    {rf}  ->  variance = {data['variance']}")
        for algo, val in data["seed_averaged_perf_per_algo"].items():
            print(f"          {algo:<10} seed-avg final_perf = {val}")
        print(f"          {data['interpretation']}")
        print()

    print("=" * 65)
    print("  End of report.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    compute_sensitivity_analysis()