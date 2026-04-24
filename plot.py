"""
plot.py
Standalone plotting script. Run this after all (or some) experiments are done.

Generates four figure types, all saved to logs/plots/:

  1. learning_curves_<ALGO>.png
     One figure per algorithm. Three subplots (one per reward function),
     each overlaying the 10-episode rolling average for all seeds.
     → Answers: "How does reward design shape this algorithm's learning?"
     → Mirrors: Reward Sensitivity axis.

  2. algo_comparison_<REWARD>.png
     One figure per reward function. Three subplots (one per algorithm),
     each overlaying all seeds.
     → Answers: "Do algorithms differ under the same reward?"
     → Mirrors: Algorithm Sensitivity axis.

  3. final_performance_bars.png
     Grouped bar chart of seed-averaged final performance for every
     (algorithm, reward) pair. One group per algorithm, one bar per
     reward function.
     → Answers: "Which combination performs best overall?"

  4. stability_heatmap.png
     Heatmap of training stability variance for each (algorithm, reward) pair.
     → Answers: "Which conditions are most sensitive to random initialisation?"

Partial results are handled gracefully — any run whose CSV is missing is
skipped with a warning rather than crashing. This lets you plot after only
one or two algorithms are complete.

Usage
-----
    python plot.py                        # uses logs/ and saves to logs/plots/
    python plot.py --logs-dir my_logs     # custom logs directory
    python plot.py --algorithms PPO DQN   # plot only specific algorithms
    python plot.py --no-show              # save only, never open windows
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------
# Constants — must match run.py / reward_functions.py
# -----------------------------------------------------------------------

ALL_ALGORITHMS   = ["PPO", "DQN", "A2C"]
ALL_REWARD_NAMES = ["dense", "sparse", "potential_based"]
DEFAULT_SEEDS    = [42, 123, 456]

SEED_COLORS      = ["#2196F3", "#FF9800", "#4CAF50"]   # blue / orange / green
REWARD_COLORS    = ["#5C6BC0", "#EF5350", "#26A69A"]   # indigo / red / teal
ALGO_COLORS      = ["#7E57C2", "#FFA726", "#66BB6A"]   # purple / amber / green

PLOTS_DIR        = os.path.join("logs", "plots")
LOGS_DIR         = "logs"


# -----------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------

def _load_run_csv(
    algorithm:   str,
    reward_name: str,
    seed:        int,
    logs_dir:    str = LOGS_DIR,
) -> object:
    """
    Returns parsed rows for one run, or None if the file is missing.
    Missing files are expected when only some team members have finished.
    """
    run_id = f"{algorithm}_{reward_name}_seed{seed}"
    path   = os.path.join(logs_dir, f"{run_id}.csv")

    if not os.path.exists(path):
        print(f"  [skip] {run_id}.csv not found — run not yet complete.")
        return None

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["episode"]        = int(r["episode"])
        r["rolling_avg_10"] = float(r["rolling_avg_10"])
        r["env_reward_sum"] = float(r["env_reward_sum"])

    return rows


def _load_summary(logs_dir: str = LOGS_DIR) -> list[dict]:
    path = os.path.join(logs_dir, "summary.csv")
    if not os.path.exists(path):
        return []

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            row["final_performance"] = float(row["final_performance"])
        except (TypeError, ValueError):
            row["final_performance"] = None
        try:
            row["stability_variance"] = float(row["stability_variance"])
        except (TypeError, ValueError):
            row["stability_variance"] = None

    return rows


# -----------------------------------------------------------------------
# Figure 1 — Learning curves per algorithm
# -----------------------------------------------------------------------

def plot_learning_curves(
    algorithms: list[str],
    seeds:      list[int],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:
    """
    One PNG per algorithm. Three subplots = one per reward function.
    Each subplot overlays the rolling-average learning curve for each seed.
    """
    for algo in algorithms:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(
            f"{algo} — Learning Curves by Reward Function\n"
            "(10-episode rolling average of raw env_reward)",
            fontsize=12, fontweight="bold",
        )

        for ax, reward_name in zip(axes, ALL_REWARD_NAMES):
            plotted = False
            for seed, color in zip(seeds, SEED_COLORS):
                rows = _load_run_csv(algo, reward_name, seed, logs_dir)
                if rows is None:
                    continue
                episodes = [r["episode"]        for r in rows]
                avgs     = [r["rolling_avg_10"] for r in rows]
                ax.plot(episodes, avgs, color=color, alpha=0.85,
                        linewidth=1.5, label=f"seed {seed}")
                plotted = True

            ax.axhline(-90, color="red", linestyle="--",
                       linewidth=0.9, label="threshold (−90)")
            ax.set_title(reward_name, fontsize=10)
            ax.set_xlabel("Episode")
            if ax is axes[0]:
                ax.set_ylabel("Rolling Avg Env Reward")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if not plotted:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=10)

        plt.tight_layout()
        _save(fig, os.path.join(plots_dir, f"learning_curves_{algo}.png"))


# -----------------------------------------------------------------------
# Figure 2 — Algorithm comparison per reward function
# -----------------------------------------------------------------------

def plot_algo_comparison(
    algorithms: list[str],
    seeds:      list[int],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:
    """
    One PNG per reward function. Three subplots = one per algorithm.
    Each subplot overlays all seeds for that algorithm.
    """
    for reward_name in ALL_REWARD_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(
            f"{reward_name} Reward — Algorithm Comparison\n"
            "(10-episode rolling average of raw env_reward)",
            fontsize=12, fontweight="bold",
        )

        for ax, algo in zip(axes, ALL_ALGORITHMS):
            # Grey out algorithms not in the requested list
            if algo not in algorithms:
                ax.set_title(algo, fontsize=10, color="grey")
                ax.text(0.5, 0.5, "Not run yet", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=10)
                ax.grid(True, alpha=0.3)
                continue

            plotted = False
            for seed, color in zip(seeds, SEED_COLORS):
                rows = _load_run_csv(algo, reward_name, seed, logs_dir)
                if rows is None:
                    continue
                episodes = [r["episode"]        for r in rows]
                avgs     = [r["rolling_avg_10"] for r in rows]
                ax.plot(episodes, avgs, color=color, alpha=0.85,
                        linewidth=1.5, label=f"seed {seed}")
                plotted = True

            ax.axhline(-90, color="red", linestyle="--",
                       linewidth=0.9, label="threshold (−90)")
            ax.set_title(algo, fontsize=10)
            ax.set_xlabel("Episode")
            if ax is axes[0]:
                ax.set_ylabel("Rolling Avg Env Reward")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if not plotted:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=10)

        plt.tight_layout()
        _save(fig, os.path.join(plots_dir, f"algo_comparison_{reward_name}.png"))


# -----------------------------------------------------------------------
# Figure 3 — Final performance bar chart
# -----------------------------------------------------------------------

def plot_final_performance_bars(
    algorithms: list[str],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:
    """
    Grouped bar chart: seed-averaged final performance per (algorithm, reward).
    One group per algorithm, one bar per reward function.
    """
    rows = _load_summary(logs_dir)
    if not rows:
        print("  [skip] summary.csv not found — skipping bar chart.")
        return

    # Aggregate across seeds
    groups: dict = defaultdict(list)
    for row in rows:
        if row["final_performance"] is not None and row["algorithm"] in algorithms:
            groups[(row["algorithm"], row["reward_fn"])].append(
                row["final_performance"]
            )

    seed_avgs = {k: float(np.mean(v)) for k, v in groups.items()}

    x     = np.arange(len(ALL_ALGORITHMS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (reward_name, color) in enumerate(zip(ALL_REWARD_NAMES, REWARD_COLORS)):
        values = [
            seed_avgs.get((algo, reward_name), float("nan"))
            for algo in ALL_ALGORITHMS
        ]
        bars = ax.bar(
            x + i * width, values, width,
            label=reward_name, color=color, alpha=0.85, edgecolor="white",
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )

    # Mark algorithms not yet run as "pending"
    for i, algo in enumerate(ALL_ALGORITHMS):
        if algo not in algorithms:
            ax.text(
                x[i] + width, -95, "pending",
                ha="center", va="top", fontsize=8, color="grey",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(ALL_ALGORITHMS)
    ax.set_ylabel("Seed-Averaged Final Performance (env_reward)")
    ax.set_title(
        "Final Performance — All Conditions\n"
        "(bars = seed-averaged mean; higher = better)",
        fontweight="bold",
    )
    ax.legend(title="Reward Function")
    ax.axhline(-90, color="red", linestyle="--", linewidth=0.8,
               label="learning threshold (−90)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(plots_dir, "final_performance_bars.png"))


# -----------------------------------------------------------------------
# Figure 4 — Stability heatmap
# -----------------------------------------------------------------------

def plot_stability_heatmap(
    algorithms: list[str],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:
    """
    Heatmap of training stability variance per (algorithm, reward) condition.
    Rows = reward functions, Columns = algorithms.
    Darker cell = higher variance = less stable across seeds.
    """
    rows = _load_summary(logs_dir)
    if not rows:
        print("  [skip] summary.csv not found — skipping heatmap.")
        return

    # Build stability variance matrix (seed-variance per condition)
    # Re-use the same seed-grouping logic as metrics.py
    groups: dict = defaultdict(list)
    for row in rows:
        if row["final_performance"] is not None:
            groups[(row["algorithm"], row["reward_fn"])].append(
                row["final_performance"]
            )

    matrix = np.full((len(ALL_REWARD_NAMES), len(ALL_ALGORITHMS)), np.nan)
    for i, reward_name in enumerate(ALL_REWARD_NAMES):
        for j, algo in enumerate(ALL_ALGORITHMS):
            vals = groups.get((algo, reward_name), [])
            if len(vals) > 1:
                matrix[i, j] = float(np.var(vals))

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    cmap.set_bad(color="#e0e0e0")   # grey for missing data

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Variance across seeds")

    ax.set_xticks(range(len(ALL_ALGORITHMS)))
    ax.set_yticks(range(len(ALL_REWARD_NAMES)))
    ax.set_xticklabels(ALL_ALGORITHMS)
    ax.set_yticklabels(ALL_REWARD_NAMES)
    ax.set_title(
        "Training Stability — Variance Across Seeds\n"
        "(darker = higher variance = less stable; grey = no data yet)",
        fontweight="bold",
    )

    # Annotate cells
    for i in range(len(ALL_REWARD_NAMES)):
        for j in range(len(ALL_ALGORITHMS)):
            val = matrix[i, j]
            label = f"{val:.1f}" if not np.isnan(val) else "—"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color="black")

    plt.tight_layout()
    _save(fig, os.path.join(plots_dir, "stability_heatmap.png"))


# -----------------------------------------------------------------------
# Save helper
# -----------------------------------------------------------------------

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot learning curves and sensitivity figures from completed experiment logs.\n"
            "Run after one or more algorithms have finished.\n\n"
            "  After PPO only:       python plot.py --algorithms PPO\n"
            "  After all three:      python plot.py\n"
            "  Custom logs dir:      python plot.py --logs-dir my_logs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=ALL_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        help="Which algorithms to include (default: all three)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Seeds that were used in the runs (default: 42 123 456)"
    )
    parser.add_argument(
        "--logs-dir", default=LOGS_DIR,
        help=f"Directory containing the experiment CSVs (default: {LOGS_DIR})"
    )
    parser.add_argument(
        "--plots-dir", default=PLOTS_DIR,
        help=f"Directory to save figures (default: {PLOTS_DIR})"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  GENERATING PLOTS")
    print(f"  Algorithms : {args.algorithms}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Logs dir   : {args.logs_dir}")
    print(f"  Output dir : {args.plots_dir}")
    print("=" * 65)

    print("\n  [1/4] Learning curves per algorithm ...")
    plot_learning_curves(args.algorithms, args.seeds, args.logs_dir, args.plots_dir)

    print("\n  [2/4] Algorithm comparison per reward function ...")
    plot_algo_comparison(args.algorithms, args.seeds, args.logs_dir, args.plots_dir)

    print("\n  [3/4] Final performance bar chart ...")
    plot_final_performance_bars(args.algorithms, args.logs_dir, args.plots_dir)

    print("\n  [4/4] Training stability heatmap ...")
    plot_stability_heatmap(args.algorithms, args.logs_dir, args.plots_dir)

    print("\n" + "=" * 65)
    print(f"  All plots saved to {args.plots_dir}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()