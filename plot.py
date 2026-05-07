"""
plot.py
Standalone plotting script. Run this after all (or some) experiments are done.

Generates four figure types, all saved to logs/plots
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

ALL_ALGORITHMS   = ["PPO", "DQN", "A2C"]
ALL_REWARD_NAMES = ["dense", "sparse", "potential_based"]
DEFAULT_SEEDS    = [42, 123, 456]

SEED_COLORS      = ["#2196F3", "#FF9800", "#4CAF50"]
REWARD_COLORS    = ["#5C6BC0", "#EF5350", "#26A69A"]
ALGO_COLORS      = ["#7E57C2", "#FFA726", "#66BB6A"]

PLOTS_DIR        = os.path.join("logs", "plots")
LOGS_DIR         = "logs"

THRESHOLD: float = -110.0   # MountainCar-v0 success threshold


# CSV helpers

def _load_run_csv(
    algorithm:   str,
    reward_name: str,
    seed:        int,
    logs_dir:    str = LOGS_DIR,
) -> object:

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


# Figure 1 - Learning curves per algorithm

def plot_learning_curves(
    algorithms: list[str],
    seeds:      list[int],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:

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

            ax.axhline(THRESHOLD, color="red", linestyle="--",
                       linewidth=0.9, label=f"threshold ({THRESHOLD:.0f})")
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


# Figure 2 - Algorithm comparison per reward function

def plot_algo_comparison(
    algorithms: list[str],
    seeds:      list[int],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:

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

            ax.axhline(THRESHOLD, color="red", linestyle="--",
                       linewidth=0.9, label=f"threshold ({THRESHOLD:.0f})")
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


# Figure 3 - Final performance bar chart

def plot_final_performance_bars(
    algorithms: list[str],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:

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
        "(bars = seed-averaged mean)",
        fontweight="bold",
    )
    ax.legend(title="Reward Function")
    ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=0.8,
               label=f"threshold ({THRESHOLD:.0f})")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(plots_dir, "final_performance_bars.png"))


# Figure 4 - Stability heatmap

def plot_stability_heatmap(
    algorithms: list[str],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:

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
        "(darker = higher variance; grey = no data yet)",
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


# Figure 5 - Noise impact: clean vs noisy side by side per algorithm

def plot_noise_comparison(
    algorithms: list[str],
    logs_dir:   str = LOGS_DIR,
    plots_dir:  str = PLOTS_DIR,
) -> None:

    rows = _load_summary(logs_dir)
    if not rows:
        print("  [skip] summary.csv not found — skipping noise comparison.")
        return

    pairs = [("dense", "dense_noisy"), ("sparse", "sparse_noisy")]
    pair_colors = [("#5C6BC0", "#9FA8DA"), ("#EF5350", "#EF9A9A")]

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5), sharey=True)
    if len(algorithms) == 1:
        axes = [axes]

    fig.suptitle(
        "Noise Impact — Clean vs Noisy Reward\n"
        "(seed-averaged final performance)",
        fontsize=12, fontweight="bold",
    )

    # Aggregate seed averages
    groups: dict = defaultdict(list)
    for row in rows:
        if row["final_performance"] is not None:
            groups[(row["algorithm"], row["reward_fn"])].append(row["final_performance"])
    seed_avgs = {k: float(np.mean(v)) for k, v in groups.items()}

    for ax, algo in zip(axes, algorithms):
        x      = np.arange(len(pairs))
        width  = 0.35

        for i, ((clean_name, noisy_name), (c_color, n_color)) in enumerate(zip(pairs, pair_colors)):
            clean_val = seed_avgs.get((algo, clean_name), float("nan"))
            noisy_val = seed_avgs.get((algo, noisy_name), float("nan"))

            b1 = ax.bar(i - width / 2, clean_val, width, label=clean_name if algo == algorithms[0] else "",
                        color=c_color, alpha=0.9, edgecolor="white")
            b2 = ax.bar(i + width / 2, noisy_val, width, label=noisy_name if algo == algorithms[0] else "",
                        color=n_color, alpha=0.9, edgecolor="white")

            for bar, val in [(b1, clean_val), (b2, noisy_val)]:
                if not np.isnan(val):
                    ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                            bar[0].get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(algo, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["dense pair", "sparse pair"], fontsize=9)
        ax.set_xlabel("Reward Pair")
        if ax is axes[0]:
            ax.set_ylabel("Seed-Averaged Final Performance")
        ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.3)

    fig.legend(loc="upper right", fontsize=8, title="Reward variant")
    plt.tight_layout()
    _save(fig, os.path.join(plots_dir, "noise_comparison.png"))


# Figure 6 - Curriculum schedule visualization

def plot_curriculum_schedule(
    n_episodes:     int   = 5000,
    transition_end: int   = 2500,
    plots_dir:      str   = PLOTS_DIR,
) -> None:

    episodes = np.arange(0, n_episodes + 1)
    alpha    = np.minimum(episodes / transition_end, 1.0)
    dense_w  = 1.0 - alpha
    sparse_w = alpha

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.fill_between(episodes, dense_w,  alpha=0.25, color="#5C6BC0", label="Dense weight  (1 − α)")
    ax.fill_between(episodes, sparse_w, alpha=0.25, color="#EF5350", label="Sparse weight  (α)")
    ax.plot(episodes, dense_w,  color="#5C6BC0", linewidth=2)
    ax.plot(episodes, sparse_w, color="#EF5350", linewidth=2)

    ax.axvline(transition_end, color="grey", linestyle="--", linewidth=1.2,
               label=f"Transition end (ep {transition_end})")

    # Annotate key phases
    ax.text(transition_end * 0.25, 0.55, "Dense phase\n(warm-start)",
            ha="center", fontsize=9, color="#3949AB")
    ax.text(transition_end * 0.75, 0.45, "Transition",
            ha="center", fontsize=9, color="#555")
    ax.text(transition_end + (n_episodes - transition_end) * 0.5, 0.55,
            "Sparse phase\n(true objective)",
            ha="center", fontsize=9, color="#C62828")

    ax.set_xlim(0, n_episodes)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Training Episode", fontsize=10)
    ax.set_ylabel("Interpolation Weight", fontsize=10)
    ax.set_title(
        "Curriculum Schedule — Linear Transition from Dense to Sparse\n"
        r"$R_{curriculum} = (1 - \alpha) \cdot R_{dense} + \alpha \cdot R_{sparse}$",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(plots_dir, "curriculum_schedule.png"))


# Save helper

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# Main

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

    print("\n  [3/6] Final performance bar chart ...")
    plot_final_performance_bars(args.algorithms, args.logs_dir, args.plots_dir)

    print("\n  [4/6] Training stability heatmap ...")
    plot_stability_heatmap(args.algorithms, args.logs_dir, args.plots_dir)

    print("\n  [5/6] Noise impact comparison ...")
    plot_noise_comparison(args.algorithms, args.logs_dir, args.plots_dir)

    print("\n  [6/6] Curriculum schedule visualization ...")
    plot_curriculum_schedule(plots_dir=args.plots_dir)

    print("\n" + "=" * 65)
    print(f"  All 6 plots saved to {args.plots_dir}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()