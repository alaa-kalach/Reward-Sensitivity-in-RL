"""
run.py
Per-algorithm entry point. Each team member runs this for their own algorithm.

Each run covers 9 experiments:
    3 reward functions (dense, sparse, potential_based)
  × 3 random seeds (42, 123, 456)

The three team members run independently:
    python run.py --algorithm PPO
    python run.py --algorithm DQN
    python run.py --algorithm A2C

All 9 CSVs land in logs/ with the standard naming convention
(e.g. PPO_dense_seed42.csv) so the shared summary.csv and plot.py
can aggregate across all three members once everyone is done.

Flow per algorithm
------------------
  1. Training   — N_EPISODES per condition, all metrics tracked via ExperimentLogger.
  2. Evaluation — EVAL_EPISODES post-training episodes per condition, goal-reaching counted.
  3. Report     — Per-algorithm sensitivity summary printed from the 9 completed runs.

Phase 2 — replacing the random policy
--------------------------------------
Each member implements _get_action() in their own agents/ file
(agents/ppo.py, agents/dqn.py, agents/a2c.py) and imports it here.
The stub below is the only line that changes between Phase 1 and Phase 2.

Usage
-----
    python run.py --algorithm PPO
    python run.py --algorithm DQN --episodes 300
    python run.py --algorithm A2C --seeds 42 123
    python run.py --algorithm PPO --episodes 200 --seeds 42
"""

import argparse
import os
import numpy as np

from environment import MountainCarWrapper
from reward_functions import REWARD_REGISTRY, get_reward_fn
from logger import make_logger, EpisodeRecord
from metrics import compute_sensitivity_analysis

# -----------------------------------------------------------------------
# Study constants
# -----------------------------------------------------------------------

ALGORITHMS     = ["PPO", "DQN", "A2C"]
DEFAULT_SEEDS  = [42, 123, 456]
N_EPISODES     = 5000     
EVAL_EPISODES  = 100        # post-training evaluation episodes per condition


# -----------------------------------------------------------------------
# Phase 2 hook — the only function each team member replaces
# -----------------------------------------------------------------------

def _get_action(algorithm: str, obs: np.ndarray, env: MountainCarWrapper) -> int:
    """
    Action selection. Phase 1 uses a random policy to validate the pipeline.

    Phase 2 replacement example (in agents/ppo.py etc.):
        from agents.ppo import PPOAgent
        agent = PPOAgent.load("checkpoints/ppo_final.zip")

        def _get_action(algorithm, obs, env):
            return agent.predict(obs)[0]
    """
    return env.action_space.sample()


# -----------------------------------------------------------------------
# Core episode runner — shared by training and evaluation
# -----------------------------------------------------------------------

def run_episode(
    env: MountainCarWrapper,
    algorithm: str,
) -> tuple[int, float, float, bool]:
    """
    Runs one full episode.

    Returns
    -------
    (episode_steps, shaped_reward_sum, env_reward_sum, reached_goal)

    shaped_reward_sum — what the algorithm actually experienced (for diagnostics)
    env_reward_sum    — raw env reward, used as ground truth in all four metrics
    """
    obs, _ = env.reset()
    ep_shaped, ep_env, steps, reached_goal = 0.0, 0.0, 0, False

    while True:
        action                                         = _get_action(algorithm, obs, env)
        next_obs, shaped_r, terminated, truncated, info = env.step(action)

        ep_shaped += shaped_r
        ep_env    += info["env_reward"]
        steps     += 1
        obs        = next_obs

        if terminated:
            reached_goal = True
        if terminated or truncated:
            break

    return steps, ep_shaped, ep_env, reached_goal


# -----------------------------------------------------------------------
# Single condition: one (reward_fn, seed) pair for the chosen algorithm
# -----------------------------------------------------------------------

def train_one_condition(
    algorithm:   str,
    reward_name: str,
    seed:        int,
    n_episodes:  int,
) -> None:
    """
    Full training + evaluation loop for one (algorithm, reward_fn, seed) triple.
    Writes one CSV to logs/ and appends one row to logs/summary.csv.
    """
    reward_fn = get_reward_fn(reward_name)
    env       = MountainCarWrapper(reward_fn=reward_fn, seed=seed)
    logger    = make_logger(algorithm=algorithm, reward_fn=reward_name, seed=seed)

    print(f"\n  [{algorithm} | {reward_name} | seed={seed}]  "
          f"training for {n_episodes} episodes ...")

    # ---- Training ----
    for ep in range(1, n_episodes + 1):
        steps, ep_shaped, ep_env, goal = run_episode(env, algorithm)

        logger.log(EpisodeRecord(
            algorithm        = algorithm,
            reward_fn        = reward_name,
            seed             = seed,
            run_id           = logger.run_id,
            episode          = ep,
            total_steps      = 0,       # overwritten inside logger.log()
            episode_steps    = steps,
            episode_reward   = round(ep_shaped, 4),
            env_reward_sum   = round(ep_env,    4),
            reached_goal     = goal,
            rolling_avg_10   = 0.0,     # overwritten inside logger.log()
            learning_reached = False,   # overwritten inside logger.log()
        ))

        if ep % 50 == 0 or ep == n_episodes:
            rolling = round(float(np.mean(logger._window_10)), 2) \
                      if logger._window_10 else "n/a"
            print(
                f"    ep {ep:>4}/{n_episodes}  "
                f"steps={steps:4d}  "
                f"env_r={ep_env:8.2f}  "
                f"rolling_avg={rolling:>8}"
            )

    # ---- Post-training evaluation ----
    successes = sum(
        1 for _ in range(EVAL_EPISODES)
        if run_episode(env, algorithm)[3]
    )
    logger.record_eval_success_rate(successes, EVAL_EPISODES)
    logger.close()


# -----------------------------------------------------------------------
# Run all 9 conditions for one algorithm
# -----------------------------------------------------------------------

def run_algorithm(algorithm: str, n_episodes: int, seeds: list[int]) -> None:
    n_conditions = len(seeds) * len(REWARD_REGISTRY)

    print("=" * 65)
    print(f"  ALGORITHM : {algorithm}")
    print(f"  Conditions: {n_conditions}  "
          f"({len(seeds)} seeds × {len(REWARD_REGISTRY)} reward functions)")
    print(f"  Episodes  : {n_episodes} per condition")
    print(f"  Seeds     : {seeds}")
    print(f"  Eval eps  : {EVAL_EPISODES} per condition (post-training)")
    print(f"  Metrics   : learning_speed | final_performance | "
          f"eval_success_rate | stability_variance")
    print("=" * 65)

    for seed in seeds:
        print(f"\n{'─' * 65}")
        print(f"  Seed {seed}")
        print(f"{'─' * 65}")
        for reward_name in REWARD_REGISTRY:
            train_one_condition(
                algorithm   = algorithm,
                reward_name = reward_name,
                seed        = seed,
                n_episodes  = n_episodes,
            )

    print("\n" + "=" * 65)
    print(f"  {algorithm} — all {n_conditions} conditions complete.")
    print(f"  CSVs written to logs/")
    print("=" * 65)


# -----------------------------------------------------------------------
# Per-algorithm sensitivity report
# -----------------------------------------------------------------------

def run_report(algorithm: str) -> None:
    """
    Prints a focused sensitivity report for the runs just completed.
    Filters summary.csv to this algorithm's rows only so partial results
    (e.g. only PPO done so far) still produce a clean, self-contained report.
    """
    from metrics import (
        load_summary, reward_sensitivity,
        training_stability, summarize_all_metrics,
    )

    rows = load_summary()
    rows = [r for r in rows if r["algorithm"] == algorithm]

    if not rows:
        print(f"\n  No summary rows found for {algorithm}. "
              "Did the training complete?\n")
        return

    print("\n" + "=" * 65)
    print(f"  RESULTS — {algorithm}  ({len(rows)} runs)")
    print("=" * 65)

    # Per-run metrics table
    print(f"\n  {'Run ID':<34} {'Learn':>7} {'FinalPerf':>10} "
          f"{'SuccRate%':>10} {'StabVar':>9}")
    print("  " + "─" * 72)
    for m in summarize_all_metrics(rows):
        ls = m["learning_speed"]
        ls = str(ls) if ls != "failed" else "FAILED"
        print(
            f"  {m['run_id']:<34} "
            f"{ls:>7} "
            f"{str(m['final_performance']):>10} "
            f"{str(m['eval_success_rate']):>10} "
            f"{str(m['stability_variance']):>9}"
        )

    # Training stability across seeds
    print(f"\n  Training Stability — variance across seeds per reward function")
    print(f"  (high variance = outcome sensitive to random initialisation)\n")
    stab = training_stability(rows)
    for (algo, reward_fn), data in sorted(stab.items()):
        seed_str = "  ".join(f"seed{s}={p}" for s, p in data["per_seed"].items())
        print(f"    {reward_fn}")
        print(f"      {seed_str}")
        print(f"      mean={data['mean']}  variance={data['variance']}")

    # Reward sensitivity for this algorithm
    print(f"\n  Reward Sensitivity — how much does reward design matter for {algorithm}?")
    print(f"  (variance in seed-averaged final_performance across reward functions)\n")
    rs = reward_sensitivity(rows)
    if algorithm in rs:
        data = rs[algorithm]
        for rf, val in data["seed_averaged_perf_per_reward"].items():
            print(f"    {rf:<20}  seed-avg final_perf = {val}")
        print(f"\n    {data['interpretation']}")

    print("\n" + "=" * 65)
    print(f"  Tip: run plot.py after all three algorithms are complete")
    print(f"  for cross-algorithm comparison figures.")
    print("=" * 65 + "\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MountainCar Reward Sensitivity Study — single-algorithm runner.\n"
            "Each team member runs this once for their own algorithm.\n\n"
            "  Person 1:  python run.py --algorithm PPO\n"
            "  Person 2:  python run.py --algorithm DQN\n"
            "  Person 3:  python run.py --algorithm A2C"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithm", required=True, choices=ALGORITHMS,
        help="Which algorithm to run: PPO, DQN, or A2C"
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Training episodes per condition (default: {N_EPISODES}). "
             "Must be > 100 for metrics to be meaningful."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 123 456)"
    )
    args = parser.parse_args()

    if args.episodes <= 100:
        print(
            f"\n  WARNING: --episodes {args.episodes} is ≤ 100. "
            "final_performance will average fewer than 100 episodes and "
            "stability variance will be unreliable. Recommend ≥ 200, ideally 500+.\n"
        )

    run_algorithm(
        algorithm  = args.algorithm,
        n_episodes = args.episodes,
        seeds      = args.seeds,
    )

    run_report(algorithm=args.algorithm)


if __name__ == "__main__":
    main()