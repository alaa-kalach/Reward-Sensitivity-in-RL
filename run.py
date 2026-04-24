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

Algorithm implementations
--------------------------
  PPO and A2C : on-policy (stable-baselines3). Trained via model.learn() using a
                callback that hooks into the existing ExperimentLogger each episode.
  DQN         : off-policy (stable-baselines3). Same approach.

  Hyperparameters are tuned per algorithm (not per reward condition) so that
  performance differences across reward types reflect reward sensitivity only.

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

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback

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
MAX_STEPS      = 200        # MountainCar-v0 max steps per episode


# -----------------------------------------------------------------------
# Hyperparameters — tuned once per algorithm, fixed across all reward types
# This isolates reward sensitivity: any performance difference across reward
# conditions reflects the reward, not incidental hyperparameter mismatches.
# -----------------------------------------------------------------------

HYPERPARAMS = {
    "PPO": dict(
        learning_rate  = 3e-4,
        n_steps        = 1024,
        batch_size     = 64,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,   
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        n_epochs       = 10,
        policy         = "MlpPolicy",
    ),
    "A2C": dict(
        learning_rate  = 7e-4,
        n_steps        = 5,
        gamma          = 0.99,
        gae_lambda     = 1.0,
        ent_coef       = 0.01,
        vf_coef        = 0.25,
        max_grad_norm  = 0.5,
        policy         = "MlpPolicy",
    ),
    "DQN": dict(
        learning_rate        = 1e-4,
        batch_size           = 64,
        gamma                = 0.99,
        buffer_size          = 50_000,
        learning_starts      = 1000,
        target_update_interval = 500,
        exploration_fraction = 0.3,   # longer exploration helps under sparse reward
        exploration_final_eps = 0.05,
        train_freq           = 4,
        policy               = "MlpPolicy",
    ),
}


# -----------------------------------------------------------------------
# SB3 callback — bridges SB3's training loop into ExperimentLogger
# -----------------------------------------------------------------------

class LoggerCallback(BaseCallback):
    """
    Hooks into SB3's training loop to log one row per episode into
    ExperimentLogger using the raw env reward (info["env_reward"]),
    not the shaped reward that the agent is actually trained on.

    This keeps all 9 experiment CSVs on a comparable scale, which is
    required for the reward sensitivity analysis.
    """

    def __init__(self, logger, algorithm: str, reward_fn_name: str, seed: int):
        super().__init__(verbose=0)
        self._logger        = logger
        self._algorithm     = algorithm
        self._reward_fn_name = reward_fn_name
        self._seed          = seed

        # Per-episode accumulators
        self._ep_shaped    = 0.0
        self._ep_env       = 0.0
        self._ep_steps     = 0
        self._ep_reached   = False
        self._ep_count     = 0

    def _on_step(self) -> bool:
        # SB3 passes lists (one entry per parallel env — we use 1 env)
        info       = self.locals["infos"][0]
        shaped_r   = self.locals["rewards"][0]
        done       = self.locals["dones"][0]

        self._ep_shaped  += shaped_r
        self._ep_env     += info.get("env_reward", shaped_r)
        self._ep_steps   += 1

        if info.get("reached_goal", False):
            self._ep_reached = True

        if done:
            self._ep_count += 1
            self._logger.log(EpisodeRecord(
                algorithm        = self._algorithm,
                reward_fn        = self._reward_fn_name,
                seed             = self._seed,
                run_id           = self._logger.run_id,
                episode          = self._ep_count,
                total_steps      = 0,       # overwritten inside logger.log()
                episode_steps    = self._ep_steps,
                episode_reward   = round(self._ep_shaped, 4),
                env_reward_sum   = round(self._ep_env,    4),
                reached_goal     = self._ep_reached,
                rolling_avg_10   = 0.0,     # overwritten inside logger.log()
                learning_reached = False,   # overwritten inside logger.log()
            ))

            if self._ep_count % 50 == 0:
                rolling = round(float(np.mean(self._logger._window_10)), 2) \
                          if self._logger._window_10 else "n/a"
                print(
                    f"    ep {self._ep_count:>4}  "
                    f"steps={self._ep_steps:4d}  "
                    f"env_r={self._ep_env:8.2f}  "
                    f"rolling_avg={rolling:>8}"
                )

            # Reset accumulators for next episode
            self._ep_shaped  = 0.0
            self._ep_env     = 0.0
            self._ep_steps   = 0
            self._ep_reached = False

        return True  # returning False would stop training early


# -----------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------

def make_model(algorithm: str, env: MountainCarWrapper, seed: int):
    """
    Instantiates the SB3 model for the given algorithm with the fixed
    hyperparameters defined above. Seed is passed for reproducibility.
    """
    params = HYPERPARAMS[algorithm].copy()
    policy = params.pop("policy")

    if algorithm == "PPO":
        return PPO(policy, env, seed=seed, verbose=0, **params)
    elif algorithm == "A2C":
        return A2C(policy, env, seed=seed, verbose=0, **params)
    elif algorithm == "DQN":
        return DQN(policy, env, seed=seed, verbose=0, **params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


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

    Training uses SB3's model.learn() with a callback that routes per-episode
    data into ExperimentLogger. Evaluation runs the trained policy deterministically
    for EVAL_EPISODES episodes to measure success rate.

    Writes one CSV to logs/ and appends/updates one row in logs/summary.csv.
    """
    reward_fn  = get_reward_fn(reward_name)
    env        = MountainCarWrapper(reward_fn=reward_fn, seed=seed)
    logger     = make_logger(algorithm=algorithm, reward_fn=reward_name, seed=seed)
    total_steps = n_episodes * MAX_STEPS   # budget in steps for model.learn()

    print(f"\n  [{algorithm} | {reward_name} | seed={seed}]  "
          f"training for {n_episodes} episodes (~{total_steps} steps) ...")

    # ---- Training ----
    model    = make_model(algorithm, env, seed)
    callback = LoggerCallback(logger, algorithm, reward_name, seed)
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=True)

    # ---- Post-training evaluation ----
    # Run the trained policy deterministically (no exploration) for EVAL_EPISODES.
    # Use a fresh env instance with the same reward wrapper so shaped reward is
    # still injected, but we count goal-reaching via info["reached_goal"].
    eval_env  = MountainCarWrapper(reward_fn=reward_fn, seed=seed)
    successes = 0

    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        reached = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            if info.get("reached_goal", False):  
                reached = True
            if terminated or truncated:
                break
        if reached:
            successes += 1

    logger.record_eval_success_rate(successes, EVAL_EPISODES)
    logger.close()
    eval_env.close()
    env.close()


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
