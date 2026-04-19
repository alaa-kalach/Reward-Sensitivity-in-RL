"""
run.py
Single entry point for Phase 1.

Flow:
  1. Smoke test  — 9 combinations, random policy, all metrics tracked
  2. Mock eval   — placeholder success rate per run
  3. Sensitivity analysis report printed from summary.csv
  4. Visualization — comparative mode aligned with the study's design

Visualization modes (chosen interactively):
  A) Fix an algorithm  → watch it run under all 3 reward functions sequentially.
     Answers: "How does reward design change this algorithm's behavior?"

  B) Fix a reward function → watch all 3 algorithms run under it sequentially.
     Answers: "How differently do algorithms behave given the same reward?"

  Each episode prints a per-step breakdown and ends with a comparison table
  across all conditions shown — directly mirroring the report structure.

Usage:
    python run.py
    python run.py --episodes 5
    python run.py --vis-episodes 2
    python run.py --no-visual
"""

import argparse
import time
import gymnasium as gym

from environment import MountainCarWrapper
from reward_functions import REWARD_REGISTRY, get_reward_fn
from logger import make_logger, EpisodeRecord
from metrics import compute_sensitivity_analysis

SEEDS         = [42, 123, 456]   # 3 seeds → 27 total runs (3 seeds × 3 rewards × 3 algos)
ALGORITHMS    = ["PPO", "DQN", "A2C"]
FRAME_DELAY   = 0.02
EVAL_EPISODES = 20


# -----------------------------------------------------------------------
# Shared: one episode, random policy, no rendering
# -----------------------------------------------------------------------

def run_episode(env: MountainCarWrapper) -> tuple:
    obs, _ = env.reset()
    ep_reward, ep_env_reward, ep_steps, reached_goal = 0.0, 0.0, 0, False

    while True:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward     += reward
        ep_env_reward += info["env_reward"]
        ep_steps      += 1
        if terminated:
            reached_goal = True
        if terminated or truncated:
            break

    return ep_steps, ep_reward, ep_env_reward, reached_goal


# -----------------------------------------------------------------------
# Step 1 — Smoke test
# -----------------------------------------------------------------------

def smoke_test(n_episodes: int) -> None:
    print("=" * 65)
    print("  Phase 1 — Smoke Test")
    print(f"  27 runs total  (3 seeds x 3 reward fns x 3 algorithms)")
    print(f"  Seeds      : {SEEDS}")
    print(f"  Threshold  : avg env_reward >= -90 over 10-episode window")
    print(f"  Final perf : mean of last 100 episodes")
    print(f"  Stability  : variance across seeds (computed in report)")
    print(f"  Success    : % eval episodes reaching goal (post-train)")
    print("=" * 65)

    for seed in SEEDS:
        print(f"\n{'=' * 65}")
        print(f"  SEED {seed}")
        print(f"{'=' * 65}")

        for reward_name, reward_fn in REWARD_REGISTRY.items():
            for algo in ALGORITHMS:
                print(f"\n  [{algo} | {reward_name} | seed={seed}]")

                env    = MountainCarWrapper(reward_fn=reward_fn, seed=seed)
                logger = make_logger(algorithm=algo, reward_fn=reward_name, seed=seed)

                for ep in range(1, n_episodes + 1):
                    steps, ep_rew, env_rew, goal = run_episode(env)

                    logger.log(EpisodeRecord(
                        algorithm        = algo,
                        reward_fn        = reward_name,
                        seed             = seed,
                        run_id           = logger.run_id,
                        episode          = ep,
                        total_steps      = 0,
                        episode_steps    = steps,
                        episode_reward   = round(ep_rew, 4),
                        env_reward_sum   = round(env_rew, 4),
                        reached_goal     = goal,
                        rolling_avg_10   = 0.0,
                        learning_reached = False,
                    ))

                    print(
                        f"    ep {ep:02d} | steps={steps:4d} | "
                        f"shaped_r={ep_rew:8.3f} | "
                        f"env_r={env_rew:8.3f} | "
                        f"goal={goal}"
                    )

                mock_successes = sum(
                    1 for _ in range(EVAL_EPISODES) if run_episode(env)[3]
                )
                logger.record_eval_success_rate(mock_successes, EVAL_EPISODES)
                logger.close()

    print("\n" + "=" * 65)
    print(f"  Smoke test complete. 27 CSV files written to logs/")
    print("=" * 65)


# -----------------------------------------------------------------------
# Step 2 — Analysis
# -----------------------------------------------------------------------

def run_analysis() -> None:
    compute_sensitivity_analysis()


# -----------------------------------------------------------------------
# Step 3 — Comparative Visualization
# -----------------------------------------------------------------------

def run_visual_episode(
    env,
    reward_fn,
    label: str,
    episode_num: int,
    n_episodes: int,
) -> dict:
    """
    Renders one episode and prints live step feedback.
    label describes the condition being shown (e.g. 'PPO | dense').
    """
    obs, _ = env.reset()
    ep_reward, ep_env_reward, step, reached_goal = 0.0, 0.0, 0, False
    prev_obs = obs.copy()

    print(f"\n  [{label}]  Episode {episode_num}/{n_episodes}")
    print(f"  {'step':>5}  {'position':>10}  {'velocity':>10}  {'shaped_r':>10}")
    print("  " + "-" * 42)

    while True:
        env.render()
        time.sleep(FRAME_DELAY)

        action = env.action_space.sample()
        next_obs, env_reward, terminated, truncated, _ = env.step(action)

        shaped = reward_fn(
            obs=prev_obs, action=action, next_obs=next_obs,
            done=terminated or truncated, env_reward=env_reward,
        )

        ep_reward     += shaped
        ep_env_reward += env_reward
        step          += 1
        prev_obs       = next_obs.copy()

        if step % 25 == 0:
            pos, vel = next_obs
            print(f"  {step:>5}  {pos:>+10.3f}  {vel:>+10.5f}  {shaped:>+10.4f}")

        if terminated:
            reached_goal = True
            print(f"\n  *** GOAL REACHED at step {step}! ***")

        if terminated or truncated:
            break

    return {
        "label":        label,
        "steps":        step,
        "shaped_r":     round(ep_reward, 4),
        "env_r":        round(ep_env_reward, 4),
        "reached_goal": reached_goal,
    }


def run_comparison_visual(conditions: list[tuple], n_episodes: int) -> None:
    """
    conditions : list of (label, reward_name) pairs to run sequentially.
    Each condition gets its own pygame window, opened and closed in turn.
    After all conditions, prints a side-by-side comparison table.
    """
    all_results = []   # list of lists — one inner list per condition

    for label, reward_name in conditions:
        reward_fn  = get_reward_fn(reward_name)
        results    = []

        print(f"\n{'=' * 65}")
        print(f"  Now showing: {label}")
        print(f"  Reward function: {reward_name}")
        print(f"  Close the window after episode {n_episodes} to continue.")
        print(f"{'=' * 65}")

        env = gym.make("MountainCar-v0", render_mode="human")
        try:
            for ep in range(1, n_episodes + 1):
                r = run_visual_episode(env, reward_fn, label, ep, n_episodes)
                results.append(r)
        except Exception as e:
            print(f"  Window closed: {e}")
        finally:
            env.close()

        all_results.append((label, results))

    # --- Comparison table ---
    print("\n" + "=" * 65)
    print("  VISUAL COMPARISON TABLE")
    print("  (mirrors the sensitivity analysis — same conditions, live behavior)")
    print("=" * 65)
    print(f"  {'Condition':<28}  {'Ep':>3}  {'Steps':>6}  {'Shaped R':>10}  {'Env R':>8}  {'Goal':>5}")
    print("  " + "-" * 63)

    for label, results in all_results:
        for r in results:
            print(
                f"  {label:<28}  "
                f"{r['steps']:>6}  "
                f"{r['shaped_r']:>10.3f}  "
                f"{r['env_r']:>8.3f}  "
                f"{'YES' if r['reached_goal'] else 'no':>5}"
            )
        # Average row per condition
        avg_steps    = sum(r["steps"]    for r in results) / len(results)
        avg_env_r    = sum(r["env_r"]    for r in results) / len(results)
        avg_shaped_r = sum(r["shaped_r"] for r in results) / len(results)
        goal_count   = sum(r["reached_goal"] for r in results)
        print(
            f"  {'  ↳ avg':<28}  "
            f"{avg_steps:>6.1f}  "
            f"{avg_shaped_r:>10.3f}  "
            f"{avg_env_r:>8.3f}  "
            f"{goal_count}/{len(results):>3}"
        )
        print("  " + "-" * 63)

    print("=" * 65)


# -----------------------------------------------------------------------
# Interactive prompt — two comparison modes
# -----------------------------------------------------------------------

def pick_visualization_mode() -> list[tuple]:
    """
    Returns a list of (label, reward_name) pairs for run_comparison_visual().
    Two modes:
      A) Fix algorithm  → vary reward functions  (reward sensitivity axis)
      B) Fix reward     → vary algorithms         (algorithm sensitivity axis)
    """
    print("\n" + "=" * 65)
    print("  VISUALIZATION — Comparative Mode")
    print("=" * 65)
    print("""
  Two views, matching the two sensitivity analyses in the report:

    [A]  Fix an algorithm, vary reward functions
         → See how reward design changes behavior for one algorithm
         → Mirrors: Reward Sensitivity Analysis

    [B]  Fix a reward function, vary algorithms
         → See how algorithms differ under the same reward
         → Mirrors: Algorithm Sensitivity Analysis
    """)

    while True:
        mode = input("  Choose mode [A / B]: ").strip().upper()
        if mode in {"A", "B"}:
            break
        print("  Please enter A or B.")

    if mode == "A":
        print("\n  Which algorithm do you want to fix?")
        for i, algo in enumerate(ALGORITHMS, 1):
            print(f"    [{i}] {algo}")
        while True:
            ch = input("  Enter 1 / 2 / 3: ").strip()
            if ch in {"1", "2", "3"}:
                algo = ALGORITHMS[int(ch) - 1]
                break
            print("  Please enter 1, 2, or 3.")

        # All 3 reward functions for the chosen algorithm
        conditions = [
            (f"{algo} | {rname}", rname)
            for rname in REWARD_REGISTRY.keys()
        ]
        print(f"\n  Will show {algo} under: dense → sparse → potential_based")

    else:  # mode == "B"
        print("\n  Which reward function do you want to fix?")
        reward_names = list(REWARD_REGISTRY.keys())
        for i, rname in enumerate(reward_names, 1):
            print(f"    [{i}] {rname}")
        while True:
            ch = input("  Enter 1 / 2 / 3: ").strip()
            if ch in {"1", "2", "3"}:
                rname = reward_names[int(ch) - 1]
                break
            print("  Please enter 1, 2, or 3.")

        # All 3 algorithms for the chosen reward function
        conditions = [
            (f"{algo} | {rname}", rname)
            for algo in ALGORITHMS
        ]
        print(f"\n  Will show {rname} reward under: PPO → DQN → A2C")

    print("  Each condition opens a separate window. Close it to move to the next.")
    return conditions


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Full Run")
    parser.add_argument("--episodes",     type=int, default=3)
    parser.add_argument("--vis-episodes", type=int, default=2)
    parser.add_argument("--no-visual",    action="store_true")
    args = parser.parse_args()

    # 1. Smoke test
    smoke_test(n_episodes=args.episodes)

    # 2. Sensitivity analysis
    run_analysis()

    # 3. Comparative visualization
    if not args.no_visual:
        conditions = pick_visualization_mode()
        run_comparison_visual(conditions=conditions, n_episodes=args.vis_episodes)


if __name__ == "__main__":
    main()