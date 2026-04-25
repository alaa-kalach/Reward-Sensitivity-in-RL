"""
tune_a2c.py
Hyperparameter tuning for A2C.
8 combinations, dense reward, seed 42, 3000 episodes.
"""

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from environment import MountainCarWrapper
from reward_functions import get_reward_fn

EPISODES  = 3000
SEED      = 42
REWARD    = "dense"
MAX_STEPS = 200

# 8 combinations — meaningfully different from each other
GRID = [
    {"learning_rate": 1e-3,  "n_steps": 5,   "ent_coef": 0.01},
    {"learning_rate": 1e-3,  "n_steps": 5,   "ent_coef": 0.1 },
    {"learning_rate": 1e-3,  "n_steps": 128, "ent_coef": 0.01},
    {"learning_rate": 1e-3,  "n_steps": 128, "ent_coef": 0.1 },
    {"learning_rate": 7e-4,  "n_steps": 5,   "ent_coef": 0.01},
    {"learning_rate": 7e-4,  "n_steps": 5,   "ent_coef": 0.1 },
    {"learning_rate": 7e-4,  "n_steps": 128, "ent_coef": 0.01},
    {"learning_rate": 7e-4,  "n_steps": 128, "ent_coef": 0.1 },
]


class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.env_rewards = []
        self.successes   = []
        self._env_reward = 0.0
        self._success    = False
        self._ep_count   = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        self._env_reward += info.get("env_reward", 0.0)
        if info.get("reached_goal", False):
            self._success = True

        if done:
            self.env_rewards.append(self._env_reward)
            self.successes.append(self._success)
            self._env_reward = 0.0
            self._success    = False
            self._ep_count  += 1

        return True


def evaluate(params, combo_num):
    print(f"\n  [{combo_num}/8] lr={params['learning_rate']} | "
          f"n_steps={params['n_steps']} | ent_coef={params['ent_coef']}")

    reward_fn = get_reward_fn(REWARD)
    env       = MountainCarWrapper(reward_fn=reward_fn, seed=SEED)

    model = A2C(
        "MlpPolicy", env,
        seed          = SEED,
        learning_rate = params["learning_rate"],
        n_steps       = params["n_steps"],
        ent_coef      = params["ent_coef"],
        gamma         = 0.99,
        gae_lambda    = 1.0,
        vf_coef       = 0.25,
        max_grad_norm = 0.5,
        verbose       = 0,
    )

    tracker = RewardTracker()
    model.learn(total_timesteps=EPISODES * MAX_STEPS, callback=tracker)
    env.close()

    final_perf   = round(float(np.mean(tracker.env_rewards[-100:])), 2)
    success_rate = round(float(np.mean(tracker.successes[-100:])) * 100, 1)

    print(f"           → final_perf={final_perf} | success_rate={success_rate}%")
    return final_perf, success_rate


def main():
    print("  A2C Hyperparameter Tuning")
    print(f"  Reward: {REWARD} | Seed: {SEED} | Episodes: {EPISODES}")
    print(f"  Total combinations: {len(GRID)}")


    results = []
    for i, params in enumerate(GRID, 1):
        final_perf, success_rate = evaluate(params, i)
        results.append({**params, 
                        "final_perf": final_perf, 
                        "success_rate": success_rate})

    # Sort by success rate first, then final performance
    results.sort(key=lambda x: (x["success_rate"], x["final_perf"]), reverse=True)


    print("  RESULTS — Best to Worst")

    print(f"  {'lr':<10} {'n_steps':<10} {'ent_coef':<10} {'final_perf':<12} {'success%'}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {str(r['learning_rate']):<10} {str(r['n_steps']):<10} "
              f"{str(r['ent_coef']):<10} {str(r['final_perf']):<12} {r['success_rate']}%")

    best = results[0]

    print("  WINNER")
    print("=" * 60)
    print(f"  learning_rate = {best['learning_rate']}")
    print(f"  n_steps       = {best['n_steps']}")
    print(f"  ent_coef      = {best['ent_coef']}")
    print(f"  final_perf    = {best['final_perf']}")
    print(f"  success_rate  = {best['success_rate']}%")
    print("=" * 60)



if __name__ == "__main__":
    main()