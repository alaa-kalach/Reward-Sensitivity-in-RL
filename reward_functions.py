"""
reward_functions.py
Defines the three reward functions used across all 9 experiments.

Each function follows the unified signature expected by MountainCarWrapper:
    reward_fn(obs, action, next_obs, done, env_reward) -> float

MountainCar-v0 state space:
    obs[0] : position   in [-1.2,  0.6]   (goal: >= 0.5)
    obs[1] : velocity   in [-0.07, 0.07]
"""

import numpy as np

# Goal threshold matches the environment's internal termination condition
GOAL_POSITION: float = 0.5


# ----------------------------------------------------------------------
# 1. Dense Reward
# ----------------------------------------------------------------------

def dense_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:
    """
    Shaped reward that provides continuous feedback on every step.

    Components:
      - Position bonus : rewards being further right (closer to goal)
                         uses absolute position so signal is always meaningful
      - Speed bonus    : rewards high absolute velocity (building momentum)
      - Goal bonus     : large one-time reward on success
    """
    # IMPORTANT: keep the shaped reward "goal-directed".
    # The original version added a large positive baseline every step (height/speed),
    # which can make "survive 200 steps without reaching the goal" optimal under the
    # *shaped* reward, even though env reward stays at -200.
    pos, vel       = float(obs[0]), float(obs[1])
    next_pos, nvel = float(next_obs[0]), float(next_obs[1])

    # Reward progress to the right (potential-like but simpler/stronger signal).
    progress_reward = (next_pos - pos) * 10.0

    # Small momentum term to encourage building speed without dominating.
    speed_reward = abs(nvel) * 0.5

    # One-time goal bonus to strongly prefer termination by success.
    goal_bonus = 50.0 if next_pos >= GOAL_POSITION else 0.0

    # Combine with true environment reward (-1 per step) to keep incentives aligned.
    return float(env_reward) + progress_reward + speed_reward + goal_bonus


# ----------------------------------------------------------------------
# 2. Sparse Reward
# ----------------------------------------------------------------------

def sparse_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:
    """
    Binary reward: +1 on reaching the goal, 0 everywhere else.

    This is the hardest exploration challenge — the agent receives
    no intermediate guidance, making it the ideal baseline for comparing
    how much reward shaping actually helps each algorithm.
    """
    return 1.0 if next_obs[0] >= GOAL_POSITION else 0.0


# ----------------------------------------------------------------------
# 3. Potential-Based Reward  (theory: Ng et al., 1999)
# ----------------------------------------------------------------------

def _potential(obs: np.ndarray) -> float:
    """
    Potential function Φ(s).
    Higher value = closer to goal, combining position and kinetic energy.
    """
    position = obs[0]
    velocity = obs[1]
    return position + 0.5 * (velocity ** 2)     # energy-inspired shaping


def potential_based_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:
    """
    F(s, s') = γ·Φ(s') - Φ(s)  added on top of the original env reward.

    Potential-based shaping is theoretically guaranteed not to alter the
    optimal policy (policy invariance), making it the most principled of
    the three designs.

    γ (discount factor here) is fixed at 0.99 to match typical RL training.
    """
    gamma: float = 0.99
    shaping = gamma * _potential(next_obs) - _potential(obs)
    return env_reward + shaping * 10.0 #added a multiplier x10 


# ----------------------------------------------------------------------
# Registry — maps experiment config strings to callables
# ----------------------------------------------------------------------

REWARD_REGISTRY: dict = {
    "dense":           dense_reward,
    "sparse":          sparse_reward,
    "potential_based": potential_based_reward,
}


def get_reward_fn(name: str):
    """Retrieve a reward function by its registry key."""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Choose from: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]