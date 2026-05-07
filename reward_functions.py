"""
reward_functions.py
Defines the three reward functions used across all 9 experiments.
"""

import numpy as np


GOAL_POSITION: float = 0.5


# 1. Dense Reward

def dense_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:

    pos, vel       = float(obs[0]), float(obs[1])
    next_pos, nvel = float(next_obs[0]), float(next_obs[1])

  
    progress_reward = (next_pos - pos) * 10.0


    speed_reward = abs(nvel) * 0.5

    
    goal_bonus = 50.0 if next_pos >= GOAL_POSITION else 0.0

 
    return float(env_reward) + progress_reward + speed_reward + goal_bonus


# 2. Sparse Reward

def sparse_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:

    return 1.0 if next_obs[0] >= GOAL_POSITION else 0.0


# 3. Potential-Based Reward  

def _potential(obs: np.ndarray) -> float:
    position = obs[0]
    velocity = obs[1]
    return (position + 0.5 * (velocity ** 2)) * 10.0    # energy-inspired shaping


def potential_based_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    done: bool,
    env_reward: float,
) -> float:

    gamma: float = 0.99
    shaping = gamma * _potential(next_obs) - _potential(obs)
    return env_reward + shaping


# Registry - maps experiment config strings to callables

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