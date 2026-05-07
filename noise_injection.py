"""
noise_injection.py
Wraps any existing reward function with Gaussian noise.

Motivation
----------
Real-world reward signals are rarely perfect — sensors drift, human feedback
is inconsistent, and proxy rewards are approximations. By injecting calibrated
Gaussian noise into our three base reward functions, we test how robust each
algorithm is when the reward signal is corrupted.

Research question this answers
-------------------------------
Does reward noise affect PPO, A2C, and DQN equally, or do some algorithms
degrade faster under a corrupted reward signal?

Design
------
add_noise(reward_fn, sigma) wraps ANY reward function with signature:
    reward_fn(obs, action, next_obs, done, env_reward) -> float

and returns a NEW function with the same signature that adds ε ~ N(0, sigma)
to whatever the base reward function returns.

This keeps noise injection decoupled from reward logic — we don't touch
reward_functions.py at all.

Pre-built variants
------------------
Two noisy variants are exported for use in run.py:
    dense_noisy    : dense reward   + N(0, NOISE_SIGMA)
    sparse_noisy   : sparse reward  + N(0, NOISE_SIGMA)

We don't add noise to potential_based because its values are already small
(typically in [-1, 1]) — the same sigma would overwhelm the signal entirely,
making the comparison unfair. Dense and sparse are the more informative pair.

Usage in run.py
---------------
    from noise_injection import dense_noisy, sparse_noisy, NOISE_SIGMA
    REWARD_REGISTRY["dense_noisy"]  = dense_noisy
    REWARD_REGISTRY["sparse_noisy"] = sparse_noisy
"""

import numpy as np
from typing import Callable
from reward_functions import dense_reward, sparse_reward

# -----------------------------------------------------------------------
# Noise level — single source of truth
# -----------------------------------------------------------------------

# sigma = 0.1 is deliberately modest:
#   - Dense reward values are typically in [-1, +50] range
#   - Sparse reward values are 0 or 1
#   - At sigma=0.1, noise is meaningful but not catastrophic for dense,
#     and large relative to the sparse signal (100% of a 0.1-magnitude step)
# This asymmetry is intentional — it lets us observe differential sensitivity.
NOISE_SIGMA: float = 0.1


# -----------------------------------------------------------------------
# Core wrapper
# -----------------------------------------------------------------------

def add_noise(reward_fn: Callable, sigma: float = NOISE_SIGMA) -> Callable:
    """
    Wraps a reward function with additive Gaussian noise.

    Parameters
    ----------
    reward_fn : Callable
        Any reward function with the standard signature:
        (obs, action, next_obs, done, env_reward) -> float

    sigma : float
        Standard deviation of the Gaussian noise N(0, sigma).
        Higher sigma = more corrupted reward signal.

    Returns
    -------
    Callable
        A new reward function with the same signature that returns:
        reward_fn(obs, action, next_obs, done, env_reward) + N(0, sigma)

    Example
    -------
        noisy_dense = add_noise(dense_reward, sigma=0.1)
        r = noisy_dense(obs, action, next_obs, done, env_reward)
    """
    def noisy_reward_fn(
        obs,
        action,
        next_obs,
        done,
        env_reward,
    ) -> float:
        base_reward = reward_fn(
            obs=obs,
            action=action,
            next_obs=next_obs,
            done=done,
            env_reward=env_reward,
        )
        noise = np.random.normal(loc=0.0, scale=sigma)
        return float(base_reward + noise)

    # Preserve the original function's name with a suffix so logger.py
    # writes a meaningful reward_fn string into the CSV (e.g. "dense_noisy")
    noisy_reward_fn.__name__ = f"{reward_fn.__name__}_noisy"
    return noisy_reward_fn


# -----------------------------------------------------------------------
# Pre-built noisy variants — import these directly into run.py
# -----------------------------------------------------------------------

dense_noisy  = add_noise(dense_reward,  sigma=NOISE_SIGMA)
sparse_noisy = add_noise(sparse_reward, sigma=NOISE_SIGMA)
