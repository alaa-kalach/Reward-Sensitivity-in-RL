"""
noise_injection.py
Wraps any existing reward function with Gaussian noise.


"""

import numpy as np
from typing import Callable
from reward_functions import dense_reward, sparse_reward


NOISE_SIGMA: float = 0.1




def add_noise(reward_fn: Callable, sigma: float = NOISE_SIGMA) -> Callable:
    
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


    noisy_reward_fn.__name__ = f"{reward_fn.__name__}_noisy"
    return noisy_reward_fn




dense_noisy  = add_noise(dense_reward,  sigma=NOISE_SIGMA)
sparse_noisy = add_noise(sparse_reward, sigma=NOISE_SIGMA)
