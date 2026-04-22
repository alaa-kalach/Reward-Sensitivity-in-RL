"""
environment.py
Wraps MountainCar-v0 with seeding, reward-function injection, and track steps.
All 9 experiments share this single wrapper for consistency.
"""

import gymnasium as gym
import numpy as np
from typing import Callable, Optional, Tuple


class MountainCarWrapper(gym.Wrapper):
    """
    A deterministic, reward-injectable wrapper around MountainCar-v0.

    Accepts any reward function with signature:
        reward_fn(obs, action, next_obs, done, env_reward) -> float
    This keeps the environment logic decoupled from reward design.
    """

    def __init__(
        self,
        reward_fn: Callable,
        seed: int = 42,
        max_steps: int = 999,
    ):
        env = gym.make("MountainCar-v0")
        super().__init__(env)

        self.reward_fn = reward_fn
        self.seed_val = seed
        self.max_steps = max_steps

        self._step_count: int = 0
        self._episode_count: int = 0
        self._prev_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        kwargs.setdefault("seed", self.seed_val + self._episode_count)
        obs, info = self.env.reset(**kwargs)

        self._step_count = 0
        self._episode_count += 1
        self._prev_obs = obs.copy()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Inject custom reward
        reward = self.reward_fn(
            obs=self._prev_obs,
            action=action,
            next_obs=next_obs,
            done=terminated or truncated,
            env_reward=env_reward,
        )

        self._step_count += 1
        self._prev_obs = next_obs.copy()

        # Enforce episode length cap
        if self._step_count >= self.max_steps:
            truncated = True

        info["step"] = self._step_count
        info["episode_num"] = self._episode_count
        info["env_reward"] = env_reward   

        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self.observation_space.shape[0]   # 2: position, velocity

    @property
    def n_actions(self) -> int:
        return self.action_space.n               # 3: push left, no-op, push right

    def __repr__(self) -> str:
        return (
            f"MountainCarWrapper("
            f"reward_fn={self.reward_fn.__name__}, "
            f"seed={self.seed_val})"
        )
