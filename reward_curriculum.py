"""
reward_curriculum.py
Implements a linear reward curriculum that transitions from dense to sparse.

Motivation
----------
Training directly on sparse rewards is hard — the agent rarely stumbles on
the goal and gets almost no learning signal. Training on dense rewards is
easier, but the shaped signal may not match the true objective.

A curriculum starts with the dense reward (easy, informative) and gradually
shifts toward the sparse reward (hard, true objective) as training progresses.
This tests whether warm-starting with a rich signal and annealing it away
outperforms training on either reward alone — a natural extension of our
3-reward comparison.

Research question this answers
-------------------------------
Does a curriculum schedule outperform training on sparse alone?
Does it match or beat dense alone?
Do PPO, A2C, and DQN benefit equally from curriculum learning?

Design
------
The curriculum interpolates linearly between dense and sparse:

    alpha(episode) = min(episode / TRANSITION_END, 1.0)

    R_curriculum = (1 - alpha) * R_dense  +  alpha * R_sparse

At episode 0      : alpha = 0.0  →  pure dense reward
At episode halfway: alpha = 0.5  →  50/50 mix
At TRANSITION_END : alpha = 1.0  →  pure sparse reward (stays there)

This linear schedule is the simplest principled choice — it makes no
assumptions about when learning plateaus and is easy to explain and replicate.

Integration with run.py
-----------------------
CurriculumReward is callable (implements __call__) so it plugs directly
into MountainCarWrapper as a reward_fn with no changes to environment.py.

IMPORTANT: run.py must call curriculum.update(episode) at the end of each
episode so the schedule advances. This is done inside LoggerCallback._on_step
when done=True and the reward_fn is a CurriculumReward instance.

Usage in run.py
---------------
    from reward_curriculum import CurriculumReward, TRANSITION_END

    curriculum = CurriculumReward()                  # fresh instance per run
    REWARD_REGISTRY["curriculum"] = curriculum

    # Inside LoggerCallback._on_step, when done=True:
    if isinstance(env.reward_fn, CurriculumReward):
        env.reward_fn.update(episode_number)
"""

import numpy as np
from reward_functions import dense_reward, sparse_reward

# -----------------------------------------------------------------------
# Curriculum schedule constant
# -----------------------------------------------------------------------

# We transition over the first half of training (default 5000 episodes → 2500).
# Rationale:
#   - Too short (e.g. 500 episodes): agent barely learns dense before switching
#   - Too long (e.g. 4500 episodes): almost all training is on dense, defeating
#     the purpose of eventually testing sparse
#   - Half of training is a reasonable middle ground that gives the agent time
#     to bootstrap from dense before being forced to operate on sparse alone.
TRANSITION_END: int = 2500   # episode at which alpha reaches 1.0 (pure sparse)


# -----------------------------------------------------------------------
# CurriculumReward
# -----------------------------------------------------------------------

class CurriculumReward:
    """
    A callable reward function that linearly interpolates from dense to sparse.

    Parameters
    ----------
    transition_end : int
        The episode number at which the schedule completes (alpha = 1.0).
        After this episode, the reward is pure sparse for the rest of training.

    Attributes
    ----------
    current_episode : int
        Tracks how many episodes have elapsed. Advance with update().
    alpha : float
        Current interpolation weight. 0.0 = pure dense, 1.0 = pure sparse.

    Methods
    -------
    __call__(obs, action, next_obs, done, env_reward) -> float
        Standard reward function interface. Returns the interpolated reward.
        Plugs directly into MountainCarWrapper as reward_fn.

    update(episode: int) -> None
        Advances the schedule to the given episode number.
        Call this at the end of every training episode.

    schedule_str() -> str
        Returns a human-readable description of the current schedule state.
        Useful for progress logging.
    """

    # Give it a __name__ so logger.py writes "curriculum" into the CSV
    __name__ = "curriculum"

    def __init__(self, transition_end: int = TRANSITION_END):
        self.transition_end    = transition_end
        self.current_episode   = 0
        self.alpha             = 0.0   # starts at pure dense

    # ------------------------------------------------------------------
    # Reward function interface — called every step by MountainCarWrapper
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs,
        action,
        next_obs,
        done,
        env_reward,
    ) -> float:
        """
        Returns: (1 - alpha) * dense_reward + alpha * sparse_reward

        alpha = 0.0  →  pure dense  (start of training)
        alpha = 1.0  →  pure sparse (after transition_end episodes)
        """
        r_dense  = dense_reward(obs, action, next_obs, done, env_reward)
        r_sparse = sparse_reward(obs, action, next_obs, done, env_reward)

        return float((1.0 - self.alpha) * r_dense + self.alpha * r_sparse)

    # ------------------------------------------------------------------
    # Schedule advancement — call once per episode from LoggerCallback
    # ------------------------------------------------------------------

    def update(self, episode: int) -> None:
        """
        Recomputes alpha based on the current episode number.

        Parameters
        ----------
        episode : int
            The episode number just completed (1-indexed).
        """
        self.current_episode = episode
        self.alpha = min(episode / self.transition_end, 1.0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def schedule_str(self) -> str:
        """Human-readable schedule state for logging/debugging."""
        phase = (
            "dense phase"    if self.alpha < 0.05 else
            "transitioning"  if self.alpha < 0.95 else
            "sparse phase"
        )
        return (
            f"CurriculumReward | episode={self.current_episode} | "
            f"alpha={self.alpha:.3f} | {phase}"
        )

    def __repr__(self) -> str:
        return (
            f"CurriculumReward("
            f"transition_end={self.transition_end}, "
            f"alpha={self.alpha:.3f}, "
            f"episode={self.current_episode})"
        )
