"""
reward_curriculum.py
Implements a linear reward curriculum that transitions from dense to sparse.
"""


import numpy as np
from reward_functions import dense_reward, sparse_reward


TRANSITION_END: int = 2500   




class CurriculumReward:
    
    __name__ = "curriculum"

    def __init__(self, transition_end: int = TRANSITION_END):
        self.transition_end    = transition_end
        self.current_episode   = 0
        self.alpha             = 0.0  


    def __call__(
        self,
        obs,
        action,
        next_obs,
        done,
        env_reward,
    ) -> float:

        r_dense  = dense_reward(obs, action, next_obs, done, env_reward)
        r_sparse = sparse_reward(obs, action, next_obs, done, env_reward)

        return float((1.0 - self.alpha) * r_dense + self.alpha * r_sparse)



    def update(self, episode: int) -> None:

        self.current_episode = episode
        self.alpha = min(episode / self.transition_end, 1.0)



    def schedule_str(self) -> str:
      
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
