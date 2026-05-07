"""
logger.py
Centralized CSV logging for all 9 experiments.

"""

import csv
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Optional


LOGS_DIR              = "logs"
PERFORMANCE_THRESHOLD = -110.0  
WINDOW_SIZE           = 10      
FINAL_WINDOW          = 100     



@dataclass
class EpisodeRecord:


    algorithm:    str
    reward_fn:    str
    seed:         int
    run_id:       str

  
    episode:      int
    total_steps:  int


    episode_steps:  int
    episode_reward: float
    env_reward_sum: float
    reached_goal:   bool

    
    rolling_avg_10:   float
    learning_reached: bool

    wall_time: float = field(default_factory=time.time)




class ExperimentLogger:


    FIELDNAMES = [
        "run_id", "algorithm", "reward_fn", "seed",
        "episode", "total_steps",
        "episode_steps", "episode_reward", "env_reward_sum",
        "reached_goal", "rolling_avg_10", "learning_reached",
        "wall_time",
    ]

    def __init__(
        self,
        algorithm: str,
        reward_fn: str,
        seed:      int,
        logs_dir:  str = LOGS_DIR,
    ):
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self.seed      = seed
        self.run_id    = f"{algorithm}_{reward_fn}_seed{seed}"

        os.makedirs(logs_dir, exist_ok=True)
        self._logs_dir = logs_dir
        filepath = os.path.join(logs_dir, f"{self.run_id}.csv")

        self._file   = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        # Internal state
        self._total_steps       = 0
        self._start_time        = time.time()
        self._episode_count     = 0
        self._threshold_crossed = False
        self._first_goal_episode: Optional[int] = None

       
        self._window_10 = deque(maxlen=WINDOW_SIZE)   
        self._last_100  = deque(maxlen=FINAL_WINDOW) 


        self.learning_speed:    Optional[int]   = None  
        self.final_performance: Optional[float] = None
        self.stability:         Optional[float] = None  
        self.eval_success_rate: Optional[float] = None 

        print(f"[Logger] {self.run_id}  →  {filepath}")



    def log(self, record: EpisodeRecord) -> None:

        self._episode_count += 1
        self._total_steps   += record.episode_steps
        record.total_steps   = self._total_steps
        record.wall_time     = round(time.time() - self._start_time, 3)

        
        self._window_10.append(record.env_reward_sum)
        self._last_100.append(record.env_reward_sum)

        
        rolling_avg = float(np.mean(self._window_10))
        record.rolling_avg_10 = round(rolling_avg, 4)

        
        if (not self._threshold_crossed
                and len(self._window_10) == WINDOW_SIZE
                and rolling_avg >= PERFORMANCE_THRESHOLD):
            self.learning_speed     = self._episode_count
            self._threshold_crossed = True

        
        if self._first_goal_episode is None and record.reached_goal:
            self._first_goal_episode = self._episode_count

        record.learning_reached = self._threshold_crossed

        self._writer.writerow(asdict(record))
        self._file.flush()

    # Success rate 

    def record_eval_success_rate(self, successes: int, total_eval_episodes: int) -> None:
        self.eval_success_rate = round(successes / total_eval_episodes * 100, 2)

    # Finalization

    def close(self) -> dict:

        if len(self._last_100) > 0:
            vals = list(self._last_100)
            self.final_performance = round(float(np.mean(vals)), 4)
            self.stability         = round(float(np.var(vals)),  4)

        self._file.close()

        summary = {
            "run_id":             self.run_id,
            "algorithm":          self.algorithm,
            "reward_fn":          self.reward_fn,
            "seed":               self.seed,
    
            "learning_speed": (
                self.learning_speed if self.learning_speed
                else f"learned_slow({self._first_goal_episode})" if (self.eval_success_rate or 0) > 0
                else "failed"
            ),
      
            "final_performance":  self.final_performance,
            
            "eval_success_rate":  self.eval_success_rate,

            "stability_variance": self.stability,
        }

        self._write_summary(summary)
        print(
            f"[Logger] Closed {self.run_id} | "
            f"learning_speed={summary['learning_speed']} | "
            f"final_perf={self.final_performance} | "
            f"stability_var={self.stability} | "
            f"success_rate={self.eval_success_rate}%"
        )
        return summary

    def _write_summary(self, summary: dict) -> None:
        summary_path = os.path.join(self._logs_dir, "summary.csv")

   
        existing_rows = []
        if os.path.exists(summary_path):
            with open(summary_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = [
                    row for row in reader
                    if row["run_id"] != self.run_id 
                ]

        existing_rows.append(summary)  

        with open(summary_path, "w", newline="") as f: 
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerows(existing_rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Factory

def make_logger(algorithm: str, reward_fn: str, seed: int) -> ExperimentLogger:
    return ExperimentLogger(algorithm=algorithm, reward_fn=reward_fn, seed=seed)