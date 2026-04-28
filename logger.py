"""
logger.py
Centralized CSV logging for all 9 experiments.

One Logger instance per experiment run. Writes one row per episode.
Tracks all four evaluation metrics in real time during training:
  - Learning speed      : episode when 10-window average first hits threshold
  - Final performance   : average reward over the last 100 episodes
  - Success rate        : tracked per episode, computed over eval episodes post-training
  - Training stability  : variance over the last 100 episodes (computed at close)
"""

import csv
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Optional


LOGS_DIR              = "logs"
# MountainCar-v0 is typically considered "solved" around -110 (avg over 100 eps).
# Using -90 is unnecessarily strict and labels successful learning as "failed".
PERFORMANCE_THRESHOLD = -110.0  # learning speed: 10-window avg must reach this
WINDOW_SIZE           = 10      # rolling window for learning speed detection
FINAL_WINDOW          = 100     # last N episodes for final performance + stability


# -----------------------------------------------------------------------
# Episode record — one CSV row
# -----------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """All per-episode data written to CSV."""

    # Experiment identity
    algorithm:    str
    reward_fn:    str
    seed:         int
    run_id:       str

    # Progress
    episode:      int
    total_steps:  int

    # Raw episode outcome
    episode_steps:  int
    episode_reward: float       # shaped reward the algorithm saw
    env_reward_sum: float       # raw env reward (ground truth for all comparisons)
    reached_goal:   bool

    # Rolling metric snapshot (written every episode so we can plot learning curves)
    rolling_avg_10:   float     # 10-episode rolling average of env_reward_sum
    learning_reached: bool      # True from the episode the threshold was first crossed

    wall_time: float = field(default_factory=time.time)


# -----------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------

class ExperimentLogger:
    """
    Tracks all four study metrics and writes one CSV per experiment.

    Metric computation
    ------------------
    Learning speed     : detected live inside log(); stored as episode number.
                         If the threshold is never reached → recorded as "failed".
    Final performance  : mean of env_reward_sum over the last 100 episodes.
    Training stability : variance of env_reward_sum over the last 100 episodes.
    Success rate       : set externally via record_eval_success_rate() after
                         post-training evaluation.

    On close(), all four metrics are appended to logs/summary.csv which
    consolidates results across all 9 runs for the analysis phase.
    """

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

        # Rolling buffers
        self._window_10 = deque(maxlen=WINDOW_SIZE)   # learning speed detection
        self._last_100  = deque(maxlen=FINAL_WINDOW)  # final performance + stability

        # Metric results — populated during logging or at close()
        self.learning_speed:    Optional[int]   = None  # episode #, or None = failed
        self.final_performance: Optional[float] = None
        self.stability:         Optional[float] = None  # variance across seeds
        self.eval_success_rate: Optional[float] = None  # % set after eval

        print(f"[Logger] {self.run_id}  →  {filepath}")

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, record: EpisodeRecord) -> None:
        """
        Append one episode to CSV, update rolling windows, detect learning speed.
        Always uses env_reward_sum (raw env reward) for metric computation
        so all 9 experiments are comparable on the same scale.
        """
        self._episode_count += 1
        self._total_steps   += record.episode_steps
        record.total_steps   = self._total_steps
        record.wall_time     = round(time.time() - self._start_time, 3)

        # Update rolling buffers with ground-truth env reward
        self._window_10.append(record.env_reward_sum)
        self._last_100.append(record.env_reward_sum)

        # Compute and attach rolling average to the record
        rolling_avg = float(np.mean(self._window_10))
        record.rolling_avg_10 = round(rolling_avg, 4)

        # Learning speed: first episode where full 10-window avg >= threshold
        if (not self._threshold_crossed
                and len(self._window_10) == WINDOW_SIZE
                and rolling_avg >= PERFORMANCE_THRESHOLD):
            self.learning_speed     = self._episode_count
            self._threshold_crossed = True

        # Track first episode the goal was ever reached (for learned_slow label)
        if self._first_goal_episode is None and record.reached_goal:
            self._first_goal_episode = self._episode_count

        record.learning_reached = self._threshold_crossed

        self._writer.writerow(asdict(record))
        self._file.flush()

    # ------------------------------------------------------------------
    # Success rate (set after post-training evaluation)
    # ------------------------------------------------------------------

    def record_eval_success_rate(self, successes: int, total_eval_episodes: int) -> None:
        """
        Call this after running the agent in evaluation mode post-training.
        Computes the percentage of eval episodes where the goal was reached.
        """
        self.eval_success_rate = round(successes / total_eval_episodes * 100, 2)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def close(self) -> dict:
        """
        Compute final performance and stability from the last 100 episodes.
        Appends a row to logs/summary.csv and closes the episode CSV.
        Returns a summary dict with all four metrics.
        """
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
            # Metric 1 — Learning speed
            # Three states:
            #   <episode number> : crossed the -110 rolling avg threshold (fast learner)
            #   "learned_slow(N)": never crossed threshold but reached goal at episode N
            #   "failed"         : never reached the goal at all (success rate = 0%)
            "learning_speed": (
                self.learning_speed if self.learning_speed
                else f"learned_slow({self._first_goal_episode})" if (self.eval_success_rate or 0) > 0
                else "failed"
            ),
            # Metric 2 — Final performance
            "final_performance":  self.final_performance,
            # Metric 3 — Success rate (None until eval is run)
            "eval_success_rate":  self.eval_success_rate,
            # Metric 4 — Training stability
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

        # Load existing rows if file exists
        existing_rows = []
        if os.path.exists(summary_path):
            with open(summary_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = [
                    row for row in reader
                    if row["run_id"] != self.run_id  # drop old version of this run
                ]

        existing_rows.append(summary)  # add the fresh row

        with open(summary_path, "w", newline="") as f:  # overwrite cleanly
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerows(existing_rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def make_logger(algorithm: str, reward_fn: str, seed: int) -> ExperimentLogger:
    return ExperimentLogger(algorithm=algorithm, reward_fn=reward_fn, seed=seed)