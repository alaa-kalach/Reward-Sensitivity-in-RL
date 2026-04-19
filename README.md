# Reward Sensitivity in Reinforcement Learning
### How Sensitive Are Different RL Algorithms to Reward Function Design?

---

## Project Overview

This project studies how three reinforcement learning algorithms — PPO, DQN, and A2C — respond to three different reward function designs on the MountainCar-v0 environment. The central question is whether an algorithm's performance is driven more by the algorithm itself or by how the reward signal is engineered.

The project is split into two phases. Phase 1 (complete) builds the shared infrastructure: the environment, reward functions, logging, and analysis pipeline. Phase 2 (upcoming) plugs in the three trained agents and produces the final results.

---

## Research Design

**Environment:** MountainCar-v0 (OpenAI Gymnasium)
A car starts at the bottom of a valley and must reach a flag on the right hill. The engine is too weak to drive straight up — the agent must learn to swing left first to build momentum. This makes it a classic hard-exploration problem and an ideal testbed for studying reward sensitivity.

**State space:** position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]
**Action space:** 3 discrete actions — push left, no-op, push right
**Goal:** position ≥ 0.5

**Total runs:** 27 — 3 algorithms × 3 reward functions × 3 random seeds

---

## Algorithms (Phase 2)

| Algorithm | Type | Key characteristic |
|---|---|---|
| PPO | On-policy, actor-critic | Clipped surrogate objective, stable updates |
| DQN | Off-policy, value-based | Replay buffer, target network |
| A2C | On-policy, actor-critic | Synchronous n-step updates, simpler than PPO |

---

## Reward Functions

All three functions share the same signature accepted by the environment wrapper:
```
reward_fn(obs, action, next_obs, done, env_reward) -> float
```

### 1. Dense
Provides a continuous shaped signal on every step. Designed to guide the agent without waiting for goal discovery.

```
reward = (position_gain × 100) + (speed_bonus × 10) − 0.1 + goal_bonus
```

- `position_gain`: difference in position between steps, scaled by 100
- `speed_bonus`: absolute velocity × 10, encouraging momentum
- `step_penalty`: −0.1 per step to discourage unnecessary exploration
- `goal_bonus`: +10.0 one-time reward on reaching the goal

### 2. Sparse
Binary signal. +1 when the goal is reached, 0 everywhere else. The agent receives no intermediate feedback, making this the hardest exploration challenge and the purest test of an algorithm's intrinsic exploration capability.

```
reward = 1.0 if position >= 0.5 else 0.0
```

### 3. Potential-Based
Theoretically grounded shaping based on Ng et al. (1999). The shaping term `F(s, s') = γ·Φ(s') − Φ(s)` is added on top of the original environment reward. This is guaranteed not to change the optimal policy (policy invariance), making it the most principled design.

```
Φ(s) = position + 0.5 × velocity²
reward = env_reward + (0.99 × Φ(s') − Φ(s))
```

---

## Evaluation Metrics

All four metrics use the raw environment reward (`env_reward`) as the ground truth, regardless of which reward function the agent was trained on. This ensures fair comparison across all 27 runs.

| Metric | Definition |
|---|---|
| **Learning speed** | Episode number when the 10-episode rolling average of env_reward first reaches ≥ −90. Recorded as `failed` if never reached within the episode budget. |
| **Final performance** | Mean env_reward over the last 100 training episodes. |
| **Success rate** | Percentage of post-training evaluation episodes where the agent reached the goal. |
| **Training stability** | Variance of final_performance across the 3 random seeds for each (algorithm, reward) condition. High variance = unstable training. |

---

## Sensitivity Analyses

**Reward Sensitivity** — per algorithm: variance in seed-averaged final performance across its 3 reward conditions. High variance means that algorithm is sensitive to reward design.

**Algorithm Sensitivity** — per reward function: variance in seed-averaged final performance across its 3 algorithms. High variance means that reward type strongly differentiates algorithm behavior.

Both analyses operate on seed-averaged values to isolate reward/algorithm effects from random initialization noise.

---

## File Structure

```
project/
├── environment.py       # MountainCarWrapper — seeding, reward injection, step tracking
├── reward_functions.py  # dense, sparse, potential_based + REWARD_REGISTRY
├── logger.py            # Per-run CSV logging + summary.csv aggregation
├── metrics.py           # Sensitivity analysis computed from summary.csv
├── run.py               # Single entry point for Phase 1
│
├── logs/
│   ├── PPO_dense_seed42.csv          # one file per run (27 total)
│   ├── PPO_dense_seed123.csv
│   ├── ...
│   └── summary.csv                   # one row per run, all four metrics
│
└── agents/                           # Phase 2 — not yet implemented
    ├── ppo.py
    ├── dqn.py
    └── a2c.py
```

---

## CSV Format

Each per-run CSV (`logs/<run_id>.csv`) has one row per episode:

| Column | Description |
|---|---|
| `run_id` | Unique identifier e.g. `PPO_dense_seed42` |
| `algorithm` | PPO, DQN, or A2C |
| `reward_fn` | dense, sparse, or potential_based |
| `seed` | 42, 123, or 456 |
| `episode` | Episode number within this run |
| `total_steps` | Cumulative environment steps so far |
| `episode_steps` | Steps taken in this episode |
| `episode_reward` | Sum of shaped reward the algorithm received |
| `env_reward_sum` | Sum of raw env reward (ground truth) |
| `reached_goal` | Whether the car reached position ≥ 0.5 |
| `rolling_avg_10` | 10-episode rolling average of env_reward_sum |
| `learning_reached` | True from the episode the −90 threshold was first crossed |
| `wall_time` | Seconds elapsed since run start |

The shared `logs/summary.csv` has one row per run with the four final metrics: `learning_speed`, `final_performance`, `eval_success_rate`, `stability_variance`.

---

## How to Run

**Install dependencies:**
```bash
pip install gymnasium stable-baselines3 pygame
```

**Run Phase 1 (full flow — smoke test, analysis, visualization):**
```bash
python run.py
```

**Flags:**
```bash
python run.py --episodes 5          # more episodes per run in smoke test
python run.py --vis-episodes 3      # more episodes in visualization
python run.py --no-visual           # skip visualization, run smoke test + report only
```

**Visualization modes** (prompted interactively after the smoke test):

- **Mode A** — Fix an algorithm, watch it under all 3 reward functions sequentially. Mirrors the Reward Sensitivity analysis.
- **Mode B** — Fix a reward function, watch all 3 algorithms run under it sequentially. Mirrors the Algorithm Sensitivity analysis.

Each condition opens its own pygame window. Close it to advance to the next. A comparison table prints after all windows close.

**Run the analysis report standalone** (after experiments are complete):
```bash
python metrics.py
```

---

## Seeds

Three fixed seeds are used: `[42, 123, 456]`. Each seed initializes both the environment and the policy network, producing distinct but reproducible training trajectories. Variance across seeds is how training stability is measured.

---

## Current Status

**Phase 1 — Complete**
- Environment wrapper with seeding and reward injection
- Three reward functions (dense, sparse, potential-based)
- Full metric tracking and CSV logging across all 27 runs
- Sensitivity analysis report
- Comparative visualization

**Phase 2 — In progress**
- Person 1: PPO agent (`agents/ppo.py`)
- Person 2: DQN agent (`agents/dqn.py`)
- Person 3: A2C agent (`agents/a2c.py`)

Phase 2 replaces the random policy in `run.py` with trained agents. The environment wrapper, reward functions, logger, and metrics pipeline remain unchanged.
