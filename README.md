# Reward Sensitivity in Reinforcement Learning
### How Sensitive Are Different RL Algorithms to Reward Function Design?

---

## Project Overview

This project studies how three reinforcement learning algorithms — **PPO**, **DQN**, and **A2C** — respond to reward design on **MountainCar-v0** (Gymnasium). The core question is whether performance is driven more by the algorithm or by how the reward signal is engineered.

Implementations use **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** directly in `run.py`, with hyperparameters fixed per algorithm across all reward conditions so differences reflect **reward sensitivity**, not incidental tuning.

Experiments write per-run CSVs and a consolidated `logs/summary.csv`; `plot.py` produces figures for learning curves, cross-algorithm comparisons, stability, noise ablations, and the curriculum schedule.

---

## Research Design

**Environment:** MountainCar-v0 ([Gymnasium](https://gymnasium.farama.org/))

A car starts at the bottom of a valley and must reach a flag on the right hill. The engine is too weak to drive straight up — the agent must learn to swing left first to build momentum. This is a classic hard-exploration problem and a practical testbed for reward sensitivity.

**State space:** position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]  
**Action space:** 3 discrete actions — push left, no-op, push right  
**Goal:** position ≥ 0.5  

**Classic factorial (paper-style core):** 3 algorithms × 3 base reward shapes × 3 seeds = **27 runs** (when restricting to `dense`, `sparse`, `potential_based`).  

The default CLI runs **six** reward conditions (below), i.e. 6 × 3 seeds = **18 runs per algorithm**. Use `--rewards dense sparse potential_based` to match the classic 9 runs per algorithm.

---

## Algorithms

All three algorithms are trained via SB3 (`model.learn`) with a custom callback (`LoggerCallback`) that logs **one row per episode** using the raw environment reward in `info["env_reward"]` as the comparison scale, regardless of shaped reward during training.

| Algorithm | Type | Notes |
|-----------|------|--------|
| PPO | On-policy actor-critic | Clipped surrogate, tuned `n_steps`, GAE |
| DQN | Off-policy value-based | Replay buffer, target network, longer exploration fraction for sparse rewards |
| A2C | On-policy actor-critic | Synchronous updates, simpler than PPO |

Hyperparameters live in `HYPERPARAMS` in ```80:116:run.py``` and are **not** switched per reward type.

---

## Reward Functions

All callables share the signature expected by `MountainCarWrapper`:

```
reward_fn(obs, action, next_obs, done, env_reward) -> float
```

### Core registry (`reward_functions.py`)

**1. Dense** — dense progress and momentum shaping on top of true step cost:

- `progress_reward = (next_pos − pos) × 10`
- `speed_reward = |next_velocity| × 0.5`
- `goal_bonus = 50` when the goal position is reached
- Returns **`env_reward` + shaping** so incentives stay aligned with the environment’s −1-per-step penalty

**2. Sparse** — binary signal: **`1.0`** if `next_obs[0] ≥ 0.5`, else **`0.0`** (hard exploration).

**3. Potential-based** (Ng et al., 1999) — **`env_reward + γ·Φ(s′) − Φ(s)`** with `γ = 0.99`,  
`Φ(s) = (position + 0.5 × velocity²) × 10`  
(policy-invariant shaping up to rescaling conventions used here).

Registered names: `dense`, `sparse`, `potential_based`.

### Extensions

**Gaussian noise (`noise_injection.py`)**

- **`dense_noisy`** / **`sparse_noisy`** — same as dense/sparse plus **ε ~ 𝒩(0, σ)** (`NOISE_SIGMA = 0.1`).
- Potential-based is **not** given a noisy variant in code: small typical magnitudes would make fixed σ disproportionately distort the signal.

**Dense → sparse curriculum (`reward_curriculum.py`)**

- **`curriculum`** — linear mix  
  **`R = (1 − α)·R_dense + α·R_sparse`** with **`α = min(episode / TRANSITION_END, 1)`**.
- **`TRANSITION_END`** is **6500** episodes in code; **default training is `N_EPISODES = 5000`**, so **α reaches ~0.77** unless you train longer or change `TRANSITION_END`. The schedule advances once per finished episode inside `LoggerCallback`.

### Choosing conditions at runtime

```bash
# Classic 9 runs per algorithm (3 rewards × 3 seeds)
python run.py --algorithm PPO --rewards dense sparse potential_based

# Only noisy comparisons
python run.py --algorithm DQN --rewards dense_noisy sparse_noisy

# Default: all six names (dense, sparse, potential_based, dense_noisy, sparse_noisy, curriculum)
python run.py --algorithm A2C
```

---

## Evaluation Metrics

Metrics use **`env_reward` (sum per episode)** as **ground truth** for learning speed and final performance, independent of shaped training reward.

**Learning speed** — defined in ```24:26:logger.py``` using a **10-episode rolling mean** of `env_reward_sum` and threshold **`PERFORMANCE_THRESHOLD = -110`**. Stored in `summary.csv` as:

- an episode number once the threshold is first met;
- or **`learned_slow(N)`** if the threshold was never met but post-training evaluation still shows goal success (`eval_success_rate > 0`), with **`N`** the first training episode where the goal was reached;
- or **`failed`** if appropriate.

**Final performance** — mean `env_reward_sum` over the **last 100** training episodes (variance over that window is **`stability_variance`** in `summary.csv`).

**Eval success rate** — percentage of **100** deterministic post-training episodes that reach the goal.

Cross-run **training stability** in reports aggregates **final performance across seeds** per `(algorithm, reward_fn)`.

**Sensitivity summaries** (`metrics.py`):

- **Reward sensitivity** — per algorithm: variance in seed-averaged final performance across reward conditions present in `summary.csv`.
- **Algorithm sensitivity** — per reward: variance across algorithms.

---

## File Structure

```
project/
├── environment.py        # MountainCarWrapper — Gymnasium wrapper, shaped reward injection, info dict
├── reward_functions.py   # dense, sparse, potential_based + REWARD_REGISTRY
├── noise_injection.py    # Gaussian wrapper; dense_noisy, sparse_noisy
├── reward_curriculum.py  # CurriculumReward (dense→sparse interpolation)
├── logger.py               # Per-run CSV + summary.csv aggregation
├── metrics.py              # Sensitivity analysis from summary.csv
├── run.py                  # SB3 training + evaluation + per-algorithm report
├── plot.py                 # Figures from logs/
│
├── logs/
│   ├── <ALGO>_<reward>_seed<SEED>.csv   # one per condition
│   ├── summary.csv
│   └── plots/
│       ├── learning_curves_PPO.png
│       ├── learning_curves_DQN.png
│       ├── learning_curves_A2C.png
│       ├── algo_comparison_dense.png
│       ├── algo_comparison_sparse.png
│       ├── algo_comparison_potential_based.png
│       ├── final_performance_bars.png
│       ├── stability_heatmap.png
│       ├── noise_comparison.png           # dense vs dense_noisy, sparse vs sparse_noisy (from summary)
│       └── curriculum_schedule.png        # schematic α schedule + formula
│
└── README.md
```

---

## CSV Format

Each per-run CSV row is one episode. Main columns:

| Column | Description |
|--------|-------------|
| `run_id` | e.g. `PPO_dense_seed42` |
| `algorithm` | PPO, DQN, A2C |
| `reward_fn` | Reward name (`dense`, `sparse`, `potential_based`, `dense_noisy`, `sparse_noisy`, `curriculum`) |
| `seed` | e.g. 42, 123, 456 |
| `episode` | Episode index |
| `total_steps` | Cumulative env steps |
| `episode_reward` | Sum of **shaped** reward seen by the learner |
| `env_reward_sum` | Sum of **raw** MountainCar reward (metric scale) |
| `reached_goal` | Goal reached this episode |
| `rolling_avg_10` | 10-episode rolling mean of `env_reward_sum` |
| `learning_reached` | True after the −110 threshold is first crossed (see `logger.py`) |
| `wall_time` | Seconds since run start |

`logs/summary.csv` adds **`learning_speed`**, **`final_performance`**, **`eval_success_rate`**, **`stability_variance`**.

---

## How to Run

**Install dependencies**

```bash
pip install gymnasium stable-baselines3 numpy matplotlib
```

**Training (one algorithm)**

```bash
python run.py --algorithm PPO
python run.py --algorithm DQN --episodes 5000
python run.py --algorithm A2C --seeds 42 123 456
```

**Useful flags**

```bash
python run.py --algorithm PPO --rewards dense sparse potential_based   # classic 27-run study across all algos
python run.py --algorithm PPO --episodes 7000                           # closer to full curriculum if using curriculum reward
python run.py --algorithm PPO --seeds 42                                # quick smoke run
```

If `--episodes` is ≤ 100, `run.py` prints a warning: final averages and variance use fewer than 100 episodes.

**Plotting**

```bash
python plot.py
python plot.py --algorithms PPO DQN --logs-dir logs --plots-dir logs/plots
```

Figures save with matplotlib’s Agg backend (`plot.py`), so no display is required for file output.

The first four figures use the **three base** rewards (`dense`, `sparse`, `potential_based`) for learning curves and algorithm grids. **`noise_comparison.png`** needs summary rows that include **`dense`** / **`dense_noisy`** and **`sparse`** / **`sparse_noisy`**. **`curriculum_schedule.png`** is a standalone diagram (no CSV required).

**Standalone metrics**

```bash
python metrics.py
```

---

## Episode Budget & Evaluation

| Constant | Default | Role |
|---------|---------|------|
| `N_EPISODES` | 5000 | Training episodes per `(algorithm, reward, seed)` |
| `EVAL_EPISODES` | 100 | Post-training deterministic eval episodes for success rate |
| `MAX_STEPS` | 200 | Gym time limit per MountainCar episode (step budget passed to SB3 via total timesteps) |

---

## Seeds

Default seeds **`[42, 123, 456]`** fix environment and SB3 RNGs for repeatable runs. **Training stability** compares outcomes across seeds for the same `(algorithm, reward_fn)`.

---

## Status

Infrastructure and RL training (**Stable-Baselines3**) are implemented end-to-end: shaped rewards, noisy variants, optional curriculum, logging, metrics, reporting, and plotting (including noise and curriculum figures). Tune `--rewards`, `--episodes`, and **`TRANSITION_END`** in `reward_curriculum.py` when you want the classic 27-run factorial, longer runs for sparse convergence, or a curriculum that reaches α = 1 within the training horizon.
