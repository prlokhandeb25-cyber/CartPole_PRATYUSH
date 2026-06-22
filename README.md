# Reward Function Analysis in CartPole using PPO

## Overview

This project investigates how different reward functions influence the behavior of Reinforcement Learning (RL) agents in the CartPole environment.

Using Proximal Policy Optimization (PPO), three separate agents were trained under different reward specifications:

1. **Baseline Reward** – Standard CartPole reward provided by Gymnasium.
2. **Angle-Based Reward** – Agent is rewarded based on the magnitude of the pole angle.
3. **Velocity-Based Reward** – Agent is rewarded based on the cart's velocity.

The objective of this experiment is to demonstrate how RL agents optimize the reward signal they receive, even when that reward does not align with the intended task objective.

This project serves as a preliminary study of reward design, reward misspecification, and specification gaming, providing a foundation for future research into reward hacking and AI alignment.

---

## Motivation

In Reinforcement Learning, agents do not understand human intentions. They only optimize the reward function provided by the designer.

A poorly designed reward function can lead to behaviors that maximize reward while failing to accomplish the intended objective.

This experiment explores that phenomenon by intentionally replacing the standard CartPole reward with alternative reward signals and observing how agent behavior changes.

---

## Environment

The experiment uses the Gymnasium implementation of:

* CartPole-v1

The standard objective of CartPole is to balance a pole attached to a moving cart for as long as possible.

Under the default environment:

* Reward = +1 for every timestep the pole remains balanced.
* Episode ends when the pole falls or the cart moves out of bounds.

---

## Reward Functions

### 1. Baseline Reward

Uses the original CartPole reward function.

**Objective:**

* Keep the pole balanced.
* Maximize survival time.

---

### 2. Angle-Based Reward

Implemented in:

```python
reward = abs(obs[2])
```

Where:

* `obs[2]` = Pole angle

The agent is rewarded for producing larger pole angles.

This creates a reward signal that conflicts with the original objective of maintaining balance.

---

### 3. Velocity-Based Reward

Implemented in:

```python
reward = abs(obs[1])
```

Where:

* `obs[1]` = Cart velocity

The agent is rewarded for moving faster regardless of pole stability.

This encourages behavior different from the intended CartPole task.

---

## Methodology

All agents were trained using:

* PPO (Proximal Policy Optimization)
* Stable-Baselines3
* Gymnasium

Training configuration:

* Policy: MLP Policy
* Algorithm: PPO
* Training Steps: 100,000 timesteps
* Environment: CartPole-v1 and modified variants

Each reward function was trained independently and evaluated separately.

---

## Project Structure

```text
cartpole-reward-hacking-analysis/
│
├── src/
│   ├── main.py
│   ├── train_cartpole.py
│   ├── train_angle.py
│   ├── train_velocity.py
│   ├── test_cartpole.py
│   ├── test_angle.py
│   ├── test_velocity.py
│   ├── angle_cartpole.py
│   ├── velocity_cartpole.py
│   ├── plot_rewards.py
│   ├── check_rewards.py
│   └── bargraph.py
│
├── models/
│   ├── ppo_cartpole.zip
│   ├── ppo_angle.zip
│   └── ppo_velocity.zip
│
├── results/
│   ├── reward_comparison.png
│   ├── bargraph.png
│   └── CARTPOLE RECORDINGS/
│
├── requirements.txt
├── README.md
├──NOTES.md
├── LICENSE
└── .gitignore
```

---

## File Descriptions

### Training Scripts

| File              | Purpose                                   |
| ----------------- | ----------------------------------------- |
| train_cartpole.py | Trains PPO using standard CartPole reward |
| train_angle.py    | Trains PPO using angle-based reward       |
| train_velocity.py | Trains PPO using velocity-based reward    |

---

### Custom Environments

| File                 | Purpose                                    |
| -------------------- | ------------------------------------------ |
| angle_cartpole.py    | Modifies reward to depend on pole angle    |
| velocity_cartpole.py | Modifies reward to depend on cart velocity |

---

### Evaluation Scripts

| File             | Purpose                    |
| ---------------- | -------------------------- |
| test_cartpole.py | Tests baseline model       |
| test_angle.py    | Tests angle-based model    |
| test_velocity.py | Tests velocity-based model |

---

### Analysis Scripts

| File             | Purpose                         |
| ---------------- | ------------------------------- |
| check_rewards.py | Examines recorded reward data   |
| plot_rewards.py  | Generates reward visualizations |
| bargraph.py      | Creates comparative plots       |
| main.py          | Entry point / experiment runner |

---

### Models Directory

Contains trained PPO checkpoints for each reward function.

```text
ppo_cartpole.zip
ppo_angle.zip
ppo_velocity.zip
```

---

### Results Directory

Contains:

* Reward comparison plots
* Visual analysis graphs
* Recorded agent demonstrations

---

## Installation

Clone the repository:

```bash
git clone https://github.com/prlokhandeb25-cyber/cartpole-reward-hacking-analysis.git
cd cartpole-reward-hacking-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

Train baseline model:

```bash
python src/train_cartpole.py
```

Train angle-based model:

```bash
python src/train_angle.py
```

Train velocity-based model:

```bash
python src/train_velocity.py
```

---

## Key Insight

This experiment highlights an important principle in Reinforcement Learning:

> Agents optimize the reward function they are given, not the objective humans intend.

When reward functions become misaligned with the desired task, agents may develop behaviors that maximize reward while failing to achieve the original goal.

Understanding these effects is critical for building reliable and aligned AI systems.

---

## Future Work

This project serves as a stepping stone toward into Project Optimum Fallax:

* Reward Misspecification
* Specification Gaming
* Reward Hacking
* AI Alignment
* Emergent Agent Behavior

Future work will extend these concepts into custom multi-agent environments where agents can evolve increasingly sophisticated strategies for exploiting reward functions.

---

## Technologies Used

* Python
* Gymnasium
* Stable-Baselines3
* PPO (Proximal Policy Optimization)
* NumPy
* Matplotlib

---

