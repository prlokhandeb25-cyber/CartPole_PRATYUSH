# NOTES.md

# CartPole PPO Experiment Notes

## What Was This Project?

This project explored how Reinforcement Learning agents behave when trained under different reward functions.

Instead of changing the learning algorithm, we changed the objective that the agent receives and observed how its behavior evolved.

Three environments were trained:

1. Baseline CartPole
2. Angle-Based Reward CartPole
3. Velocity-Based Reward CartPole

The experiment demonstrates a fundamental RL principle:

> Agents optimize the reward function they receive, not the objective humans intend.

This idea forms the basis of reward misspecification, specification gaming, reward hacking, and AI alignment research.

---

# Reinforcement Learning Fundamentals

## Agent

The learner.

Examples:

* CartPole agent
* Chess AI
* AlphaGo
* Autonomous robot

The agent chooses actions.

---

## Environment

The world in which the agent operates.

Examples:

* CartPole
* Atari games
* Chess board
* Robot simulator

The environment provides observations and rewards.

---

## State (Observation)

Information available to the agent.

For CartPole:

```text
obs[0] = Cart Position
obs[1] = Cart Velocity
obs[2] = Pole Angle
obs[3] = Pole Angular Velocity
```

Example:

```python
[0.02, -0.15, 0.03, 0.22]
```

This represents the current situation of the environment.

---

## Action

Decision made by the agent.

CartPole actions:

```text
0 = Move Left
1 = Move Right
```

---

## Reward

Feedback given after an action.

Example:

```text
Good action → Positive Reward
Bad action → Low Reward
```

The entire learning process revolves around maximizing cumulative reward.

---

## Episode

One complete attempt.

CartPole episode:

```text
Reset Environment
      ↓
Take Actions
      ↓
Collect Rewards
      ↓
Pole Falls
      ↓
Episode Ends
```

Then a new episode begins.

---

# How CartPole Works

Goal:

```text
Keep Pole Balanced
```

Default reward:

```text
+1 per timestep alive
```

Therefore:

```text
Stay alive longer
       ↓
Receive more reward
       ↓
Learn balancing behavior
```

---

# Why We Modified Rewards

We wanted to see how behavior changes when objectives change.

---

## Baseline Reward

```python
reward = 1
```

Objective:

```text
Stay alive
```

Expected behavior:

```text
Balanced pole
Stable movements
```

---

## Angle Reward

```python
reward = abs(obs[2])
```

Objective:

```text
Increase pole angle
```

Expected behavior:

```text
Swing pole aggressively
```

The agent no longer cares about balance.

---

## Velocity Reward

```python
reward = abs(obs[1])
```

Objective:

```text
Increase cart speed
```

Expected behavior:

```text
Move rapidly
Ignore stability
```

Again, the intended task is ignored.

---

# PPO (Proximal Policy Optimization)

The learning algorithm used.
---

# What Is a Policy?

Policy = Agent's brain

Mathematically:

```text
State → Action
```

Example:

```text
Pole leaning left
        ↓
Move cart left
```

The policy learns this mapping.

---

# PPO Overview

```text
Environment State
        ↓
Neural Network Policy
        ↓
Choose Action
        ↓
Execute Action
        ↓
Receive Reward
        ↓
Store Experience
        ↓
Update Policy
        ↓
Repeat
```

PPO clips updates to keep learning stable.

Hence:

```text
Proximal Policy Optimization
```

"Proximal" means:

```text
Do not change policy too much at once.
```

---

# model.learn()

Code:

```python
model.learn(total_timesteps=100000)
```

Internally:

```text
Initialize Neural Network
            ↓
Collect Experiences
            ↓
Fill Rollout Buffer
            ↓
Compute Advantages
            ↓
Update Policy
            ↓
Update Value Network
            ↓
Repeat
            ↓
100000 Steps Reached
            ↓
Training Ends
```

---

# Neural Networks Inside PPO

PPO contains two networks.

---

## Actor Network

Responsible for actions.

```text
State
   ↓
Actor
   ↓
Action Probabilities
```

Example:

```text
Left = 80%
Right = 20%
```

---

## Critic Network

Responsible for value estimation.

```text
State
   ↓
Critic
   ↓
Expected Future Reward
```

The critic helps PPO judge whether actions were good.

---

# Understanding Training Metrics

During training Stable-Baselines3 prints statistics.

Example:

```text
ep_len_mean
ep_rew_mean
fps
iterations
time_elapsed
total_timesteps
approx_kl
clip_fraction
entropy_loss
explained_variance
learning_rate
loss
policy_gradient_loss
value_loss
```

---

# ep_len_mean

Average episode length.

```text
Higher = Better
```

CartPole maximum:

```text
500
```

Example:

```text
ep_len_mean = 450
```

Means:

```text
Agent survives ~450 steps
on average.
```

---

# ep_rew_mean

Average reward per episode.

Example:

```text
ep_rew_mean = 450
```

For standard CartPole:

```text
Longer survival
      ↓
More reward
```

For custom rewards:

```text
Interpret carefully
```

High reward may not mean good behavior.

---

# fps

Frames per second.

Training speed.

Example:

```text
fps = 2500
```

Means:

```text
2500 environment steps per second
```

---

# iterations

Number of PPO update cycles completed.

Each iteration:

```text
Collect Data
      ↓
Train Network
```

---

# time_elapsed

Total training time.

Example:

```text
12 seconds
```

---

# total_timesteps

Steps collected so far.

Example:

```text
75000
```

Means:

```text
75,000 interactions
```

---

# approx_kl

KL Divergence.

Measures:

```text
How much policy changed
after an update.
```

Small:

```text
Stable learning
```

Large:

```text
Dangerous update
```

---

# clip_fraction

Unique PPO metric.

Measures:

```text
How often PPO clipped updates.
```

High values indicate:

```text
Large policy changes
```

---

# entropy_loss

Exploration measure.

High entropy:

```text
Random exploration
```

Low entropy:

```text
Confident behavior
```

As training progresses:

```text
Entropy usually decreases.
```

---

# explained_variance

Critic quality.

Range:

```text
-∞ to 1
```

Ideal:

```text
Near 1
```

Interpretation:

```text
1.0 = Excellent value predictions
0.0 = No predictive power
Negative = Very poor critic
```

---

# learning_rate

Gradient descent step size.

Example:

```text
0.0003
```

Controls how aggressively weights change.

---

# loss

Overall training objective.

Combination of:

* Policy Loss
* Value Loss
* Entropy Term

Used internally by PPO.

---

# policy_gradient_loss

Measures policy update magnitude.

Represents:

```text
How strongly PPO
is adjusting behavior.
```

---

# value_loss

Critic prediction error.

Large:

```text
Critic making poor estimates
```

Small:

```text
Critic understands environment
```

# Code Walkthrough

---

# Training Scripts

The training scripts are responsible for creating the environment, initializing PPO, training the agent, and saving the trained model.

All three training scripts follow the same structure.

General flow:

```text
Create Environment
        ↓
Wrap Environment with Monitor
        ↓
Create PPO Model
        ↓
Train using model.learn()
        ↓
Save Trained Model
```

---

## train_cartpole.py

Purpose:

Train a PPO agent using the original CartPole reward function.

Code Flow:

```python
env = Monitor(gym.make("CartPole-v1"))
```

Creates the standard CartPole environment.

The Monitor wrapper records episode statistics such as:

* Episode length
* Episode reward
* Training logs

---

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)
```

Creates a PPO agent.

Parameters:

* MlpPolicy → Multi-Layer Perceptron neural network
* env → Training environment
* verbose=1 → Print training statistics

---

```python
model.learn(total_timesteps=100000)
```

Starts training.

The agent interacts with the environment for 100,000 timesteps and continuously updates its policy.

---

```python
model.save("ppo_cartpole")
```

Stores the trained neural network weights.

The saved file can later be loaded without retraining.

---

## train_angle.py

Purpose:

Train PPO using the custom AngleCartPole environment.

Main difference:

```python
env = Monitor(AngleCartPoleEnv())
```

Instead of the default reward:

```python
reward = 1
```

the environment uses:

```python
reward = abs(obs[2])
```

The PPO algorithm remains exactly the same.

Only the reward function changes.

This allows us to observe how reward design influences behavior.

---

## train_velocity.py

Purpose:

Train PPO using the VelocityCartPole environment.

Main difference:

```python
reward = abs(obs[1])
```

The agent now attempts to maximize cart velocity instead of balancing the pole.

Again:

```text
Same PPO
Same Network
Same Training Steps

Only Reward Changes
```

This isolates the effect of reward design.

---

# Custom Environment Files

The custom environment files inherit from Gymnasium's CartPole implementation and override the reward function.

---

## angle_cartpole.py

Purpose :

Modify CartPole so that reward depends on pole angle.

We used Inheritance :

```text
Use all original CartPole mechanics
        ↓
Change only the reward calculation
```

Inside step():

```python
obs,reward,terminated,truncated,info = super().step(action)
```

The original CartPole step is executed first.

This updates:

* Position
* Velocity
* Pole Angle
* Pole Angular Velocity

---

Then:

```python
reward = abs(obs[2])
```

The original reward is replaced.

The new reward becomes proportional to pole angle.

---

Finally:

```python
return obs,reward,terminated,truncated,info
```

Returns the modified transition information.

---

## velocity_cartpole.py

Purpose:

Modify CartPole so that reward depends on cart velocity.

Implementation:

```python
reward = abs(obs[1])
```

where:

```python
obs[1]
```

represents cart velocity.

The environment dynamics remain unchanged.

Only the reward calculation changes.

---

# Testing Scripts

Training teaches the agent.

Testing evaluates the learned behavior.

The testing scripts load trained models and allow visual inspection.

General flow:

```text
Create Environment
        ↓
Load Saved PPO Model
        ↓
Reset Environment
        ↓
Predict Actions
        ↓
Run Environment
        ↓
Observe Behavior
```

---

## test_cartpole.py

Purpose:

Evaluate the baseline PPO model.

Typical process:

```python
model = PPO.load("ppo_cartpole")
```

Loads saved neural network weights.

No retraining occurs.

---

```python
obs,_ = env.reset()
```

Starts a fresh episode.

---

```python
action,_ = model.predict(obs)
```

Uses the trained policy network to select an action.

The neural network receives:

```text
Current State
       ↓
Action Prediction
```

---

```python
obs,reward,done,...
```

The environment executes the action and returns the next state.

This loop continues until the episode ends.

---

Observation:

The baseline agent should attempt to balance the pole for as long as possible.

---

## test_angle.py

Purpose:

Evaluate the angle-based reward agent.

Expected behavior:

```text
Seek Larger Pole Angles
```

instead of:

```text
Maintain Balance
```

The testing script helps verify whether the learned behavior matches the reward function.

---

## test_velocity.py

Purpose:

Evaluate the velocity-based reward agent.

Expected behavior:

```text
Maximize Cart Speed
```

The agent may move aggressively because velocity contributes directly to reward.

---

# Model Files

The models directory contains serialized PPO agents.

Files:

```text
ppo_cartpole.zip
ppo_angle.zip
ppo_velocity.zip
```

# Complete Project Flow

```text
Create Environment
        ↓
Define Reward Function
        ↓
Train PPO Agent
        ↓
Save Model
        ↓
Load Model
        ↓
Run Evaluation
        ↓
Compare Behaviors
        ↓
Study Effect of Reward Design
```

The lessons learned here directly inform the design of future multi-agent environments where agents may discover increasingly sophisticated methods of exploiting reward functions.
