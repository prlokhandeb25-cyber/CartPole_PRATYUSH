import gymnasium as gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from stable_baselines3.common.monitor import Monitor # type: ignore
import os
env = Monitor(gym.make("CartPole-v1"),filename="logs/baseline")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

os.makedirs("models", exist_ok=True)
model.learn(total_timesteps=100000)
model.save("models/ppo_cartpole")