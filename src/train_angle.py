from stable_baselines3 import PPO
from angle_cartpole import AngleCartPoleEnv
from stable_baselines3.common.monitor import Monitor
import os
env = Monitor(AngleCartPoleEnv(),filename="logs/angle")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)
os.makedirs("models", exist_ok=True)
model.learn(total_timesteps = 100000)
model.save("models/ppo_angle")