import gymnasium as gym 
from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("CartPole-v1"),filename="logs/baseline")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=100000)
model.save("ppo_cartpole")