from stable_baselines3 import PPO
from velocity_cartpole import VelocityCartPoleEnv
from stable_baselines3.common.monitor import Monitor

env = Monitor(VelocityCartPoleEnv(),filename="logs/velocity")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)
model.learn(total_timesteps = 100000)
model.save("ppo_velocity")