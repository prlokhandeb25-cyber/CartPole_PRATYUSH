import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

model = PPO.load("ppo_cartpole")

obs,info = env.reset()

terminated = False
truncated = False
steps = 0

while not terminated and not truncated :
    action,_ = model.predict(obs)
    obs,reward,terminated,truncated,info = env.step(action)
    steps +=1
print("Episode Length :",steps)
env.close()