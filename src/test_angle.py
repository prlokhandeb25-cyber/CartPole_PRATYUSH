from stable_baselines3 import PPO
from angle_cartpole import AngleCartPoleEnv

env = AngleCartPoleEnv(render_mode="human")

model = PPO.load("ppo_angle") 

obs,info = env.reset()

terminated = False
truncated = False
steps = 0 

while not terminated and not truncated :
    action,_= model.predict(obs)
    obs,reward,terminated,truncated,info = env.step(action)
    steps += 1
print("Episode lenght: ",steps)
env.close()
    


