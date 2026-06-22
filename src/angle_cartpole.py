from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class AngleCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode = None):
        super().__init__(render_mode=render_mode)
    def step(self,action):
        obs,reward,terminated,truncated,info = super().step(action)
        reward = abs(obs[2])
        
        return obs,reward,terminated,truncated,info