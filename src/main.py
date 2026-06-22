import  gymnasium as gym  # type: ignore

env = gym.make("CartPole-v1",render_mode="human")

observation, info = env.reset()

print("START")
print(observation)

for step in range(10):
    observation,reward,terminated,truncated,info = env.step(0)

    print(f"\n Step {step+1}")
    print(observation)

    if terminated or truncated :
        print("episode ended")
        break

env.close()