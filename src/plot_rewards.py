import pandas as pd
import matplotlib.pyplot as plt

def load_monitor_file(filename):
    df = pd.read_csv(filename,skiprows = 1)
    rewards = df["r"]
    smoothed = rewards.rolling(window=20,min_periods=1).mean()
    return smoothed

baseline = load_monitor_file("logs/baseline.monitor.csv")
velocity = load_monitor_file("logs/velocity.monitor.csv")
angle = load_monitor_file("logs/angle.monitor.csv")

plt.figure(figsize=(10,6))

plt.plot(baseline,label="Baseline Reward")
plt.plot(velocity,label="Velocity Reward")
plt.plot(angle,label="Angle Reward")

plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("CartPole Training under Different Reward Functions")

plt.legend()
plt.grid(True)

plt.savefig("reward_comparison.png")
plt.show()