import matplotlib.pyplot as plt

experiments = ["Baseline", "Velocity", "Angle"]
rewards = [500.00, 124.39, 56.08]

plt.figure(figsize=(8,5))

bars = plt.bar(experiments, rewards)

plt.title("Average Reward Over Last 20 Episodes")
plt.xlabel("Experiment")
plt.ylabel("Average Reward")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center"
    )

plt.show()
plt.savefig("reward_comparison_bar.png")