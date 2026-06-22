import pandas as pd

baseline = pd.read_csv("logs/baseline.monitor.csv", skiprows=1)
velocity = pd.read_csv("logs/velocity.monitor.csv", skiprows=1)
angle = pd.read_csv("logs/angle.monitor.csv", skiprows=1)

print("Baseline")
print("Last reward:", baseline["r"].iloc[-1])
print("Best reward:", baseline["r"].max())
print("Average of last 20:", baseline["r"].tail(20).mean())

print("\nVelocity")
print("Last reward:", velocity["r"].iloc[-1])
print("Best reward:", velocity["r"].max())
print("Average of last 20:", velocity["r"].tail(20).mean())

print("\nAngle")
print("Last reward:", angle["r"].iloc[-1])
print("Best reward:", angle["r"].max())
print("Average of last 20:", angle["r"].tail(20).mean())