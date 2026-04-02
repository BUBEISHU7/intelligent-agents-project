import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
import time
import sys
sys.path.append("..")
from env.robot_env import RobotEnvironment

def run_test_once_and_save():
    env = RobotEnvironment(num_bots=1, num_dirt=300)
    obs = env.reset()
    start_time = time.time()

    trajectory = []
    strategy = "random"
    
    max_steps = 500 
    for step in range(max_steps):
        print("Current step:", step)
        
        obs, reward, done, info = env.step(None)
        if done:
            break
            
        x, y = env.agents[0].x, env.agents[0].y
        trajectory.append([step, x, y])

    total_runtime = time.time() - start_time
    coverage = info["coverage"]
    remaining_dirt = info["remaining_dirt"] / env.num_dirt

    os.makedirs("../data", exist_ok=True)

    csv_path  = f"../data/{strategy}_trajectory.csv"
    json_path = f"../data/{strategy}_metrics.json"
    img_path  = f"../data/{strategy}_trajectory.png"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y"])
        writer.writerows(trajectory)  

    metrics = {
        "strategy": strategy,
        "total_steps": step,
        "coverage_rate": float(coverage),
        "remaining_dirt_ratio": float(remaining_dirt),
        "runtime_seconds": round(total_runtime, 2),
        "stopped_by_max_steps": step >= max_steps - 1 
    }
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    traj_np = np.array(trajectory)
    plt.figure(figsize=(6,6))
    plt.plot(traj_np[:,1], traj_np[:,2], linewidth=1)
    plt.scatter(traj_np[0,1], traj_np[0,2], c="red", s=40, label="Start")
    plt.scatter(traj_np[-1,1], traj_np[-1,2], c="green", s=40, label="End")
    plt.title(f"Robot Trajectory ({strategy})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    env.close()

    print(f"Done. Data saved. Steps taken: {step}/{max_steps}")

if __name__ == "__main__":
    run_test_once_and_save()