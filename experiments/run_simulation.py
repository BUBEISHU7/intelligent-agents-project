import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
import time
import sys
sys.path.append("..")
from env.robot_env import RobotEnvironment

def run_test_once_and_save(num_bots=1, num_dirt=50, max_steps=500):
    env = RobotEnvironment(num_bots=num_bots, num_dirt=num_dirt)
    obs = env.reset()
    start_time = time.time()

    trajectories = {f"robot_{i}": [] for i in range(num_bots)}
    strategy = "random"
    
    for step in range(max_steps):
        print("Current step:", step)
        
        obs, reward, done, info = env.step(None)
        if done:
            break
        
        for i, agent in enumerate(env.agents):
            x, y = agent.x, agent.y
            trajectories[f"robot_{i}"].append([step, x, y])

    total_runtime = time.time() - start_time
    coverage = info["coverage"]
    remaining_dirt = info["remaining_dirt"] / env.num_dirt

    os.makedirs("../data", exist_ok=True)

    for robot_name, traj in trajectories.items():
        csv_path = f"../data/{strategy}_{robot_name}_trajectory.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y"])
            writer.writerows(traj)

  
    metrics = {
        "strategy": strategy,
        "num_bots": num_bots,
        "total_steps": step,
        "coverage_rate": float(coverage),
        "remaining_dirt_ratio": float(remaining_dirt),
        "runtime_seconds": round(total_runtime, 2),
        "stopped_by_max_steps": step >= max_steps - 1,
        "trajectories": {name: traj for name, traj in trajectories.items()}  
    }
    json_path = f"../data/{strategy}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    num_robots = len(trajectories)
    fig, axes = plt.subplots(1, num_robots, figsize=(6*num_robots, 6))
    if num_robots == 1:
        axes = [axes]
    
    for idx, (robot_name, traj) in enumerate(trajectories.items()):
        traj_np = np.array(traj)
        ax = axes[idx]
        ax.plot(traj_np[:,1], traj_np[:,2], linewidth=1)
        ax.scatter(traj_np[0,1], traj_np[0,2], c="red", s=40, label="Start")
        ax.scatter(traj_np[-1,1], traj_np[-1,2], c="green", s=40, label="End")
        ax.set_title(f"{robot_name} Trajectory ({strategy})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    img_path = f"../data/{strategy}_trajectory.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    env.close()
    print(f"Done. {num_bots} robots, Steps taken: {step}/{max_steps}")

if __name__ == "__main__":
    run_test_once_and_save(num_bots=1, num_dirt=300, max_steps=500)