import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
import time
import sys
sys.path.append("..")
from env.robot_env import RobotEnvironment
from agents.goap_agent import GOAPAgent

def run_test_once_and_save(num_bots=1, num_dirt=50, max_steps=500):
    env = RobotEnvironment(num_bots=num_bots, num_dirt=num_dirt)
    obs = env.reset()
    start_time = time.time()

    trajectories = {f"robot_{i}": [] for i in range(num_bots)}
    strategy = "goap"
    agents = [GOAPAgent() for _ in range(num_bots)]

    for step in range(max_steps):
        # print("Current step:", step)


        actions = []

        for i in range(num_bots):
            action = agents[i].act(obs[i], env.agents[i], env.passive_objects)
            actions.append(action)

        obs, reward, done, info = env.step(actions)
        for i, agent in enumerate(env.agents):
            trajectories[f"robot_{i}"].append([step, agent.x, agent.y])

            # 每 100 步打印一次标准化进度
        if step % 100 == 0:
            print(f"[{step:04d} 步] 覆盖率: {info['coverage']:.2%} | 碰撞: {info['collisions']} 次")
        if done:
            break

        # for i, agent in enumerate(env.agents):
        #     x, y = agent.x, agent.y
        #     trajectories[f"robot_{i}"].append([step, x, y])

    final_metrics = env.get_metrics()
    total_runtime = time.time() - start_time

    save_data = {
        "simulation_metadata": {
            "strategy": strategy,
            "num_bots": num_bots,
            "runtime_seconds": round(total_runtime, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance_results": final_metrics,  # 包含覆盖率、碰撞、路径效率等
        "trajectories": trajectories
    }

    # coverage = info["coverage"]
    # remaining_dirt = info["remaining_dirt"] / env.num_dirt

    os.makedirs("../data", exist_ok=True)

    for robot_name, traj in trajectories.items():
        csv_path = f"../data/{strategy}_{robot_name}_trajectory.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y"])
            writer.writerows(traj)


    # metrics = {
    #     "strategy": strategy,
    #     "num_bots": num_bots,
    #     "total_steps": step,
    #     "coverage_rate": float(coverage),
    #     "remaining_dirt_ratio": float(remaining_dirt),
    #     "runtime_seconds": round(total_runtime, 2),
    #     "stopped_by_max_steps": step >= max_steps - 1,
    #     "trajectories": {name: traj for name, traj in trajectories.items()}
    # }
    json_path = f"../data/{strategy}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)

    print("\n" + "=" * 45)
    print("Experimental Performance Evaluation Report")
    print("=" * 45)
    for key, value in final_metrics.items():
        print(f"{key:20}: {value}")
    print("-" * 45)
    print(f"Actual running time consumed: {total_runtime:.2f} 秒")
    print("=" * 45)

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