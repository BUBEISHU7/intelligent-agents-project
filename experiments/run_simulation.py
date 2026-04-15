import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from agents.goap_agent import GOAPTeamController
from env.robot_env import RobotEnvironment


def run_test_once_and_save(
    num_bots=1,
    num_dirt=50,
    max_steps=500,
    planner_algorithm="dstar",
    num_dynamic_obstacles=2,
    sensor_noise=2.0,
    execution_noise=0.3,
):
    env = RobotEnvironment(
        num_bots=num_bots,
        num_dirt=num_dirt,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_steps=max_steps,
        noise_config={
            "sensor_gaussian_std": sensor_noise,
            "sensor_miss_rate": 0.05 if sensor_noise > 0 else 0.0,
            "execution_noise_std": execution_noise,
        },
    )
    obs = env.reset()
    start_time = time.time()

    trajectories = {f"robot_{i}": [] for i in range(num_bots)}
    strategy = f"goap_{planner_algorithm}"
    controller = GOAPTeamController(env, planner_algorithm=planner_algorithm)

    for step in range(max_steps):
        actions = controller.compute_actions(obs)
        obs, reward, done, info = env.step(actions)

        for i, agent in enumerate(env.agents):
            trajectories[f"robot_{i}"].append([step, agent.x, agent.y])

        if step % 100 == 0:
            print(
                f"[{step:04d} 步] 覆盖率: {info['coverage']:.2%} | "
                f"碰撞: {info['collisions']} 次 | 动态碰撞: {info['dynamic_collisions']} 次"
            )
        if done:
            break

    final_metrics = env.get_metrics()
    total_runtime = time.time() - start_time

    save_data = {
        "simulation_metadata": {
            "strategy": strategy,
            "num_bots": num_bots,
            "num_dynamic_obstacles": num_dynamic_obstacles,
            "sensor_noise": sensor_noise,
            "execution_noise": execution_noise,
            "runtime_seconds": round(total_runtime, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "performance_results": final_metrics,
        "trajectories": trajectories,
    }

    os.makedirs("../data", exist_ok=True)

    for robot_name, traj in trajectories.items():
        csv_path = f"../data/{strategy}_{robot_name}_trajectory.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y"])
            writer.writerows(traj)

    json_path = f"../data/{strategy}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 45)
    print("Experimental Performance Evaluation Report")
    print("=" * 45)
    for key, value in final_metrics.items():
        print(f"{key:20}: {value}")
    print("-" * 45)
    print(f"Actual running time consumed: {total_runtime:.2f} 秒")
    print("=" * 45)

    num_robots = len(trajectories)
    fig, axes = plt.subplots(1, num_robots, figsize=(6 * num_robots, 6))
    if num_robots == 1:
        axes = [axes]

    for idx, (robot_name, traj) in enumerate(trajectories.items()):
        traj_np = np.array(traj)
        ax = axes[idx]
        ax.plot(traj_np[:, 1], traj_np[:, 2], linewidth=1)
        ax.scatter(traj_np[0, 1], traj_np[0, 2], c="red", s=40, label="Start")
        ax.scatter(traj_np[-1, 1], traj_np[-1, 2], c="green", s=40, label="End")
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
    print(f"Done. {num_bots} robots, Steps taken: {step + 1}/{max_steps}")


if __name__ == "__main__":
    run_test_once_and_save(num_bots=1, num_dirt=300, max_steps=500)