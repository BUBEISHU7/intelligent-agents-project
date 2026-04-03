import csv
import json
import math
import os
import random
import sys
import time
from statistics import mean
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.append("..")
from env.robot_env import RobotEnvironment
from env.simpleBot2 import Obstacle


def _clamp(value, low, high):
    return max(low, min(high, value))


def _wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class CoordinationController:
    """
    Week1 coordination controller for member C.
    Modes:
    - no_coordination: independent random headings
    - implicit_coordination: neighbor/wall/obstacle repulsion
    - explicit_coordination: implicit + region attraction (task partition)
    """

    def __init__(self, mode, num_bots, seed=0, canvas_size=1000):
        self.mode = mode
        self.num_bots = num_bots
        self.canvas_size = canvas_size
        self.rng = random.Random(seed)
        self.base_speed = 5.0
        # 轮速差异只用于转向，避免 sl/sr 变负导致原地旋转
        self.max_turn = 3.2
        self.min_forward_speed = 1.5
        self.max_wheel_speed = 7.0
        self.robot_repulsion_radius = 140.0
        self.robot_emergency_radius = 85.0
        self.obstacle_repulsion_radius = 140.0
        self.wall_margin = 90.0
        self.heading = [self.rng.uniform(-math.pi, math.pi) for _ in range(num_bots)]

        # Stuck detection: if a robot doesn't move for many controller calls,
        # force a stronger heading reset so it can escape local oscillations.
        self.prev_pos = [None for _ in range(num_bots)]
        self.stuck_count = [0 for _ in range(num_bots)]
        self.stuck_threshold = 18
        self.stuck_move_eps = 1.0

    def __call__(self, idx, bot, agents, passive_objects):
        # Update stuck counter based on recent motion.
        if self.prev_pos[idx] is not None:
            last_x, last_y = self.prev_pos[idx]
            moved = math.hypot(bot.x - last_x, bot.y - last_y)
            if moved < self.stuck_move_eps:
                self.stuck_count[idx] += 1
            else:
                self.stuck_count[idx] = 0

        self.prev_pos[idx] = (bot.x, bot.y)
        if self.stuck_count[idx] >= self.stuck_threshold:
            # "Escape" by resetting heading and injecting more randomness once.
            self.heading[idx] = self.rng.uniform(-math.pi, math.pi)
            self.stuck_count[idx] = 0
            self.heading[idx] += self.rng.uniform(-0.6, 0.6)

        self.heading[idx] += self.rng.uniform(-0.18, 0.18)
        self.heading[idx] = _wrap_angle(self.heading[idx])

        vx = math.cos(self.heading[idx])
        vy = math.sin(self.heading[idx])

        if self.mode in ("implicit_coordination", "explicit_coordination"):
            nearest_dist = 1e9
            nearest_dx, nearest_dy = 0.0, 0.0
            for j, other in enumerate(agents):
                if j == idx:
                    continue
                dx = bot.x - other.x
                dy = bot.y - other.y
                dist = math.hypot(dx, dy) + 1e-6
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_dx, nearest_dy = dx, dy
                if dist < self.robot_repulsion_radius:
                    force = (self.robot_repulsion_radius - dist) / self.robot_repulsion_radius
                    vx += 2.8 * force * (dx / dist)
                    vy += 2.8 * force * (dy / dist)

            for obj in passive_objects:
                if not isinstance(obj, Obstacle):
                    continue
                dx = bot.x - obj.centreX
                dy = bot.y - obj.centreY
                dist = math.hypot(dx, dy) + 1e-6
                if dist < self.obstacle_repulsion_radius:
                    force = (self.obstacle_repulsion_radius - dist) / self.obstacle_repulsion_radius
                    vx += 1.2 * force * (dx / dist)
                    vy += 1.2 * force * (dy / dist)

            if bot.x < self.wall_margin:
                vx += 1.2
            elif bot.x > self.canvas_size - self.wall_margin:
                vx -= 1.2
            if bot.y < self.wall_margin:
                vy += 1.2
            elif bot.y > self.canvas_size - self.wall_margin:
                vy -= 1.2

            # Emergency rule: if another robot is too close, turn away immediately.
            if nearest_dist < self.robot_emergency_radius:
                away_theta = math.atan2(nearest_dy, nearest_dx)
                hard_err = _wrap_angle(away_theta - bot.theta)
                # Still enforce forward motion to avoid spinning in place.
                hard_turn = _clamp(2.2 * hard_err, -self.max_turn, self.max_turn)
                safe_speed = 2.6
                sl = _clamp(safe_speed - hard_turn, self.min_forward_speed, self.max_wheel_speed)
                sr = _clamp(safe_speed + hard_turn, self.min_forward_speed, self.max_wheel_speed)
                return sl, sr

        if self.mode == "explicit_coordination":
            sector_w = self.canvas_size / max(1, self.num_bots)
            target_x = min(self.canvas_size - 50.0, (idx + 0.5) * sector_w)
            target_y = self.canvas_size * 0.5
            vx += 0.003 * (target_x - bot.x)
            vy += 0.003 * (target_y - bot.y)

        desired_theta = math.atan2(vy, vx)
        err = _wrap_angle(desired_theta - bot.theta)
        turn = _clamp(1.6 * err, -self.max_turn, self.max_turn)

        sl = _clamp(self.base_speed - turn, self.min_forward_speed, self.max_wheel_speed)
        sr = _clamp(self.base_speed + turn, self.min_forward_speed, self.max_wheel_speed)
        return sl, sr


def run_one(mode, num_bots=4, num_dirt=300, max_steps=200, seed=0):
    env = RobotEnvironment(num_bots=num_bots, num_dirt=num_dirt, seed=seed)
    env.reset()
    controller = CoordinationController(mode=mode, num_bots=num_bots, seed=seed)
    start = time.time()

    trajectories = {f"robot_{i}": [] for i in range(num_bots)}
    done = False
    final_info = {}

    for step in range(max_steps):
        # Always drive robots via this controller so that we can apply
        # the anti-stuck logic consistently across modes.
        _, _, done, info = env.step(actions=controller)
        final_info = info
        for i, bot in enumerate(env.agents):
            trajectories[f"robot_{i}"].append([step, round(bot.x, 3), round(bot.y, 3)])
        if done:
            break

    elapsed = round(time.time() - start, 2)
    metrics = {
        "mode": mode,
        "seed": seed,
        "num_bots": num_bots,
        "num_dirt": num_dirt,
        "runtime_seconds": elapsed,
    }
    metrics.update(final_info)
    env.close()
    return metrics, trajectories, done


def _generate_visual_reports(all_rows, output_dir):
    if not all_rows:
        return

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["mode"], row["num_bots"])].append(row)

    modes = ["no_coordination", "implicit_coordination", "explicit_coordination"]
    robot_counts = sorted({int(r["num_bots"]) for r in all_rows})
    metrics = ["coverage", "collisions", "performance_score"]
    metric_titles = {
        "coverage": "Average Coverage",
        "collisions": "Average Collisions",
        "performance_score": "Average Performance Score",
    }

    summary_rows = []
    for mode in modes:
        for n in robot_counts:
            rows = grouped.get((mode, n), [])
            if not rows:
                continue
            summary_rows.append(
                {
                    "mode": mode,
                    "num_bots": n,
                    "coverage": mean([r["coverage"] for r in rows]),
                    "collisions": mean([r["collisions"] for r in rows]),
                    "performance_score": mean([r["performance_score"] for r in rows]),
                }
            )

    # 1) Three metric charts in one figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for mode in modes:
            x_vals = []
            y_vals = []
            for n in robot_counts:
                vals = [r[metric] for r in summary_rows if r["mode"] == mode and r["num_bots"] == n]
                if vals:
                    x_vals.append(n)
                    y_vals.append(vals[0])
            if x_vals:
                ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=mode)
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Number of Robots")
        ax.grid(True, alpha=0.3)
        if metric == "collisions":
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("Value")
    axes[0].legend()
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "member_c_week1_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Summary table image
    table_data = [
        [
            r["mode"],
            r["num_bots"],
            f"{r['coverage']:.4f}",
            f"{r['collisions']:.2f}",
            f"{r['performance_score']:.3f}",
        ]
        for r in summary_rows
    ]
    fig2, ax2 = plt.subplots(figsize=(11, 0.45 * max(4, len(table_data)) + 1.8))
    ax2.axis("off")
    table = ax2.table(
        cellText=table_data,
        colLabels=["mode", "num_bots", "avg_coverage", "avg_collisions", "avg_score"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    plt.title("Member C Week1 Summary", pad=12)
    table_path = os.path.join(output_dir, "member_c_week1_summary_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved visualization: {chart_path}")
    print(f"Saved table image: {table_path}")


def run_batch(output_dir="../data/member_c_week1", repetitions=5, max_total_runs=None):
    os.makedirs(output_dir, exist_ok=True)
    modes = ["no_coordination", "implicit_coordination", "explicit_coordination"]
    robot_counts = [2, 4, 6]
    all_rows = []
    completed_runs = 0
    planned_runs = len(modes) * len(robot_counts) * repetitions
    target_runs = planned_runs if max_total_runs is None else min(max_total_runs, planned_runs)

    for mode in modes:
        for num_bots in robot_counts:
            mode_rows = []
            for rep in range(repetitions):
                if completed_runs >= target_runs:
                    break
                seed = 20260403 + rep + num_bots * 100
                metrics, trajectories, done = run_one(
                    mode=mode,
                    num_bots=num_bots,
                    max_steps=200,
                    seed=seed,
                )
                row = dict(metrics)
                row["finished"] = int(done)
                all_rows.append(row)
                mode_rows.append(row)
                completed_runs += 1
                print(f"[progress] {completed_runs}/{target_runs} runs finished")

                traj_path = os.path.join(
                    output_dir, f"traj_{mode}_{num_bots}bots_seed{seed}.json"
                )
                with open(traj_path, "w", encoding="utf-8") as f:
                    json.dump(trajectories, f, ensure_ascii=False, indent=2)

            if not mode_rows:
                continue

            summary = {
                "mode": mode,
                "num_bots": num_bots,
                "repetitions": repetitions,
                "avg_coverage": round(mean([r["coverage"] for r in mode_rows]), 4),
                "avg_collisions": round(mean([r["collisions"] for r in mode_rows]), 4),
                "avg_steps": round(mean([r["steps"] for r in mode_rows]), 2),
                "avg_performance_score": round(mean([r["performance_score"] for r in mode_rows]), 4),
            }
            summary_path = os.path.join(output_dir, f"summary_{mode}_{num_bots}bots.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[done] {mode} / {num_bots} bots -> {summary}")
        if completed_runs >= target_runs:
            break

    csv_path = os.path.join(output_dir, "member_c_week1_all_runs.csv")
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)

    _generate_visual_reports(all_rows, output_dir)
    print(f"Saved all outputs to: {os.path.abspath(output_dir)}")
    print(f"Batch completed automatically: {completed_runs}/{target_runs} runs.")


if __name__ == "__main__":
    # 默认只跑 9 次（便于快速出结果），每次最多 200 步
    run_batch(repetitions=3, max_total_runs=9)
