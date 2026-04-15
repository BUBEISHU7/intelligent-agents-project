import csv
import json
import math
import os
import random
import sys
import time
from statistics import mean
from tkinter import TclError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from env.robot_env import RobotEnvironment


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class SharedBlackboard:
    """
    Central blackboard for explicit communication:
    - shared dirt map (global latest observations)
    - simple nearest-dirt task assignment (one target per robot)
    """

    def __init__(self):
        self.shared_dirt_map = {}  # dirt_name -> (x, y, last_seen_step)
        self.robot_target = {}  # robot_idx -> dirt_name or None

    def update(self, env, step):
        current = env.get_dirt_positions()
        current_names = set()
        for dirt_name, x, y in current:
            self.shared_dirt_map[dirt_name] = (float(x), float(y), int(step))
            current_names.add(dirt_name)

        # Remove cleaned dirt from map.
        stale = [name for name in self.shared_dirt_map if name not in current_names]
        for name in stale:
            del self.shared_dirt_map[name]
            for ridx, target_name in list(self.robot_target.items()):
                if target_name == name:
                    self.robot_target[ridx] = None

    def assign_nearest_tasks(self, agents):
        unassigned = set(self.shared_dirt_map.keys())
        # Keep still-valid previous assignment first.
        for ridx, target_name in list(self.robot_target.items()):
            if target_name in unassigned:
                unassigned.remove(target_name)
            else:
                self.robot_target[ridx] = None

        # Greedy nearest assignment for robots without target.
        for ridx, bot in enumerate(agents):
            if self.robot_target.get(ridx):
                continue
            best_name = None
            best_dist = float("inf")
            for dirt_name in unassigned:
                x, y, _ = self.shared_dirt_map[dirt_name]
                d = math.hypot(bot.x - x, bot.y - y)
                if d < best_dist:
                    best_dist = d
                    best_name = dirt_name
            self.robot_target[ridx] = best_name
            if best_name is not None:
                unassigned.remove(best_name)

    def get_target_position(self, ridx):
        target_name = self.robot_target.get(ridx)
        if not target_name:
            return None
        if target_name not in self.shared_dirt_map:
            self.robot_target[ridx] = None
            return None
        x, y, _ = self.shared_dirt_map[target_name]
        return x, y


class Week2CoordinationController:
    """
    Strategies:
    - no_coordination: random wandering
    - implicit_coordination: keep safe distance, no shared map
    - explicit_coordination: shared blackboard + nearest-dirt assignment
    """

    def __init__(self, mode, num_bots, seed=0):
        self.mode = mode
        self.num_bots = num_bots
        self.rng = random.Random(seed)
        self.base_speed = 4.8
        self.max_turn = 3.0
        self.min_forward = 1.2
        self.max_wheel = 7.0
        self.repulsion_radius = 130.0
        self.wall_margin = 80.0
        self.heading = [self.rng.uniform(-math.pi, math.pi) for _ in range(num_bots)]
        self.blackboard = SharedBlackboard()

    def update_shared_state(self, env, step):
        if self.mode == "explicit_coordination":
            self.blackboard.update(env, step)
            self.blackboard.assign_nearest_tasks(env.agents)

    def __call__(self, ridx, bot, agents, passive_objects):
        self.heading[ridx] += self.rng.uniform(-0.15, 0.15)
        self.heading[ridx] = _wrap_angle(self.heading[ridx])
        vx = math.cos(self.heading[ridx])
        vy = math.sin(self.heading[ridx])

        # Robot-robot repulsion for implicit/explicit.
        if self.mode in ("implicit_coordination", "explicit_coordination"):
            for j, other in enumerate(agents):
                if j == ridx:
                    continue
                dx = bot.x - other.x
                dy = bot.y - other.y
                d = math.hypot(dx, dy) + 1e-6
                if d < self.repulsion_radius:
                    force = (self.repulsion_radius - d) / self.repulsion_radius
                    vx += 2.2 * force * (dx / d)
                    vy += 2.2 * force * (dy / d)

            # Simple wall repulsion.
            if bot.x < self.wall_margin:
                vx += 1.0
            elif bot.x > 1000 - self.wall_margin:
                vx -= 1.0
            if bot.y < self.wall_margin:
                vy += 1.0
            elif bot.y > 1000 - self.wall_margin:
                vy -= 1.0

        # Explicit coordination: move to blackboard-assigned nearest dirt.
        if self.mode == "explicit_coordination":
            target = self.blackboard.get_target_position(ridx)
            if target is not None:
                tx, ty = target
                vx += 0.008 * (tx - bot.x)
                vy += 0.008 * (ty - bot.y)

        desired = math.atan2(vy, vx)
        err = _wrap_angle(desired - bot.theta)
        turn = _clamp(1.8 * err, -self.max_turn, self.max_turn)
        sl = _clamp(self.base_speed - turn, self.min_forward, self.max_wheel)
        sr = _clamp(self.base_speed + turn, self.min_forward, self.max_wheel)
        return sl, sr


def run_one(mode, num_bots=4, num_dirt=300, max_steps=300, seed=0):
    env = RobotEnvironment(num_bots=num_bots, num_dirt=num_dirt, seed=seed)
    env.reset()
    controller = Week2CoordinationController(mode=mode, num_bots=num_bots, seed=seed)
    start = time.time()
    trajectories = {f"robot_{i}": [] for i in range(num_bots)}
    final_info = {}
    done = False

    for step in range(max_steps):
        controller.update_shared_state(env, step)
        try:
            if mode == "no_coordination":
                obs, reward, done, info = env.step(actions=None)
            else:
                obs, reward, done, info = env.step(actions=controller)
        except TclError:
            # If Tk canvas is unexpectedly closed, end this run gracefully.
            done = True
            info = env.get_metrics()
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
    try:
        env.close()
    except TclError:
        pass
    return metrics, trajectories, done


def run_batch(output_dir=None, repetitions=30, max_total_runs=30):
    """
    Week2 experimental flow (default):
    - strategies: 3
    - robot counts: 2/4/6
    - repetitions: configurable (default 30 in design)
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "member_c_week2")
    os.makedirs(output_dir, exist_ok=True)
    modes = ["no_coordination", "implicit_coordination", "explicit_coordination"]
    robot_counts = [2, 4, 6]
    all_rows = []
    completed = 0
    target_runs = max_total_runs

    for mode in modes:
        for num_bots in robot_counts:
            grouped_rows = []
            for rep in range(repetitions):
                if completed >= target_runs:
                    break
                seed = 20260406 + rep + num_bots * 100
                metrics, trajectories, done = run_one(
                    mode=mode,
                    num_bots=num_bots,
                    max_steps=300,
                    seed=seed,
                )
                row = dict(metrics)
                row["finished"] = int(done)
                grouped_rows.append(row)
                all_rows.append(row)
                completed += 1
                print(f"[progress] {completed}/{target_runs}")

                traj_path = os.path.join(
                    output_dir, f"traj_{mode}_{num_bots}bots_seed{seed}.json"
                )
                with open(traj_path, "w", encoding="utf-8") as f:
                    json.dump(trajectories, f, ensure_ascii=False, indent=2)

            if grouped_rows:
                summary = {
                    "mode": mode,
                    "num_bots": num_bots,
                    "repetitions_done": len(grouped_rows),
                    "avg_coverage": round(mean([r["coverage"] for r in grouped_rows]), 4),
                    "avg_collisions": round(mean([r["collisions"] for r in grouped_rows]), 4),
                    "avg_steps": round(mean([r["steps"] for r in grouped_rows]), 2),
                    "avg_performance_score": round(
                        mean([r["performance_score"] for r in grouped_rows]), 4
                    ),
                }
                summary_path = os.path.join(output_dir, f"summary_{mode}_{num_bots}bots.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"[done] {mode}/{num_bots} -> {summary}")
        if completed >= target_runs:
            break

    if all_rows:
        keys = list(all_rows[0].keys())
        csv_path = os.path.join(output_dir, "member_c_week2_all_runs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Saved outputs to: {os.path.abspath(output_dir)}")
    print(f"Batch completed: {completed}/{target_runs} runs.")


if __name__ == "__main__":
    # Quick demo: 9 runs = 3 strategies × 3 robot counts (one rep each).
    # Full study per design: use repetitions=30 and max_total_runs=270 (or omit max_total_runs).
    run_batch(repetitions=1, max_total_runs=9)
