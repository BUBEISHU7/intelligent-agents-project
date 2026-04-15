from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.robot_env import RobotEnvironment
from agents.goap_agent import GOAPTeamController
from env.astar import AStarPlanner
from env.simpleBot2 import Dirt
from experiments.member_c_week2_blackboard import Week2CoordinationController


@dataclass
class RunRow:
    phase: str
    condition: str
    seed: int
    planner_type: str
    coordination: str
    num_bots: int
    noise_level: str
    completed: int
    success: int
    steps: int
    coverage: float
    collisions: int
    dynamic_collisions: int
    total_distance: float
    path_efficiency: float
    runtime_seconds: float


class ReactiveController:
    def __init__(self, num_bots: int):
        self.num_bots = num_bots

    def compute_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        out = []
        for _ in obs:
            if random.random() < 0.25:
                turn = random.choice([-4.5, 4.5])
                out.append((-turn, turn))
            else:
                v = 6.5 + random.uniform(-1.0, 1.0)
                out.append((v, v))
        return out


class SimplePlannerController:
    """Single-robot planner controller for clean P2 baseline."""

    def __init__(self, env: RobotEnvironment):
        self.env = env
        self.max_speed = 8.5
        self.min_forward = 2.0
        self.turn_gain = 2.6
        self.waypoint_reach = 26.0
        self.replan_interval = 6
        self.clean_trigger_dist = 30.0
        # Do not spend multiple idle steps at each target; keep sweeping motion continuous.
        self.clean_hold_steps = 0
        self.current_target: Optional[Tuple[float, float]] = None
        self.path: List[Tuple[float, float]] = []
        self.steps_since_replan = 0
        self.clean_hold = 0
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.stuck_steps = 0
        self.last_target_dist: float = float("inf")
        self.no_progress_steps = 0

    def _collect_dirt(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for obj in self.env.passive_objects:
            if isinstance(obj, Dirt):
                out.append((float(obj.centreX), float(obj.centreY)))
        return out

    def _pick_target(self, pos: Tuple[float, float], dirt: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not dirt:
            return None
        best = None
        best_score = float("inf")
        for d in dirt:
            dist = math.hypot(pos[0] - d[0], pos[1] - d[1])
            # Prefer dirt in local clusters to improve short-horizon cleaning yield.
            local_density = 0
            for q in dirt:
                if math.hypot(d[0] - q[0], d[1] - q[1]) <= 110.0:
                    local_density += 1
            # Reduce over-bias to dense local clusters and keep some incentive to expand outward.
            score = 0.82 * dist - 8.0 * local_density
            if score < best_score:
                best_score = score
                best = d
        return best

    def _plan_path(self, start: Tuple[float, float], target: Optional[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if target is None:
            return []
        planner = AStarPlanner(self.env.get_grid_map(include_dynamic=True), resolution=self.env.grid_resolution)
        path = planner.plan(start, target) or []
        if path and len(path) > 1:
            path = path[1:]
        if not path and math.hypot(start[0] - target[0], start[1] - target[1]) > self.waypoint_reach:
            path = [target]
        return path

    def _track(self, pos: Tuple[float, float], theta: float, waypoint: Tuple[float, float]) -> Tuple[float, float]:
        dx = waypoint[0] - pos[0]
        dy = waypoint[1] - pos[1]
        desired_heading = math.atan2(dy, dx)
        angle_diff = math.atan2(math.sin(desired_heading - theta), math.cos(desired_heading - theta))
        dist = math.hypot(dx, dy)
        # Keep moving forward while steering; avoid in-place spins that kill coverage.
        forward = min(self.max_speed, max(self.min_forward, dist * 0.18))
        turn = self.turn_gain * angle_diff
        turn = max(-forward * 0.85, min(forward * 0.85, turn))
        left = max(-self.max_speed, min(self.max_speed, forward - turn))
        right = max(-self.max_speed, min(self.max_speed, forward + turn))
        return (left, right)

    def compute_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        if not obs:
            return []
        o = obs[0]
        pos = (float(o["x"]), float(o["y"]))
        theta = float(o["theta"])

        if self.prev_pos is not None:
            moved = math.hypot(pos[0] - self.prev_pos[0], pos[1] - self.prev_pos[1])
            self.stuck_steps = self.stuck_steps + 1 if moved < 2.0 else 0
        self.prev_pos = pos

        dirt = self._collect_dirt()
        if self.current_target is not None:
            near_target = math.hypot(pos[0] - self.current_target[0], pos[1] - self.current_target[1]) < 35.0
            target_exists = any(math.hypot(d[0] - self.current_target[0], d[1] - self.current_target[1]) < 8.0 for d in dirt)
            if not target_exists:
                self.current_target = None
                self.path = []
                self.clean_hold = 0
                self.last_target_dist = float("inf")
                self.no_progress_steps = 0
            else:
                dist_t = math.hypot(pos[0] - self.current_target[0], pos[1] - self.current_target[1])
                if dist_t <= self.clean_trigger_dist:
                    self.clean_hold += 1
                else:
                    self.clean_hold = 0
                if dist_t > self.last_target_dist - 2.0:
                    self.no_progress_steps += 1
                else:
                    self.no_progress_steps = 0
                self.last_target_dist = dist_t
                if near_target and self.no_progress_steps > 24:
                    # If we are close but not getting cleaner, pick another target.
                    self.current_target = None
                    self.path = []
                    self.clean_hold = 0
                    self.last_target_dist = float("inf")
                    self.no_progress_steps = 0

        if self.current_target is None:
            self.current_target = self._pick_target(pos, dirt)
            self.path = []

        self.steps_since_replan += 1
        changed_cells = self.env.consume_changed_cells() if hasattr(self.env, "consume_changed_cells") else self.env.get_changed_cells()
        if self.steps_since_replan >= self.replan_interval or not self.path or bool(changed_cells):
            self.path = self._plan_path(pos, self.current_target)
            self.steps_since_replan = 0

        if self.stuck_steps >= 10:
            self.stuck_steps = 0
            return [(-3.0, 2.0)]

        if self.current_target is not None and self.clean_hold > 0 and self.clean_hold <= self.clean_hold_steps:
            # Pause briefly at target to guarantee dirt collection trigger.
            return [(0.0, 0.0)]

        if not self.path:
            return [(0.0, 0.0)]
        while len(self.path) > 1 and math.hypot(pos[0] - self.path[0][0], pos[1] - self.path[0][1]) < self.waypoint_reach:
            self.path.pop(0)
        return [self._track(pos, theta, self.path[0])]


def _noise_cfg(level: str) -> Dict[str, float]:
    table = {
        "0": {"sensor_gaussian_std": 0.0, "sensor_miss_rate": 0.0, "execution_noise_std": 0.0},
        "low": {"sensor_gaussian_std": 1.0, "sensor_miss_rate": 0.02, "execution_noise_std": 0.1},
        "mid": {"sensor_gaussian_std": 2.0, "sensor_miss_rate": 0.05, "execution_noise_std": 0.25},
        "high": {"sensor_gaussian_std": 4.0, "sensor_miss_rate": 0.12, "execution_noise_std": 0.5},
    }
    return table[level]


def _make_controller(kind: str, env: RobotEnvironment, coordination: str, phase: str, p2_controller: str):
    if kind == "planning":
        if phase == "P2_single_agent" and len(env.agents) == 1 and p2_controller == "simple":
            return SimplePlannerController(env)
        ctrl = GOAPTeamController(env, planner_algorithm="dstar")
        # Make coordination level materially affect planning behavior.
        if coordination == "none":
            ctrl.config["partition_cleaning"] = False
            ctrl.config["yield_speed_scale"] = 1.0
            ctrl.config["team_deconflict_dist"] = 0.0
            ctrl.config["avoidance_mode"] = "ttc"
        elif coordination == "basic":
            ctrl.config["partition_cleaning"] = True
            ctrl.config["partition_overlap"] = 60.0
            ctrl.config["yield_speed_scale"] = 0.75
            ctrl.config["team_deconflict_dist"] = 65.0
            ctrl.config["avoidance_mode"] = "ttc"
        elif coordination == "enhanced":
            ctrl.config["partition_cleaning"] = True
            ctrl.config["partition_overlap"] = 90.0
            ctrl.config["yield_speed_scale"] = 0.72
            ctrl.config["team_deconflict_dist"] = 70.0
            ctrl.config["avoidance_mode"] = "orca"
            ctrl.config["auction_interval"] = 10
            ctrl.config["replan_interval"] = 10
        else:
            ctrl.config["avoidance_mode"] = "orca"
        return ctrl
    if kind == "reactive":
        if coordination in ("basic", "enhanced"):
            mode = "implicit_coordination" if coordination == "basic" else "explicit_coordination"
            return Week2CoordinationController(mode, num_bots=len(env.agents), seed=random.randint(0, 10**6))
        return ReactiveController(len(env.agents))
    raise ValueError(kind)


def _step_actions(controller, env: RobotEnvironment, obs, coordination: str):
    if isinstance(controller, GOAPTeamController):
        return controller.compute_actions(obs)
    if isinstance(controller, SimplePlannerController):
        return controller.compute_actions(obs)
    if isinstance(controller, ReactiveController):
        return controller.compute_actions(obs)
    # Week2CoordinationController
    controller.update_shared_state(env, env.total_steps)
    return [controller(i, env.agents[i], env.agents, env.passive_objects) for i in range(len(env.agents))]


def run_episode(
    *,
    planner_type: str,
    coordination: str,
    num_bots: int,
    noise_level: str,
    num_dirt: int,
    max_steps: int,
    seed: int,
    phase: str = "",
    num_dynamic_obstacles: int = 3,
    p2_controller: str = "simple",
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    env = RobotEnvironment(
        num_bots=num_bots,
        num_dirt=num_dirt,
        seed=seed,
        num_dynamic_obstacles=num_dynamic_obstacles,
        dynamic_obstacle_speed=2.2,
        noise_config=_noise_cfg(noise_level),
        render=False,
        max_steps=max_steps,
    )
    obs = env.reset()
    controller = _make_controller(planner_type, env, coordination, phase, p2_controller)
    t0 = time.time()
    done = False
    info = {}
    for _ in range(max_steps):
        actions = _step_actions(controller, env, obs, coordination)
        obs, _, done, info = env.step(actions)
        if done:
            break
    runtime = time.time() - t0
    m = env.get_metrics()
    env.close()
    return {
        "completed": int(done),
        "success": int(m.get("success", 0)),
        "steps": int(m.get("steps", max_steps)),
        "coverage": float(m.get("coverage", 0.0)),
        "collisions": int(m.get("collisions", 0)),
        "dynamic_collisions": int(m.get("dynamic_collisions", 0)),
        "total_distance": float(m.get("total_distance", 0.0)),
        "path_efficiency": float(m.get("path_efficiency", 0.0)),
        "runtime_seconds": float(runtime),
    }


def _perm_pvalue(a: Sequence[float], b: Sequence[float], n_perm: int = 2000) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(pool)
        aa = pool[: len(a)]
        bb = pool[len(a) :]
        if abs(aa.mean() - bb.mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def _stable_seed(base_seed: int, parts: Sequence[object]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False) % (2**31 - 1)


def _holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: float(pvals[i]))
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        p = float(pvals[i])
        val = (m - rank) * p
        if val < prev:
            val = prev
        prev = val
        adj[i] = min(1.0, val)
    return adj


def _write_plots(df: pd.DataFrame, out_dir: str) -> None:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # P4: Noise-performance curve (path efficiency)
    sub = (
        df[df["phase"] == "P4_noise_robustness"]
        .groupby(["noise_level", "planner_type"])["path_efficiency"]
        .mean()
        .reset_index()
    )
    order = ["0", "low", "mid", "high"]
    plt.figure(figsize=(8, 4))
    for p in sorted(sub["planner_type"].unique()):
        g = sub[sub["planner_type"] == p]
        xs = [order.index(x) for x in g["noise_level"]]
        plt.plot(xs, g["path_efficiency"], marker="o", label=p)
    plt.xticks(range(len(order)), order)
    plt.xlabel("noise_level")
    plt.ylabel("mean path_efficiency")
    plt.title("Noise vs Path Efficiency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "noise_path_efficiency_curve.png"), dpi=150)
    plt.close()

    # P4: Noise-performance curve (coverage)
    sub_cov = (
        df[df["phase"] == "P4_noise_robustness"]
        .groupby(["noise_level", "planner_type"])["coverage"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 4))
    for p in sorted(sub_cov["planner_type"].unique()):
        g = sub_cov[sub_cov["planner_type"] == p]
        xs = [order.index(x) for x in g["noise_level"]]
        plt.plot(xs, g["coverage"], marker="o", label=p)
    plt.xticks(range(len(order)), order)
    plt.xlabel("noise_level")
    plt.ylabel("mean coverage")
    plt.title("Noise vs Coverage")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "noise_coverage_curve.png"), dpi=150)
    plt.close()

    # Global: Coverage boxplot by planner
    plt.figure(figsize=(7, 4))
    vals = [df[df["planner_type"] == p]["coverage"].values for p in ["planning", "reactive"]]
    plt.boxplot(vals, labels=["planning", "reactive"])
    plt.ylabel("coverage")
    plt.title("Coverage: planning vs reactive")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "coverage_planner_boxplot.png"), dpi=150)
    plt.close()

    # P2: Path efficiency boxplot by planner
    p2 = df[df["phase"] == "P2_single_agent"]
    if not p2.empty:
        plt.figure(figsize=(7, 4))
        vals = [p2[p2["planner_type"] == p]["path_efficiency"].values for p in ["planning", "reactive"]]
        plt.boxplot(vals, labels=["planning", "reactive"])
        plt.ylabel("path_efficiency")
        plt.title("P2 Path Efficiency: planning vs reactive")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "p2_path_efficiency_boxplot.png"), dpi=150)
        plt.close()

    # P3: Multi-agent coordination coverage line
    p3 = df[df["phase"] == "P3_multi_agent"]
    if not p3.empty:
        p3s = p3.groupby(["num_bots", "coordination"])["coverage"].mean().reset_index()
        plt.figure(figsize=(8, 4))
        for coord in ["none", "basic", "enhanced"]:
            g = p3s[p3s["coordination"] == coord].sort_values("num_bots")
            if g.empty:
                continue
            plt.plot(g["num_bots"], g["coverage"], marker="o", label=coord)
        plt.xlabel("num_bots")
        plt.ylabel("mean coverage")
        plt.title("P3 Coordination Effect on Coverage")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "p3_coordination_coverage_line.png"), dpi=150)
        plt.close()

    # P5: Collision grouped bar by noise and bots, faceted by coordination
    p5 = df[df["phase"] == "P5_full_matrix"]
    if not p5.empty:
        p5s = p5.groupby(["coordination", "noise_level", "num_bots"])["collisions"].mean().reset_index()
        noise_levels = ["0", "low", "mid", "high"]
        bots_order = [1, 3, 5]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, coord in zip(axes, ["none", "basic"]):
            subc = p5s[p5s["coordination"] == coord]
            x = np.arange(len(noise_levels))
            width = 0.22
            for i, bots in enumerate(bots_order):
                ys = []
                for nl in noise_levels:
                    row = subc[(subc["noise_level"] == nl) & (subc["num_bots"] == bots)]
                    ys.append(float(row["collisions"].iloc[0]) if not row.empty else np.nan)
                ax.bar(x + (i - 1) * width, ys, width=width, label=f"bots={bots}")
            ax.set_xticks(x)
            ax.set_xticklabels(noise_levels)
            ax.set_xlabel("noise_level")
            ax.set_title(f"P5 collisions ({coord})")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("mean collisions")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "p5_collisions_grouped_bar.png"), dpi=150)
        plt.close(fig)


def run_full_matrix(
    output_dir: str,
    reps: int,
    fast: bool = False,
    fast_reps: int = 3,
    p2_controller: str = "simple",
    base_seed: int = 12345,
    n_perm: int = 10000,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    reps = max(1, fast_reps) if fast else reps
    # Always use long-horizon episodes so coverage reflects sustained cleaning, not a short budget.
    max_steps = 1200
    rows: List[RunRow] = []
    run_config = {
        "fast": bool(fast),
        "reps": int(reps),
        "max_steps": int(max_steps),
        "p2_controller": str(p2_controller),
        "base_seed": int(base_seed),
        "n_perm": int(n_perm),
        "phases": {
            "P2_single_agent": {
                "num_bots": 1,
                "noise_level": "0",
                "num_dirt": 60,
                "num_dynamic_obstacles": 0,
            },
            "P3_multi_agent": {"num_bots": [1, 2, 3, 5], "coordination": ["none", "basic", "enhanced"]},
            "P4_noise_robustness": {"num_bots": 3, "noise_level": ["0", "low", "mid", "high"]},
            "P5_full_matrix": {"num_bots": [1, 3, 5], "coordination": ["none", "basic"], "noise_level": ["0", "low", "mid", "high"]},
        },
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # Phase 2: single-agent planning vs reactive (>=10 by default in full)
    p2_reps = reps if fast else max(10, reps)
    for planner in ["planning", "reactive"]:
        for rep in range(p2_reps):
            seed = _stable_seed(base_seed, ("P2_single_agent", planner, "none", 1, "0", rep))
            metric = run_episode(
                planner_type=planner,
                coordination="none",
                num_bots=1,
                noise_level="0",
                num_dirt=60,
                max_steps=max_steps,
                seed=seed,
                phase="P2_single_agent",
                num_dynamic_obstacles=0,
                p2_controller=p2_controller,
            )
            rows.append(RunRow("P2_single_agent", f"{planner}_single", seed, planner, "none", 1, "0", **metric))

    # Phase 3: multi-agent coordination (2/3/5 vs single)
    for bots in [1, 2, 3, 5]:
        for coord in ["none", "basic", "enhanced"]:
            if bots == 1 and coord != "none":
                continue
            for rep in range(reps):
                seed = _stable_seed(base_seed, ("P3_multi_agent", "planning", coord, bots, "low", rep))
                metric = run_episode(
                    planner_type="planning",
                    coordination=coord,
                    num_bots=bots,
                    noise_level="low",
                    num_dirt=100,
                    max_steps=max_steps,
                    seed=seed,
                    phase="P3_multi_agent",
                    num_dynamic_obstacles=3,
                    p2_controller=p2_controller,
                )
                rows.append(RunRow("P3_multi_agent", f"bots{bots}_{coord}", seed, "planning", coord, bots, "low", **metric))

    # Phase 4: noise robustness planning vs reactive, 30 each condition
    for planner in ["planning", "reactive"]:
        for nl in ["0", "low", "mid", "high"]:
            for rep in range(reps):
                seed = _stable_seed(base_seed, ("P4_noise_robustness", planner, "none", 3, nl, rep))
                metric = run_episode(
                    planner_type=planner,
                    coordination="none",
                    num_bots=3,
                    noise_level=nl,
                    num_dirt=90,
                    max_steps=max_steps,
                    seed=seed,
                    phase="P4_noise_robustness",
                    num_dynamic_obstacles=3,
                    p2_controller=p2_controller,
                )
                rows.append(RunRow("P4_noise_robustness", f"{planner}_{nl}", seed, planner, "none", 3, nl, **metric))

    # Phase 5: full matrix required in prompt
    for planner in ["planning", "reactive"]:
        for bots in [1, 3, 5]:
            for nl in ["0", "low", "mid", "high"]:
                for coord in ["none", "basic"]:
                    if bots == 1 and coord != "none":
                        continue
                    for rep in range(reps):
                        seed = _stable_seed(base_seed, ("P5_full_matrix", planner, coord, bots, nl, rep))
                        metric = run_episode(
                            planner_type=planner,
                            coordination=coord,
                            num_bots=bots,
                            noise_level=nl,
                            num_dirt=110,
                            max_steps=max_steps,
                            seed=seed,
                            phase="P5_full_matrix",
                            num_dynamic_obstacles=3,
                            p2_controller=p2_controller,
                        )
                        rows.append(
                            RunRow(
                                "P5_full_matrix",
                                f"{planner}_bots{bots}_{nl}_{coord}",
                                seed,
                                planner,
                                coord,
                                bots,
                                nl,
                                **metric,
                            )
                        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(os.path.join(output_dir, "all_runs.csv"), index=False)

    # Aggregate
    summary = (
        df.groupby(["phase", "planner_type", "coordination", "num_bots", "noise_level"])
        .agg(
            n=("seed", "count"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            steps_mean=("steps", "mean"),
            collisions_mean=("collisions", "mean"),
            path_eff_mean=("path_efficiency", "mean"),
            success_mean=("success", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

    # Significance tests: planning vs reactive under same (phase, bots, noise, coordination)
    tests = []
    key_cols = ["phase", "num_bots", "noise_level", "coordination"]
    for key, g in df.groupby(key_cols):
        gp = g[g["planner_type"] == "planning"]
        gr = g[g["planner_type"] == "reactive"]
        if len(gp) < 3 or len(gr) < 3:
            continue
        p_cov = _perm_pvalue(gp["coverage"].values, gr["coverage"].values, n_perm=n_perm)
        p_eff = _perm_pvalue(gp["path_efficiency"].values, gr["path_efficiency"].values, n_perm=n_perm)
        cov_diff = float(gp["coverage"].mean() - gr["coverage"].mean())
        eff_diff = float(gp["path_efficiency"].mean() - gr["path_efficiency"].mean())
        tests.append(
            {
                "phase": key[0],
                "num_bots": key[1],
                "noise_level": key[2],
                "coordination": key[3],
                "n_planning": len(gp),
                "n_reactive": len(gr),
                "planning_cov_mean": float(gp["coverage"].mean()),
                "reactive_cov_mean": float(gr["coverage"].mean()),
                "planning_eff_mean": float(gp["path_efficiency"].mean()),
                "reactive_eff_mean": float(gr["path_efficiency"].mean()),
                "coverage_mean_diff": cov_diff,
                "path_eff_mean_diff": eff_diff,
                "p_perm_coverage": p_cov,
                "p_perm_path_eff": p_eff,
            }
        )
    cols = [
        "phase",
        "num_bots",
        "noise_level",
        "coordination",
        "n_planning",
        "n_reactive",
        "planning_cov_mean",
        "reactive_cov_mean",
        "planning_eff_mean",
        "reactive_eff_mean",
        "coverage_mean_diff",
        "path_eff_mean_diff",
        "p_perm_coverage",
        "p_perm_path_eff",
        "p_holm_coverage",
        "p_holm_path_eff",
    ]
    tests_df = pd.DataFrame(tests)
    if tests_df.empty:
        tests_df = pd.DataFrame(columns=cols)
    else:
        tests_df["p_holm_coverage"] = _holm_bonferroni(tests_df["p_perm_coverage"].fillna(1.0).tolist())
        tests_df["p_holm_path_eff"] = _holm_bonferroni(tests_df["p_perm_path_eff"].fillna(1.0).tolist())
        tests_df = tests_df.reindex(columns=cols)
    tests_df.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)

    _write_plots(df, output_dir)
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run requirement-aligned full experiment matrix.")
    ap.add_argument("--output-dir", default="research_requirements_results")
    ap.add_argument("--reps", type=int, default=30, help="Runs per condition in phase 3/4/5.")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fast-reps", type=int, default=3, help="Runs per condition when --fast is enabled.")
    ap.add_argument("--p2-controller", choices=["simple", "goap"], default="simple")
    ap.add_argument("--base-seed", type=int, default=12345, help="Base seed for deterministic per-condition seed derivation.")
    ap.add_argument("--n-perm", type=int, default=10000, help="Permutation test iterations.")
    args = ap.parse_args()
    out = run_full_matrix(
        args.output_dir,
        reps=args.reps,
        fast=args.fast,
        fast_reps=args.fast_reps,
        p2_controller=args.p2_controller,
        base_seed=args.base_seed,
        n_perm=args.n_perm,
    )
    print(f"Done. Results saved to: {out}")


if __name__ == "__main__":
    main()

