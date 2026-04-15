from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.robot_env import RobotEnvironment
from agents.planner_agent import PlannerAgent
from experiments.member_c_week2_blackboard import Week2CoordinationController


def _seed(seed: int) -> None:
    random.seed(seed)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _action_noise(action: Tuple[float, float], sigma: float) -> Tuple[float, float]:
    if sigma <= 0:
        return action
    return (
        _clamp(action[0] + random.gauss(0, sigma), -10.0, 10.0),
        _clamp(action[1] + random.gauss(0, sigma), -10.0, 10.0),
    )


class FSMController:
    def __init__(self):
        self.state = "search"

    def get_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        actions = []
        for o in obs:
            batt = float(o.get("battery", 1000.0))
            if batt < 260:
                self.state = "charge"
            elif self.state == "charge" and batt > 820:
                self.state = "search"
            if self.state == "charge":
                actions.append((2.0, -2.0))
            else:
                actions.append((6.0, 6.0) if random.random() > 0.2 else (4.0, -4.0))
        return actions


class GOAPLikeController:
    def get_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        actions = []
        for o in obs:
            batt = float(o.get("battery", 1000.0))
            if batt < 260:
                actions.append((2.0, -2.0))
            else:
                actions.append((7.0, 7.0) if random.random() > 0.15 else (3.5, -3.5))
        return actions


class ReactiveController:
    def get_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        out = []
        for _ in obs:
            out.append((6.0, 6.0) if random.random() > 0.2 else (4.0, -4.0))
        return out


class MPCStyleController:
    def get_actions(self, obs: Sequence[Dict]) -> List[Tuple[float, float]]:
        candidates = [(7.0, 7.0), (6.0, 6.0), (5.0, 5.0), (4.0, -4.0), (-4.0, 4.0), (0.0, 0.0)]
        out = []
        for _ in obs:
            scored = []
            for a in candidates:
                score = 0.9 * (a[0] + a[1]) - 0.12 * abs(a[0] - a[1]) + random.uniform(-0.3, 0.3)
                scored.append((score, a))
            scored.sort(key=lambda x: x[0], reverse=True)
            out.append(scored[0][1])
        return out


@dataclass
class RunResult:
    qid: str
    condition: str
    seed: int
    completed: int
    completion_time_steps: int
    coverage: float
    path_efficiency: float
    planning_time_ms: float
    decision_time_ms: float
    action_seq_len: int
    repeat_coverage_rate: float
    success_rate: float
    runtime_seconds: float


def _repeat_rate(obs: Sequence[Dict], cell_size: float = 20.0) -> float:
    visit = {}
    for o in obs:
        cell = (int(float(o["x"]) // cell_size), int(float(o["y"]) // cell_size))
        visit[cell] = visit.get(cell, 0) + 1
    repeats = sum(v - 1 for v in visit.values() if v > 1)
    total_visits = sum(visit.values()) if visit else 1
    return repeats / total_visits


def _run_episode(
    env: RobotEnvironment,
    action_fn: Callable[[Sequence[Dict]], Optional[List[Tuple[float, float]]]],
    max_steps: int,
    action_noise_sigma: float = 0.0,
    per_step_hook: Optional[Callable[[int], None]] = None,
) -> Dict[str, float]:
    obs = env.reset()
    start = time.time()
    planning_time = 0.0
    decision_time = 0.0
    action_seq = 0
    done = False
    info = {}

    for step in range(max_steps):
        if per_step_hook is not None:
            per_step_hook(step)

        t0 = time.perf_counter()
        actions = action_fn(obs)
        decision_time += (time.perf_counter() - t0) * 1000.0

        if actions is not None:
            actions = [_action_noise(a, action_noise_sigma) for a in actions]
            action_seq += len(actions)

        t1 = time.perf_counter()
        obs, reward, done, info = env.step(actions)
        planning_time += (time.perf_counter() - t1) * 1000.0

        if done:
            break

    repeat_rate = _repeat_rate(obs)
    runtime = time.time() - start
    return {
        "completed": int(done),
        "completion_time_steps": int(info.get("steps", max_steps)),
        "coverage": float(info.get("coverage", 0.0)),
        "path_efficiency": float(info.get("path_efficiency", 0.0)),
        "planning_time_ms": planning_time,
        "decision_time_ms": decision_time,
        "action_seq_len": action_seq,
        "repeat_coverage_rate": repeat_rate,
        "success_rate": float(info.get("success", 0)),
        "runtime_seconds": runtime,
    }


def run_suite(output_dir: str, fast: bool = False, only: Optional[Sequence[str]] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results: List[RunResult] = []
    only_set = {x.upper() for x in (only or [])}
    run_all = not only_set

    # Q1: A*+replanning vs D* Lite, low/high dynamic environments
    if run_all or "Q1" in only_set:
        reps = 6 if fast else 30
        conditions = [
            ("astar_replan_low_dyn", "astar", 1, 1.5, 0.0),
            ("astar_replan_high_dyn", "astar", 4, 3.0, 0.0),
            ("dstar_low_dyn", "dstar_lite", 1, 1.5, 0.0),
            ("dstar_high_dyn", "dstar_lite", 4, 3.0, 0.0),
            ("dstar_high_dyn_noise", "dstar_lite", 4, 3.0, 0.25),
        ]
        for cond, algo, dyn_n, dyn_speed, noise in conditions:
            for rep in range(reps):
                seed = 1000 + rep
                _seed(seed)
                env = RobotEnvironment(
                    num_bots=1,
                    num_dirt=80,
                    seed=seed,
                    num_dynamic_obstacles=dyn_n,
                    dynamic_obstacle_speed=dyn_speed,
                    render=False,
                    max_steps=1200,
                )
                agent = PlannerAgent(algo, env, config={"replan_interval": 1 if algo == "astar" else 8})
                metric = _run_episode(
                    env,
                    action_fn=lambda obs, a=agent: a.get_action(obs),
                    max_steps=1200,
                    action_noise_sigma=noise,
                )
                env.close()
                results.append(RunResult("Q1", cond, seed, **metric))

    # Q2: GOAP vs FSM, static/dynamic goals
    if run_all or "Q2" in only_set:
        reps = 6 if fast else 30
        conditions = [
            ("goap_static", GOAPLikeController(), 0),
            ("goap_dynamic", GOAPLikeController(), 20),
            ("goap_dynamic_high_freq", GOAPLikeController(), 8),
            ("fsm_static", FSMController(), 0),
            ("fsm_dynamic", FSMController(), 20),
            ("fsm_dynamic_high_freq", FSMController(), 8),
        ]
        for cond, controller, dyn_freq in conditions:
            for rep in range(reps):
                seed = 2000 + rep
                _seed(seed)
                env = RobotEnvironment(num_bots=1, num_dirt=70, seed=seed, render=False, max_steps=1200)

                def goal_hook(step: int, e=env, f=dyn_freq):
                    if f <= 0 or step <= 0 or step % f != 0:
                        return
                    # move a few dirt targets (dynamic goals)
                    for obj in e.passive_objects:
                        if getattr(obj, "__class__", None) and obj.__class__.__name__ == "Dirt" and random.random() < 0.05:
                            obj.centreX = _clamp(obj.centreX + random.uniform(-60, 60), 20, 980)
                            obj.centreY = _clamp(obj.centreY + random.uniform(-60, 60), 20, 980)

                metric = _run_episode(
                    env,
                    action_fn=lambda obs, c=controller: c.get_actions(obs),
                    max_steps=1200,
                    per_step_hook=goal_hook,
                )
                env.close()
                results.append(RunResult("Q2", cond, seed, **metric))

    # Q3: coordination modes, bots=2/4/6
    if run_all or "Q3" in only_set:
        reps = 6 if fast else 30
        modes = ["no_coordination", "implicit_coordination", "explicit_coordination"]
        for mode in modes:
            for bots in [2, 4, 6]:
                for rep in range(reps):
                    seed = 3000 + rep
                    _seed(seed)
                    env = RobotEnvironment(num_bots=bots, num_dirt=120, seed=seed, render=False, max_steps=1200)
                    ctrl = Week2CoordinationController(mode, num_bots=bots, seed=seed)
                    metric = _run_episode(
                        env,
                        per_step_hook=lambda step, c=ctrl, e=env: c.update_shared_state(e, step),
                        action_fn=lambda obs, c=ctrl, e=env, m=mode: (
                            [c(i, e.agents[i], e.agents, e.passive_objects) for i in range(len(e.agents))]
                            if m != "no_coordination"
                            else None
                        ),
                        max_steps=1200,
                    )
                    env.close()
                    results.append(RunResult("Q3", f"{mode}_bots{bots}", seed, **metric))

    # Q4: MPC-style vs reactive, 4 noise levels
    if run_all or "Q4" in only_set:
        reps = 8 if fast else 50
        noise_levels = [0.0, 0.1, 0.25, 0.4]
        for label, controller in [("mpc", MPCStyleController()), ("reactive", ReactiveController())]:
            for nl in noise_levels:
                for rep in range(reps):
                    seed = 4000 + rep
                    _seed(seed)
                    env = RobotEnvironment(num_bots=2, num_dirt=90, seed=seed, render=False, max_steps=1200)
                    metric = _run_episode(
                        env,
                        action_fn=lambda obs, c=controller: c.get_actions(obs),
                        max_steps=1200,
                        action_noise_sigma=nl,
                    )
                    env.close()
                    results.append(RunResult("Q4", f"{label}_noise{nl}", seed, **metric))

    # Q5: coordination strategy × obstacle speed
    if run_all or "Q5" in only_set:
        reps = 6 if fast else 30
        speed_map = {"slow": 1.0, "mid": 2.0, "fast": 3.2}
        strat_map = {"none": "no_coordination", "basic": "implicit_coordination", "enhanced": "explicit_coordination"}
        for sname, mode in strat_map.items():
            for vname, spd in speed_map.items():
                for rep in range(reps):
                    seed = 5000 + rep
                    _seed(seed)
                    env = RobotEnvironment(
                        num_bots=4,
                        num_dirt=140,
                        seed=seed,
                        num_dynamic_obstacles=4,
                        dynamic_obstacle_speed=spd,
                        render=False,
                        max_steps=1200,
                    )
                    ctrl = Week2CoordinationController(mode, num_bots=4, seed=seed)
                    metric = _run_episode(
                        env,
                        per_step_hook=lambda step, c=ctrl, e=env: c.update_shared_state(e, step),
                        action_fn=lambda obs, c=ctrl, e=env, m=mode: (
                            [c(i, e.agents[i], e.agents, e.passive_objects) for i in range(len(e.agents))]
                            if m != "no_coordination"
                            else None
                        ),
                        max_steps=1200,
                    )
                    env.close()
                    results.append(RunResult("Q5", f"{sname}_{vname}", seed, **metric))

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(os.path.join(output_dir, "all_runs.csv"), index=False)

    # anomalies: coverage z-score within each qid
    anomalies = []
    for qid, g in df.groupby("qid"):
        m = g["coverage"].mean()
        s = g["coverage"].std() or 1e-6
        z = (g["coverage"] - m).abs() / s
        bad = g[z > 3.0]
        for _, r in bad.iterrows():
            anomalies.append({"qid": qid, "condition": r["condition"], "seed": int(r["seed"]), "coverage_z": float((r["coverage"] - m) / s)})
    with open(os.path.join(output_dir, "anomalies.json"), "w", encoding="utf-8") as f:
        json.dump(anomalies, f, ensure_ascii=False, indent=2)

    # summary
    grouped = (
        df.groupby(["qid", "condition"])
        .agg(
            n=("seed", "count"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            completion_time_mean=("completion_time_steps", "mean"),
            path_eff_mean=("path_efficiency", "mean"),
            success_mean=("success_rate", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(os.path.join(output_dir, "summary_by_condition.csv"), index=False)

    # plots (coverage boxplot + path_eff line)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for qid in sorted(df["qid"].unique()):
        sub = df[df["qid"] == qid]
        conds = sorted(sub["condition"].unique())
        plt.figure(figsize=(12, 4))
        plt.boxplot([sub[sub["condition"] == c]["coverage"] for c in conds], labels=conds)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("coverage")
        plt.title(f"{qid} Coverage Boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{qid.lower()}_coverage_boxplot.png"), dpi=150)
        plt.close()

        summary = sub.groupby("condition")[["path_efficiency"]].mean().reset_index()
        plt.figure(figsize=(12, 4))
        plt.plot(summary["condition"], summary["path_efficiency"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("path_efficiency")
        plt.title(f"{qid} Path Efficiency Line")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{qid.lower()}_path_eff_line.png"), dpi=150)
        plt.close()

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q1–Q5 unified research experiments (current project).")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "research_results"))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    out = run_suite(args.output_dir, fast=args.fast, only=args.only)
    print(f"Done. Results saved to: {out}")


if __name__ == "__main__":
    main()

