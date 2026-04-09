"""
批量实验脚本框架：
- 可循环不同算法、参数、随机种子
- 自动保存：config/metrics/step_log/trajectories + 汇总 summary.csv

示例：
  python experiments/run_batch.py --algorithms random astar dstar --num-bots 1 --num-dirt 50 100 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from env.robot_env import RobotEnvironment
from agents.planner_agent import PlannerAgent


def _now_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_int_list(values: Optional[List[str]]) -> List[int]:
    if not values:
        return []
    return [int(v) for v in values]


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class ExperimentParams:
    algorithm: str
    num_bots: int
    num_dirt: int
    seed: Optional[int]
    max_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "num_bots": self.num_bots,
            "num_dirt": self.num_dirt,
            "seed": self.seed,
            "max_steps": self.max_steps,
        }


def _make_agent(params: ExperimentParams, env: RobotEnvironment):
    if params.algorithm == "random":
        return None
    if params.algorithm == "astar":
        return PlannerAgent("astar", env)
    if params.algorithm in ("dstar", "dstar_lite"):
        return PlannerAgent("dstar_lite", env)
    raise ValueError(f"Unknown algorithm: {params.algorithm}")


def run_single_experiment(exp_id: str, params: ExperimentParams, exp_dir: str) -> Dict[str, Any]:
    """运行单次实验并落盘。失败也会写 error.json。"""
    os.makedirs(exp_dir, exist_ok=True)

    _seed_everything(params.seed)

    # 保存配置
    _write_json(os.path.join(exp_dir, "config.json"), {"exp_id": exp_id, **params.to_dict()})

    env = RobotEnvironment(
        num_bots=params.num_bots,
        num_dirt=params.num_dirt,
        seed=params.seed,
    )

    obs = env.reset()
    agent = _make_agent(params, env)

    trajectories: Dict[str, List[List[float]]] = {f"robot_{i}": [] for i in range(params.num_bots)}
    step_log: List[Dict[str, Any]] = []
    start_time = time.time()

    done = False
    info: Dict[str, Any] = {}

    try:
        for step in range(params.max_steps):
            if agent is None:
                actions = None
            else:
                actions = agent.get_action(obs)

            obs, reward, done, info = env.step(actions)

            for i, robot in enumerate(env.agents):
                trajectories[f"robot_{i}"].append([step, float(robot.x), float(robot.y)])

            step_log.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "coverage": info.get("coverage"),
                    "remaining_dirt": info.get("remaining_dirt"),
                    "collisions": info.get("collisions"),
                    "total_distance": info.get("total_distance"),
                    "path_efficiency": info.get("path_efficiency"),
                    "performance_score": info.get("performance_score"),
                    "success": info.get("success"),
                }
            )

            if done:
                break

        runtime = time.time() - start_time
        total_steps = step + 1

        # 保存轨迹与逐步日志
        traj_dir = os.path.join(exp_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        for robot_name, traj in trajectories.items():
            csv_path = os.path.join(traj_dir, f"{robot_name}.csv")
            pd.DataFrame(traj, columns=["step", "x", "y"]).to_csv(csv_path, index=False)
        pd.DataFrame(step_log).to_csv(os.path.join(exp_dir, "step_log.csv"), index=False)

        metrics = {
            "exp_id": exp_id,
            **params.to_dict(),
            "total_steps": int(total_steps),
            "completed": bool(done),
            "runtime_seconds": round(runtime, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            # 末步指标（来自 env.get_metrics()/step 返回的 metrics）
            "coverage": info.get("coverage"),
            "remaining_dirt": info.get("remaining_dirt"),
            "collisions": info.get("collisions"),
            "total_distance": info.get("total_distance"),
            "path_efficiency": info.get("path_efficiency"),
            "performance_score": info.get("performance_score"),
            "success": info.get("success"),
        }
        _write_json(os.path.join(exp_dir, "metrics.json"), metrics)
        return metrics
    except Exception as e:
        err = {
            "exp_id": exp_id,
            **params.to_dict(),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(os.path.join(exp_dir, "error.json"), err)
        raise
    finally:
        env.close()


def _iter_experiments(
    algorithms: List[str],
    num_bots_list: List[int],
    num_dirt_list: List[int],
    seeds: List[int],
    max_steps: int,
) -> Iterable[Tuple[str, ExperimentParams]]:
    for algo, nb, nd, seed in product(algorithms, num_bots_list, num_dirt_list, seeds):
        exp_id = f"{algo}_bots{nb}_dirt{nd}_seed{seed}"
        yield exp_id, ExperimentParams(algorithm=algo, num_bots=nb, num_dirt=nd, seed=seed, max_steps=max_steps)


def run_batch(
    algorithms: List[str],
    num_bots_list: List[int],
    num_dirt_list: List[int],
    seeds: List[int],
    max_steps: int,
    out_dir: str,
    run_name: Optional[str] = None,
    skip_existing: bool = True,
) -> str:
    run_dir = os.path.join(out_dir, run_name or f"run-{_now_compact()}")
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "run_name": os.path.basename(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": os.path.abspath(run_dir),
        "grid": {
            "algorithms": algorithms,
            "num_bots": num_bots_list,
            "num_dirt": num_dirt_list,
            "seeds": seeds,
            "max_steps": max_steps,
        },
    }
    _write_json(os.path.join(run_dir, "run_manifest.json"), manifest)

    experiments = list(_iter_experiments(algorithms, num_bots_list, num_dirt_list, seeds, max_steps))
    total_exps = len(experiments)
    print(f"共生成 {total_exps} 个实验，输出目录: {run_dir}")

    results: List[Dict[str, Any]] = []
    for idx, (exp_id, params) in enumerate(experiments):
        exp_dir = os.path.join(run_dir, exp_id)
        metrics_path = os.path.join(exp_dir, "metrics.json")

        if skip_existing and os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                print(f"[{idx+1}/{total_exps}] 跳过已完成: {exp_id}")
                continue
            except Exception:
                # 读取失败就重跑并覆盖
                pass

        print(f"[{idx+1}/{total_exps}] 运行: {exp_id}")
        try:
            metrics = run_single_experiment(exp_id, params, exp_dir)
            results.append(metrics)
        except Exception as e:
            print(f"实验失败: {e}")
            results.append({"exp_id": exp_id, **params.to_dict(), "error": str(e)})

        pd.DataFrame(results).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    # Windows 控制台常见为 GBK 编码，避免使用 emoji 导致 UnicodeEncodeError
    print(f"批量实验完成！汇总表: {os.path.join(run_dir, 'summary.csv')}")
    return run_dir


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch experiments runner")
    # 默认按“3 算法 × 2 种子 × 2 灰尘数 = 12 个实验”配置
    p.add_argument("--algorithms", nargs="+", default=["random", "astar", "dstar"])
    p.add_argument("--num-bots", nargs="+", default=["1"])
    p.add_argument("--num-dirt", nargs="+", default=["50", "100"])
    p.add_argument("--seeds", nargs="+", default=["0", "1"])
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out-dir", default="../batch_results")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-skip-existing", action="store_true", help="do not skip completed experiments")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    run_batch(
        algorithms=args.algorithms,
        num_bots_list=_safe_int_list(args.num_bots),
        num_dirt_list=_safe_int_list(args.num_dirt),
        seeds=_safe_int_list(args.seeds),
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        run_name=args.run_name,
        skip_existing=not args.no_skip_existing,
    )