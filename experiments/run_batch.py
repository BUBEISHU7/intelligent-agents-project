"""
批量实验脚本框架：
- 可循环不同算法、参数、随机种子（全因子设计）
- 自动保存：config/metrics/step_log/trajectories + 汇总 summary.csv

实验矩阵（自变量与水平，见 experiments/matrix_presets.json）：
- algorithm：random / astar / dstar（规划策略）
- num_bots：机器人数量
- num_dirt：灰尘数量（任务难度）
- seed：随机种子（重复与方差估计）

示例：
  python experiments/run_batch.py --algorithms random astar dstar --num-bots 1 --num-dirt 50 100 --seeds 0 1
  python experiments/run_batch.py --preset default
  python experiments/run_batch.py --preset quick --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_MATRIX_FILE = os.path.join(os.path.dirname(__file__), "matrix_presets.json")
_DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "batch_results")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from env.robot_env import RobotEnvironment
from agents.goap_agent import GOAPTeamController, RandomTeamController


def _now_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_int_list(values: Optional[List[str]]) -> List[int]:
    if not values:
        return []
    return [int(v) for v in values]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_grid_dict(g: Dict[str, Any]) -> Dict[str, Any]:
    required = ("algorithms", "num_bots", "num_dirt", "seeds", "max_steps")
    for k in required:
        if k not in g:
            raise ValueError(f"grid 中缺少 '{k}'")
    return {
        "algorithms": [str(x) for x in g["algorithms"]],
        "num_bots": [int(x) for x in g["num_bots"]],
        "num_dirt": [int(x) for x in g["num_dirt"]],
        "seeds": [int(x) for x in g["seeds"]],
        "max_steps": int(g["max_steps"]),
        "num_dynamic_obstacles": [int(x) for x in g.get("num_dynamic_obstacles", [0])],
        "sensor_noise": [float(x) for x in g.get("sensor_noise", [0.0])],
        "execution_noise": [float(x) for x in g.get("execution_noise", [0.0])],
    }


def _grid_from_preset_block(block: Dict[str, Any]) -> Dict[str, Any]:
    if "grid" not in block:
        raise ValueError("预设块中缺少 'grid' 字段")
    return _normalize_grid_dict(block["grid"])


def load_experiment_grid(matrix_path: str, preset: Optional[str]) -> Dict[str, Any]:
    """从 matrix_presets.json（多预设）或仅含单个 grid 的 JSON 解析实验网格。"""
    data = _load_json(matrix_path)
    presets = data.get("presets")
    if isinstance(presets, dict) and presets:
        if not preset:
            raise ValueError(
                f"文件 {matrix_path} 含多个预设，请指定 --preset <name>，或运行 --list-presets"
            )
        if preset not in presets:
            raise KeyError(f"预设 '{preset}' 不存在，可用: {sorted(presets.keys())}")
        return _grid_from_preset_block(presets[preset])
    if "grid" in data:
        return _normalize_grid_dict(data["grid"])
    raise ValueError(f"无法从 {matrix_path} 解析 grid（需要顶层 'grid' 或 'presets'）")


def list_matrix_presets(matrix_path: str) -> None:
    data = _load_json(matrix_path)
    presets = data.get("presets") or {}
    if not presets:
        print("(该文件无 presets，可直接用作单矩阵: 顶层含 grid 即可)")
        return
    for name in sorted(presets.keys()):
        meta = presets[name]
        label = meta.get("label", "")
        g = meta.get("grid") or {}
        algs = g.get("algorithms", [])
        nb = g.get("num_bots", [])
        nd = g.get("num_dirt", [])
        sd = g.get("seeds", [])
        n = len(algs) * len(nb) * len(nd) * len(sd) if algs and nb and nd and sd else 0
        print(f"{name}\t{n} runs\t{label}")


def dry_run_print(grid: Dict[str, Any], sample: int = 8) -> None:
    a, nb, nd, sd, ndo, sn, en = (
        grid["algorithms"],
        grid["num_bots"],
        grid["num_dirt"],
        grid["seeds"],
        grid["num_dynamic_obstacles"],
        grid["sensor_noise"],
        grid["execution_noise"],
    )
    total = len(a) * len(nb) * len(nd) * len(sd) * len(ndo) * len(sn) * len(en)
    print(
        "实验矩阵: "
        f"|algorithm|={len(a)} × |num_bots|={len(nb)} × |num_dirt|={len(nd)} × |seeds|={len(sd)} "
        f"× |num_dynamic_obstacles|={len(ndo)} × |sensor_noise|={len(sn)} × |execution_noise|={len(en)} "
        f"=> {total} 次运行 (max_steps={grid['max_steps']})"
    )
    exps = list(_iter_experiments(a, nb, nd, sd, grid["max_steps"], ndo, sn, en))
    for i, (exp_id, _) in enumerate(exps[:sample]):
        print(f"  {i+1}. {exp_id}")
    if total > sample:
        print(f"  ... 共 {total} 条，仅展示前 {sample} 条")


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
    num_dynamic_obstacles: int = 0
    sensor_noise: float = 0.0
    execution_noise: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "num_bots": self.num_bots,
            "num_dirt": self.num_dirt,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "num_dynamic_obstacles": self.num_dynamic_obstacles,
            "sensor_noise": self.sensor_noise,
            "execution_noise": self.execution_noise,
        }


def _make_agent(params: ExperimentParams, env: RobotEnvironment):
    if params.algorithm == "random":
        return RandomTeamController(env)
    if params.algorithm in ("astar", "goap_astar"):
        return GOAPTeamController(env, planner_algorithm="astar")
    if params.algorithm in ("dstar", "dstar_lite", "goap_dstar"):
        return GOAPTeamController(env, planner_algorithm="dstar")
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
        num_dynamic_obstacles=params.num_dynamic_obstacles,
        max_steps=params.max_steps,
        render=False,
        noise_config={
            "sensor_gaussian_std": params.sensor_noise,
            "sensor_miss_rate": 0.05 if params.sensor_noise > 0 else 0.0,
            "execution_noise_std": params.execution_noise,
        },
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
                actions = agent.compute_actions(obs)

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
                    "dynamic_collisions": info.get("dynamic_collisions"),
                    "robot_collisions": info.get("robot_collisions"),
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
            "dynamic_collisions": info.get("dynamic_collisions"),
            "robot_collisions": info.get("robot_collisions"),
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
    num_dynamic_obstacles_list: List[int],
    sensor_noise_list: List[float],
    execution_noise_list: List[float],
) -> Iterable[Tuple[str, ExperimentParams]]:
    for algo, nb, nd, seed, ndo, sn, en in product(
        algorithms,
        num_bots_list,
        num_dirt_list,
        seeds,
        num_dynamic_obstacles_list,
        sensor_noise_list,
        execution_noise_list,
    ):
        exp_id = (
            f"{algo}_bots{nb}_dirt{nd}_dyn{ndo}_"
            f"snoise{str(sn).replace('.', 'p')}_enoise{str(en).replace('.', 'p')}_seed{seed}"
        )
        yield exp_id, ExperimentParams(
            algorithm=algo,
            num_bots=nb,
            num_dirt=nd,
            seed=seed,
            max_steps=max_steps,
            num_dynamic_obstacles=ndo,
            sensor_noise=sn,
            execution_noise=en,
        )


def run_batch(
    algorithms: List[str],
    num_bots_list: List[int],
    num_dirt_list: List[int],
    seeds: List[int],
    max_steps: int,
    num_dynamic_obstacles_list: List[int],
    sensor_noise_list: List[float],
    execution_noise_list: List[float],
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
            "num_dynamic_obstacles": num_dynamic_obstacles_list,
            "sensor_noise": sensor_noise_list,
            "execution_noise": execution_noise_list,
        },
    }
    _write_json(os.path.join(run_dir, "run_manifest.json"), manifest)

    experiments = list(
        _iter_experiments(
            algorithms,
            num_bots_list,
            num_dirt_list,
            seeds,
            max_steps,
            num_dynamic_obstacles_list,
            sensor_noise_list,
            execution_noise_list,
        )
    )
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
    p = argparse.ArgumentParser(description="Batch experiments runner (full factorial grid)")
    p.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help=f"从 --matrix-file 中选择预设（默认文件: {_DEFAULT_MATRIX_FILE}）",
    )
    p.add_argument(
        "--matrix-file",
        default=None,
        help="实验矩阵 JSON：含 presets 或顶层 grid；与 --preset 联用",
    )
    p.add_argument(
        "--list-presets",
        action="store_true",
        help="列出矩阵文件中的预设名并退出",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印因子组合数量与示例 exp_id，不启动仿真",
    )
    # 未指定 --preset 时：与下方默认等价于原「3×2×2」基线
    p.add_argument("--algorithms", nargs="+", default=["random", "astar", "dstar"])
    p.add_argument("--num-bots", nargs="+", default=["1"])
    p.add_argument("--num-dirt", nargs="+", default=["50", "100"])
    p.add_argument("--seeds", nargs="+", default=["0", "1"])
    p.add_argument("--num-dynamic-obstacles", nargs="+", default=["0", "2"])
    p.add_argument("--sensor-noise", nargs="+", default=["0.0", "2.0"])
    p.add_argument("--execution-noise", nargs="+", default=["0.0", "0.3"])
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-skip-existing", action="store_true", help="do not skip completed experiments")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    matrix_path = args.matrix_file or _DEFAULT_MATRIX_FILE

    if args.list_presets:
        list_matrix_presets(matrix_path)
        raise SystemExit(0)

    if args.preset:
        grid = load_experiment_grid(matrix_path, args.preset)
        algorithms = grid["algorithms"]
        num_bots_list = grid["num_bots"]
        num_dirt_list = grid["num_dirt"]
        seeds = grid["seeds"]
        max_steps = grid["max_steps"]
        num_dynamic_obstacles_list = grid["num_dynamic_obstacles"]
        sensor_noise_list = grid["sensor_noise"]
        execution_noise_list = grid["execution_noise"]
    elif args.matrix_file:
        grid = load_experiment_grid(matrix_path, None)
        algorithms = grid["algorithms"]
        num_bots_list = grid["num_bots"]
        num_dirt_list = grid["num_dirt"]
        seeds = grid["seeds"]
        max_steps = grid["max_steps"]
        num_dynamic_obstacles_list = grid["num_dynamic_obstacles"]
        sensor_noise_list = grid["sensor_noise"]
        execution_noise_list = grid["execution_noise"]
    else:
        algorithms = args.algorithms
        num_bots_list = _safe_int_list(args.num_bots)
        num_dirt_list = _safe_int_list(args.num_dirt)
        seeds = _safe_int_list(args.seeds)
        max_steps = args.max_steps
        num_dynamic_obstacles_list = _safe_int_list(args.num_dynamic_obstacles)
        sensor_noise_list = [float(v) for v in args.sensor_noise]
        execution_noise_list = [float(v) for v in args.execution_noise]

    if args.dry_run:
        dry_run_print(
            {
                "algorithms": algorithms,
                "num_bots": num_bots_list,
                "num_dirt": num_dirt_list,
                "seeds": seeds,
                "max_steps": max_steps,
                "num_dynamic_obstacles": num_dynamic_obstacles_list,
                "sensor_noise": sensor_noise_list,
                "execution_noise": execution_noise_list,
            }
        )
        raise SystemExit(0)

    run_batch(
        algorithms=algorithms,
        num_bots_list=num_bots_list,
        num_dirt_list=num_dirt_list,
        seeds=seeds,
        max_steps=max_steps,
        num_dynamic_obstacles_list=num_dynamic_obstacles_list,
        sensor_noise_list=sensor_noise_list,
        execution_noise_list=execution_noise_list,
        out_dir=args.out_dir,
        run_name=args.run_name,
        skip_existing=not args.no_skip_existing,
    )