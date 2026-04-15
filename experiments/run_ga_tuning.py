"""
Genetic Algorithm baseline: tune controller parameters to maximize coverage.

This is a lightweight "GA" component for the coursework requirement (AI technique beyond classical planning).
It searches over a small parameter vector and reports the best configuration.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from env.robot_env import RobotEnvironment
from agents.goap_agent import GOAPTeamController


@dataclass
class Individual:
    replan_interval: int
    max_speed: float
    lookahead_distance: float

    def to_config(self) -> Dict[str, float]:
        return {
            "replan_interval": int(self.replan_interval),
            "max_speed": float(self.max_speed),
            "lookahead_distance": float(self.lookahead_distance),
        }


def _mutate(ind: Individual, rng: random.Random) -> Individual:
    if rng.random() < 0.4:
        ind.replan_interval = int(max(3, min(25, ind.replan_interval + rng.choice([-2, -1, 1, 2]))))
    if rng.random() < 0.4:
        ind.max_speed = float(max(4.0, min(10.0, ind.max_speed + rng.uniform(-1.0, 1.0))))
    if rng.random() < 0.4:
        ind.lookahead_distance = float(max(30.0, min(120.0, ind.lookahead_distance + rng.uniform(-15.0, 15.0))))
    return ind


def _crossover(a: Individual, b: Individual, rng: random.Random) -> Individual:
    return Individual(
        replan_interval=rng.choice([a.replan_interval, b.replan_interval]),
        max_speed=rng.choice([a.max_speed, b.max_speed]),
        lookahead_distance=rng.choice([a.lookahead_distance, b.lookahead_distance]),
    )


def _evaluate(ind: Individual, seed: int, steps: int, dyn: int) -> float:
    random.seed(seed)
    np.random.seed(seed)
    env = RobotEnvironment(num_bots=2, num_dirt=80, seed=seed, num_dynamic_obstacles=dyn, render=False, max_steps=steps)
    obs = env.reset()
    ctl = GOAPTeamController(env, planner_algorithm="dstar", config=ind.to_config())
    done = False
    info = {}
    for _ in range(steps):
        actions = ctl.compute_actions(obs)
        obs, _, done, info = env.step(actions)
        if done:
            break
    m = env.get_metrics()
    env.close()
    return float(m.get("coverage", 0.0))


def run_ga(pop: int, gens: int, steps: int, dyn: int, seed0: int) -> Tuple[Individual, float]:
    rng = random.Random(seed0)
    population = [
        Individual(
            replan_interval=rng.randint(4, 18),
            max_speed=rng.uniform(6.0, 10.0),
            lookahead_distance=rng.uniform(40.0, 110.0),
        )
        for _ in range(pop)
    ]

    best = None
    best_fit = -1.0
    for g in range(gens):
        scored: List[Tuple[float, Individual]] = []
        for i, ind in enumerate(population):
            fit = _evaluate(ind, seed=seed0 + g * 100 + i, steps=steps, dyn=dyn)
            scored.append((fit, ind))
            if fit > best_fit:
                best_fit = fit
                best = Individual(ind.replan_interval, ind.max_speed, ind.lookahead_distance)
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind in scored[: max(2, pop // 4)]]
        next_pop = elites[:]
        while len(next_pop) < pop:
            p1, p2 = rng.choice(elites), rng.choice(elites)
            child = _crossover(p1, p2, rng)
            child = _mutate(child, rng)
            next_pop.append(child)
        population = next_pop
        print(f"[gen {g+1}/{gens}] best_fit={best_fit:.4f}")
    return best, best_fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dyn", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    start = time.time()
    best, fit = run_ga(args.pop, args.gens, args.steps, args.dyn, args.seed)
    print("Best individual:", best)
    print("Best fitness:", fit)
    print("Elapsed seconds:", round(time.time() - start, 2))


if __name__ == "__main__":
    main()

