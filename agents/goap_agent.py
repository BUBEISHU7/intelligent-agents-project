import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from env.astar import AStarPlanner
from env.simpleBot2 import Charger, Dirt
from agents.goap_planner import Action, GoapPlanner
from agents.orca import Neighbor, compute_orca_velocity


@dataclass
class WorldState:
    low_battery: bool
    on_charger: bool
    dirt_visible: bool
    dirt_remaining: bool


class GoapAction:
    name = "base"
    cost = 1.0
    preconditions: Tuple[str, ...] = ()
    effects: Dict[str, bool] = {}

    def applicable(self, state: WorldState) -> bool:
        return True

    def is_applicable_predicates(self, predicates: Dict[str, bool]) -> bool:
        return all(predicates.get(p, False) for p in self.preconditions)

    def apply(self, predicates: Dict[str, bool]) -> Dict[str, bool]:
        nxt = dict(predicates)
        for key, val in self.effects.items():
            nxt[key] = val
        return nxt


class ChargeAction(GoapAction):
    name = "charge"
    cost = 2.0
    preconditions = ("at_charger",)
    effects = {"battery_ok": True}

    def applicable(self, state: WorldState) -> bool:
        return state.low_battery or state.on_charger


class CleanAction(GoapAction):
    name = "clean"
    cost = 1.0
    preconditions = ("at_target", "target_exists")
    effects = {"target_cleaned": True}

    def applicable(self, state: WorldState) -> bool:
        return state.dirt_visible and not state.low_battery


class MoveAction(GoapAction):
    name = "move"
    cost = 1.2
    preconditions = ("battery_ok", "target_exists")
    effects = {"at_target": True}

    def applicable(self, state: WorldState) -> bool:
        return state.dirt_remaining and not state.low_battery


class WaitAction(GoapAction):
    name = "wait"
    cost = 5.0

    def applicable(self, state: WorldState) -> bool:
        return True


class DStarLitePlanner:
    def __init__(self, grid: List[List[int]], resolution: float):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.resolution = resolution
        self.km = 0.0
        self.rhs: Dict[Tuple[int, int], float] = {}
        self.g: Dict[Tuple[int, int], float] = {}
        self.open: List[Tuple[float, float, Tuple[int, int]]] = []
        self.open_best: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None

    def _world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        return int(pos[0] / self.resolution), int(pos[1] / self.resolution)

    def _grid_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        return (
            pos[0] * self.resolution + self.resolution / 2.0,
            pos[1] * self.resolution + self.resolution / 2.0,
        )

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _is_free(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def _neighbors(self, cell: Tuple[int, int]):
        x, y = cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height:
                yield nxt, math.hypot(dx, dy)

    def _calculate_key(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        g = self.g.get(cell, float("inf"))
        rhs = self.rhs.get(cell, float("inf"))
        m = min(g, rhs)
        return (m + self._heuristic(self.start, cell) + self.km, m)

    def _push(self, cell: Tuple[int, int], key: Tuple[float, float]) -> None:
        best = self.open_best.get(cell)
        if best is None or key < best:
            self.open_best[cell] = key
            self.open.append((key[0], key[1], cell))

    def _heapify(self) -> None:
        if self.open and not hasattr(self, "_heapified"):
            import heapq

            heapq.heapify(self.open)
            self._heapified = True

    def _top_key(self) -> Tuple[float, float]:
        import heapq

        self._heapify()
        while self.open:
            k1, k2, cell = self.open[0]
            best = self.open_best.get(cell)
            if best is None or (k1, k2) != best:
                heapq.heappop(self.open)
                continue
            return (k1, k2)
        return (float("inf"), float("inf"))

    def _pop(self):
        import heapq

        self._heapify()
        while self.open:
            k1, k2, cell = heapq.heappop(self.open)
            best = self.open_best.get(cell)
            if best is None or (k1, k2) != best:
                continue
            del self.open_best[cell]
            return (k1, k2), cell
        return (float("inf"), float("inf")), None

    def _update_vertex(self, cell: Tuple[int, int]) -> None:
        if cell != self.goal:
            min_rhs = float("inf")
            for nxt, cost in self._neighbors(cell):
                if self._is_free(nxt):
                    min_rhs = min(min_rhs, self.g.get(nxt, float("inf")) + cost)
            self.rhs[cell] = min_rhs
        if self.g.get(cell, float("inf")) != self.rhs.get(cell, float("inf")):
            self._push(cell, self._calculate_key(cell))
        elif cell in self.open_best:
            del self.open_best[cell]

    def initialize(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> None:
        self.start = self._world_to_grid(start_pos)
        self.goal = self._world_to_grid(goal_pos)
        self.km = 0.0
        self.rhs = {self.goal: 0.0}
        self.g = {}
        self.open = []
        self.open_best = {}
        if hasattr(self, "_heapified"):
            delattr(self, "_heapified")
        self._push(self.goal, self._calculate_key(self.goal))

    def update_grid(self, grid: List[List[int]], changed_cells: Sequence[Tuple[int, int, int]]) -> None:
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        if self.start is None or self.goal is None:
            return
        for x, y, _occupied in changed_cells:
            cell = (x, y)
            self._update_vertex(cell)
            for nxt, _ in self._neighbors(cell):
                self._update_vertex(nxt)

    def compute_shortest_path(self, max_iters: int = 100000) -> None:
        iters = 0
        while iters < max_iters and (
            self._top_key() < self._calculate_key(self.start)
            or self.rhs.get(self.start, float("inf")) != self.g.get(self.start, float("inf"))
        ):
            iters += 1
            old_key, cell = self._pop()
            if cell is None:
                break
            new_key = self._calculate_key(cell)
            if old_key < new_key:
                self._push(cell, new_key)
                continue
            if self.g.get(cell, float("inf")) > self.rhs.get(cell, float("inf")):
                self.g[cell] = self.rhs[cell]
                for nxt, _ in self._neighbors(cell):
                    self._update_vertex(nxt)
            else:
                self.g[cell] = float("inf")
                self._update_vertex(cell)
                for nxt, _ in self._neighbors(cell):
                    self._update_vertex(nxt)

    def replan(self, start_pos: Tuple[float, float], changed_cells: Sequence[Tuple[int, int, int]]) -> List[Tuple[float, float]]:
        current_start = self._world_to_grid(start_pos)
        if self.start is not None:
            self.km += self._heuristic(self.start, current_start)
        self.start = current_start
        self.update_grid(self.grid, changed_cells)
        self.compute_shortest_path()
        return self.get_path()

    def get_path(self, max_len: int = 5000) -> List[Tuple[float, float]]:
        if self.start is None or self.goal is None or not self._is_free(self.start) or not self._is_free(self.goal):
            return []
        current = self.start
        path = [current]
        for _ in range(max_len):
            if current == self.goal:
                break
            best = None
            best_cost = float("inf")
            for nxt, move_cost in self._neighbors(current):
                if not self._is_free(nxt):
                    continue
                candidate = move_cost + self.g.get(nxt, float("inf"))
                if candidate < best_cost:
                    best_cost = candidate
                    best = nxt
            if best is None or best_cost == float("inf"):
                break
            path.append(best)
            current = best
        return [self._grid_to_world(p) for p in path]


class GOAPTeamController:
    def __init__(self, env, planner_algorithm: str = "dstar", config: Optional[Dict[str, float]] = None):
        self.env = env
        self.planner_algorithm = planner_algorithm
        self.config = {
            "grid_resolution": getattr(env, "grid_resolution", 20),
            "replan_interval": 12,
            "auction_interval": 8,
            "waypoint_reach_dist": 24.0,
            "max_speed": 9.0,
            "min_speed": 1.5,
            "turn_gain": 3.0,
            "battery_low_threshold": 220.0,
            "battery_resume_threshold": 520.0,
            "avoidance_radius": 90.0,
            "robot_radius": 30.0,
            "ttc_horizon": 6.0,
            "ttc_threshold": 1.2,
            "safety_margin": 10.0,
            "sample_speeds": (0.0, 0.4, 0.7, 1.0),
            "sample_angle_offsets_deg": (-80, -50, -30, -15, 0, 15, 30, 50, 80),
            "partition_cleaning": True,
            "partition_overlap": 40.0,
            "team_deconflict_dist": 85.0,
            "yield_speed_scale": 0.45,
            "charge_hysteresis_dist": 140.0,
            "target_cluster_radius": 120.0,
            "target_cluster_k": 12,
            "cleaning_priority_mode": True,
            "lookahead_distance": 70.0,
            "path_deviation_replan_dist": 55.0,
            "avoidance_mode": "ttc",  # 'ttc' | 'orca'
            "orca_time_horizon": 4.0,
        }
        if config:
            self.config.update(config)
        if self.planner_algorithm == "dstar":
            # D* replans more often around moving obstacles; keep controller slightly conservative.
            self.config["ttc_threshold"] = max(float(self.config["ttc_threshold"]), 1.8)
            self.config["sample_speeds"] = (0.0, 0.35, 0.6, 0.85)
            self.config["sample_angle_offsets_deg"] = (-60, -40, -25, -12, 0, 12, 25, 40, 60)
        self.actions = [ChargeAction(), CleanAction(), MoveAction(), WaitAction()]
        self.goap_planner = GoapPlanner(
            actions=[
                Action("move", 1.2, frozenset({"battery_ok", "target_exists"}), frozenset({"at_target"})),
                Action("clean", 1.0, frozenset({"at_target", "target_exists"}), frozenset({"target_cleaned"})),
                Action("charge", 2.0, frozenset({"at_charger"}), frozenset({"battery_ok"})),
                Action("wait", 5.0, frozenset(), frozenset()),
            ]
        )
        self.state_by_robot: Dict[int, Dict[str, object]] = {}

    def _get_state(self, robot_idx: int) -> Dict[str, object]:
        state = self.state_by_robot.get(robot_idx)
        if state is None:
            state = {
                "planner": None,
                "path": [],
                "target": None,
                "goal_kind": None,
                "steps_since_replan": 0,
                "last_velocity": (0.0, 0.0),
                "last_action_name": "wait",
                "prev_pos": None,
                "stuck_steps": 0,
                "escape_steps": 0,
                "escape_sign": 1.0,
            }
            self.state_by_robot[robot_idx] = state
        return state

    def _goap_plan(self, predicates: Dict[str, bool], goals: Dict[str, bool]) -> List[str]:
        start = frozenset([k for k, v in predicates.items() if v])
        goal = frozenset([k for k, v in goals.items() if v])
        plan = self.goap_planner.plan(start, goal, max_expansions=2000)
        return plan or ["wait"]

    def _build_world_state(self, obs_one: Dict[str, object], robot) -> WorldState:
        charger_pos = self._charger_position()
        on_charger = math.hypot(robot.x - charger_pos[0], robot.y - charger_pos[1]) < 60.0
        dirt_visible = bool(obs_one.get("detected_dirt"))
        dirt_remaining = any(isinstance(obj, Dirt) for obj in self.env.passive_objects)
        low_battery = float(obs_one.get("battery", 0.0)) < self.config["battery_low_threshold"]
        return WorldState(
            low_battery=low_battery,
            on_charger=on_charger,
            dirt_visible=dirt_visible,
            dirt_remaining=dirt_remaining,
        )

    def _charger_position(self) -> Tuple[float, float]:
        for obj in self.env.passive_objects:
            if isinstance(obj, Charger):
                return float(obj.centreX), float(obj.centreY)
        return self.env.width / 2.0, self.env.height / 2.0

    def _collect_candidate_targets(self) -> List[Tuple[float, float]]:
        dirt_positions = []
        for obj in self.env.passive_objects:
            if isinstance(obj, Dirt):
                dirt_positions.append((float(obj.centreX), float(obj.centreY)))
        return dirt_positions

    def _auction_assignments(self, obs: Sequence[Dict[str, object]]) -> Dict[int, Optional[Tuple[float, float]]]:
        candidates = self._collect_candidate_targets()
        if not candidates:
            return {idx: None for idx in range(len(obs))}

        remaining = candidates[:]
        assignments: Dict[int, Optional[Tuple[float, float]]] = {}
        ordered_robots = sorted(range(len(obs)), key=lambda idx: float(obs[idx].get("battery", 0.0)), reverse=True)
        partition_mode = bool(self.config.get("partition_cleaning", True))
        overlap = float(self.config.get("partition_overlap", 40.0))
        width = float(getattr(self.env, "width", 1000.0))
        n = max(1, len(obs))

        def in_partition(target: Tuple[float, float], robot_idx: int) -> bool:
            if not partition_mode:
                return True
            stripe = width / n
            left = robot_idx * stripe - overlap
            right = (robot_idx + 1) * stripe + overlap
            return left <= target[0] <= right

        for robot_idx in ordered_robots:
            pos = (float(obs[robot_idx]["x"]), float(obs[robot_idx]["y"]))
            battery = float(obs[robot_idx]["battery"])
            best = None
            best_bid = float("inf")
            preferred_targets = [t for t in remaining if in_partition(t, robot_idx)]
            fallback_targets = remaining if not preferred_targets else preferred_targets
            for target in fallback_targets:
                bid_dist = math.hypot(pos[0] - target[0], pos[1] - target[1])
                density = self._target_density(target, candidates)
                bid = bid_dist - 6.0 * density
                bid += max(0.0, self.config["battery_low_threshold"] - battery) * 0.2
                if partition_mode and not in_partition(target, robot_idx):
                    bid += 60.0
                if bid < best_bid:
                    best_bid = bid
                    best = target
            assignments[robot_idx] = best
            if best is not None:
                remaining.remove(best)
        for idx in range(len(obs)):
            assignments.setdefault(idx, None)
        return assignments

    def _target_density(self, center: Tuple[float, float], all_targets: Sequence[Tuple[float, float]]) -> int:
        radius = float(self.config.get("target_cluster_radius", 120.0))
        k = int(self.config.get("target_cluster_k", 12))
        ranked = sorted(all_targets, key=lambda t: math.hypot(t[0] - center[0], t[1] - center[1]))[: max(1, k)]
        return sum(1 for t in ranked if math.hypot(t[0] - center[0], t[1] - center[1]) <= radius)

    def _plan_path(
        self,
        robot_idx: int,
        state: Dict[str, object],
        start_pos: Tuple[float, float],
        goal_pos: Optional[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if goal_pos is None:
            return []

        grid = self.env.get_grid_map(include_dynamic=True)
        resolution = self.config["grid_resolution"]
        if self.planner_algorithm == "astar":
            planner = AStarPlanner(grid, resolution=resolution)
            return planner.plan(start_pos, goal_pos)

        planner = state.get("planner")
        goal_changed = state.get("planner_goal") != self.env.world_to_grid(goal_pos)
        if planner is None or goal_changed:
            planner = DStarLitePlanner(grid, resolution=resolution)
            planner.initialize(start_pos, goal_pos)
            planner.compute_shortest_path()
            state["planner"] = planner
            state["planner_goal"] = self.env.world_to_grid(goal_pos)
            return planner.get_path()

        planner.grid = grid
        return planner.replan(start_pos, self.env.get_changed_cells())

    def _follow_path(
        self,
        obs_one: Dict[str, object],
        path: List[Tuple[float, float]],
        nearby_robots: Sequence[Dict[str, float]],
        nearby_dynamic: Sequence[Dict[str, float]],
        robot_state: Dict[str, object],
    ) -> Tuple[float, float]:
        if not path:
            return (0.0, 0.0)

        pos = (float(obs_one["x"]), float(obs_one["y"]))
        theta = float(obs_one["theta"])
        target = self._select_lookahead_target(pos, path)

        desired_vx = target[0] - pos[0]
        desired_vy = target[1] - pos[1]
        mag = math.hypot(desired_vx, desired_vy)
        if mag > 1e-6:
            desired_vx = (desired_vx / mag) * self.config["max_speed"]
            desired_vy = (desired_vy / mag) * self.config["max_speed"]

        v_pref = (desired_vx, desired_vy)
        v_cur = robot_state.get("last_velocity", (0.0, 0.0))
        if str(self.config.get("avoidance_mode", "ttc")).lower() == "orca":
            v_safe = self._orca_velocity(
                pos=pos,
                current_velocity=v_cur,
                preferred_velocity=v_pref,
                nearby_robots=nearby_robots,
                nearby_dynamic=nearby_dynamic,
            )
        else:
            v_safe = self._safe_velocity_sampling(
                pos=pos,
                current_velocity=v_cur,
                preferred_velocity=v_pref,
                theta=theta,
                nearby_robots=nearby_robots,
                nearby_dynamic=nearby_dynamic,
            )
        robot_state["last_velocity"] = v_safe

        heading = math.atan2(v_safe[1], v_safe[0]) if abs(v_safe[0]) + abs(v_safe[1]) > 1e-9 else theta
        angle_diff = math.atan2(math.sin(heading - theta), math.cos(heading - theta))

        speed = math.hypot(v_safe[0], v_safe[1])
        if speed < 1e-6:
            return (0.0, 0.0)

        # Prevent oscillatory collisions while still making forward progress.
        angle_abs = abs(angle_diff)
        align_scale = max(0.18, 1.0 - angle_abs / 1.3)
        forward = speed * align_scale
        if mag > self.config["waypoint_reach_dist"] * 2.0:
            forward = max(forward, self.config["min_speed"] * 0.8)
        elif mag > self.config["waypoint_reach_dist"]:
            forward = max(forward, self.config["min_speed"] * 0.5)
        turn = self.config["turn_gain"] * angle_diff
        left = max(-self.config["max_speed"], min(self.config["max_speed"], forward - turn))
        right = max(-self.config["max_speed"], min(self.config["max_speed"], forward + turn))
        return (left, right)

    def _select_lookahead_target(
        self,
        pos: Tuple[float, float],
        path: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        if not path:
            return pos
        lookahead = float(self.config.get("lookahead_distance", 70.0))
        # discard consumed waypoints
        while len(path) > 1 and math.hypot(pos[0] - path[0][0], pos[1] - path[0][1]) < self.config["waypoint_reach_dist"]:
            path.pop(0)
        for wp in path:
            if math.hypot(pos[0] - wp[0], pos[1] - wp[1]) >= lookahead:
                return wp
        return path[-1]

    def _safe_velocity_sampling(
        self,
        pos: Tuple[float, float],
        current_velocity: Tuple[float, float],
        preferred_velocity: Tuple[float, float],
        theta: float,
        nearby_robots: Sequence[Dict[str, float]],
        nearby_dynamic: Sequence[Dict[str, float]],
    ) -> Tuple[float, float]:
        max_speed = self.config["max_speed"]
        pref_heading = theta
        if abs(preferred_velocity[0]) + abs(preferred_velocity[1]) > 1e-9:
            pref_heading = math.atan2(preferred_velocity[1], preferred_velocity[0])

        candidates: List[Tuple[float, float]] = [(0.0, 0.0)]
        for s_factor in self.config["sample_speeds"]:
            speed = max_speed * float(s_factor)
            for angle_deg in self.config["sample_angle_offsets_deg"]:
                heading = pref_heading + math.radians(float(angle_deg))
                candidates.append((speed * math.cos(heading), speed * math.sin(heading)))

        best = (0.0, 0.0)
        best_score = float("inf")
        threshold = self.config["ttc_threshold"]

        for cand in candidates:
            ttc = self._min_ttc(pos, cand, nearby_robots, nearby_dynamic)
            if ttc < 1e-6:
                continue
            pref_err = math.hypot(cand[0] - preferred_velocity[0], cand[1] - preferred_velocity[1])
            smooth = math.hypot(cand[0] - current_velocity[0], cand[1] - current_velocity[1])
            if ttc >= threshold:
                score = pref_err + 0.25 * smooth - 0.08 * math.hypot(cand[0], cand[1])
            else:
                score = pref_err + 0.25 * smooth + 100.0 * (threshold - ttc)
            if score < best_score:
                best_score = score
                best = cand

        return best

    def _orca_velocity(
        self,
        pos: Tuple[float, float],
        current_velocity: Tuple[float, float],
        preferred_velocity: Tuple[float, float],
        nearby_robots: Sequence[Dict[str, float]],
        nearby_dynamic: Sequence[Dict[str, float]],
    ) -> Tuple[float, float]:
        neighbors: List[Neighbor] = []
        for other in nearby_robots:
            neighbors.append(
                Neighbor(
                    position=(float(other["x"]), float(other["y"])),
                    velocity=(0.0, 0.0),
                    radius=float(self.config.get("robot_radius", 30.0)),
                )
            )
        for other in nearby_dynamic:
            neighbors.append(
                Neighbor(
                    position=(float(other["x"]), float(other["y"])),
                    velocity=(float(other.get("vx", 0.0)), float(other.get("vy", 0.0))),
                    radius=float(other.get("radius", self.config.get("robot_radius", 30.0))),
                )
            )
        return compute_orca_velocity(
            position=pos,
            velocity=current_velocity,
            preferred_velocity=preferred_velocity,
            radius=float(self.config.get("robot_radius", 30.0)),
            neighbors=neighbors,
            time_horizon=float(self.config.get("orca_time_horizon", 4.0)),
            max_speed=float(self.config.get("max_speed", 9.0)),
        )

    def _min_ttc(
        self,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        nearby_robots: Sequence[Dict[str, float]],
        nearby_dynamic: Sequence[Dict[str, float]],
    ) -> float:
        min_t = float("inf")
        own_r = self.config["robot_radius"]
        margin = self.config["safety_margin"]

        for other in nearby_robots:
            t = self._ttc_two_bodies(
                pos,
                vel,
                (float(other["x"]), float(other["y"])),
                (0.0, 0.0),
                own_r + own_r + margin,
            )
            min_t = min(min_t, t)

        for other in nearby_dynamic:
            t = self._ttc_two_bodies(
                pos,
                vel,
                (float(other["x"]), float(other["y"])),
                (float(other.get("vx", 0.0)), float(other.get("vy", 0.0))),
                own_r + float(other.get("radius", own_r)) + margin,
            )
            min_t = min(min_t, t)

        return min_t

    def _ttc_two_bodies(
        self,
        p1: Tuple[float, float],
        v1: Tuple[float, float],
        p2: Tuple[float, float],
        v2: Tuple[float, float],
        combined_radius: float,
    ) -> float:
        rx = p2[0] - p1[0]
        ry = p2[1] - p1[1]
        rvx = v2[0] - v1[0]
        rvy = v2[1] - v1[1]

        c = rx * rx + ry * ry - combined_radius * combined_radius
        if c <= 0.0:
            return 0.0
        a = rvx * rvx + rvy * rvy
        if a < 1e-9:
            return float("inf")
        b = 2.0 * (rx * rvx + ry * rvy)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return float("inf")
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        horizon = self.config["ttc_horizon"]
        candidates = [t for t in (t1, t2) if 0.0 <= t <= horizon]
        if not candidates:
            return float("inf")
        return min(candidates)

    def compute_actions(self, obs: Sequence[Dict[str, object]]) -> List[Tuple[float, float]]:
        assignments = self._auction_assignments(obs)
        actions: List[Tuple[float, float]] = []

        for robot_idx, obs_one in enumerate(obs):
            robot = self.env.agents[robot_idx]
            state = self._get_state(robot_idx)
            world_state = self._build_world_state(obs_one, robot)
            pos_now = (float(obs_one["x"]), float(obs_one["y"]))
            prev_pos = state.get("prev_pos")
            if prev_pos is not None:
                moved = math.hypot(pos_now[0] - prev_pos[0], pos_now[1] - prev_pos[1])
                if moved < 2.0:
                    state["stuck_steps"] = int(state.get("stuck_steps", 0)) + 1
                else:
                    state["stuck_steps"] = 0
            state["prev_pos"] = pos_now

            assigned_target = assignments.get(robot_idx)
            charger_pos = self._charger_position()
            at_charger = math.hypot(float(obs_one["x"]) - charger_pos[0], float(obs_one["y"]) - charger_pos[1]) < 60.0
            at_target = (
                assigned_target is not None
                and math.hypot(float(obs_one["x"]) - assigned_target[0], float(obs_one["y"]) - assigned_target[1])
                < self.config["waypoint_reach_dist"] * 1.2
            )
            predicates = {
                "battery_ok": not world_state.low_battery,
                "at_charger": at_charger,
                "target_exists": assigned_target is not None,
                "at_target": at_target,
                "target_cleaned": False if assigned_target is not None else True,
            }
            nearby_charger = math.hypot(float(obs_one["x"]) - charger_pos[0], float(obs_one["y"]) - charger_pos[1]) < float(
                self.config.get("charge_hysteresis_dist", 140.0)
            )
            force_charge = world_state.low_battery and nearby_charger
            if self.config.get("cleaning_priority_mode", True) and (not force_charge) and assigned_target is not None:
                next_action = "move"
            else:
                goals = {"battery_ok": True} if force_charge else {"target_cleaned": True}
                plan = self._goap_plan(predicates, goals)
                next_action = plan[0] if plan else "wait"
            state["last_action_name"] = next_action

            if force_charge and next_action == "charge":
                goal_kind = "charge"
                target = charger_pos
            elif next_action in ("move", "clean"):
                goal_kind = "clean"
                target = assigned_target
            else:
                goal_kind = "wait"
                target = None

            if goal_kind == "charge" and float(obs_one["battery"]) >= self.config["battery_resume_threshold"]:
                goal_kind = "clean"
                target = assignments.get(robot_idx)

            state["steps_since_replan"] += 1
            need_replan = (
                target != state.get("target")
                or state["steps_since_replan"] >= self.config["replan_interval"]
                or bool(self.env.get_changed_cells())
                or not state.get("path")
            )
            if not need_replan and state.get("path"):
                closest = min(
                    (math.hypot(float(obs_one["x"]) - p[0], float(obs_one["y"]) - p[1]) for p in state["path"]),
                    default=float("inf"),
                )
                if closest > float(self.config.get("path_deviation_replan_dist", 55.0)):
                    need_replan = True

            if need_replan:
                start = (float(obs_one["x"]), float(obs_one["y"]))
                state["path"] = self._plan_path(robot_idx, state, start, target)
                if state["path"] and len(state["path"]) > 1:
                    state["path"] = state["path"][1:]
                if (not state["path"]) and target is not None:
                    if math.hypot(start[0] - target[0], start[1] - target[1]) > self.config["waypoint_reach_dist"]:
                        state["path"] = [target]
                state["steps_since_replan"] = 0
                state["target"] = target
                state["goal_kind"] = goal_kind

            if goal_kind == "wait":
                actions.append((0.0, 0.0))
                continue

            if int(state.get("escape_steps", 0)) > 0:
                state["escape_steps"] = int(state["escape_steps"]) - 1
                s = float(state.get("escape_sign", 1.0))
                actions.append((-2.5 * s, 1.5 * s))
                continue

            if int(state.get("stuck_steps", 0)) >= 12:
                state["escape_steps"] = 8
                state["stuck_steps"] = 0
                state["escape_sign"] = 1.0 if (robot_idx % 2 == 0) else -1.0
                s = float(state["escape_sign"])
                actions.append((-2.5 * s, 1.5 * s))
                continue

            path = state.get("path") or []
            action = self._follow_path(
                obs_one,
                path,
                obs_one.get("nearby_robots", []),
                obs_one.get("detected_dynamic_obstacles", []),
                state,
            )
            actions.append(action)

        return self._apply_team_deconfliction(obs, actions)

    def _apply_team_deconfliction(
        self,
        obs: Sequence[Dict[str, object]],
        actions: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        adjusted = list(actions)
        threshold = float(self.config.get("team_deconflict_dist", 85.0))
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                pi = (float(obs[i]["x"]), float(obs[i]["y"]))
                pj = (float(obs[j]["x"]), float(obs[j]["y"]))
                if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) > threshold:
                    continue
                # Lower-energy robot yields when robots are too close.
                if float(obs[i].get("battery", 0.0)) <= float(obs[j].get("battery", 0.0)):
                    adjusted[i] = (
                        adjusted[i][0] * self.config["yield_speed_scale"],
                        adjusted[i][1] * self.config["yield_speed_scale"],
                    )
                else:
                    adjusted[j] = (
                        adjusted[j][0] * self.config["yield_speed_scale"],
                        adjusted[j][1] * self.config["yield_speed_scale"],
                    )
        return adjusted


class GOAPAgent:
    """
    Compatibility wrapper for single-robot simulations.
    """

    def __init__(self, env=None, planner_algorithm: str = "dstar", config: Optional[Dict[str, float]] = None):
        self.env = env
        self.planner_algorithm = planner_algorithm
        self.config = config or {}
        self._controller = None

    def bind_env(self, env) -> None:
        self.env = env
        self._controller = GOAPTeamController(env, planner_algorithm=self.planner_algorithm, config=self.config)

    def act(self, obs_one: Dict[str, object], robot, passive_objects, robot_index: int = 0):
        if self._controller is None:
            if self.env is None:
                raise ValueError("GOAPAgent requires env binding before act().")
            self.bind_env(self.env)
        return self._controller.compute_actions([obs_one])[robot_index]


class RandomTeamController:
    """True random baseline controller for fair algorithm comparison."""

    def __init__(self, env, config: Optional[Dict[str, float]] = None):
        self.env = env
        self.config = {
            "max_speed": 8.0,
            "turn_prob": 0.35,
            "turn_speed": 5.0,
            "forward_speed": 6.0,
        }
        if config:
            self.config.update(config)

    def compute_actions(self, obs: Sequence[Dict[str, object]]) -> List[Tuple[float, float]]:
        actions: List[Tuple[float, float]] = []
        for _ in obs:
            if random.random() < float(self.config["turn_prob"]):
                turn = random.choice([-1.0, 1.0]) * float(self.config["turn_speed"])
                actions.append((-turn, turn))
            else:
                base = float(self.config["forward_speed"]) + random.uniform(-1.0, 1.0)
                base = max(-float(self.config["max_speed"]), min(float(self.config["max_speed"]), base))
                actions.append((base, base))
        return actions
