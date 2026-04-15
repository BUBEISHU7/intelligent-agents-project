# agents/planner_agent.py
import math
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from env.astar import AStarPlanner
from env.simpleBot2 import Dirt
from agents.goap_agent import DStarLitePlanner as UnifiedDStarLitePlanner


class _LegacyDStarLitePlanner:
    """
    纯 D* Lite（Koenig & Likhachev, 2002）的一个最小可用实现。
    目的：替代项目内 env/dstarlite.py 可能导致的卡死/未响应问题（不修改 env 文件）。
    """

    def __init__(self, grid, resolution=1.0):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.resolution = resolution
        self.km = 0.0
        self.rhs = {}
        self.g = {}
        self.open = []
        self.open_best = {}  # s -> best key for stale-skip
        self.start = None
        self.goal = None

    def _is_free(self, s):
        x, y = s
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _calculate_key(self, s):
        g = self.g.get(s, float("inf"))
        rhs = self.rhs.get(s, float("inf"))
        m = min(g, rhs)
        return (m + self._heuristic(self.start, s) + self.km, m)

    def _push(self, s, key):
        best = self.open_best.get(s)
        if best is None or key < best:
            self.open_best[s] = key
            # flatten key for heap ordering speed
            self.open.append((key[0], key[1], s))

    def _heapify_if_needed(self):
        # we keep appending to list; heapify lazily when needed
        if self.open and not hasattr(self, "_heapified"):
            import heapq

            heapq.heapify(self.open)
            self._heapified = True

    def _top_key(self):
        import heapq

        self._heapify_if_needed()
        while self.open:
            k1, k2, s = self.open[0]
            best = self.open_best.get(s)
            if best is None or (k1, k2) != best:
                heapq.heappop(self.open)
                continue
            return (k1, k2)
        return (float("inf"), float("inf"))

    def _pop(self):
        import heapq

        self._heapify_if_needed()
        while self.open:
            k1, k2, s = heapq.heappop(self.open)
            best = self.open_best.get(s)
            if best is None or (k1, k2) != best:
                continue
            del self.open_best[s]
            return (k1, k2), s
        return (float("inf"), float("inf")), None

    def _neighbors(self, s):
        x, y = s
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny), math.hypot(dx, dy)

    def _update_vertex(self, u):
        if u != self.goal:
            min_rhs = float("inf")
            for s, cost in self._neighbors(u):
                if not self._is_free(s):
                    continue
                min_rhs = min(min_rhs, self.g.get(s, float("inf")) + cost)
            self.rhs[u] = min_rhs

        if self.g.get(u, float("inf")) != self.rhs.get(u, float("inf")):
            self._push(u, self._calculate_key(u))
        else:
            # make queued entry stale if exists
            if u in self.open_best:
                del self.open_best[u]

    def _world_to_grid(self, pos):
        x, y = pos
        return (int(x / self.resolution), int(y / self.resolution))

    def _grid_to_world(self, s):
        gx, gy = s
        return (
            gx * self.resolution + self.resolution / 2.0,
            gy * self.resolution + self.resolution / 2.0,
        )

    def initialize(self, start_pos, goal_pos):
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

    def compute_shortest_path(self, max_iters=200000):
        it = 0
        while it < max_iters and (
            self._top_key() < self._calculate_key(self.start)
            or self.rhs.get(self.start, float("inf")) != self.g.get(self.start, float("inf"))
        ):
            it += 1
            k_old, u = self._pop()
            if u is None:
                break
            k_new = self._calculate_key(u)
            if k_old < k_new:
                self._push(u, k_new)
                continue

            gu = self.g.get(u, float("inf"))
            rhsu = self.rhs.get(u, float("inf"))
            if gu > rhsu:
                self.g[u] = rhsu
                for s, _ in self._neighbors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u)
                for s, _ in self._neighbors(u):
                    self._update_vertex(s)

    def get_path(self, max_len=5000):
        # greedy extraction following rhs-minimizing neighbor
        if self.start is None or self.goal is None:
            return []
        if not self._is_free(self.start) or not self._is_free(self.goal):
            return []

        path = [self.start]
        current = self.start
        for _ in range(max_len):
            if current == self.goal:
                break
            best = None
            best_cost = float("inf")
            for s, cost in self._neighbors(current):
                if not self._is_free(s):
                    continue
                v = cost + self.g.get(s, float("inf"))
                if v < best_cost:
                    best_cost = v
                    best = s
            if best is None or best_cost == float("inf"):
                break
            path.append(best)
            current = best
        return [self._grid_to_world(s) for s in path]


class PlannerAgent:
    """
    使用 A* 或 D* Lite 风格的路径规划做清扫控制。

    允许修改的部分：目标选择（更适合清扫），但路径规划算法保持“纯 A* / 纯 D* Lite”。
    """
    
    def __init__(self, planner_type, env, config=None):
        self.type = planner_type
        self.env = env
        config = config or {}

        self.grid_resolution = int(config.get("grid_resolution", 20))
        self.replan_interval = int(config.get("replan_interval", 8))
        self.waypoint_reach_dist = float(config.get("waypoint_reach_dist", 10.0))

        self.base_forward = float(config.get("base_forward", 7.0))
        self.min_forward = float(config.get("min_forward", 2.0))
        self.turn_gain = float(config.get("turn_gain", 2.4))

        # 目标选择参数（允许修改）
        self.target_mode = str(config.get("target_mode", "cluster"))  # 'nearest' | 'cluster'
        self.cluster_k = int(config.get("cluster_k", 6))
        self.cluster_radius = float(config.get("cluster_radius", 120.0))

        # 每个机器人一套独立规划状态（支持多机器人）
        self._state_by_robot = {}

    def _get_grid_map(self):
        # env/robot_env.py 里不一定提供 get_grid_map，这里在 agent 内部构建栅格地图：
        # 0=free, 1=obstacle
        resolution = self.grid_resolution
        width = int(getattr(self.env, "width", 1000))
        height = int(getattr(self.env, "height", 1000))

        w_grid = max(1, width // resolution)
        h_grid = max(1, height // resolution)
        grid = [[0 for _ in range(w_grid)] for _ in range(h_grid)]

        for obj in getattr(self.env, "passive_objects", []):
            # 只把障碍物标为 1；灰尘/充电桩/WiFi 不做障碍
            if obj.__class__.__name__ != "Obstacle":
                continue
            if not hasattr(obj, "centreX") or not hasattr(obj, "centreY"):
                continue
            gx = int(float(obj.centreX) // resolution)
            gy = int(float(obj.centreY) // resolution)
            if 0 <= gx < w_grid and 0 <= gy < h_grid:
                grid[gy][gx] = 1
        return grid

    def _list_dirt_positions(self):
        dirt = []
        for obj in self.env.passive_objects:
            if isinstance(obj, Dirt):
                dirt.append((float(obj.centreX), float(obj.centreY)))
        return dirt

    def _find_nearest_dirt(self, pos):
        dirt = self._list_dirt_positions()
        if not dirt:
            return None

        if self.target_mode == "nearest":
            return min(dirt, key=lambda d: math.hypot(pos[0] - d[0], pos[1] - d[1]))

        # cluster mode: 从最近的 K 个灰尘中，选“周围灰尘最多的点”（清扫更连贯，减少来回折返）
        candidates = sorted(dirt, key=lambda d: math.hypot(pos[0] - d[0], pos[1] - d[1]))[: max(1, self.cluster_k)]
        r2 = self.cluster_radius * self.cluster_radius

        best = None
        best_score = (-1, float("inf"))  # (density, dist) - density higher better, dist lower better
        for c in candidates:
            cx, cy = c
            density = 0
            for (x, y) in dirt:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= r2:
                    density += 1
            dist = math.hypot(pos[0] - cx, pos[1] - cy)
            score = (density, -dist)
            if score > best_score:
                best_score = score
                best = c
        return best

    def _get_state(self, robot_idx):
        state = self._state_by_robot.get(robot_idx)
        if state is None:
            state = {
                "path": [],
                "step_counter": 0,
                "planner": None,
                "current_target": None,
            }
            self._state_by_robot[robot_idx] = state
        return state

    def _plan_path(self, state, start_pos, goal_pos):
        if goal_pos is None:
            return []
        grid = self._get_grid_map()
        if self.type == 'astar':
            if state["planner"] is None:
                state["planner"] = AStarPlanner(grid, resolution=self.grid_resolution)
            return state["planner"].plan(start_pos, goal_pos)
        elif self.type == 'dstar_lite':
            if state["planner"] is None:
                state["planner"] = _DStarLitePlanner(grid, resolution=self.grid_resolution)
                state["planner"].initialize(start_pos, goal_pos)
                state["last_goal_grid"] = state["planner"]._world_to_grid(goal_pos)
            else:
                # 目标变化则重新初始化（纯 D* Lite 用法）
                current_goal_grid = state["planner"]._world_to_grid(goal_pos)
                if state.get("last_goal_grid") != current_goal_grid:
                    state["planner"] = _DStarLitePlanner(grid, resolution=self.grid_resolution)
                    state["planner"].initialize(start_pos, goal_pos)
                    state["last_goal_grid"] = current_goal_grid
                else:
                    state["planner"].grid = grid
                    state["planner"].width = len(grid[0])
                    state["planner"].height = len(grid)
                    state["planner"].resolution = self.grid_resolution
                    state["planner"].start = state["planner"]._world_to_grid(start_pos)
                    state["planner"].compute_shortest_path(max_iters=200000)
            return state["planner"].get_path()
        return []

    def _action_for_one(self, robot_idx, obs_one):
        state = self._get_state(robot_idx)
        state["step_counter"] += 1

        pos = (float(obs_one['x']), float(obs_one['y']))
        theta = obs_one.get('theta', 0.0)

        # 边界保持：靠近画布边缘时，优先朝画布中心转回去（不改 env，只在动作层做约束）
        w = float(getattr(self.env, "width", 1000))
        h = float(getattr(self.env, "height", 1000))
        margin = 60.0
        if pos[0] < margin or pos[0] > (w - margin) or pos[1] < margin or pos[1] > (h - margin):
            cx, cy = w / 2.0, h / 2.0
            desired = math.atan2(cy - pos[1], cx - pos[0])
            aerr = math.atan2(math.sin(desired - theta), math.cos(desired - theta))
            forward = max(self.min_forward, self.base_forward * 0.8)
            turn = self.turn_gain * aerr
            left = max(-10.0, min(10.0, forward - turn))
            right = max(-10.0, min(10.0, forward + turn))
            return (left, right)

        need_replan = (not state["path"]) or (state["step_counter"] % self.replan_interval == 0)
        if need_replan:
            state["current_target"] = self._find_nearest_dirt(pos)
            if state["current_target"]:
                state["path"] = self._plan_path(state, pos, state["current_target"])
                if state["path"] and len(state["path"]) > 1:
                    state["path"] = state["path"][1:]

        if not state["path"]:
            return (0.0, 0.0)

        target = state["path"][0]
        if math.hypot(pos[0] - target[0], pos[1] - target[1]) < self.waypoint_reach_dist:
            state["path"].pop(0)
            if not state["path"]:
                return (0.0, 0.0)
            target = state["path"][0]

        # 计算朝向目标的角度差
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - theta
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        forward = self.base_forward * (1.0 - min(1.0, abs(angle_diff) / math.pi))
        forward = max(self.min_forward, forward)
        turn = self.turn_gain * angle_diff
        left_speed = forward - turn
        right_speed = forward + turn

        left_speed = max(-10.0, min(10.0, left_speed))
        right_speed = max(-10.0, min(10.0, right_speed))

        return (left_speed, right_speed)

    def get_action(self, obs):
        """
        返回外部控制动作列表（与 RobotEnvironment.step(actions) 兼容）：
        - obs 为 dict：返回 [(left_speed, right_speed)]（单机器人也返回长度为1的列表）
        - obs 为 list[dict]：返回 [(l0,r0), (l1,r1), ...]，长度与机器人数量一致
        """
        if isinstance(obs, list):
            return [self._action_for_one(i, obs_i) for i, obs_i in enumerate(obs)]
        return [self._action_for_one(0, obs)]


# Use the unified D* Lite implementation shared with GOAPTeamController.
# The legacy in-file implementation above is intentionally unused.
_DStarLitePlanner = UnifiedDStarLitePlanner