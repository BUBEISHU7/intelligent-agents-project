import math
import random
import tkinter as tk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .simpleBot2 import Bot, Brain, Charger, Dirt, Obstacle, WiFiHub


class DynamicObstacle:
    """Simple moving obstacle with bouncing boundary behavior."""

    def __init__(self, name: str, x: float, y: float, vx: float, vy: float, radius: float = 18.0):
        self.name = name
        self.centreX = float(x)
        self.centreY = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.radius = float(radius)

    def step(self, width: float, height: float, dt: float) -> None:
        self.centreX += self.vx * dt
        self.centreY += self.vy * dt

        if self.centreX < self.radius or self.centreX > width - self.radius:
            self.vx *= -1.0
            self.centreX = min(max(self.radius, self.centreX), width - self.radius)
        if self.centreY < self.radius or self.centreY > height - self.radius:
            self.vy *= -1.0
            self.centreY = min(max(self.radius, self.centreY), height - self.radius)

    def draw(self, canvas: tk.Canvas) -> None:
        canvas.create_oval(
            self.centreX - self.radius,
            self.centreY - self.radius,
            self.centreX + self.radius,
            self.centreY + self.radius,
            fill="orange",
            tags=self.name,
        )
        canvas.create_text(self.centreX, self.centreY, text="M", fill="black", tags=self.name)

    def getLocation(self) -> Tuple[float, float]:
        return self.centreX, self.centreY

    def getRadius(self) -> float:
        return self.radius


class RobotEnvironment:
    """Environment with dynamic obstacles, noisy sensing, and shared-map support."""

    def __init__(
        self,
        canvas_width: int = 1000,
        canvas_height: int = 1000,
        num_bots: int = 1,
        num_dirt: int = 300,
        seed: Optional[int] = None,
        num_obstacles: int = 5,
        num_dynamic_obstacles: int = 0,
        dynamic_obstacle_speed: float = 4.0,
        grid_resolution: int = 20,
        max_steps: int = 2000,
        render: bool = True,
        noise_config: Optional[Dict[str, float]] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = int(canvas_width)
        self.height = int(canvas_height)
        self.num_bots = int(num_bots)
        self.num_dirt = int(num_dirt)
        self.initial_num_dirt = int(num_dirt)
        self.num_obstacles = int(num_obstacles)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.dynamic_obstacle_speed = float(dynamic_obstacle_speed)
        self.grid_resolution = int(grid_resolution)
        self.max_steps = int(max_steps)
        self.render_enabled = bool(render)

        self.noise_config = {
            "sensor_gaussian_std": 0.0,
            "sensor_bias_x": 0.0,
            "sensor_bias_y": 0.0,
            "sensor_miss_rate": 0.0,
            "execution_noise_std": 0.0,
        }
        if noise_config:
            self.noise_config.update(noise_config)
        self.robot_collision_radius = 26.0
        self.static_collision_margin = 8.0
        self.dynamic_collision_margin = 4.0
        self.dynamic_collision_cooldown_steps = 5

        self.window = tk.Tk()
        self.window.resizable(False, False)
        if not self.render_enabled:
            self.window.withdraw()
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        self.agents: List[Bot] = []
        self.passive_objects: List[object] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.shared_map: Dict[str, object] = {}
        self._last_dynamic_grid: set = set()
        self._changed_cells: List[Tuple[int, int, int]] = []
        self.cell_size = self.grid_resolution
        self.cleaned_cells = set()
        self._prev_coverage = 0.0

        self._create_objects()
        self.canvas.bind("<Button-1>", self._on_click)
        self._reset_metrics()
        self._rebuild_shared_map()

    def _reset_metrics(self) -> None:
        self.total_steps = 0
        self.total_distance = 0.0
        self.collision_count = 0
        self.dynamic_collision_count = 0
        self.robot_collision_count = 0
        self.bot_collision_state = [False for _ in self.agents]
        self.dynamic_collision_cooldown = [0 for _ in self.agents]

    def _random_free_pose(self, margin: float = 80.0) -> Tuple[float, float]:
        for _ in range(200):
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            if all(math.hypot(x - bot.x, y - bot.y) > 80.0 for bot in self.agents):
                return x, y
        return self.width / 2.0, self.height / 2.0

    def _create_objects(self) -> None:
        for i in range(self.num_bots):
            bot = Bot(f"Bot{i}")
            bot.x, bot.y = self._random_free_pose()
            brain = Brain(bot)
            bot.setBrain(brain)
            self.agents.append(bot)
            bot.draw(self.canvas)

        charger = Charger("Charger")
        charger.centreX = self.width / 2.0
        charger.centreY = self.height / 2.0
        self.passive_objects.append(charger)
        charger.draw(self.canvas)

        hub1 = WiFiHub("Hub1", self.width - 50, 50)
        hub2 = WiFiHub("Hub2", 50, self.height / 2.0)
        self.passive_objects.extend([hub1, hub2])
        hub1.draw(self.canvas)
        hub2.draw(self.canvas)

        for i in range(self.num_obstacles):
            ox, oy = self._random_free_pose(margin=120.0)
            obstacle = Obstacle(f"Obstacle{i}", ox, oy)
            self.passive_objects.append(obstacle)
            obstacle.draw(self.canvas)

        for i in range(self.num_dynamic_obstacles):
            ox, oy = self._random_free_pose(margin=150.0)
            angle = random.uniform(0.0, 2.0 * math.pi)
            vx = self.dynamic_obstacle_speed * math.cos(angle)
            vy = self.dynamic_obstacle_speed * math.sin(angle)
            obstacle = DynamicObstacle(f"DynamicObstacle{i}", ox, oy, vx, vy)
            self.dynamic_obstacles.append(obstacle)
            obstacle.draw(self.canvas)

        for i in range(self.num_dirt):
            dirt = Dirt(f"Dirt{i}")
            self.passive_objects.append(dirt)
            dirt.draw(self.canvas)

    def _on_click(self, event) -> None:
        for rr in self.agents:
            rr.x = event.x
            rr.y = event.y

    def reset(self):
        self.canvas.delete("all")
        self.agents = []
        self.passive_objects = []
        self.dynamic_obstacles = []
        self.cleaned_cells = set()
        self._prev_coverage = 0.0
        self._changed_cells = []
        self._last_dynamic_grid = set()
        self._create_objects()
        self.initial_num_dirt = len([d for d in self.passive_objects if isinstance(d, Dirt)])
        self._reset_metrics()
        self._rebuild_shared_map()
        return self._get_observation()

    def world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        gx = int(pos[0] // self.grid_resolution)
        gy = int(pos[1] // self.grid_resolution)
        w = max(1, self.width // self.grid_resolution)
        h = max(1, self.height // self.grid_resolution)
        return min(max(0, gx), w - 1), min(max(0, gy), h - 1)

    def grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return (
            cell[0] * self.grid_resolution + self.grid_resolution / 2.0,
            cell[1] * self.grid_resolution + self.grid_resolution / 2.0,
        )

    def _mark_disk(self, cells: set, x: float, y: float, radius: float) -> None:
        gx, gy = self.world_to_grid((x, y))
        r = max(1, int(math.ceil(radius / self.grid_resolution)))
        w = max(1, self.width // self.grid_resolution)
        h = max(1, self.height // self.grid_resolution)
        for yy in range(max(0, gy - r), min(h, gy + r + 1)):
            for xx in range(max(0, gx - r), min(w, gx + r + 1)):
                wx, wy = self.grid_to_world((xx, yy))
                if math.hypot(wx - x, wy - y) <= radius + self.grid_resolution * 0.75:
                    cells.add((xx, yy))

    def get_grid_map(self, include_dynamic: bool = True) -> List[List[int]]:
        w = max(1, self.width // self.grid_resolution)
        h = max(1, self.height // self.grid_resolution)
        occupied = set()

        for obj in self.passive_objects:
            if isinstance(obj, Obstacle):
                self._mark_disk(occupied, obj.centreX, obj.centreY, getattr(obj, "radius", 18.0) + 12.0)

        if include_dynamic:
            for obj in self.dynamic_obstacles:
                self._mark_disk(occupied, obj.centreX, obj.centreY, obj.radius + 14.0)

        grid = [[0 for _ in range(w)] for _ in range(h)]
        for x, y in occupied:
            grid[y][x] = 1
        return grid

    def get_changed_cells(self) -> List[Tuple[int, int, int]]:
        return list(self._changed_cells)

    def _dynamic_grid_cells(self) -> set:
        occupied = set()
        for obj in self.dynamic_obstacles:
            self._mark_disk(occupied, obj.centreX, obj.centreY, obj.radius + 14.0)
        return occupied

    def _update_dynamic_obstacles(self, dt: float) -> None:
        for obj in self.dynamic_obstacles:
            self.canvas.delete(obj.name)
            obj.step(self.width, self.height, dt)
            obj.draw(self.canvas)

        current = self._dynamic_grid_cells()
        changed = []
        for cell in current - self._last_dynamic_grid:
            changed.append((cell[0], cell[1], 1))
        for cell in self._last_dynamic_grid - current:
            changed.append((cell[0], cell[1], 0))
        self._changed_cells = changed
        self._last_dynamic_grid = current

    def _apply_execution_noise(self, action: Tuple[float, float]) -> Tuple[float, float]:
        std = float(self.noise_config.get("execution_noise_std", 0.0))
        left, right = action
        if std > 0.0:
            left += float(np.random.normal(0.0, std))
            right += float(np.random.normal(0.0, std))
        return max(-10.0, min(10.0, left)), max(-10.0, min(10.0, right))

    def _resolve_robot_collision(self, i: int, old_pos: Tuple[float, float]) -> bool:
        rr = self.agents[i]
        collided = False
        radius = self.robot_collision_radius

        for obj in self.passive_objects:
            if not isinstance(obj, Obstacle):
                continue
            obj_radius = getattr(obj, "radius", 15.0) + radius + self.static_collision_margin
            dist = math.hypot(rr.x - obj.centreX, rr.y - obj.centreY)
            if dist < obj_radius:
                if dist < 1e-6:
                    nx, ny = math.cos(rr.theta), math.sin(rr.theta)
                else:
                    nx, ny = (rr.x - obj.centreX) / dist, (rr.y - obj.centreY) / dist
                # slide out instead of hard rollback to reduce oscillation
                rr.x = obj.centreX + nx * obj_radius
                rr.y = obj.centreY + ny * obj_radius
                rr.theta = (rr.theta + random.uniform(math.pi / 2.0, math.pi)) % (2.0 * math.pi)
                collided = True
                break

        for obj in self.dynamic_obstacles:
            obj_radius = obj.radius + radius + self.dynamic_collision_margin
            dist = math.hypot(rr.x - obj.centreX, rr.y - obj.centreY)
            if dist < obj_radius:
                if dist < 1e-6:
                    nx, ny = math.cos(rr.theta), math.sin(rr.theta)
                else:
                    nx, ny = (rr.x - obj.centreX) / dist, (rr.y - obj.centreY) / dist
                rr.x = obj.centreX + nx * obj_radius
                rr.y = obj.centreY + ny * obj_radius
                rr.theta = (rr.theta + random.uniform(math.pi / 2.0, math.pi)) % (2.0 * math.pi)
                collided = True
                if self.dynamic_collision_cooldown[i] <= 0:
                    self.dynamic_collision_count += 1
                    self.dynamic_collision_cooldown[i] = self.dynamic_collision_cooldown_steps
                break

        rr.x = min(max(radius, rr.x), self.width - radius)
        rr.y = min(max(radius, rr.y), self.height - radius)
        return collided

    def _resolve_inter_robot_collisions(self) -> None:
        radius = self.robot_collision_radius + 4.0
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                a = self.agents[i]
                b = self.agents[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy)
                if dist == 0.0:
                    dx, dy, dist = 1.0, 0.0, 1.0
                if dist < 2.0 * radius:
                    overlap = 2.0 * radius - dist
                    nx, ny = dx / dist, dy / dist
                    a.x -= nx * overlap / 2.0
                    a.y -= ny * overlap / 2.0
                    b.x += nx * overlap / 2.0
                    b.y += ny * overlap / 2.0
                    a.theta = (a.theta + math.pi / 3.0) % (2.0 * math.pi)
                    b.theta = (b.theta - math.pi / 3.0) % (2.0 * math.pi)
                    self.collision_count += 1
                    self.robot_collision_count += 1
                    a.draw(self.canvas)
                    b.draw(self.canvas)

    def _update_shared_map(self, robot_idx: int, sensed: Dict[str, object]) -> None:
        obstacle_cells = set()
        dynamic_cells = set()
        dirt_cells = set()

        for item in sensed["obstacles"]:
            obstacle_cells.add(self.world_to_grid((item["x"], item["y"])))
        for item in sensed["dynamic_obstacles"]:
            dynamic_cells.add(self.world_to_grid((item["x"], item["y"])))
        for item in sensed["dirt"]:
            dirt_cells.add(self.world_to_grid((item["x"], item["y"])))

        self.shared_map["obstacles"].update(obstacle_cells)
        self.shared_map["dynamic_obstacles"].update(dynamic_cells)
        self.shared_map["dirt"].update(dirt_cells)
        self.shared_map["robot_updates"][robot_idx] = {
            "position": (self.agents[robot_idx].x, self.agents[robot_idx].y),
            "obstacles": sorted(obstacle_cells),
            "dynamic_obstacles": sorted(dynamic_cells),
            "dirt": sorted(dirt_cells),
            "step": self.total_steps,
        }

    def _rebuild_shared_map(self) -> None:
        self.shared_map = {
            "obstacles": set(),
            "dynamic_obstacles": set(),
            "dirt": set(),
            "robot_updates": {},
            "grid_resolution": self.grid_resolution,
        }

    def _noisy_detection(self, obj, miss_rate: float, std: float, bias_x: float, bias_y: float):
        if random.random() < miss_rate:
            return None
        x = float(obj.centreX) + bias_x + float(np.random.normal(0.0, std))
        y = float(obj.centreY) + bias_y + float(np.random.normal(0.0, std))
        return {
            "id": obj.name,
            "x": max(0.0, min(float(self.width), x)),
            "y": max(0.0, min(float(self.height), y)),
            "radius": float(getattr(obj, "radius", 2.0)),
        }

    def _sense_robot_environment(self, robot: Bot) -> Dict[str, object]:
        std = float(self.noise_config.get("sensor_gaussian_std", 0.0))
        bias_x = float(self.noise_config.get("sensor_bias_x", 0.0))
        bias_y = float(self.noise_config.get("sensor_bias_y", 0.0))
        miss_rate = float(self.noise_config.get("sensor_miss_rate", 0.0))
        sensor_range = 240.0

        dirt_obs = []
        obstacle_obs = []
        dynamic_obs = []
        charger_obs = []
        nearby_robots = []

        for obj in self.passive_objects:
            if not hasattr(obj, "centreX"):
                continue
            if math.hypot(robot.x - obj.centreX, robot.y - obj.centreY) > sensor_range:
                continue
            detection = self._noisy_detection(obj, miss_rate, std, bias_x, bias_y)
            if detection is None:
                continue
            if isinstance(obj, Dirt):
                dirt_obs.append(detection)
            elif isinstance(obj, Charger):
                charger_obs.append(detection)
            elif isinstance(obj, Obstacle):
                obstacle_obs.append(detection)

        for obj in self.dynamic_obstacles:
            if math.hypot(robot.x - obj.centreX, robot.y - obj.centreY) > sensor_range:
                continue
            detection = self._noisy_detection(obj, miss_rate, std, bias_x, bias_y)
            if detection is not None:
                detection["vx"] = obj.vx
                detection["vy"] = obj.vy
                dynamic_obs.append(detection)

        for other in self.agents:
            if other is robot:
                continue
            if math.hypot(robot.x - other.x, robot.y - other.y) <= sensor_range:
                nearby_robots.append(
                    {
                        "id": other.name,
                        "x": float(other.x),
                        "y": float(other.y),
                        "theta": float(other.theta),
                        "battery": float(other.battery),
                    }
                )

        return {
            "dirt": dirt_obs,
            "obstacles": obstacle_obs,
            "dynamic_obstacles": dynamic_obs,
            "chargers": charger_obs,
            "nearby_robots": nearby_robots,
        }

    def step(self, actions: Optional[Sequence[Tuple[float, float]]] = None):
        if not hasattr(self, "total_steps"):
            self._reset_metrics()

        self.total_steps += 1
        for i in range(len(self.dynamic_collision_cooldown)):
            self.dynamic_collision_cooldown[i] = max(0, self.dynamic_collision_cooldown[i] - 1)
        self._update_dynamic_obstacles(1.0)

        use_external = actions is not None and len(actions) == len(self.agents)
        step_collisions = 0

        for i, rr in enumerate(self.agents):
            old_pos = (rr.x, rr.y)

            if use_external:
                rr.sl, rr.sr = self._apply_execution_noise(actions[i])
            else:
                rr.thinkAndAct(self.agents, self.passive_objects)
                rr.sl, rr.sr = self._apply_execution_noise((rr.sl, rr.sr))

            rr.update(self.canvas, self.passive_objects, 1.0)
            self.passive_objects = rr.collectDirt(self.canvas, self.passive_objects)

            step_dist = math.hypot(rr.x - old_pos[0], rr.y - old_pos[1])
            self.total_distance += step_dist

            collided = self._resolve_robot_collision(i, old_pos)
            if collided:
                step_collisions += 1
            self.bot_collision_state[i] = collided
            rr.draw(self.canvas)

        self._resolve_inter_robot_collisions()
        self.collision_count += step_collisions
        self._rebuild_shared_map()

        obs = self._get_observation()
        metrics = self.get_metrics()
        reward = metrics["coverage"] - self._prev_coverage
        reward -= 0.01 * step_collisions
        self._prev_coverage = metrics["coverage"]

        if self.render_enabled:
            self.canvas.update()
            self.canvas.after(20)
        else:
            self.window.update_idletasks()

        done = metrics["success"] or (self.total_steps >= self.max_steps)
        return obs, reward, done, metrics

    def _get_observation(self):
        obs = []
        for idx, rr in enumerate(self.agents):
            lightL, lightR = rr.senseLight(self.passive_objects)
            sensed = self._sense_robot_environment(rr)
            self._update_shared_map(idx, sensed)
            obs.append(
                {
                    "robot_id": rr.name,
                    "x": float(rr.x),
                    "y": float(rr.y),
                    "theta": float(rr.theta),
                    "battery": float(rr.battery),
                    "sensor_left": float(lightL),
                    "sensor_right": float(lightR),
                    "detected_dirt": sensed["dirt"],
                    "detected_obstacles": sensed["obstacles"],
                    "detected_dynamic_obstacles": sensed["dynamic_obstacles"],
                    "detected_chargers": sensed["chargers"],
                    "nearby_robots": sensed["nearby_robots"],
                    "shared_map": {
                        "obstacles": sorted(self.shared_map["obstacles"]),
                        "dynamic_obstacles": sorted(self.shared_map["dynamic_obstacles"]),
                        "dirt": sorted(self.shared_map["dirt"]),
                        "grid_resolution": self.grid_resolution,
                    },
                    "grid_resolution": self.grid_resolution,
                }
            )
        return obs

    def _compute_coverage(self) -> float:
        for rr in self.agents:
            self.cleaned_cells.add(self.world_to_grid((rr.x, rr.y)))
        total_cells = max(1, (self.width // self.cell_size) * (self.height // self.cell_size))
        return len(self.cleaned_cells) / total_cells

    def get_metrics(self) -> Dict[str, float]:
        remaining_dirt = len([d for d in self.passive_objects if isinstance(d, Dirt)])
        exploration_coverage = self._compute_coverage()
        if self.initial_num_dirt > 0:
            task_coverage = (self.initial_num_dirt - remaining_dirt) / float(self.initial_num_dirt)
        else:
            task_coverage = 1.0
        success = remaining_dirt == 0
        path_efficiency = 1000.0 * task_coverage / (self.total_distance + 1e-6)
        return {
            "success": int(success),
            "coverage": round(task_coverage, 4),
            "exploration_coverage": round(exploration_coverage, 4),
            "steps": self.total_steps,
            "collisions": self.collision_count,
            "dynamic_collisions": self.dynamic_collision_count,
            "robot_collisions": self.robot_collision_count,
            "total_distance": round(self.total_distance, 2),
            "path_efficiency": round(path_efficiency, 4),
            "remaining_dirt": remaining_dirt,
            "performance_score": self.compute_performance_score(task_coverage, success),
        }

    def get_dirt_positions(self):
        """
        Compatibility helper for blackboard-style experiments.
        Returns list of (dirt_name, x, y).
        """
        out = []
        for obj in self.passive_objects:
            if isinstance(obj, Dirt):
                out.append((obj.name, float(obj.centreX), float(obj.centreY)))
        return out

    def compute_performance_score(self, coverage: float, success: bool) -> float:
        success_bonus = 1.0 if success else 0.0
        score = (
            100.0 * coverage
            + 20.0 * success_bonus
            - 0.01 * self.total_steps
            - 2.0 * self.collision_count
            - 0.001 * self.total_distance
        )
        return round(score, 3)

    def render(self):
        if self.render_enabled:
            self.canvas.update()

    def close(self):
        self.window.destroy()