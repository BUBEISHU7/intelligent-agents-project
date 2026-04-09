# robot_env.py
# Encapsulation of the course simpleBot2.py, providing a unified step/reset/render interface.

import tkinter as tk
import random
import math
import numpy as np

# Import required classes from simpleBot2 (relative import from current package)

from .simpleBot2 import Bot, Brain, Lamp, Charger, WiFiHub, Dirt, Obstacle

class RobotEnvironment:
    """Robot environment that encapsulates the simulation and provides a standard interface."""

    def __init__(self, canvas_width=1000, canvas_height=1000, num_bots=1, num_dirt=300, seed=None):
        """
        Initialize the environment.

        :param canvas_width: Width of the canvas (simulation area)
        :param canvas_height: Height of the canvas
        :param num_bots: Number of robots
        :param num_dirt: Number of dirt particles
        :param seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = canvas_width
        self.height = canvas_height
        self.num_bots = num_bots
        self.num_dirt = num_dirt

        # Create Tkinter window and canvas
        self.window = tk.Tk()
        self.window.resizable(False, False)
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        # Track scheduled Tk callbacks so we can cancel them on close/reset.
        self._after_id = None

        # Lists to hold agents and passive objects
        self.agents = []
        self.passive_objects = []

        self._create_objects()

        # Bind mouse click for teleportation (debugging)
        self.canvas.bind("<Button-1>", self._on_click)

        # For coverage tracking
        self.cleaned_cells = set()
        self.cell_size = 20          # grid cell size for coverage estimation
        self._prev_coverage = 0.0    # previous coverage to compute reward
        self.no_progress_steps = 0   # 连续无覆盖提升的步数
        self.no_progress_limit = 150 # 超过该阈值则提前终止，避免空转
        self._reset_metrics()

    def _reset_metrics(self):
            """初始化或重置实验数据计数器"""
            self.total_steps = 0  # 时间（步数）
            self.total_distance = 0.0  # 路径长度
            self.collision_count = 0  # 碰撞次数
            self.robot_overlap_corrections = 0  # 仅用于可视化平滑：机器人-机器人最小间距修正次数
            self.bot_collision_state = [False for _ in self.agents]# 碰撞状态追踪，防止同一次碰撞重复计数
            # 记录每个机器人的上一个位置，用于计算位移
            self.prev_bot_pos = []
            for bot in self.agents:
                self.prev_bot_pos.append((bot.x, bot.y))

    def _create_objects(self):
        """Create robots, chargers, WiFi hubs,obstacles and dirt."""



        # Create robots
        for i in range(self.num_bots):
            bot = Bot(f"Bot{i}")
            brain = Brain(bot)       # use the built-in Brain from simpleBot2
            bot.setBrain(brain)
            self.agents.append(bot)
            bot.draw(self.canvas)

        # Create a charger
        charger = Charger("Charger")
        self.passive_objects.append(charger)
        charger.draw(self.canvas)

        # Create two WiFi hubs (fixed positions)
        hub1 = WiFiHub("Hub1", 950, 50)
        self.passive_objects.append(hub1)
        hub1.draw(self.canvas)

        hub2 = WiFiHub("Hub2", 50, 500)
        self.passive_objects.append(hub2)
        hub2.draw(self.canvas)

        # Create dirt particles
        for i in range(self.num_dirt):
            dirt = Dirt(f"Dirt{i}")
            self.passive_objects.append(dirt)
            dirt.draw(self.canvas)

            # ========== 新增：创建随机障碍物 ==========
        import random
        num_obstacles = 5  # 可以改成参数
        for i in range(num_obstacles):
            # 避免障碍物重叠在充电桩和WiFi位置上
            while True:
                ox = random.randint(50, 950)
                oy = random.randint(50, 950)
                # 检查是否与充电桩重叠
                charger_pos = (self.passive_objects[0].centreX, self.passive_objects[0].centreY)
                hub1_pos = (950, 50)
                hub2_pos = (50, 500)
                if (abs(ox - charger_pos[0]) > 50 and abs(oy - charger_pos[1]) > 50 and
                        abs(ox - hub1_pos[0]) > 50 and abs(oy - hub1_pos[1]) > 50 and
                        abs(ox - hub2_pos[0]) > 50 and abs(oy - hub2_pos[1]) > 50):
                    break
            obstacle = Obstacle(f"Obstacle{i}", ox, oy)
            self.passive_objects.append(obstacle)
            obstacle.draw(self.canvas)

        self._create_step_hud()

    def _create_step_hud(self):
        """左上角显示当前仿真步数（与 metrics['steps'] 一致）。"""
        self._step_hud_bg = self.canvas.create_rectangle(
            4, 4, 152, 44, fill="white", outline="#999999", width=1, tags="step_hud"
        )
        self._step_hud_id = self.canvas.create_text(
            12, 24, anchor="w", text="步数: 0",
            fill="#003366", font=("Segoe UI", 12, "bold"), tags="step_hud"
        )

    def _update_step_hud(self):
        if getattr(self, "_step_hud_id", None) is None:
            return
        self.canvas.itemconfig(self._step_hud_id, text=f"步数: {self.total_steps}")
        self.canvas.tag_raise("step_hud")

    def _on_click(self, event):
        """Teleport all robots to the clicked position (for debugging)."""
        for rr in self.agents:
            rr.x = event.x
            rr.y = event.y

    def _resolve_robot_robot_overlap(self, min_dist=55.0):
        """
        Keep robots separated to reduce visual collisions.
        Returns number of overlap corrections applied in this step.
        """
        corrections = 0
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a = self.agents[i]
                b = self.agents[j]
                dx = a.x - b.x
                dy = a.y - b.y
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    corrections += 1
                    if dist < 1e-6:
                        # Perfect overlap: push in random opposite directions.
                        ang = random.uniform(0.0, 2.0 * math.pi)
                        ux, uy = math.cos(ang), math.sin(ang)
                    else:
                        ux, uy = dx / dist, dy / dist
                    push = 0.5 * (min_dist - max(dist, 1e-6))
                    a.x += push * ux
                    a.y += push * uy
                    b.x -= push * ux
                    b.y -= push * uy
                    a.x = max(20, min(self.width - 20, a.x))
                    a.y = max(20, min(self.height - 20, a.y))
                    b.x = max(20, min(self.width - 20, b.x))
                    b.y = max(20, min(self.height - 20, b.y))
                    a.theta += random.uniform(-0.5, 0.5)
                    b.theta += random.uniform(-0.5, 0.5)
        return corrections

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation (list of robot states)
        """
        self.canvas.delete("all")
        if getattr(self, "_after_id", None) is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.agents = []
        self.passive_objects = []
        self._create_objects()
        self.cleaned_cells = set()
        self._prev_coverage = 0.0
        self.no_progress_steps = 0
        self._reset_metrics()
        return self._get_observation()

    def _resolve_action(self, actions, idx, bot):
        """
        Resolve external action for one robot.

        Supported formats:
        - None: use robot's internal brain
        - list/tuple: [(sl, sr), ...]
        - dict: {idx: (sl, sr)}
        - callable: fn(idx, bot, agents, passive_objects) -> (sl, sr) or None
        """
        if actions is None:
            return None
        if callable(actions):
            return actions(idx, bot, self.agents, self.passive_objects)
        if isinstance(actions, (list, tuple)):
            if idx < len(actions):
                return actions[idx]
            return None
        if isinstance(actions, dict):
            return actions.get(idx)
        return None

    def step(self, actions=None):
        """
        Advance the environment by one time step.

        :param actions: Optional external control for multi-agent experiments.
        :return: (observation, reward, done, info)
        """
        if not hasattr(self, 'total_steps'):
            self._reset_metrics()  # 如果还没初始化指标，先初始化

        self.total_steps += 1
        # Let each robot decide and act
        for i, rr in enumerate(self.agents):
            # 记录移动前的位置
            old_x, old_y = rr.x, rr.y
            external_action = self._resolve_action(actions, i, rr)
            if external_action is None:
                rr.thinkAndAct(self.agents, self.passive_objects)
            else:
                rr.sl, rr.sr = float(external_action[0]), float(external_action[1])
            rr.update(self.canvas, self.passive_objects, 1.0)
            self.passive_objects = rr.collectDirt(self.canvas, self.passive_objects)
            # 计算位移
            step_dist = math.sqrt((rr.x - old_x) ** 2 + (rr.y - old_y) ** 2)
            self.total_distance += step_dist

            # 只有从“未碰撞”变为“碰撞”时才计数一次
            colliding_now = False
            for obj in self.passive_objects:
                if isinstance(obj, Obstacle):
                    dist_to_obs = math.sqrt((rr.x - obj.centreX) ** 2 + (rr.y - obj.centreY) ** 2)
                    if dist_to_obs < 30:
                        colliding_now = True
                        # Push robot away from obstacle center to avoid
                        # getting stuck in repeated "touch-turn-touch" loops.
                        dx = rr.x - obj.centreX
                        dy = rr.y - obj.centreY
                        norm = math.hypot(dx, dy)
                        if norm < 1e-6:
                            ang = random.uniform(0.0, 2.0 * math.pi)
                            ux, uy = math.cos(ang), math.sin(ang)
                        else:
                            ux, uy = dx / norm, dy / norm
                        safe_dist = 42.0
                        rr.x = obj.centreX + safe_dist * ux
                        rr.y = obj.centreY + safe_dist * uy
                        rr.x = max(20, min(self.width - 20, rr.x))
                        rr.y = max(20, min(self.height - 20, rr.y))
                        rr.theta = math.atan2(uy, ux) + random.uniform(-0.35, 0.35)
                        rr.sl = max(2.0, rr.sl)
                        rr.sr = max(2.0, rr.sr)
                        break

            if colliding_now and not self.bot_collision_state[i]:
                self.collision_count += 1
            self.bot_collision_state[i] = colliding_now

        # Safety layer for robot-robot near-overlap.
        # We do not add these corrections to `collision_count`, because it's not a physical collision with objects.
        self.robot_overlap_corrections += self._resolve_robot_robot_overlap(min_dist=55.0)
        self._update_step_hud()
        # Update canvas
        self.canvas.update()
        if getattr(self, "_after_id", None) is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.canvas.after(50)
        # 获取最新的指标并计算 reward
        metrics = self.get_metrics()
        reward = metrics['coverage'] - self._prev_coverage
        if reward <= 1e-9:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        self._prev_coverage = metrics['coverage']

        all_battery_empty = all(rr.battery <= 0 for rr in self.agents)
        no_progress_timeout = self.no_progress_steps >= self.no_progress_limit
        done = (
            metrics['success']
            or all_battery_empty
            or no_progress_timeout
            or (self.total_steps >= 1200)
        )

        return self._get_observation(), reward, done, metrics

        #统一的实验结果输出接口

    def get_metrics(self):
        """获取标准化的性能度量字典"""
        remaining_dirt = len([d for d in self.passive_objects if isinstance(d, Dirt)])
        coverage = self._compute_coverage()
        success = (remaining_dirt == 0)

        # 路径效率：单位位移带来的覆盖收益 (乘1000使数值易读)
        path_efficiency = 1000 * coverage / (self.total_distance + 1e-6)

        return {
            "success": int(success),
            "coverage": round(coverage, 4),
            "steps": self.total_steps,
            "collisions": self.collision_count,
            "robot_overlap_corrections": self.robot_overlap_corrections,
            "total_distance": round(self.total_distance, 2),
            "path_efficiency": round(path_efficiency, 4),
            "remaining_dirt": remaining_dirt,
            "performance_score": self.compute_performance_score(coverage, success)
        }

        # 综合性能分数计算

    def compute_performance_score(self, coverage, success):
        """计算综合分数用于鲁棒性对比"""
        success_bonus = 1.0 if success else 0.0
        # 权重分配：覆盖率(100) + 完成奖励(20) - 步数惩罚 - 碰撞惩罚 - 位移惩罚
        score = (
                100 * coverage
                + 20 * success_bonus
                - 0.01 * self.total_steps
                - 2.0 * self.collision_count
                - 0.001 * self.total_distance
        )
        return round(score, 3)

    def _get_observation(self):
        """
        Build the observation for all robots.

        :return: List of dictionaries, each containing robot state.
        """
        obs = []
        for rr in self.agents:
            # Sense light (just for demonstration)
            lightL, lightR = rr.senseLight(self.passive_objects)
            obs.append({
                'x': rr.x,
                'y': rr.y,
                'theta': rr.theta,
                'battery': rr.battery,
                'sensor_left': lightL,
                'sensor_right': lightR
            })
        return obs

    def _compute_coverage(self):
        """
        Compute the coverage ratio based on robot positions.

        :return: Coverage ratio (0 to 1).
        """
        # Mark the grid cell of each robot as visited
        for rr in self.agents:
            cell_x = int(rr.x // self.cell_size)
            cell_y = int(rr.y // self.cell_size)
            self.cleaned_cells.add((cell_x, cell_y))

        total_cells = (self.width // self.cell_size) * (self.height // self.cell_size)
        if total_cells == 0:
            return 0.0
        return len(self.cleaned_cells) / total_cells

    def get_dirt_positions(self):
        """Return current dirt positions as [(name, x, y), ...]."""
        dirt_positions = []
        for obj in self.passive_objects:
            if isinstance(obj, Dirt):
                dirt_positions.append((obj.name, obj.centreX, obj.centreY))
        return dirt_positions

    def render(self):
        """Render the current state (already updated in step)."""
        pass

    def close(self):
        """Close the Tkinter window."""
        if getattr(self, "_after_id", None) is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.window.destroy()