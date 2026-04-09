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

    def __init__(self, canvas_width=1000, canvas_height=1000, num_bots=1, num_dirt=300, seed=None,noise_config=None):
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
        self._reset_metrics()
        self.noise_config = {
            'sensor_pos_std': 0.0,  # 位置感知的标准差 (像素)
            'sensor_angle_std': 0.0,  # 角度感知的标准差 (弧度)
            'sensor_light_std': 0.0,  # 光感传感器的标准差
            'actuator_slip_std': 0.0,  # 移动打滑的标准差 (比例)
            'actuator_turn_std': 0.0  # 转向误差的标准差 (弧度)
        }
        # 如果传入了自定义配置，则更新默认值
        if noise_config is not None:
            self.noise_config.update(noise_config)



    def _reset_metrics(self):
            """初始化或重置实验数据计数器"""
            self.total_steps = 0  # 时间（步数）
            self.total_distance = 0.0  # 路径长度
            self.collision_count = 0  # 碰撞次数
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


    def _on_click(self, event):
        """Teleport all robots to the clicked position (for debugging)."""
        for rr in self.agents:
            rr.x = event.x
            rr.y = event.y

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation (list of robot states)
        """
        self.canvas.delete("all")
        self.agents = []
        self.passive_objects = []
        self._create_objects()
        self.cleaned_cells = set()
        self._prev_coverage = 0.0
        self._reset_metrics()
        return self._get_observation()

    def step(self, actions):
        """
        Advance the environment by one time step.

        :param actions: Ignored here; robots use their internal brains.
        :return: (observation, reward, done, info)
        """
        if not hasattr(self, 'total_steps'):
            self._reset_metrics()  # 如果还没初始化指标，先初始化

        self.total_steps += 1
        # Let each robot decide and act
        for i, rr in enumerate(self.agents):
            # 记录移动前的位置
            old_x, old_y = rr.x, rr.y
            old_theta = rr.theta
            rr.thinkAndAct(self.agents, self.passive_objects)
            rr.update(self.canvas, self.passive_objects, 1.0)

            intended_dx = rr.x - old_x
            intended_dy = rr.y - old_y
            step_dist = math.sqrt(intended_dx ** 2 + intended_dy ** 2)
            if step_dist > 0.001:
                # 打滑误差与移动距离成正比
                slip_x = np.random.normal(0, self.noise_config['actuator_slip_std'] * step_dist)
                slip_y = np.random.normal(0, self.noise_config['actuator_slip_std'] * step_dist)
                rr.x += slip_x
                rr.y += slip_y

                # 只有当机器人发生转向时，才会产生转向误差
                dtheta = rr.theta - old_theta
                if abs(dtheta) > 0.001:
                    turn_error = np.random.normal(0, self.noise_config['actuator_turn_std'])
                    rr.theta += turn_error
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
                        # 物理反弹处理（保持现状）
                        rr.x, rr.y = old_x, old_y
                        rr.theta += random.uniform(math.pi / 2, math.pi)
                        break

            if colliding_now and not self.bot_collision_state[i]:
                self.collision_count += 1
            self.bot_collision_state[i] = colliding_now
        # Update canvas
        self.canvas.update()
        self.canvas.after(50)
        # 获取最新的指标并计算 reward
        metrics = self.get_metrics()
        reward = metrics['coverage'] - self._prev_coverage
        self._prev_coverage = metrics['coverage']

        done = metrics['success'] or (self.total_steps >= 2000)

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
            real_lightL, real_lightR = rr.senseLight(self.passive_objects)

            # 1. 注入光感噪声 (使用 max 确保光线值不为负)
            noisy_lightL = max(0.0, np.random.normal(real_lightL, self.noise_config['sensor_light_std']))
            noisy_lightR = max(0.0, np.random.normal(real_lightR, self.noise_config['sensor_light_std']))

            # 2. 注入里程计/定位噪声
            noisy_x = np.random.normal(rr.x, self.noise_config['sensor_pos_std'])
            noisy_y = np.random.normal(rr.y, self.noise_config['sensor_pos_std'])
            noisy_theta = np.random.normal(rr.theta, self.noise_config['sensor_angle_std'])
            obs.append({
                'x': rr.x,
                'y': rr.y,
                'theta': rr.theta,
                'battery': rr.battery,
                'sensor_left': noisy_lightL,
                'sensor_right': noisy_lightR,
                'real_x': rr.x,
                'real_y': rr.y,
                'real_theta': rr.theta
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

    def render(self):
        """Render the current state (already updated in step)."""
        pass

    def close(self):
        """Close the Tkinter window."""
        self.window.destroy()