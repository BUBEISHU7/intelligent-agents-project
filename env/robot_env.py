# robot_env.py
# Encapsulation of the course simpleBot2.py, providing a unified step/reset/render interface.

import tkinter as tk
import random
import math
import numpy as np

# Import required classes from simpleBot2 (relative import from current package)
from .simpleBot2 import Bot, Brain, Lamp, Charger, WiFiHub, Dirt

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

    def _create_objects(self):
        """Create robots, chargers, WiFi hubs, and dirt."""
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
        return self._get_observation()

    def step(self, actions):
        """
        Advance the environment by one time step.

        :param actions: Ignored here; robots use their internal brains.
        :return: (observation, reward, done, info)
        """
        # Let each robot decide and act
        for rr in self.agents:
            rr.thinkAndAct(self.agents, self.passive_objects)
            rr.update(self.canvas, self.passive_objects, 1.0)
            self.passive_objects = rr.collectDirt(self.canvas, self.passive_objects)

        # Update canvas
        self.canvas.update()
        self.canvas.after(50)   # slow down for visibility

        # Compute coverage and reward
        coverage = self._compute_coverage()
        reward = coverage - self._prev_coverage
        self._prev_coverage = coverage

        # Check if all dirt is collected
        remaining_dirt = len([d for d in self.passive_objects if isinstance(d, Dirt)])
        done = (remaining_dirt == 0)

        info = {
            'coverage': coverage,
            'remaining_dirt': remaining_dirt
        }
        return self._get_observation(), reward, done, info

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

    def render(self):
        """Render the current state (already updated in step)."""
        pass

    def close(self):
        """Close the Tkinter window."""
        self.window.destroy()