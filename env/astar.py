# env/astar.py
# A* path planning on a grid map.

import heapq
import math

class AStarPlanner:
    def __init__(self, grid, resolution=1.0):
        """
        grid: 2D list, 0 = free, 1 = obstacle
        resolution: grid cell size (meters per cell)
        """
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.resolution = resolution

    def plan(self, start, goal):
        """
        start, goal: (x, y) in world coordinates (continuous)
        returns: list of (x, y) waypoints in world coordinates, or [] if no path
        """
        # Convert world to grid indices
        sx, sy = self._world_to_grid(start)
        gx, gy = self._world_to_grid(goal)

        if not self._is_valid(sx, sy) or not self._is_valid(gx, gy):
            return []

        open_set = []
        heapq.heappush(open_set, (0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0}
        f_score = {(sx, sy): self._heuristic(sx, sy, gx, gy)}

        while open_set:
            _, x, y = heapq.heappop(open_set)
            if (x, y) == (gx, gy):
                path = self._reconstruct_path(came_from, (x, y))
                return [self._grid_to_world(p) for p in path]

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = x + dx, y + dy
                if not self._is_valid(nx, ny):
                    continue
                # Diagonal movement cost = sqrt(2), else 1
                move_cost = math.hypot(dx, dy)
                tentative_g = g_score[(x, y)] + move_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self._heuristic(nx, ny, gx, gy)
                    f_score[(nx, ny)] = f
                    heapq.heappush(open_set, (f, nx, ny))
        return []

    def _heuristic(self, x, y, gx, gy):
        # Euclidean distance
        return math.hypot(x - gx, y - gy)

    def _is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def _world_to_grid(self, pos):
        x, y = pos
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return gx, gy

    def _grid_to_world(self, grid_pos):
        gx, gy = grid_pos
        return (gx * self.resolution + self.resolution/2,
                gy * self.resolution + self.resolution/2)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]