# env/dstarlite.py
# D* Lite for dynamic environments.
# Based on Koenig & Likhachev 2002.

import heapq
import math

class DStarLite:
    def __init__(self, grid, resolution=1.0):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.resolution = resolution
        self.km = 0
        self.rhs = {}
        self.g = {}
        self.open_set = {}

    def initialize(self, start, goal):
        self.start = self._world_to_grid(start)
        self.goal = self._world_to_grid(goal)
        self.rhs[self.goal] = 0
        self.g[self.goal] = float('inf')
        self._insert(self.goal, self._calculate_key(self.goal))

    def _calculate_key(self, s):
        return (min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))) + self.km,
                min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))))

    def _insert(self, s, key):
        heapq.heappush(self.open_set, (key, s))

    def _update_vertex(self, s):
        if s != self.goal:
            min_rhs = float('inf')
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = s[0]+dx, s[1]+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cost = math.hypot(dx, dy)
                    if self.grid[ny][nx] == 0:  # free
                        min_rhs = min(min_rhs, self.g.get((nx, ny), float('inf')) + cost)
            self.rhs[s] = min_rhs
        # Remove from open set if present
        # (simplified: we'll just push new entry; duplicates handled by comparing keys)
        key = self._calculate_key(s)
        heapq.heappush(self.open_set, (key, s))

    def compute_shortest_path(self):
        while self.open_set:
            k_old, s = heapq.heappop(self.open_set)
            if k_old < self._calculate_key(s):
                self._insert(s, self._calculate_key(s))
            elif self.g.get(s, float('inf')) > self.rhs.get(s, float('inf')):
                self.g[s] = self.rhs[s]
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    nx, ny = s[0]+dx, s[1]+dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self._update_vertex((nx, ny))
            else:
                self.g[s] = float('inf')
                self._update_vertex(s)
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    nx, ny = s[0]+dx, s[1]+dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self._update_vertex((nx, ny))

    def replan(self, changed_cells):
        # changed_cells: list of (x, y) grid cells that have changed (obstacle added/removed)
        self.km += self._heuristic(self.start, self.goal)
        # Update grid and rhs for affected cells (simplified: just update neighbors)
        for (x,y) in changed_cells:
            self.grid[y][x] = 1  # assume obstacle added; for removal set 0
            self._update_vertex((x,y))
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self._update_vertex((nx, ny))
        self.compute_shortest_path()
        return self._get_path()

    def _heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _get_path(self):
        # Extract path from start to goal following decreasing g values
        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = []
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = current[0]+dx, current[1]+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in self.g:
                        neighbors.append((nx, ny))
            if not neighbors:
                break
            best = min(neighbors, key=lambda n: self.g.get(n, float('inf')))
            if self.g.get(best, float('inf')) >= self.g.get(current, float('inf')):
                break
            path.append(best)
            current = best
        return [self._grid_to_world(p) for p in path]

    def _world_to_grid(self, pos):
        x, y = pos
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return (gx, gy)

    def _grid_to_world(self, grid_pos):
        gx, gy = grid_pos
        return (gx * self.resolution + self.resolution/2,
                gy * self.resolution + self.resolution/2)