from env.astar import AStarPlanner

# 简单栅格地图 (5x5)
grid = [
    [0,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
]
planner = AStarPlanner(grid, resolution=1.0)
path = planner.plan((0.5, 0.5), (4.5, 4.5))
print("Path:", path)