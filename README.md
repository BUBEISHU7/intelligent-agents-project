# Intelligent Agents Project

This repository contains a complete agent-based system for coursework in **Designing Intelligent Agents**.
The project is centered on a **multi-robot cleaning simulation** where autonomous agents plan, coordinate, and adapt under uncertainty.

## System Goal

The overall target is to build and evaluate an intelligent agent system with:

1. **Environment**  
   A robotics simulation with static/dynamic obstacles, charging station, dirt targets, shared map, and configurable sensing/execution noise.

2. **Autonomous Agents**  
   Multi-agent control combining GOAP-style decision making, A*/D* Lite path planning, task auction, and distributed collision avoidance.

3. **Concrete Research Question**  
   Compare algorithmic choices (e.g., A* vs D* Lite) and coordination strategies under dynamic obstacles and noise to evaluate robustness.

4. **Repeatable Experiments**  
   Run factorial experiment matrices across seeds and conditions; aggregate metrics and plots to answer the research question statistically.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Research Workflow](#research-workflow)
- [Code Attribution](#code-attribution)
- [Contributing](#contributing)
- [License](#license)

---

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
Run the following command in the project root:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet available, install the core packages manually:

bash

```
pip install numpy matplotlib
```

> **Note**: The environment uses `tkinter` for graphics, which is usually included with Python on macOS and Linux. On Windows, it comes with the standard Python installation. If you encounter issues, ensure `tkinter` is installed.

## Quick Start

### **1.Clone the repository** (if you haven't already):

```bash 
git clone https://github.com/你的用户名/intelligent-agents-project.git
cd intelligent-agents-project
```

### 2.**Run a simple test** to verify the environment works:

```bash
python test_env.py
```

You should see a window with a blue robot moving around and collecting grey dirt particles. After a few seconds, the terminal will print the final coverage percentage.

### **3.Explore the environment** in your own scripts:

python

```
from env.robot_env import RobotEnvironment

env = RobotEnvironment(num_bots=1, num_dirt=50)
obs = env.reset()
for _ in range(500):
    obs, reward, done, info = env.step()   # robots use their built-in Brain
    if done:
        break
env.close()
```

### 4. Run member C coordination baseline experiment:

```bash
python experiments/member_c_week1_coordination.py
```

Outputs are saved in `data/member_c_week1/` (CSV + JSON summaries and trajectories).

### 5. Run member C week2 blackboard experiment:

```bash
python experiments/member_c_week2_blackboard.py
```

Week2 outputs are saved in `data/member_c_week2/`.

## Project Structure

text

```
intelligent-agents-project/
├── env/                     # Environment code (maintained by member A)
│   ├── __init__.py          # Makes env a Python package
│   ├── simpleBot1.py        # Course-provided basic robot (with lights)
│   ├── simpleBot2.py        # Course-provided robot with battery, charger, dirt
│   └── robot_env.py         # Unified RL-style wrapper (step, reset, render)
├── agents/                  # Agent implementations (other members)
│   └── (to be added)
├── experiments/             # Experiment scripts (member E)
│   └── (to be added)
├── data/                    # Experimental results (gitignored)
├── notebooks/               # Analysis notebooks
├── test_env.py              # Simple test script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Research Workflow

1. Run a preset matrix:

```bash
python experiments/run_batch.py --preset noise_stress
```

2. Analyze one run directory:

```bash
python experiments/analyze_results.py --run-dir batch_results/<run_name>
```

3. Use generated outputs under `batch_results/<run_name>/analysis/`:
- `mean_by_algorithm.csv`
- `mean_by_dynamic_obstacles.csv`
- `coverage_vs_dynamic_obstacles.png`
- `dynamic_collisions_heatmap.png`

### Unified Q1–Q5 Research Suite (recommended for the report)

Run the full suite (outputs: `all_runs.csv`, `summary_by_condition.csv`, plots, anomalies):

```bash
python experiments/run_research_suite.py --output-dir research_results_full
```

Fast sanity check:

```bash
python experiments/run_research_suite.py --fast --output-dir research_results_fast
```

## Code Attribution

- `env/simpleBot1.py` and `env/simpleBot2.py` are provided by the course instructors. Their original comments and structure have been preserved.
- `env/robot_env.py` and `test_env.py` are original work created by the project team, under the supervision of the course.

If you extend or reuse any code from other sources, please add proper attribution here.

## Contributing

This repository is used by the project team (5 members).

- Each member works on their own feature branch.
- Merge into `main` after code review.
- Keep the `main` branch always runnable.

**Branch naming convention**: `feature/your-name` (e.g., `feature/alice-planning`).

## License

This project is part of the coursework and is not intended for external distribution.
All rights reserved to the University of Nottingham and the project team.