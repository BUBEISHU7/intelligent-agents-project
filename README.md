# Intelligent Agents Project

This repository contains the coursework project for the module **Designing Intelligent Agents**.  
We focus on a **robotics-based environment** with **complex planning**, **multi-agent coordination**, and **noise robustness**.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
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
