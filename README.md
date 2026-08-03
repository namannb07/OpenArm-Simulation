# OpenArm MuJoCo Simulation

This repository contains a [MuJoCo](https://mujoco.org/) simulation environment for the OpenArm robotic arm. It allows you to load and visualize the OpenArm model using the MuJoCo physics engine in Python.

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Installation

This project uses `uv` for fast dependency management. You can install the dependencies by syncing the project:

```bash
uv sync
```

This will create a virtual environment (`.venv`) and install the required packages, including `mujoco`.

Alternatively, if you are not using `uv`, you can install the dependencies using `pip`:

```bash
pip install mujoco>=3.11.0
```

## Usage

### Viewing the Model

You can load and visualize the OpenArm model in a passive MuJoCo viewer using the provided script.

Run the following command from the root of the workspace:

```bash
uv run python scripts/load_openarm.py
```

Or, if you have activated the virtual environment manually:

```bash
python scripts/load_openarm.py
```

This script will parse the URDF model located at `models/openarm_mujoco.urdf` and open a MuJoCo interactive viewer.

## Project Structure

- `assets/`: Contains assets (like meshes or textures) referenced by the model.
- `models/`: Contains the robot descriptions. The primary model is `openarm_mujoco.urdf`.
- `scripts/`: Contains executable scripts, such as `load_openarm.py` for visualizing the robot.
- `src/openarm_mujoco/`: The main Python package source code.
- `tools/`: Additional tools or utilities for the project.
- `pyproject.toml`: The Python project configuration file.
