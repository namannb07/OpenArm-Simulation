# OpenArm MuJoCo Simulation

Interactive [MuJoCo](https://mujoco.org/) simulation of the **OpenArm v2.0** bimanual robot with real-time keyboard control of every joint and gripper.

![MuJoCo](https://img.shields.io/badge/MuJoCo-≥3.11-blue)
![Python](https://img.shields.io/badge/Python-≥3.12-3776AB)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## Features

- **Full bimanual control** — independently move all 7 joints on each arm
- **Gripper operation** — open and close the pinch grippers with mimic-joint synchronisation
- **Real-time feedback** — joint positions and limits printed to the terminal as you move
- **Joint limit enforcement** — all movements are clamped to URDF-defined safe ranges
- **Zero additional dependencies** — runs entirely on `mujoco` and the Python standard library
- **MuJoCo viewer** — full 3D visualisation with mouse-based camera controls (rotate, pan, zoom)

## Robot Overview

The OpenArm v2.0 is a bimanual robot with two 7-DOF arms and pinch grippers, mounted on a shared body.

| Component | Joints | DOF |
|-----------|--------|-----|
| Left arm  | `openarm_left_joint1` – `openarm_left_joint7` | 7 |
| Left gripper | `openarm_left_finger_joint1` + mimic | 1 |
| Right arm | `openarm_right_joint1` – `openarm_right_joint7` | 7 |
| Right gripper | `openarm_right_finger_joint1` + mimic | 1 |
| **Total** | | **16 independent DOF** |

---

## Prerequisites

- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- A display server (X11 / Wayland) for the MuJoCo viewer window

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/openarm_mujoco.git
cd openarm_mujoco

# Install dependencies (creates a virtual environment automatically)
uv sync
```

> **Note:** `uv sync` reads `pyproject.toml` and `uv.lock` to install exact, reproducible dependencies — currently just `mujoco ≥ 3.11.0`.

---

## Quick Start

```bash
# Run via the script
uv run python scripts/load_openarm.py

# Or via the installed package entry point
uv run openarm-mujoco
```

A MuJoCo viewer window will open showing the OpenArm robot, and the terminal will display the keyboard control reference.

---

## Keyboard Controls

Focus the **MuJoCo viewer window** and use these keys:

### Joint Selection

| Key | Action |
|-----|--------|
| `1` – `7` | Select joint 1 through 7 on the active arm |
| `Tab` | Toggle active arm between **Left** and **Right** |

### Joint Movement

| Key | Action |
|-----|--------|
| `↑` | Increase selected joint angle (fine: +0.05 rad) |
| `↓` | Decrease selected joint angle (fine: −0.05 rad) |
| `→` | Increase selected joint angle (coarse: +0.25 rad) |
| `←` | Decrease selected joint angle (coarse: −0.25 rad) |

### Gripper

| Key | Action |
|-----|--------|
| `G` | Close the gripper on the active arm |
| `H` | Open the gripper on the active arm |

### Utilities

| Key | Action |
|-----|--------|
| `R` | Reset all joints to zero (home position) |
| `P` | Print the full joint state of both arms to the terminal |

### Mouse (built-in MuJoCo viewer)

| Action | Control |
|--------|---------|
| Rotate camera | Left-click + drag |
| Pan camera | Right-click + drag |
| Zoom | Scroll wheel |

---

## Project Structure

```
openarm_mujoco/
├── assets/                          # Robot description assets
│   ├── __init__.py                  # Asset path helpers
│   ├── robot/
│   │   ├── openarm_v1.0/           # V1.0 robot (meshes, URDF, config)
│   │   └── openarm_v2.0/           # V2.0 robot (meshes, URDF, config)
│   │       ├── config/             # Joint limits, axes, origins (YAML)
│   │       ├── meshes/             # Visual (.dae) and collision (.stl) meshes
│   │       └── urdf/               # URDF xacro source files
│   ├── end_effector/
│   │   ├── pinch_gripper/          # Pinch gripper meshes and URDF
│   │   └── parallel_link/          # Parallel-link gripper meshes and URDF
│   └── sensor/
│       └── zed/                    # ZED camera sensor
├── models/
│   ├── openarm_mujoco.urdf         # Compiled URDF (used by the simulation)
│   └── output.urdf                 # Raw xacro output (pre-path-fix)
├── scripts/
│   └── load_openarm.py             # Quick-launch script
├── src/
│   └── openarm_mujoco/
│       ├── __init__.py             # Package entry point (main)
│       └── sim_controller.py       # Interactive simulation controller
├── tools/
│   └── fix_urdf_paths.py           # Converts ROS package:// paths to relative
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
└── README.md
```

---

## URDF Pipeline

The simulation model is built from xacro sources in `assets/`:

```
assets/.../openarm_v20.urdf.xacro
        ↓  (xacro expansion)
models/output.urdf
        ↓  (tools/fix_urdf_paths.py)
models/openarm_mujoco.urdf          ← loaded by MuJoCo
```

The `fix_urdf_paths.py` tool rewrites `package://openarm_description/` prefixes to relative paths (`../`) so MuJoCo can resolve the mesh files without a ROS workspace.

If you modify the xacro source, regenerate the URDF:

```bash
# 1. Expand xacro (requires ROS 2 / xacro installed)
xacro assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro -o models/output.urdf

# 2. Fix mesh paths for MuJoCo
uv run python tools/fix_urdf_paths.py
```

---

## Joint Limits

All joints enforce their URDF-defined limits. Here are the ranges for the right arm (left arm is mirrored):

| Joint | Lower (rad) | Upper (rad) | Lower (deg) | Upper (deg) | Max Effort (Nm) |
|-------|-------------|-------------|-------------|-------------|-----------------|
| J1 | −1.40 | +3.49 | −80° | +200° | 40 |
| J2 | −0.17 | +3.32 | −10° | +190° | 40 |
| J3 | −1.57 | +1.57 | −90° | +90° | 27 |
| J4 | 0.00 | +2.44 | 0° | +140° | 27 |
| J5 | −1.57 | +1.57 | −90° | +90° | 7 |
| J6 | −0.79 | +0.79 | −45° | +45° | 7 |
| J7 | −0.79 | +0.79 | −45° | +45° | 7 |
| Gripper | −1.57 | 0.00 | −90° | 0° | 7 |

---

## License

This project uses assets from [OpenArm](https://github.com/enactic) by Enactic, Inc., licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
