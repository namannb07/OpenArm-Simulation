# OpenArm MuJoCo Simulation

This repository contains a [MuJoCo](https://mujoco.org/) simulation environment for the OpenArm robotic arm. It allows you to load and visualize the OpenArm model using the MuJoCo physics engine in Python.

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- A webcam (optional — required only for hand-gesture control)

## Installation

This project uses `uv` for fast dependency management. You can install the dependencies by syncing the project:

```bash
uv sync
```

This will create a virtual environment (`.venv`) and install the required packages, including `mujoco`, `opencv-python`, and `mediapipe`.

Alternatively, if you are not using `uv`, you can install the dependencies using `pip`:

```bash
pip install mujoco>=3.11.0 opencv-python>=4.8.0 mediapipe>=0.10.0 numpy>=1.24.0
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

### Keyboard Controls

Once the viewer is open, you can control the robot using your keyboard:

| Key | Action |
|-----|--------|
| `1` – `7` | Select joint 1 through 7 |
| `Tab` | Toggle between Left / Right arm |
| `↑` / `↓` | Fine step (±0.05 rad) |
| `→` / `←` | Coarse step (±0.25 rad) |
| `G` | Close gripper |
| `H` | Open gripper |
| `R` | Reset all joints to zero |
| `P` | Print all joint states |
| `V` | Toggle hand gesture control |

### Hand Gesture Control 🖐️

The simulation supports real-time control of the robot arms using hand gestures captured by your webcam.

#### Starting Gesture Control

**Option 1 — Toggle at runtime:** Press `V` in the MuJoCo viewer to enable/disable gesture control.

**Option 2 — Start with gesture mode:** Launch with the `--gesture` flag:

```bash
uv run python scripts/load_openarm.py --gesture
```

#### How It Works

A background thread captures your webcam feed and uses [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) to detect hand landmarks in real time. Your hand movements are then mapped to robot joint commands:

| Your Hand Gesture | Robot Joint | Description |
|---|---|---|
| Wrist position (horizontal) | Joint 1 | Base yaw — move hand left/right |
| Wrist position (vertical) | Joint 2 | Shoulder pitch — move hand up/down |
| Hand distance from camera | Joint 3 | Elbow — move hand closer/further |
| Palm pitch (tilt forward/back) | Joint 5 | Wrist pitch |
| Palm roll (tilt left/right) | Joint 6 | Wrist roll |
| Thumb-index pinch | Gripper | Pinch to close, spread to open |

> **Note:** Joints 4 and 7 (twist/fine-rotation) remain keyboard-controlled as they are difficult to map intuitively to hand gestures.

#### Bimanual Control

- Your **left hand** controls the **left arm**
- Your **right hand** controls the **right arm**
- Both arms can be controlled simultaneously when both hands are visible

#### Tips

- Stand about 1–2 feet from your webcam for best tracking
- An OpenCV window shows the camera feed with hand landmark overlay
- The system applies smoothing to reduce jitter — movements feel natural but slightly delayed
- When you remove your hand from view, the corresponding arm holds its last position
- Keyboard controls remain fully functional while gesture mode is active

## Project Structure

- `assets/`: Contains assets (like meshes or textures) referenced by the model.
- `models/`: Contains the robot descriptions. The primary model is `openarm_mujoco.urdf`.
- `scripts/`: Contains executable scripts, such as `load_openarm.py` for visualizing the robot.
- `src/openarm_mujoco/`: The main Python package source code.
  - `sim_controller.py`: Interactive simulation controller with keyboard and gesture input.
  - `hand_tracker.py`: Webcam-based hand detection using MediaPipe Hands.
  - `gesture_mapper.py`: Translates hand poses to robot joint targets.
- `tools/`: Additional tools or utilities for the project.
- `pyproject.toml`: The Python project configuration file.
