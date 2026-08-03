"""Launch the OpenArm MuJoCo simulation.

Supports an optional ``--gesture`` flag to start with hand-gesture
control already active.
"""

from pathlib import Path
import argparse

from openarm_mujoco.sim_controller import SimController

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "models" / "openarm_mujoco.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenArm v2.0 MuJoCo Simulation"
    )
    parser.add_argument(
        "--gesture",
        action="store_true",
        help="Start with hand-gesture control enabled (requires webcam)",
    )
    args = parser.parse_args()

    print(f"Loading: {URDF}")
    controller = SimController(str(URDF))

    if args.gesture:
        controller._toggle_gesture_mode()

    controller.run()


if __name__ == "__main__":
    main()
