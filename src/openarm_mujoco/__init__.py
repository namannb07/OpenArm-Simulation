from pathlib import Path

from openarm_mujoco.sim_controller import SimController


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    urdf = root / "models" / "openarm_mujoco.urdf"
    print(f"Loading: {urdf}")
    controller = SimController(str(urdf))
    controller.run()
