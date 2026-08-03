from pathlib import Path

from openarm_mujoco.sim_controller import SimController

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "models" / "openarm_mujoco.urdf"

print(f"Loading: {URDF}")

controller = SimController(str(URDF))
controller.run()
