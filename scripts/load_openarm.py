from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "models" / "openarm_mujoco.urdf"

print(f"Loading: {URDF}")

model = mujoco.MjModel.from_xml_path(str(URDF))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
