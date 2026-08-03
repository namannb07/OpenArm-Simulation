from pathlib import Path

# Project root = parent directory of this script's folder
project_root = Path(__file__).resolve().parent.parent

input_urdf = project_root / "models" / "output.urdf"
output_urdf = project_root / "models" / "openarm_mujoco.urdf"

print(f"Reading : {input_urdf}")

text = input_urdf.read_text(encoding="utf-8")

# Convert ROS package paths to local paths
text = text.replace(
    "package://openarm_description/",
    "../"
)

output_urdf.write_text(text, encoding="utf-8")

print(f"Written : {output_urdf}")
