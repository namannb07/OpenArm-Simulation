"""Interactive simulation controller for OpenArm v2.0 bimanual robot.

Provides keyboard-driven control of all arm joints and grippers within
the MuJoCo passive viewer.  No additional dependencies beyond ``mujoco``.
"""

import time

import mujoco
import mujoco.viewer

# Joint step size in radians per keypress
JOINT_STEP_FINE = 0.05    # Up / Down arrows
JOINT_STEP_COARSE = 0.25  # Right / Left arrows
GRIPPER_STEP = 0.1

# GLFW key codes used by the MuJoCo viewer callback
_KEY_TAB = 258
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_DOWN = 264
_KEY_UP = 265


class SimController:
    """Manages interactive keyboard control of the OpenArm simulation.

    Parameters
    ----------
    model_path : str
        Absolute path to the URDF / MJCF model file.
    """

    def __init__(self, model_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Discover joint / actuator mapping from the loaded model
        self._build_joint_map()

        # Control state
        self.active_arm: str = "left"   # "left" or "right"
        self.selected_joint: int = 0    # 0-6  →  joints 1-7

    # ------------------------------------------------------------------
    # Joint discovery
    # ------------------------------------------------------------------
    def _build_joint_map(self) -> None:
        """Build mapping from arm side + joint index to MuJoCo qpos indices."""
        self.arms: dict = {}
        for side in ("left", "right"):
            # 7-DOF arm joints
            joints: list[dict] = []
            for i in range(1, 8):
                name = f"openarm_{side}_joint{i}"
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if jnt_id < 0:
                    continue
                joints.append(
                    {
                        "name": name,
                        "jnt_id": jnt_id,
                        "qpos_adr": self.model.jnt_qposadr[jnt_id],
                        "lower": float(self.model.jnt_range[jnt_id, 0]),
                        "upper": float(self.model.jnt_range[jnt_id, 1]),
                    }
                )

            # Gripper – only finger_joint1 is independent; finger_joint2
            # is a mimic joint (multiplier = -1).
            grip_info = self._lookup_joint(f"openarm_{side}_finger_joint1")
            mimic_info = self._lookup_joint(f"openarm_{side}_finger_joint2")

            self.arms[side] = {
                "joints": joints,
                "gripper": grip_info,
                "gripper_mimic": mimic_info,
            }

    def _lookup_joint(self, name: str) -> dict | None:
        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jnt_id < 0:
            return None
        return {
            "name": name,
            "jnt_id": jnt_id,
            "qpos_adr": self.model.jnt_qposadr[jnt_id],
            "lower": float(self.model.jnt_range[jnt_id, 0]),
            "upper": float(self.model.jnt_range[jnt_id, 1]),
        }

    # ------------------------------------------------------------------
    # Keyboard callback
    # ------------------------------------------------------------------
    def _key_callback(self, keycode: int) -> None:
        """Handle keyboard input forwarded by the MuJoCo viewer."""
        # 1-7: select joint
        if ord("1") <= keycode <= ord("7"):
            self.selected_joint = keycode - ord("1")
            self._print_status()

        # Tab: toggle arm
        elif keycode == _KEY_TAB:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            self._print_status()

        # Arrow keys: move selected joint
        elif keycode == _KEY_UP:
            self._move_joint(JOINT_STEP_FINE)
        elif keycode == _KEY_DOWN:
            self._move_joint(-JOINT_STEP_FINE)
        elif keycode == _KEY_RIGHT:
            self._move_joint(JOINT_STEP_COARSE)
        elif keycode == _KEY_LEFT:
            self._move_joint(-JOINT_STEP_COARSE)

        # G / g: close gripper
        elif keycode in (ord("G"), ord("g")):
            self._move_gripper(-GRIPPER_STEP)

        # H / h: open gripper
        elif keycode in (ord("H"), ord("h")):
            self._move_gripper(GRIPPER_STEP)

        # R / r: reset all joints
        elif keycode in (ord("R"), ord("r")):
            self._reset_joints()

        # P / p: print all joint states
        elif keycode in (ord("P"), ord("p")):
            self._print_all_joints()

    # ------------------------------------------------------------------
    # Joint manipulation helpers
    # ------------------------------------------------------------------
    def _move_joint(self, delta: float) -> None:
        """Move the currently selected joint by *delta* radians, clamped."""
        arm = self.arms[self.active_arm]
        if self.selected_joint >= len(arm["joints"]):
            return
        jinfo = arm["joints"][self.selected_joint]
        current = float(self.data.qpos[jinfo["qpos_adr"]])
        new_val = max(jinfo["lower"], min(jinfo["upper"], current + delta))
        self.data.qpos[jinfo["qpos_adr"]] = new_val
        self._print_status()

    def _move_gripper(self, delta: float) -> None:
        """Open / close the gripper on the active arm."""
        arm = self.arms[self.active_arm]
        grip = arm["gripper"]
        if grip is None:
            return

        current = float(self.data.qpos[grip["qpos_adr"]])
        new_val = max(grip["lower"], min(grip["upper"], current + delta))
        self.data.qpos[grip["qpos_adr"]] = new_val

        # Synchronise mimic joint (multiplier = -1)
        mimic = arm["gripper_mimic"]
        if mimic is not None:
            mimic_val = max(mimic["lower"], min(mimic["upper"], -new_val))
            self.data.qpos[mimic["qpos_adr"]] = mimic_val

        print(f"  Gripper ({self.active_arm}): {new_val:+.3f} rad")

    def _reset_joints(self) -> None:
        """Reset every joint on both arms to zero."""
        for side in ("left", "right"):
            arm = self.arms[side]
            for jinfo in arm["joints"]:
                self.data.qpos[jinfo["qpos_adr"]] = 0.0
            if arm["gripper"] is not None:
                self.data.qpos[arm["gripper"]["qpos_adr"]] = 0.0
            if arm["gripper_mimic"] is not None:
                self.data.qpos[arm["gripper_mimic"]["qpos_adr"]] = 0.0
        print("  ✓ All joints reset to 0.")
        self._print_status()

    # ------------------------------------------------------------------
    # Terminal feedback
    # ------------------------------------------------------------------
    def _print_status(self) -> None:
        """Print the currently-selected joint and its value."""
        arm = self.arms[self.active_arm]
        if self.selected_joint < len(arm["joints"]):
            jinfo = arm["joints"][self.selected_joint]
            val = float(self.data.qpos[jinfo["qpos_adr"]])
            print(
                f"  [{self.active_arm.upper():>5}] "
                f"Joint {self.selected_joint + 1}: {val:+.3f} rad  "
                f"(range: {jinfo['lower']:.2f} .. {jinfo['upper']:.2f})"
            )

    def _print_all_joints(self) -> None:
        """Dump the full joint state for both arms."""
        print("\n" + "=" * 60)
        for side in ("left", "right"):
            arm = self.arms[side]
            marker = " ◄ active" if side == self.active_arm else ""
            print(f"  {side.upper()} ARM{marker}")
            for i, jinfo in enumerate(arm["joints"]):
                val = float(self.data.qpos[jinfo["qpos_adr"]])
                sel = "►" if (side == self.active_arm and i == self.selected_joint) else " "
                print(
                    f"    {sel} J{i + 1}: {val:+.4f} rad  "
                    f"[{jinfo['lower']:.2f}, {jinfo['upper']:.2f}]"
                )
            if arm["gripper"] is not None:
                val = float(self.data.qpos[arm["gripper"]["qpos_adr"]])
                print(f"      Gripper: {val:+.4f} rad")
            print()
        print("=" * 60)

    # ------------------------------------------------------------------
    # Controls help banner
    # ------------------------------------------------------------------
    @staticmethod
    def _print_controls() -> None:
        """Print the keyboard-control reference to stdout."""
        print(
            """
╔══════════════════════════════════════════════════════════╗
║          OpenArm Interactive Simulation Controls         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Joint Selection                                         ║
║    1 – 7        Select joint 1 through 7                 ║
║    Tab           Toggle between Left / Right arm         ║
║                                                          ║
║  Joint Movement                                          ║
║    ↑ / ↓        Fine step   (±0.05 rad)                  ║
║    → / ←        Coarse step (±0.25 rad)                  ║
║                                                          ║
║  Gripper                                                 ║
║    G             Close gripper                           ║
║    H             Open gripper                            ║
║                                                          ║
║  Utilities                                               ║
║    R             Reset all joints to zero                ║
║    P             Print all joint states                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Launch the viewer and enter the interactive simulation loop."""
        self._print_controls()

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._key_callback,
        ) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)  # ≈ 1 kHz – avoids busy-waiting
