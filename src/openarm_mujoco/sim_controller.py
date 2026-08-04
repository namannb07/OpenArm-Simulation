"""Interactive simulation controller for OpenArm v2.0 bimanual robot.

Provides keyboard-driven control of all arm joints and grippers within
the MuJoCo passive viewer.  Supports an optional **hand-gesture control**
mode using a webcam and MediaPipe Hands.

Physics design
--------------
The model is built from the Anvil URDF using ``mujoco.MjSpec``, which lets
us add position actuators (PD controllers) **before** compiling the model.
Joint targets are written to ``data.ctrl`` -- not directly to ``data.qpos``
-- so the physics engine moves each joint with realistic motor torques instead
of teleporting it every frame (which caused violent jerkiness).

Actuator gains are sourced from the DAMIAO motor datasheets used by Anvil:
  J1, J2  (shoulder) : DM-J8009P-2EC  kp=400  kv=40  limit=40 Nm
  J3      (upper-arm): DM-J4340P-2EC  kp=300  kv=30  limit=27 Nm
  J4      (elbow)    : DM-J4340-2EC   kp=300  kv=30  limit=27 Nm
  J5-J7   (wrist)    : DM-J4310-2EC   kp=100  kv=10  limit=7  Nm
  Gripper (finger)   : DM-J4310-2EC   kp=80   kv=8   limit=7  Nm
"""

# Force UTF-8 output so print() works on Windows cp1252 consoles
import sys
import io as _io
try:
    sys.stdout = _io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
except Exception:
    pass


import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from openarm_mujoco.hand_tracker import HandTracker
from openarm_mujoco.gesture_mapper import GestureMapper

# Joint step size in radians per keypress
JOINT_STEP_FINE   = 0.05   # Up / Down arrows
JOINT_STEP_COARSE = 0.20   # Right / Left arrows
GRIPPER_STEP      = 0.1

# GLFW key codes used by the MuJoCo viewer callback
_KEY_TAB   = 258
_KEY_RIGHT = 262
_KEY_LEFT  = 263
_KEY_DOWN  = 264
_KEY_UP    = 265

# ---------------------------------------------------------------------------
# Actuator specification table
# (joint_name_template, act_name_template, kp, kv, force_limit)
# {side} is substituted with "left" / "right" at build time.
# ---------------------------------------------------------------------------
_ACTUATOR_SPECS = [
    # joint-name pattern               act-name pattern      kp   kv  F_lim
    ("openarm_{side}_joint1",          "{side}_j1_act",      400, 40, 40.0),
    ("openarm_{side}_joint2",          "{side}_j2_act",      400, 40, 40.0),
    ("openarm_{side}_joint3",          "{side}_j3_act",      300, 30, 27.0),
    ("openarm_{side}_joint4",          "{side}_j4_act",      300, 30, 27.0),
    ("openarm_{side}_joint5",          "{side}_j5_act",      100, 10,  7.0),
    ("openarm_{side}_joint6",          "{side}_j6_act",      100, 10,  7.0),
    ("openarm_{side}_joint7",          "{side}_j7_act",      100, 10,  7.0),
    ("openarm_{side}_finger_joint1",   "{side}_grip_act",     80,  8,  7.0),
]


def _build_model(urdf_path: str) -> mujoco.MjModel:
    """Load the Anvil URDF and inject position actuators via MjSpec.

    Using MjSpec (MuJoCo ≥ 3.0) lets us add actuators to the compiled model
    without needing a hand-written MJCF that tries to ``<include>`` the URDF
    (which MuJoCo does not support — ``<include>`` only works for MJCF files).

    Actuator model
    --------------
    Each joint gets a ``position`` servo equivalent:
      torque = kp * (ctrl - q) - kv * qd

    In MjSpec terms this is a ``general`` actuator with:
      gaintype  = FIXED     gainprm[0]  =  kp
      biastype  = AFFINE    biasprm     = [0, -kp, -kv]
    """
    spec = mujoco.MjSpec.from_file(urdf_path)

    # ── Physics options ────────────────────────────────────────────────
    spec.option.timestep    = 0.002                          # 500 Hz
    spec.option.integrator  = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity     = [0.0, 0.0, -9.81]
    spec.option.cone        = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio    = 10.0

    # ── Add position actuators ─────────────────────────────────────────
    for side in ("left", "right"):
        for jnt_tmpl, act_tmpl, kp, kv, flim in _ACTUATOR_SPECS:
            jnt_name = jnt_tmpl.format(side=side)
            act_name = act_tmpl.format(side=side)

            act = spec.add_actuator()
            act.name       = act_name
            act.trntype    = mujoco.mjtTrn.mjTRN_JOINT
            act.target     = jnt_name

            # gaintype=FIXED: output = gainprm[0] * u  (u = ctrl - q)
            act.gaintype   = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm[0] = kp

            # biastype=AFFINE: bias = biasprm[0] + biasprm[1]*q + biasprm[2]*qd
            act.biastype   = mujoco.mjtBias.mjBIAS_AFFINE
            act.biasprm[0] = 0.0
            act.biasprm[1] = -kp   # position spring
            act.biasprm[2] = -kv   # velocity damper

            act.forcelimited = True
            act.forcerange   = [-flim, flim]

    # ── Gripper equality constraints ──────────────────────────────────
    # finger_joint2 = -1 × finger_joint1 (mimic, already declared in URDF
    # via <mimic> tag, but MuJoCo reads it as an equality constraint).
    # Nothing extra needed here — MuJoCo converts <mimic> automatically.

    return spec.compile()


class SimController:
    """Manages interactive keyboard and gesture control of OpenArm.

    Parameters
    ----------
    model_path : str
        Absolute path to the Anvil URDF file (``openarm_mujoco.urdf``).
        Actuators are injected programmatically via MjSpec.
    """

    def __init__(self, model_path: str) -> None:
        self.model = _build_model(model_path)
        self.data  = mujoco.MjData(self.model)

        # Discover joint / actuator mapping from the compiled model
        self._build_joint_map()

        # Control state
        self.active_arm:    str = "left"   # "left" or "right"
        self.selected_joint: int = 0       # 0-6 → joints 1-7

        # Gesture control state
        self._gesture_mode:    bool = False
        self._hand_tracker:    Optional[HandTracker] = None
        self._gesture_mappers: dict[str, GestureMapper] = {}
        self._init_gesture_mappers()

        # Seed ctrl targets from initial qpos so arm holds its rest pose
        self._sync_ctrl_to_qpos()

    # ------------------------------------------------------------------
    # Joint and actuator discovery
    # ------------------------------------------------------------------
    def _build_joint_map(self) -> None:
        """Map arm side + joint index to qpos address and actuator ctrl index."""
        self.arms: dict = {}
        for side in ("left", "right"):
            joints: list[dict] = []
            for i in range(1, 8):
                jnt_name = f"openarm_{side}_joint{i}"
                act_name = f"{side}_j{i}_act"
                info = self._lookup(jnt_name, act_name)
                if info:
                    joints.append(info)
                else:
                    print(f"  [WARNING] Joint not found: {jnt_name}")

            grip_info = self._lookup(
                f"openarm_{side}_finger_joint1",
                f"{side}_grip_act",
            )
            self.arms[side] = {
                "joints":  joints,
                "gripper": grip_info,
            }

    def _lookup(self, jnt_name: str, act_name: str) -> dict | None:
        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name
        )
        if jnt_id < 0:
            return None
        act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
        )
        return {
            "name":     jnt_name,
            "jnt_id":   jnt_id,
            "qpos_adr": self.model.jnt_qposadr[jnt_id],
            "act_id":   act_id,                          # -1 if missing
            "lower":    float(self.model.jnt_range[jnt_id, 0]),
            "upper":    float(self.model.jnt_range[jnt_id, 1]),
        }

    # ------------------------------------------------------------------
    # Ctrl helpers
    # ------------------------------------------------------------------
    def _sync_ctrl_to_qpos(self) -> None:
        """Copy initial joint positions into actuator set-points."""
        for side in ("left", "right"):
            for jinfo in self.arms[side]["joints"]:
                self._set_ctrl(jinfo, float(self.data.qpos[jinfo["qpos_adr"]]))
            grip = self.arms[side]["gripper"]
            if grip:
                self._set_ctrl(grip, float(self.data.qpos[grip["qpos_adr"]]))

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _set_ctrl(self, jinfo: dict, value: float) -> None:
        """Write a clamped position target to the actuator ctrl slot."""
        clamped = self._clamp(value, jinfo["lower"], jinfo["upper"])
        act_id  = jinfo["act_id"]
        if act_id >= 0:
            self.data.ctrl[act_id] = clamped
        else:
            # Fallback (no actuator): write directly to qpos
            self.data.qpos[jinfo["qpos_adr"]] = clamped

    def _get_ctrl(self, jinfo: dict) -> float:
        """Return the current actuator set-point."""
        act_id = jinfo["act_id"]
        if act_id >= 0:
            return float(self.data.ctrl[act_id])
        return float(self.data.qpos[jinfo["qpos_adr"]])

    # ------------------------------------------------------------------
    # Gesture mapper initialisation
    # ------------------------------------------------------------------
    def _init_gesture_mappers(self) -> None:
        for side in ("left", "right"):
            arm = self.arms[side]
            joint_limits = [(j["lower"], j["upper"]) for j in arm["joints"]]
            grip = arm["gripper"]
            gripper_limits = (grip["lower"], grip["upper"]) if grip else (0.0, 0.0)
            self._gesture_mappers[side] = GestureMapper(
                joint_limits=joint_limits,
                gripper_limits=gripper_limits,
            )

    # ------------------------------------------------------------------
    # Keyboard callback
    # ------------------------------------------------------------------
    def _key_callback(self, keycode: int) -> None:
        if ord("1") <= keycode <= ord("7"):
            self.selected_joint = keycode - ord("1")
            self._print_status()
        elif keycode == _KEY_TAB:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            self._print_status()
        elif keycode == _KEY_UP:
            self._move_joint(JOINT_STEP_FINE)
        elif keycode == _KEY_DOWN:
            self._move_joint(-JOINT_STEP_FINE)
        elif keycode == _KEY_RIGHT:
            self._move_joint(JOINT_STEP_COARSE)
        elif keycode == _KEY_LEFT:
            self._move_joint(-JOINT_STEP_COARSE)
        elif keycode in (ord("G"), ord("g")):
            self._move_gripper(-GRIPPER_STEP)
        elif keycode in (ord("H"), ord("h")):
            self._move_gripper(GRIPPER_STEP)
        elif keycode in (ord("R"), ord("r")):
            self._reset_joints()
        elif keycode in (ord("V"), ord("v")):
            self._toggle_gesture_mode()
        elif keycode in (ord("P"), ord("p")):
            self._print_all_joints()

    # ------------------------------------------------------------------
    # Joint manipulation
    # ------------------------------------------------------------------
    def _move_joint(self, delta: float) -> None:
        arm = self.arms[self.active_arm]
        if self.selected_joint >= len(arm["joints"]):
            return
        jinfo   = arm["joints"][self.selected_joint]
        current = self._get_ctrl(jinfo)
        self._set_ctrl(jinfo, current + delta)
        self._print_status()

    def _move_gripper(self, delta: float) -> None:
        arm  = self.arms[self.active_arm]
        grip = arm["gripper"]
        if grip is None:
            return
        self._set_ctrl(grip, self._get_ctrl(grip) + delta)
        print(f"  Gripper ({self.active_arm}): {self._get_ctrl(grip):+.3f} rad")

    def _reset_joints(self) -> None:
        for side in ("left", "right"):
            arm = self.arms[side]
            for jinfo in arm["joints"]:
                rest = self._clamp(0.0, jinfo["lower"], jinfo["upper"])
                self._set_ctrl(jinfo, rest)
            grip = arm["gripper"]
            if grip:
                rest = self._clamp(0.0, grip["lower"], grip["upper"])
                self._set_ctrl(grip, rest)
        self.data.qvel[:] = 0.0
        print("  [OK] All joints reset to 0.")
        self._print_status()

    # ------------------------------------------------------------------
    # Terminal feedback
    # ------------------------------------------------------------------
    def _print_status(self) -> None:
        arm = self.arms[self.active_arm]
        if self.selected_joint < len(arm["joints"]):
            jinfo  = arm["joints"][self.selected_joint]
            target = self._get_ctrl(jinfo)
            actual = float(self.data.qpos[jinfo["qpos_adr"]])
            print(
                f"  [{self.active_arm.upper():>5}] "
                f"J{self.selected_joint + 1}: "
                f"target={target:+.3f}  actual={actual:+.3f} rad  "
                f"[{jinfo['lower']:.2f}, {jinfo['upper']:.2f}]"
            )

    def _print_all_joints(self) -> None:
        print("\n" + "=" * 68)
        for side in ("left", "right"):
            arm    = self.arms[side]
            marker = " << active" if side == self.active_arm else ""
            print(f"  {side.upper()} ARM{marker}")
            for i, jinfo in enumerate(arm["joints"]):
                target = self._get_ctrl(jinfo)
                actual = float(self.data.qpos[jinfo["qpos_adr"]])
                sel = ">" if (side == self.active_arm and i == self.selected_joint) else " "
                print(
                    f"    {sel} J{i+1}: target={target:+.4f}  "
                    f"actual={actual:+.4f} rad  "
                    f"[{jinfo['lower']:.2f}, {jinfo['upper']:.2f}]"
                )
            if arm["gripper"]:
                grip   = arm["gripper"]
                target = self._get_ctrl(grip)
                actual = float(self.data.qpos[grip["qpos_adr"]])
                print(f"      Gripper: target={target:+.4f}  actual={actual:+.4f} rad")
            print()
        print("=" * 68)

    # ------------------------------------------------------------------
    # Gesture control
    # ------------------------------------------------------------------
    def _toggle_gesture_mode(self) -> None:
        if self._gesture_mode:
            if self._hand_tracker is not None:
                self._hand_tracker.stop()
                self._hand_tracker = None
            for mapper in self._gesture_mappers.values():
                mapper.reset()
            self._gesture_mode = False
            print("  [GESTURE] Disabled")
        else:
            tracker = HandTracker(camera_index=0, show_feed=True)
            if tracker.start():
                self._hand_tracker  = tracker
                self._gesture_mode  = True
                self._seed_mappers_from_ctrl()
                print(
                    "  [GESTURE] ENABLED — show your hands to the camera!"
                )
                self._print_gesture_reference()
            else:
                print("  [WARNING] Could not open webcam. Gesture control unavailable.")

    def _seed_mappers_from_ctrl(self) -> None:
        """Pre-fill EMA with current ctrl to avoid startup jump."""
        for side in ("left", "right"):
            arm    = self.arms[side]
            mapper = self._gesture_mappers[side]
            for i, jinfo in enumerate(arm["joints"][:7]):
                val       = self._get_ctrl(jinfo)
                lo, hi    = jinfo["lower"], jinfo["upper"]
                norm      = (val - lo) / (hi - lo) if (hi - lo) > 0 else 0.5
                mapper._smoothed[i]  = val
                mapper._prev_raw[i]  = max(0.0, min(1.0, norm))
            grip = arm["gripper"]
            if grip:
                mapper._smoothed_grip = self._get_ctrl(grip)

    def _apply_gesture_commands(self) -> None:
        if not self._gesture_mode or self._hand_tracker is None:
            return
        gestures = self._hand_tracker.get_gestures()
        for side in ("left", "right"):
            gesture = gestures.get(side)
            if gesture is None:
                continue
            mapper  = self._gesture_mappers[side]
            targets = mapper.update(gesture)
            arm     = self.arms[side]
            for i, target in enumerate(targets.joint_positions):
                if target is not None and i < len(arm["joints"]):
                    self._set_ctrl(arm["joints"][i], target)
            if targets.gripper_position is not None and arm["gripper"]:
                self._set_ctrl(arm["gripper"], targets.gripper_position)

    # ------------------------------------------------------------------
    # Help banners
    # ------------------------------------------------------------------
    @staticmethod
    def _print_controls() -> None:
        print(
            """
+----------------------------------------------------------+
|       OpenArm Interactive Simulation Controls            |
+----------------------------------------------------------+
| Joint Selection                                          |
|   1-7        Select joint 1 through 7                   |
|   Tab        Toggle between Left / Right arm            |
| Joint Movement                                           |
|   Up/Down    Fine step   (+/-0.05 rad)                   |
|   Rt/Left    Coarse step (+/-0.20 rad)                   |
| Gripper   G = Close   H = Open                          |
| Utilities R = Reset   P = Print joints   V = Gesture     |
+----------------------------------------------------------+
"""
        )

    @staticmethod
    def _print_gesture_reference() -> None:
        print(
            """
Hand Gesture -> Robot Joint Mapping
  Hand Cue              Joint     Robot Motion
  --------------------  --------  ---------------------------------
  Wrist LEFT / RIGHT    J1        Base yaw
  Wrist UP / DOWN       J2        Shoulder pitch
  Palm tilt fwd/back    J3        Upper-arm rotation
  Fingers open/fist     J4        Elbow extend / flex
  Palm roll L / R       J5        Wrist pitch
  Palm face up/down     J6        Wrist roll
  (keyboard only)       J7        Fine end-effector twist
  Pinch thumb+index     Gripper   Open / close fingers

  Tips: keep hand 40-80 cm from camera, move slowly, V to disable
"""
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._print_controls()
        mujoco.mj_forward(self.model, self.data)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._key_callback,
        ) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                self._apply_gesture_commands()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                # Hold ~500 Hz (matches the 0.002 s timestep)
                elapsed    = time.perf_counter() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if self._hand_tracker is not None:
                self._hand_tracker.stop()
