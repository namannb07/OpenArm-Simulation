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
from openarm_mujoco.gesture_mapper import FingerJointMapper

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
    ("openarm_{side}_finger_joint1",   "{side}_grip1_act",    80,  8,  7.0),
    ("openarm_{side}_finger_joint2",   "{side}_grip2_act",    80,  8,  7.0),
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

    # ── Gripper Collision Exclusions ───────────────────────────────────
    # Exclude internal contact between finger links and end-effector base link
    # to prevent physics contact forces from tearing the mimic equality constraint.
    for side in ("left", "right"):
        ex1 = spec.add_exclude()
        ex1.bodyname1 = f"openarm_{side}_ee_base_link"
        ex1.bodyname2 = f"openarm_{side}_ee_link1"
        ex2 = spec.add_exclude()
        ex2.bodyname1 = f"openarm_{side}_ee_base_link"
        ex2.bodyname2 = f"openarm_{side}_ee_link2"
        ex3 = spec.add_exclude()
        ex3.bodyname1 = f"openarm_{side}_ee_link1"
        ex3.bodyname2 = f"openarm_{side}_ee_link2"

    # ── Simulation Environment Setup ──────────────────────────────────
    # 1. Skybox Texture
    sky_tex = spec.add_texture()
    sky_tex.name = "sky_tex"
    sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky_tex.rgb1 = [0.4, 0.5, 0.6]  # Light sky blue
    sky_tex.rgb2 = [0.0, 0.0, 0.0]  # Ground reflection / horizon blend
    sky_tex.width = 800
    sky_tex.height = 800

    # 2. Tiled Floor Texture (Checkered Pattern)
    floor_tex = spec.add_texture()
    floor_tex.name = "tiled_floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.15, 0.15, 0.15]  # Dark grey tiles
    floor_tex.rgb2 = [0.30, 0.30, 0.30]  # Light grey tiles
    floor_tex.width = 512
    floor_tex.height = 512
    floor_tex.mark = mujoco.mjtMark.mjMARK_CROSS
    floor_tex.markrgb = [0.1, 0.1, 0.1]

    # 3. Floor Material
    floor_mat = spec.add_material()
    floor_mat.name = "tiled_floor_mat"
    floor_mat.textures = ["tiled_floor_tex"] + [""] * 9
    floor_mat.roughness = 0.5
    floor_mat.shininess = 0.1

    # 4. Floor Geom
    floor_geom = spec.worldbody.add_geom()
    floor_geom.name = "floor"
    floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_geom.size = [10.0, 10.0, 0.1]
    floor_geom.pos = [0.0, 0.0, 0.0]
    floor_geom.material = "tiled_floor_mat"

    # 5. Table Setup
    # Size: width=44cm, depth=90cm, height=30cm (half-extents: [0.22, 0.45, 0.15])
    table_size = [0.22, 0.45, 0.15]
    table_body = spec.worldbody.add_body()
    table_body.name = "table"
    table_body.pos = [0.45, 0.0, table_size[2]]

    table_geom = table_body.add_geom()
    table_geom.name = "table_geom"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = table_size
    table_geom.rgba = [0.4, 0.3, 0.25, 1.0]  # Dark wood color
    table_geom.friction = [0.5, 0.005, 0.0001]

    # 6. Lighting Setup (Key + Fill)
    sun = spec.worldbody.add_light()
    sun.name = "sun"
    sun.pos = [0.0, 0.0, 3.0]
    sun.dir = [0.0, 0.0, -1.0]
    sun.castshadow = True
    sun.diffuse = [0.8, 0.8, 0.8]
    sun.ambient = [0.3, 0.3, 0.3]
    sun.specular = [0.2, 0.2, 0.2]

    fill_light = spec.worldbody.add_light()
    fill_light.name = "fill_light"
    fill_light.pos = [2.0, 2.0, 2.0]
    fill_light.dir = [-1.0, -1.0, -1.0]
    fill_light.castshadow = False
    fill_light.diffuse = [0.4, 0.4, 0.4]

    # 7. Open Cardboard Box with 4 Hinged Flaps
    # Container size: 15cm x 15cm x 15cm outer cube
    box_body = spec.worldbody.add_body()
    box_body.name = "cardboard_box"
    box_body.pos = [0.38, 0.0, 0.30]  # Sitting on table top
    box_body.add_freejoint()

    # Bottom plate
    bottom = box_body.add_geom()
    bottom.name = "cardboard_box_bottom"
    bottom.type = mujoco.mjtGeom.mjGEOM_BOX
    bottom.size = [0.075, 0.075, 0.0025]
    bottom.pos = [0.0, 0.0, 0.0025]
    bottom.rgba = [0.76, 0.60, 0.42, 1.0]  # Cardboard brown
    bottom.mass = 0.04
    bottom.friction = [0.8, 0.005, 0.0001]
    bottom.solref = [0.04, 1.0]
    bottom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    # 4 side walls
    wall_defs = [
        ("box_wall_front", [0.075, 0.002, 0.075], [0.0, 0.073, 0.0775]),
        ("box_wall_back",  [0.075, 0.002, 0.075], [0.0, -0.073, 0.0775]),
        ("box_wall_left",  [0.002, 0.071, 0.075], [-0.073, 0.0, 0.0775]),
        ("box_wall_right", [0.002, 0.071, 0.075], [0.073, 0.0, 0.0775]),
    ]
    for wname, wsize, wpos in wall_defs:
        wgeom = box_body.add_geom()
        wgeom.name = wname
        wgeom.type = mujoco.mjtGeom.mjGEOM_BOX
        wgeom.size = wsize
        wgeom.pos = wpos
        wgeom.rgba = [0.76, 0.60, 0.42, 1.0]
        wgeom.mass = 0.02
        wgeom.friction = [0.8, 0.005, 0.0001]
        wgeom.solref = [0.04, 1.0]
        wgeom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    # 4 hinged top flaps (hinged at top edges of walls z = 0.1525)
    # q = 0.0 rad represents FLAP CLOSED (flat over box top).
    # q = 1.0 rad represents FLAP OPEN (~57 deg angled upward/outward).
    # Range [0.0, 2.4] rad allows full rotation when pushed.
    flaps_info = [
        ("box_flap_front", [0.0, 0.073, 0.1525], [-1, 0, 0], [0.0, -0.036, 0.0], [0.073, 0.036, 0.0015]),
        ("box_flap_back",  [0.0, -0.073, 0.1525], [1, 0, 0], [0.0, +0.036, 0.0], [0.073, 0.036, 0.0015]),
        ("box_flap_left",  [-0.073, 0.0, 0.1525], [0, 1, 0], [+0.036, 0.0, 0.0], [0.036, 0.071, 0.0015]),
        ("box_flap_right", [+0.073, 0.0, 0.1525], [0, -1, 0], [-0.036, 0.0, 0.0], [0.036, 0.071, 0.0015]),
    ]
    for fname, fpos, faxis, fgpos, fgsize in flaps_info:
        f_body = box_body.add_body()
        f_body.name = fname
        f_body.pos = fpos
        fj = f_body.add_joint()
        fj.name = f"{fname}_joint"
        fj.type = mujoco.mjtJoint.mjJNT_HINGE
        fj.axis = faxis
        fj.range[0] = 0.0
        fj.range[1] = 2.4
        fj.damping[0] = 0.01
        fj.stiffness[0] = 0.05   # Moderate cardboard crease spring
        fj.springref = 1.8       # Holds flap standing wide OPEN (~103 deg) under gravity
        fg = f_body.add_geom()
        fg.name = f"{fname}_geom"
        fg.type = mujoco.mjtGeom.mjGEOM_BOX
        fg.size = fgsize
        fg.pos = fgpos
        fg.rgba = [0.80, 0.64, 0.45, 1.0]
        fg.mass = 0.01
        fg.friction = [0.8, 0.005, 0.0001]

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
        self._gesture_mappers: dict[str, FingerJointMapper] = {}
        self._init_gesture_mappers()

        # Seed initial flap joint positions to wide open rest angles (1.8 rad ~ 103 deg)
        for flap_jnt in (
            "box_flap_front_joint",
            "box_flap_back_joint",
            "box_flap_left_joint",
            "box_flap_right_joint",
        ):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, flap_jnt)
            if j_id >= 0:
                adr = self.model.jnt_qposadr[j_id]
                self.model.qpos0[adr] = 1.8
                self.data.qpos[adr] = 1.8

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

            grip1_info = self._lookup(
                f"openarm_{side}_finger_joint1",
                f"{side}_grip1_act",
            )
            grip2_info = self._lookup(
                f"openarm_{side}_finger_joint2",
                f"{side}_grip2_act",
            )
            self.arms[side] = {
                "joints":   joints,
                "gripper":  grip1_info,
                "gripper2": grip2_info,
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
            grip1 = self.arms[side]["gripper"]
            if grip1:
                self._set_ctrl(grip1, float(self.data.qpos[grip1["qpos_adr"]]))
            grip2 = self.arms[side].get("gripper2")
            if grip2:
                self._set_ctrl(grip2, float(self.data.qpos[grip2["qpos_adr"]]))

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
        """Create one FingerJointMapper per robot arm (left/right)."""
        for side in ("left", "right"):
            arm = self.arms[side]
            joint_limits = [(j["lower"], j["upper"]) for j in arm["joints"]]
            grip = arm["gripper"]
            gripper_limits = (grip["lower"], grip["upper"]) if grip else (0.0, 0.0)
            self._gesture_mappers[side] = FingerJointMapper(
                joint_limits=joint_limits,
                gripper_limits=gripper_limits,
            )

    # ------------------------------------------------------------------
    # Keyboard callback
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Keyboard callback
    # ------------------------------------------------------------------
    def _key_callback(self, keycode: int) -> None:
        if ord("1") <= keycode <= ord("7"):
            self.selected_joint = keycode - ord("1")
            self._print_status()
        elif keycode in (ord("G"), ord("g"), ord("8")):
            self.selected_joint = 7  # 7 = Gripper mode
            self._print_status()
        elif keycode == _KEY_TAB:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            # Reset EMA on the newly activated arm so the first finger sample
            # doesn't cause a sudden jump in joint position.
            if self._gesture_mode:
                self._gesture_mappers[self.active_arm].reset()
                self._seed_mappers_from_ctrl(arm_side=self.active_arm)
            self._print_status()
        elif keycode == _KEY_UP:
            if self.selected_joint == 7:
                self._move_gripper(JOINT_STEP_FINE)  # Up closes gripper (towards 0 rad)
            else:
                self._move_joint(JOINT_STEP_FINE)
        elif keycode == _KEY_DOWN:
            if self.selected_joint == 7:
                self._move_gripper(-JOINT_STEP_FINE)  # Down opens gripper (towards -0.80 rad)
            else:
                self._move_joint(-JOINT_STEP_FINE)
        elif keycode == _KEY_RIGHT:
            if self.selected_joint == 7:
                self._move_gripper(JOINT_STEP_COARSE)
            else:
                self._move_joint(JOINT_STEP_COARSE)
        elif keycode == _KEY_LEFT:
            if self.selected_joint == 7:
                self._move_gripper(-JOINT_STEP_COARSE)
            else:
                self._move_joint(-JOINT_STEP_COARSE)
        elif keycode in (ord("H"), ord("h")):
            # H also selects the Gripper
            self.selected_joint = 7
            self._print_status()
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
        arm   = self.arms[self.active_arm]
        grip1 = arm["gripper"]
        grip2 = arm.get("gripper2")
        if grip1 is None:
            return
        new_val = self._get_ctrl(grip1) + delta
        self._set_ctrl(grip1, new_val)
        if grip2 is not None:
            self._set_ctrl(grip2, -new_val)
        self._print_status()

    def _reset_joints(self) -> None:
        self.data.qpos[:] = self.model.qpos0[:]
        self.data.qvel[:] = 0.0
        self._sync_ctrl_to_qpos()
        print("  [OK] Simulation reset to initial state (robot & box).")
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
        elif self.selected_joint == 7 and arm["gripper"]:
            grip   = arm["gripper"]
            target = self._get_ctrl(grip)
            actual = float(self.data.qpos[grip["qpos_adr"]])
            print(
                f"  [{self.active_arm.upper():>5}] "
                f"Gripper: "
                f"target={target:+.3f}  actual={actual:+.3f} rad  "
                f"[{grip['lower']:.2f}, {grip['upper']:.2f}]"
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
                sel = ">" if (side == self.active_arm and self.selected_joint == 7) else " "
                print(
                    f"    {sel} Gripper: target={target:+.4f}  "
                    f"actual={actual:+.4f} rad  "
                    f"[{grip['lower']:.2f}, {grip['upper']:.2f}]"
                )
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

    def _seed_mappers_from_ctrl(self, arm_side: str | None = None) -> None:
        """Pre-fill EMA with current ctrl so there is no jump on first frame.

        Parameters
        ----------
        arm_side
            If given, seed only that arm's mapper.  If *None*, seed both.
        """
        sides = (arm_side,) if arm_side else ("left", "right")
        for side in sides:
            arm    = self.arms[side]
            mapper = self._gesture_mappers[side]
            joint_values = [
                self._get_ctrl(jinfo) for jinfo in arm["joints"][:7]
            ]
            mapper.seed(joint_values)

    def _apply_gesture_commands(self) -> None:
        """Read latest finger curls and drive the currently active robot arm.

        Both camera hands (left + right) are always passed to the active arm's
        :class:`~openarm_mujoco.gesture_mapper.FingerJointMapper`.  The mapper
        uses the right-camera-hand for J1–J5 and the left-camera-hand for J6–J7.
        Switching arms with Tab simply switches *which robot arm* receives those
        joint targets — the camera hands keep the same finger assignments.
        """
        if not self._gesture_mode or self._hand_tracker is None:
            return
        gestures = self._hand_tracker.get_gestures()
        mapper   = self._gesture_mappers[self.active_arm]
        targets  = mapper.update(gestures)
        arm      = self.arms[self.active_arm]
        for i, target in enumerate(targets.joint_positions):
            if target is not None and i < len(arm["joints"]):
                self._set_ctrl(arm["joints"][i], target)
        # Gripper is always keyboard-only (G / H keys or Arrow keys when selected)

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
| Selection & Control                                      |
|   1-7        Select joint 1 through 7                    |
|   G / 8      Select Gripper                              |
|   Tab        Toggle active arm (Left / Right)            |
| Joint / Gripper Movement                                 |
|   Up/Down    Fine step   (+/-0.05 rad)                   |
|   Rt/Left    Coarse step (+/-0.20 rad)                   |
| Utilities    R = Reset   P = Print joints   V = Finger Ctrl|
+----------------------------------------------------------+
"""
        )

    @staticmethod
    def _print_gesture_reference() -> None:
        print(
            """
Finger -> Robot Joint Mapping
  Finger       Camera Hand   Joint   Robot Motion
  -----------  -----------   ------  ----------------------------
  Thumb        Right         J1      Base yaw
  Index        Right         J2      Shoulder pitch
  Middle       Right         J3      Upper-arm rotation
  Ring         Right         J4      Elbow flex / extend
  Pinky        Right         J5      Wrist pitch
  Index        Left          J6      Wrist roll
  Middle       Left          J7      Fine end-effector twist
  G / H keys   (keyboard)    Grip    Open / Close gripper

  Tab                               Switch active arm (LEFT <-> RIGHT)
  V                                 Disable finger control mode

  Tips:
   - Keep both hands 40-80 cm from the camera
   - Bend ONE finger at a time for clean, isolated joint control
   - Straight fingers = joint at lower limit; fully curled = upper limit
   - Tab switches which ROBOT ARM receives the finger commands
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
