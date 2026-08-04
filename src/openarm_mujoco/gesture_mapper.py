"""Map hand gesture data to robot joint targets.

Translates :class:`~openarm_mujoco.hand_tracker.GestureState` objects
produced by the hand tracker into concrete joint position targets that can
be applied to the MuJoCo simulation via ``data.ctrl``.

Hand ↔ Robot Joint Mapping
---------------------------
The OpenArm v2.0 is a 7-DOF arm.  We use the following intuitive mapping
so that the operator's hand posture mirrors the arm's posture:

=========================================  =========  ========================
Hand cue                                   Joint      Robot motion
=========================================  =========  ========================
Wrist X position (left ↔ right in frame)  J1 (idx 0) Base yaw
Wrist Y position (up ↔ down in frame)     J2 (idx 1) Shoulder pitch
Palm forward tilt (pitch angle)            J3 (idx 2) Upper-arm / elbow region
Finger curl (mean PIP bend, index→pinky)  J4 (idx 3) Elbow flex/extend
Palm tilt left/right (roll angle)          J5 (idx 4) Wrist pitch
Palm face-up / face-down (palm normal Z)  J6 (idx 5) Wrist roll
*(unmapped — keyboard only)*               J7 (idx 6) Fine end-effector twist
Thumb–index pinch distance                 Gripper    Open / close fingers
=========================================  =========  ========================

Design choices
--------------
* **EMA smoothing** (configurable alpha) on every output channel to suppress
  high-frequency noise from the ML tracker.
* **Dead-zone** on the *raw normalised input*: if the change from the last
  accepted sample is below ``dead_zone``, we reuse the previous EMA output
  without advancing the filter.  This prevents microscopically-noisy stable
  poses from walking the joint.
* **Hysteresis band** on the gripper to avoid rapid open/close chatter.
"""

from __future__ import annotations

import dataclasses
import math

from openarm_mujoco.hand_tracker import GestureState


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JointTargets:
    """Target joint positions computed from a single hand gesture frame.

    ``joint_positions`` is a list of length 7 (joints 1–7).  Entries that
    could not be determined from the gesture are set to *None*.
    """

    joint_positions: list[float | None]
    """Target for each of the 7 arm joints (radians).  *None* = no change."""

    gripper_position: float | None
    """Target for the gripper joint (radians).  *None* = no change."""


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

# Gripper pinch thresholds (normalised by frame diagonal).
# Below CLOSE → fully closed.  Above OPEN → fully open.  Between → interpolate.
_PINCH_CLOSE_THRESHOLD = 0.04
_PINCH_OPEN_THRESHOLD  = 0.08


class GestureMapper:
    """Convert normalised gesture data into joint-position targets.

    Parameters
    ----------
    joint_limits
        Sequence of ``(lower, upper)`` tuples for each of the 7 arm joints.
    gripper_limits
        ``(lower, upper)`` for the independent gripper joint.
    smoothing_alpha
        EMA coefficient in (0, 1].  Higher → more responsive (less smooth),
        lower → smoother (more lag).  ``1.0`` disables smoothing.
        Recommended: 0.08–0.15 for gesture teleop.
    dead_zone
        Minimum change in the *normalised* input [0, 1] before a new target
        is produced.  Suppresses jitter when the hand is roughly stationary.
        Recommended: 0.015–0.03.
    """

    def __init__(
        self,
        joint_limits: list[tuple[float, float]],
        gripper_limits: tuple[float, float],
        smoothing_alpha: float = 0.10,
        dead_zone: float = 0.02,
    ) -> None:
        if len(joint_limits) != 7:
            raise ValueError(
                f"Expected 7 joint limits, got {len(joint_limits)}"
            )
        self._joint_limits = list(joint_limits)
        self._gripper_limits = gripper_limits
        self._alpha = smoothing_alpha
        self._dead_zone = dead_zone

        # Internal EMA state — *None* until the first update
        self._smoothed: list[float | None] = [None] * 7
        self._smoothed_grip: float | None = None

        # Previous raw inputs (for dead-zone comparison)
        self._prev_raw: list[float | None] = [None] * 7
        self._prev_pinch: float | None = None

        # Gripper hysteresis state: True = closed, False = open
        self._grip_closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, gesture: GestureState) -> JointTargets:
        """Compute new joint targets from the latest gesture."""
        raw_inputs = self._extract_raw(gesture)
        targets: list[float | None] = [None] * 7

        for i in range(7):
            raw = raw_inputs[i]
            if raw is None:
                targets[i] = self._smoothed[i]   # hold last known position
                continue

            # Dead-zone check — compare against last *accepted* raw value
            if self._prev_raw[i] is not None:
                if abs(raw - self._prev_raw[i]) < self._dead_zone:
                    # Movement too small — hold the current smoothed value
                    targets[i] = self._smoothed[i]
                    continue

            self._prev_raw[i] = raw

            # Map normalised [0, 1] → joint angle range [lo, hi]
            lo, hi = self._joint_limits[i]
            mapped = lo + raw * (hi - lo)

            # EMA smoothing
            if self._smoothed[i] is None:
                self._smoothed[i] = mapped
            else:
                self._smoothed[i] = (
                    self._alpha * mapped
                    + (1 - self._alpha) * self._smoothed[i]
                )
            targets[i] = self._smoothed[i]

        # Gripper — hysteresis-based open / close
        grip_target = self._compute_gripper(gesture.pinch_distance)

        return JointTargets(
            joint_positions=targets,
            gripper_position=grip_target,
        )

    def reset(self) -> None:
        """Clear all internal smoothing state."""
        self._smoothed = [None] * 7
        self._smoothed_grip = None
        self._prev_raw = [None] * 7
        self._prev_pinch = None
        self._grip_closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, g: GestureState) -> list[float | None]:
        """Convert gesture fields to normalised [0, 1] values per joint.

        Mapping table
        -------------
        J1  wrist_x     Horizontal wrist position → base yaw
        J2  wrist_y     Vertical wrist position   → shoulder pitch (inverted)
        J3  palm_pitch  Forward/backward palm tilt → upper-arm rotation
        J4  finger_curl Mean finger bend (0=straight, 1=fist) → elbow flex
        J5  palm_roll   Palm roll angle            → wrist pitch
        J6  palm_facing Palm Z-component           → wrist roll
        J7  (unmapped)  Keyboard only
        """
        # J1: wrist_x already in [0, 1] from MediaPipe
        j1 = g.wrist_x

        # J2: wrist_y in [0, 1] (0=top, 1=bottom) — invert so raising the
        #     hand drives the shoulder upward toward its upper limit.
        j2 = 1.0 - g.wrist_y

        # J3: palm_pitch (±π/2 rad) → [0, 1]
        #     Palm tilted forward  (pitch > 0) → upper value (arm extends)
        #     Palm tilted backward (pitch < 0) → lower value (arm retracts)
        j3 = _clamp01((g.palm_pitch / math.pi) + 0.5)

        # J4: finger_curl proxy — use index-finger MCP-to-tip fold angle.
        #     GestureState exposes this via `finger_curl` (computed in tracker).
        #     Straight fingers (curl≈0) → elbow extended (upper range)
        #     Fist (curl≈1)              → elbow bent    (lower range for J4)
        j4 = 1.0 - _clamp01(g.finger_curl)   # invert: fist → elbow closed

        # J5: palm_roll (±π rad) → [0, 1]
        j5 = _clamp01((g.palm_roll / math.pi) + 0.5)

        # J6: palm_facing — use sin(palm_pitch) as a proxy for face-up/face-down.
        #     Palm facing up (pitch≈−π/2) → one extreme; down (pitch≈+π/2) → other.
        j6 = _clamp01(0.5 + 0.5 * math.sin(g.palm_pitch))

        # J7: unmapped — hold last position (handled by main loop)
        j7 = None

        return [j1, j2, j3, j4, j5, j6, j7]

    def _compute_gripper(self, pinch_distance: float) -> float | None:
        """Hysteresis-based gripper target from thumb–index pinch distance."""
        lo, hi = self._gripper_limits

        if pinch_distance < _PINCH_CLOSE_THRESHOLD:
            target = lo   # Fully closed
        elif pinch_distance > _PINCH_OPEN_THRESHOLD:
            target = hi   # Fully open
        else:
            # Interpolate within the hysteresis band
            t = (pinch_distance - _PINCH_CLOSE_THRESHOLD) / (
                _PINCH_OPEN_THRESHOLD - _PINCH_CLOSE_THRESHOLD
            )
            target = lo + t * (hi - lo)

        # Smooth the gripper as well (separate, slightly slower alpha)
        alpha_grip = self._alpha * 0.7
        if self._smoothed_grip is None:
            self._smoothed_grip = target
        else:
            self._smoothed_grip = (
                alpha_grip * target
                + (1 - alpha_grip) * self._smoothed_grip
            )
        return self._smoothed_grip


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return max(0.0, min(1.0, v))
