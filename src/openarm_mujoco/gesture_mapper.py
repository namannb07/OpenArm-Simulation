"""Map per-finger curl data to robot joint targets.

Translates per-hand :class:`~openarm_mujoco.hand_tracker.GestureState`
objects (keyed by ``"left"`` / ``"right"``) into concrete joint position
targets that can be applied to the MuJoCo simulation via ``data.ctrl``.

Finger ↔ Robot Joint Mapping
------------------------------
The OpenArm v2.0 is a 7-DOF arm.  Each joint is controlled by bending a
single finger.  Five fingers of the **right hand** cover joints 1–5; two
fingers of the **left hand** cover joints 6–7.

============  ===========  =========  ========================
Finger        Camera hand  Joint      Robot motion
============  ===========  =========  ========================
Thumb         Right        J1 (idx 0) Base yaw
Index         Right        J2 (idx 1) Shoulder pitch
Middle        Right        J3 (idx 2) Upper-arm rotation
Ring          Right        J4 (idx 3) Elbow flex / extend
Pinky         Right        J5 (idx 4) Wrist pitch
Index         Left         J6 (idx 5) Wrist roll
Middle        Left         J7 (idx 6) Fine end-effector twist
*(keyboard)*  —            Gripper    Open / close fingers
============  ===========  =========  ========================

Control model — absolute position
----------------------------------
Each finger curl value is in [0, 1]:
    0 = finger fully straight → joint set to its *lower* limit
    1 = finger fully curled   → joint set to its *upper* limit

This gives a direct, intuitive mirror: the angle the finger bends is
proportional to the joint angle.

Design choices
--------------
* **EMA smoothing** (configurable alpha) on every output channel to
  suppress high-frequency noise from the ML tracker.
* **Dead-zone** on the *raw normalised input*: if the change from the last
  accepted sample is below ``dead_zone``, we reuse the previous EMA output
  without advancing the filter.  This prevents microscopically-noisy stable
  poses from walking the joint.
* **Curl threshold**: finger curls below ``curl_straight_threshold`` are
  treated as 0 (straight), so the joint rests at its lower limit when the
  finger is roughly open.  This eliminates drift when the hand is flat.
"""

from __future__ import annotations

import dataclasses

from openarm_mujoco.hand_tracker import GestureState


# ---------------------------------------------------------------------------
# Output data class (unchanged public API)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JointTargets:
    """Target joint positions computed from a single control frame.

    ``joint_positions`` is a list of length 7 (joints 1–7).  Entries that
    could not be determined from the gesture are set to *None*.
    """

    joint_positions: list[float | None]
    """Target for each of the 7 arm joints (radians).  *None* = no change."""

    gripper_position: float | None
    """Target for the gripper joint (radians).  *None* = no change."""


# ---------------------------------------------------------------------------
# Finger → joint mapping table
# ---------------------------------------------------------------------------

# Each entry: (camera_hand_side, GestureState_attribute_name)
# Ordered by joint index J1 … J7.
_FINGER_JOINT_MAP: list[tuple[str, str]] = [
    ("right", "thumb_curl"),   # J1 — base yaw
    ("right", "index_curl"),   # J2 — shoulder pitch
    ("right", "middle_curl"),  # J3 — upper-arm rotation
    ("right", "ring_curl"),    # J4 — elbow flex / extend
    ("right", "pinky_curl"),   # J5 — wrist pitch
    ("left",  "index_curl"),   # J6 — wrist roll
    ("left",  "middle_curl"),  # J7 — fine end-effector twist
]


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class FingerJointMapper:
    """Convert per-finger curl values from both camera hands into joint targets.

    Parameters
    ----------
    joint_limits
        Sequence of ``(lower, upper)`` tuples for each of the 7 arm joints.
    gripper_limits
        ``(lower, upper)`` for the independent gripper joint.
    smoothing_alpha
        EMA coefficient in (0, 1].  Higher → more responsive (less smooth),
        lower → smoother (more lag).  ``1.0`` disables smoothing.
        Recommended: 0.08–0.15 for finger teleop.
    dead_zone
        Minimum change in the *normalised* finger curl [0, 1] before a new
        target is produced.  Suppresses jitter when a finger is roughly still.
        Recommended: 0.015–0.03.
    curl_straight_threshold
        Finger curl values below this are treated as 0 (hand fully open).
        Eliminates baseline drift caused by imperfectly flat fingers.
        Recommended: 0.10–0.20.
    """

    def __init__(
        self,
        joint_limits: list[tuple[float, float]],
        gripper_limits: tuple[float, float],
        smoothing_alpha: float = 0.10,
        dead_zone: float = 0.02,
        curl_straight_threshold: float = 0.12,
    ) -> None:
        if len(joint_limits) != 7:
            raise ValueError(
                f"Expected 7 joint limits, got {len(joint_limits)}"
            )
        self._joint_limits    = list(joint_limits)
        self._gripper_limits  = gripper_limits
        self._alpha           = smoothing_alpha
        self._dead_zone       = dead_zone
        self._curl_threshold  = curl_straight_threshold

        # Internal EMA state — *None* until the first update
        self._smoothed: list[float | None] = [None] * 7
        self._prev_raw: list[float | None] = [None] * 7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, gestures: dict[str, GestureState]) -> JointTargets:
        """Compute new joint targets from the latest per-hand gesture states.

        Parameters
        ----------
        gestures
            Dict mapping ``"left"`` / ``"right"`` to their respective
            :class:`~openarm_mujoco.hand_tracker.GestureState`.
            Missing hands produce *None* targets for their joints (hold last).
        """
        targets: list[float | None] = [None] * 7

        for joint_idx, (hand_side, finger_attr) in enumerate(_FINGER_JOINT_MAP):
            gesture = gestures.get(hand_side)
            if gesture is None:
                # Hand not visible — hold last known position
                targets[joint_idx] = self._smoothed[joint_idx]
                continue

            # Raw curl from the appropriate finger attribute
            raw_curl: float = getattr(gesture, finger_attr)

            # Apply straight-finger threshold (treat near-zero curl as 0)
            if raw_curl < self._curl_threshold:
                raw_curl = 0.0

            # Normalise to [0, 1] after threshold removal
            # Remap: [threshold, 1] → [0, 1] only when curl > threshold
            if raw_curl > 0.0:
                raw_curl = _clamp01(
                    (raw_curl - self._curl_threshold)
                    / (1.0 - self._curl_threshold)
                )

            # Dead-zone check
            prev = self._prev_raw[joint_idx]
            if prev is not None and abs(raw_curl - prev) < self._dead_zone:
                targets[joint_idx] = self._smoothed[joint_idx]
                continue

            self._prev_raw[joint_idx] = raw_curl

            # Map normalised [0, 1] → joint angle range [lo, hi]
            lo, hi = self._joint_limits[joint_idx]
            mapped = lo + raw_curl * (hi - lo)

            # EMA smoothing
            if self._smoothed[joint_idx] is None:
                self._smoothed[joint_idx] = mapped
            else:
                self._smoothed[joint_idx] = (
                    self._alpha * mapped
                    + (1 - self._alpha) * self._smoothed[joint_idx]
                )
            targets[joint_idx] = self._smoothed[joint_idx]

        return JointTargets(
            joint_positions=targets,
            gripper_position=None,  # Gripper controlled via keyboard (G/H)
        )

    def reset(self) -> None:
        """Clear all internal smoothing state."""
        self._smoothed = [None] * 7
        self._prev_raw = [None] * 7

    def seed(self, joint_values: list[float]) -> None:
        """Pre-fill EMA with known joint values to prevent startup jump.

        Parameters
        ----------
        joint_values
            Current actual joint angles (length 7), in radians.
        """
        for i, val in enumerate(joint_values[:7]):
            lo, hi = self._joint_limits[i]
            norm = (val - lo) / (hi - lo) if (hi - lo) > 0 else 0.0
            self._smoothed[i] = val
            self._prev_raw[i] = _clamp01(norm)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return max(0.0, min(1.0, v))
