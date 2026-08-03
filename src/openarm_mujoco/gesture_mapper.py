"""Map hand gesture data to robot joint targets.

Translates :class:`~openarm_mujoco.hand_tracker.GestureState` objects
produced by the hand tracker into concrete joint position targets that can
be applied to the MuJoCo simulation.

Features
--------
* **Linear mapping** from normalised gesture space to each joint's range.
* **Dead-zone filtering** — ignores tiny movements to reduce jitter.
* **Exponential moving average (EMA) smoothing** — configurable alpha.
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

# Normalised hand-scale range observed empirically.  When the hand is
# very close to the camera the scale is ~0.25; far away ~0.06.
_SCALE_MIN = 0.06
_SCALE_MAX = 0.25

# Pinch threshold — below this value the gripper is considered "closed".
_PINCH_CLOSE_THRESHOLD = 0.04
_PINCH_OPEN_THRESHOLD = 0.07


class GestureMapper:
    """Convert normalised gesture data into joint-position targets.

    Parameters
    ----------
    joint_limits
        Sequence of ``(lower, upper)`` tuples for each of the 7 arm joints.
    gripper_limits
        ``(lower, upper)`` for the independent gripper joint.
    smoothing_alpha
        EMA coefficient in (0, 1].  Higher → more responsive, lower →
        smoother.  ``1.0`` disables smoothing entirely.
    dead_zone
        Minimum change in the normalised input required before a new
        target is produced.  Helps suppress jitter at rest.
    """

    def __init__(
        self,
        joint_limits: list[tuple[float, float]],
        gripper_limits: tuple[float, float],
        smoothing_alpha: float = 0.3,
        dead_zone: float = 0.03,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, gesture: GestureState) -> JointTargets:
        """Compute new joint targets from the latest gesture.

        Mapping
        -------
        =================  ===========  =====================================
        Gesture input      Joint        Description
        =================  ===========  =====================================
        ``wrist_x``        J1 (idx 0)   Base yaw — horizontal hand position
        ``wrist_y``        J2 (idx 1)   Shoulder pitch — vertical position
        ``hand_scale``     J3 (idx 2)   Elbow — hand proximity (depth proxy)
        *(unmapped)*       J4 (idx 3)   Twist — keyboard only
        ``palm_pitch``     J5 (idx 4)   Wrist pitch — palm tilt
        ``palm_roll``      J6 (idx 5)   Wrist roll — palm rotation
        *(unmapped)*       J7 (idx 6)   Fine rotation — keyboard only
        =================  ===========  =====================================
        ``pinch_distance``  Gripper     Open / close
        """
        raw_inputs = self._extract_raw(gesture)
        targets: list[float | None] = [None] * 7

        for i in range(7):
            raw = raw_inputs[i]
            if raw is None:
                # Unmapped joint — leave at current position
                targets[i] = None
                continue

            # Dead-zone check
            if self._prev_raw[i] is not None:
                if abs(raw - self._prev_raw[i]) < self._dead_zone:
                    # Movement too small — reuse previous smoothed value
                    targets[i] = self._smoothed[i]
                    continue
            self._prev_raw[i] = raw

            # Map [0, 1] → joint range
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, g: GestureState) -> list[float | None]:
        """Convert gesture fields to normalised [0, 1] values per joint.

        Returns a list of 7 elements.  *None* for unmapped joints.
        """
        # J1: wrist_x already in [0, 1]
        j1 = g.wrist_x

        # J2: wrist_y already in [0, 1] — invert so raising the hand
        #     lifts the shoulder (0 = top of frame → upper limit)
        j2 = 1.0 - g.wrist_y

        # J3: hand_scale → normalise to [0, 1]
        j3 = _clamp01(
            (g.hand_scale - _SCALE_MIN) / (_SCALE_MAX - _SCALE_MIN)
        )

        # J4: unmapped
        j4 = None

        # J5: palm_pitch (roughly ±π/2) → normalise to [0, 1]
        j5 = _clamp01((g.palm_pitch / math.pi) + 0.5)

        # J6: palm_roll (roughly ±π/2) → normalise to [0, 1]
        j6 = _clamp01((g.palm_roll / math.pi) + 0.5)

        # J7: unmapped
        j7 = None

        return [j1, j2, j3, j4, j5, j6, j7]

    def _compute_gripper(self, pinch_distance: float) -> float | None:
        """Hysteresis-based gripper target from pinch distance."""
        lo, hi = self._gripper_limits

        if pinch_distance < _PINCH_CLOSE_THRESHOLD:
            target = lo  # Fully closed
        elif pinch_distance > _PINCH_OPEN_THRESHOLD:
            target = hi  # Fully open
        else:
            # In the hysteresis band — interpolate
            t = (pinch_distance - _PINCH_CLOSE_THRESHOLD) / (
                _PINCH_OPEN_THRESHOLD - _PINCH_CLOSE_THRESHOLD
            )
            target = lo + t * (hi - lo)

        # Smooth the gripper as well
        if self._smoothed_grip is None:
            self._smoothed_grip = target
        else:
            self._smoothed_grip = (
                self._alpha * target
                + (1 - self._alpha) * self._smoothed_grip
            )
        return self._smoothed_grip


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return max(0.0, min(1.0, v))
