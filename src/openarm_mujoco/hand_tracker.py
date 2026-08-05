"""Background hand-tracking using a webcam and MediaPipe Hands.

This module is intentionally **independent of MuJoCo** – it only captures
frames, detects hands, and exposes processed :class:`GestureState` objects
through a thread-safe interface.

Uses the **MediaPipe Tasks** API (mediapipe >= 1.0) with the
``HandLandmarker`` task running in ``VIDEO`` mode.

Per-finger control
------------------
Each :class:`GestureState` now exposes five individual finger-curl values
(``thumb_curl``, ``index_curl``, ``middle_curl``, ``ring_curl``,
``pinky_curl``) instead of a single aggregated ``finger_curl``.  These are
used by :class:`~openarm_mujoco.gesture_mapper.FingerJointMapper` to drive
each of the 7 robot joints directly from a single finger bend.

Dependencies
------------
* ``opencv-python >= 4.8``
* ``mediapipe >= 1.0``
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GestureState:
    """Processed gesture data for a single detected hand.

    All spatial values are **normalised to [0, 1]** relative to the camera
    frame (after horizontal flipping) unless noted otherwise.

    Per-finger curl fields
    ----------------------
    Each finger's curl is measured independently at its PIP joint
    (MCP→PIP→DIP angle, where 0 = straight and 1 = fully curled).
    The thumb uses CMC→MCP→IP joints instead.
    """

    handedness: str
    """``"left"`` or ``"right"`` — from the *user's* perspective."""

    wrist_x: float
    """Horizontal position of the wrist (0 = left edge, 1 = right edge)."""

    wrist_y: float
    """Vertical position of the wrist (0 = top edge, 1 = bottom edge)."""

    hand_scale: float
    """Proxy for Z-depth: distance between wrist and middle-finger MCP,
    normalised by frame diagonal.  Larger → hand is closer to the camera."""

    palm_roll: float
    """Roll angle of the palm in radians (rotation around wrist-to-MCP axis)."""

    palm_pitch: float
    """Pitch angle of the palm in radians (tilt forward / backward)."""

    thumb_curl: float
    """Curl of the thumb [0, 1].  0 = straight, 1 = fully bent.
    Measured at the CMC→MCP→IP joints."""

    index_curl: float
    """Curl of the index finger [0, 1].  0 = straight, 1 = fully bent.
    Measured at the MCP→PIP→DIP joints."""

    middle_curl: float
    """Curl of the middle finger [0, 1].  0 = straight, 1 = fully bent."""

    ring_curl: float
    """Curl of the ring finger [0, 1].  0 = straight, 1 = fully bent."""

    pinky_curl: float
    """Curl of the pinky finger [0, 1].  0 = straight, 1 = fully bent."""

    pinch_distance: float
    """Euclidean distance between thumb tip and index-finger tip,
    normalised by frame diagonal.  Small → pinching (gripper close)."""

    timestamp: float
    """``time.monotonic()`` at the moment the gesture was computed."""


# ---------------------------------------------------------------------------
# MediaPipe Tasks API aliases
# ---------------------------------------------------------------------------

_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode
_BaseOptions = mp.tasks.BaseOptions
_HandConnections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
_draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks

# MediaPipe landmark indices (same as the old HandLandmark enum)
# Wrist / palm
_WRIST             = 0
# Thumb (CMC=1, MCP=2, IP=3, TIP=4)
_THUMB_CMC         = 1
_THUMB_MCP         = 2
_THUMB_IP          = 3
_THUMB_TIP         = 4
# Index (MCP=5, PIP=6, DIP=7, TIP=8)
_INDEX_FINGER_MCP  = 5
_INDEX_FINGER_PIP  = 6
_INDEX_FINGER_DIP  = 7
_INDEX_FINGER_TIP  = 8
# Middle (MCP=9, PIP=10, DIP=11, TIP=12)
_MIDDLE_FINGER_MCP = 9
_MIDDLE_FINGER_PIP = 10
_MIDDLE_FINGER_DIP = 11
# Ring (MCP=13, PIP=14, DIP=15, TIP=16)
_RING_FINGER_MCP   = 13
_RING_FINGER_PIP   = 14
_RING_FINGER_DIP   = 15
# Pinky (MCP=17, PIP=18, DIP=19, TIP=20)
_PINKY_MCP         = 17
_PINKY_PIP         = 18
_PINKY_DIP         = 19

# Default model path (relative to project root)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"
)


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def _lm_to_px(landmark, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert a single NormalizedLandmark to pixel-space (x, y)."""
    return np.array([landmark.x * frame_w, landmark.y * frame_h])


def _lm_to_3d(landmark, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert a single NormalizedLandmark to a 3-D vector (x, y, z)."""
    return np.array([
        landmark.x * frame_w,
        landmark.y * frame_h,
        landmark.z * frame_w,  # z is scaled the same as x in MediaPipe
    ])


def _compute_hand_scale(landmarks: list, frame_w: int, frame_h: int) -> float:
    """Return wrist↔middle-finger-MCP distance normalised by frame diagonal."""
    wrist = _lm_to_px(landmarks[_WRIST], frame_w, frame_h)
    mcp   = _lm_to_px(landmarks[_MIDDLE_FINGER_MCP], frame_w, frame_h)
    diag  = math.hypot(frame_w, frame_h)
    return float(np.linalg.norm(mcp - wrist) / diag) if diag else 0.0


def _compute_palm_orientation(
    landmarks: list, frame_w: int, frame_h: int
) -> tuple[float, float]:
    """Estimate palm roll and pitch from three coplanar landmarks.

    Uses wrist (0), index-finger MCP (5) and pinky MCP (17) to form two
    edge vectors.  Their cross product gives the palm normal, from which
    roll and pitch are extracted.

    Returns
    -------
    (roll, pitch) in radians.
    """
    p0  = _lm_to_3d(landmarks[_WRIST],            frame_w, frame_h)
    p5  = _lm_to_3d(landmarks[_INDEX_FINGER_MCP],  frame_w, frame_h)
    p17 = _lm_to_3d(landmarks[_PINKY_MCP],         frame_w, frame_h)

    v1 = p5  - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return 0.0, 0.0
    normal /= norm

    # Roll  = atan2(nx, nz) — tilt left / right
    roll  = math.atan2(normal[0], normal[2])
    # Pitch = asin(clamped ny) — tilt forward / back
    pitch = math.asin(float(np.clip(normal[1], -1.0, 1.0)))
    return roll, pitch


def _compute_per_finger_curls(
    landmarks: list, frame_w: int, frame_h: int
) -> dict[str, float]:
    """Compute curl [0, 1] independently for each of the 5 fingers.

    For each finger we measure the angle at the middle joint
    (MCP→PIP→DIP for index–pinky, CMC→MCP→IP for the thumb) in 3-D.
    A straight finger gives angle ≈ π (180°) → curl = 0.
    A fully bent finger gives angle ≈ 0        → curl = 1.

    Returns
    -------
    dict with keys ``"thumb"``, ``"index"``, ``"middle"``, ``"ring"``,
    ``"pinky"``; values in [0, 1].
    """
    # (proximal, middle, distal) landmark indices per finger
    # Thumb uses CMC(1)→MCP(2)→IP(3); others use MCP→PIP→DIP
    finger_groups: dict[str, tuple[int, int, int]] = {
        "thumb":  (_THUMB_CMC, _THUMB_MCP, _THUMB_IP),
        "index":  (_INDEX_FINGER_MCP, _INDEX_FINGER_PIP, _INDEX_FINGER_DIP),
        "middle": (_MIDDLE_FINGER_MCP, _MIDDLE_FINGER_PIP, _MIDDLE_FINGER_DIP),
        "ring":   (_RING_FINGER_MCP, _RING_FINGER_PIP, _RING_FINGER_DIP),
        "pinky":  (_PINKY_MCP, _PINKY_PIP, _PINKY_DIP),
    }
    result: dict[str, float] = {}
    for name, (prox_i, mid_i, dist_i) in finger_groups.items():
        prox = _lm_to_3d(landmarks[prox_i], frame_w, frame_h)
        mid  = _lm_to_3d(landmarks[mid_i],  frame_w, frame_h)
        dist = _lm_to_3d(landmarks[dist_i], frame_w, frame_h)

        v1 = prox - mid
        v2 = dist - mid
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            result[name] = 0.0
            continue

        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = math.acos(cos_angle)          # 0 (bent) … π (straight)
        result[name] = 1.0 - angle / math.pi  # 0 = straight, 1 = fully curled

    return result


def _compute_pinch_distance(
    landmarks: list, frame_w: int, frame_h: int
) -> float:
    """Thumb-tip ↔ index-tip distance normalised by frame diagonal."""
    thumb = _lm_to_px(landmarks[_THUMB_TIP],        frame_w, frame_h)
    index = _lm_to_px(landmarks[_INDEX_FINGER_TIP],  frame_w, frame_h)
    diag  = math.hypot(frame_w, frame_h)
    return float(np.linalg.norm(index - thumb) / diag) if diag else 0.0


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class HandTracker:
    """Captures webcam frames and detects hand landmarks via MediaPipe.

    Designed to run in a **background daemon thread** so the MuJoCo
    simulation loop is never blocked by camera I/O or ML inference.

    Parameters
    ----------
    camera_index
        ``cv2.VideoCapture`` device index (default ``0``).
    show_feed
        If *True*, an OpenCV window showing the annotated camera feed is
        displayed while tracking is active.
    model_path
        Path to the ``hand_landmarker.task`` model file.  Defaults to
        ``<project>/models/hand_landmarker.task``.
    """

    def __init__(
        self,
        camera_index: int = 0,
        show_feed: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._show_feed = show_feed
        self._model_path = str(model_path or _DEFAULT_MODEL_PATH)

        # Created lazily on start()
        self._cap: Optional[cv2.VideoCapture] = None
        self._landmarker: Optional[_HandLandmarker] = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_gestures: dict[str, GestureState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the camera and begin tracking in a background thread.

        Returns *True* on success, *False* if the camera cannot be opened.
        """
        if self._running:
            return True

        # Validate model file
        if not Path(self._model_path).is_file():
            print(
                f"  [WARNING] Hand landmark model not found: {self._model_path}\n"
                f"     Download it from:\n"
                f"     https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/"
                f"hand_landmarker.task"
            )
            return False

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f"  [WARNING] Cannot open camera index {self._camera_index}")
            cap.release()
            return False

        # Request a sensible resolution for faster inference
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._cap = cap

        # Create HandLandmarker in VIDEO mode (synchronous per-frame)
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=self._model_path),
            running_mode=_RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)

        self._running = True
        self._thread = threading.Thread(
            target=self._tracking_loop, daemon=True, name="hand-tracker"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the tracking thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._show_feed:
            cv2.destroyWindow("Hand Gesture Control")
        with self._lock:
            self._latest_gestures.clear()

    def get_gestures(self) -> dict[str, GestureState]:
        """Return the latest gesture state for each detected hand.

        Keys are ``"left"`` and/or ``"right"`` (from the user's perspective).
        Thread-safe — may be called from the main simulation thread.
        """
        with self._lock:
            return dict(self._latest_gestures)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        """Continuously capture frames and run hand detection."""
        assert self._cap is not None
        assert self._landmarker is not None

        # Use real wall-clock milliseconds for VIDEO mode timestamps.
        # MediaPipe VIDEO mode requires monotonically increasing timestamps
        # and the increment must reflect the actual elapsed time, otherwise
        # the internal tracker resets itself every few frames (causing the
        # "hand jumps back" jitter symptom).
        start_wall = time.monotonic()

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            # Mirror the image so the user's left hand appears on the left
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )

            # Use actual elapsed time (ms) as the frame timestamp.
            # This keeps MediaPipe's VIDEO mode happy and prevents its
            # internal Kalman filter from resetting between frames.
            frame_ts_ms = int((time.monotonic() - start_wall) * 1000)
            result = self._landmarker.detect_for_video(
                mp_image, frame_ts_ms
            )

            new_gestures: dict[str, GestureState] = {}

            if result.hand_landmarks and result.handedness:
                for hand_lms, handedness_list in zip(
                    result.hand_landmarks, result.handedness
                ):
                    # MediaPipe labels from the camera's perspective.
                    # After horizontal flip, MediaPipe "Right" → user's left.
                    mp_label = handedness_list[0].category_name
                    side = "left" if mp_label == "Right" else "right"

                    now = time.monotonic()
                    roll, pitch = _compute_palm_orientation(hand_lms, w, h)
                    curls = _compute_per_finger_curls(hand_lms, w, h)

                    gesture = GestureState(
                        handedness=side,
                        wrist_x=hand_lms[_WRIST].x,
                        wrist_y=hand_lms[_WRIST].y,
                        hand_scale=_compute_hand_scale(hand_lms, w, h),
                        palm_roll=roll,
                        palm_pitch=pitch,
                        thumb_curl=curls["thumb"],
                        index_curl=curls["index"],
                        middle_curl=curls["middle"],
                        ring_curl=curls["ring"],
                        pinky_curl=curls["pinky"],
                        pinch_distance=_compute_pinch_distance(hand_lms, w, h),
                        timestamp=now,
                    )
                    new_gestures[side] = gesture

                    # Draw landmarks on the frame for the feed window
                    if self._show_feed:
                        _draw_landmarks(
                            frame,
                            hand_lms,
                            _HandConnections,
                        )

            with self._lock:
                self._latest_gestures = new_gestures

            if self._show_feed:
                # Overlay: detected hands + live per-finger curl values
                if new_gestures:
                    y_off = 30
                    for side, g in new_gestures.items():
                        # Header row with hand label
                        header_color = (0, 220, 255) if side == "right" else (255, 180, 0)
                        cv2.putText(
                            frame,
                            f"{side.upper()} HAND",
                            (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            header_color, 2, cv2.LINE_AA,
                        )
                        y_off += 22
                        # Per-finger curl bars
                        finger_data = [
                            ("Thumb",  g.thumb_curl,  "J1" if side == "right" else "--"),
                            ("Index",  g.index_curl,  "J2" if side == "right" else "J6"),
                            ("Middle", g.middle_curl, "J3" if side == "right" else "J7"),
                            ("Ring",   g.ring_curl,   "J4" if side == "right" else "--"),
                            ("Pinky",  g.pinky_curl,  "J5" if side == "right" else "--"),
                        ]
                        for fname, curl, jlabel in finger_data:
                            label = f"{fname:<6} [{jlabel}]: {curl:.2f}"
                            cv2.putText(
                                frame, label, (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 255, 128), 1, cv2.LINE_AA,
                            )
                            # Mini progress bar
                            bar_x, bar_y = 160, y_off - 10
                            bar_w = int(curl * 80)
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 80, bar_y + 10), (60, 60, 60), -1)
                            if bar_w > 0:
                                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), header_color, -1)
                            y_off += 18
                        y_off += 6  # gap between hands
                else:
                    cv2.putText(
                        frame, "No hands detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Hand Gesture Control", frame)
                cv2.waitKey(1)

        # Cleanup when loop exits
        if self._show_feed:
            cv2.destroyWindow("Hand Gesture Control")
