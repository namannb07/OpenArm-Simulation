"""Background hand-tracking using a webcam and MediaPipe Hands.

This module is intentionally **independent of MuJoCo** – it only captures
frames, detects hands, and exposes processed :class:`GestureState` objects
through a thread-safe interface.

Uses the **MediaPipe Tasks** API (mediapipe >= 1.0) with the
``HandLandmarker`` task running in ``VIDEO`` mode.

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

    pinch_distance: float
    """Euclidean distance between thumb tip and index-finger tip,
    normalised by frame diagonal.  Small → pinching."""

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
_WRIST = 0
_THUMB_TIP = 4
_INDEX_FINGER_MCP = 5
_INDEX_FINGER_TIP = 8
_MIDDLE_FINGER_MCP = 9
_PINKY_MCP = 17

# Default model path (relative to project root)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def _lm_to_px(
    landmark, frame_w: int, frame_h: int
) -> np.ndarray:
    """Convert a single NormalizedLandmark to pixel-space (x, y)."""
    return np.array([landmark.x * frame_w, landmark.y * frame_h])


def _compute_hand_scale(
    landmarks: list, frame_w: int, frame_h: int
) -> float:
    """Return wrist↔middle-finger-MCP distance normalised by frame diagonal."""
    wrist = _lm_to_px(landmarks[_WRIST], frame_w, frame_h)
    mcp = _lm_to_px(landmarks[_MIDDLE_FINGER_MCP], frame_w, frame_h)
    diag = math.hypot(frame_w, frame_h)
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
    def _to_3d(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([
            lm.x * frame_w,
            lm.y * frame_h,
            lm.z * frame_w,  # z is scaled like x in MediaPipe
        ])

    p0 = _to_3d(_WRIST)
    p5 = _to_3d(_INDEX_FINGER_MCP)
    p17 = _to_3d(_PINKY_MCP)

    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return 0.0, 0.0
    normal /= norm

    # Roll  = atan2(nx, nz) — tilt left / right
    roll = math.atan2(normal[0], normal[2])
    # Pitch = asin(clamped ny) — tilt forward / back
    pitch = math.asin(float(np.clip(normal[1], -1.0, 1.0)))
    return roll, pitch


def _compute_pinch_distance(
    landmarks: list, frame_w: int, frame_h: int
) -> float:
    """Thumb-tip ↔ index-tip distance normalised by frame diagonal."""
    thumb = _lm_to_px(landmarks[_THUMB_TIP], frame_w, frame_h)
    index = _lm_to_px(landmarks[_INDEX_FINGER_TIP], frame_w, frame_h)
    diag = math.hypot(frame_w, frame_h)
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
                f"  ⚠  Hand landmark model not found: {self._model_path}\n"
                f"     Download it from:\n"
                f"     https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/"
                f"hand_landmarker.task"
            )
            return False

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f"  ⚠  Cannot open camera index {self._camera_index}")
            cap.release()
            return False

        self._cap = cap

        # Create HandLandmarker in VIDEO mode (synchronous per-frame)
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=self._model_path),
            running_mode=_RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
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

        frame_ts_ms = 0  # Monotonically increasing timestamp for VIDEO mode

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Mirror the image so the user's left hand appears on the left
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )

            # Detect hands (VIDEO mode — synchronous)
            frame_ts_ms += 33  # ~30 fps timestamp increment
            result = self._landmarker.detect_for_video(
                mp_image, frame_ts_ms
            )

            new_gestures: dict[str, GestureState] = {}

            if result.hand_landmarks and result.handedness:
                for hand_lms, handedness_list in zip(
                    result.hand_landmarks, result.handedness
                ):
                    # handedness_list is a list of Category objects
                    # MediaPipe labels from the camera's perspective.
                    # After horizontal flip, MediaPipe "Right" → user's left.
                    mp_label = handedness_list[0].category_name
                    side = "left" if mp_label == "Right" else "right"

                    now = time.monotonic()
                    roll, pitch = _compute_palm_orientation(hand_lms, w, h)

                    gesture = GestureState(
                        handedness=side,
                        wrist_x=hand_lms[_WRIST].x,
                        wrist_y=hand_lms[_WRIST].y,
                        hand_scale=_compute_hand_scale(hand_lms, w, h),
                        palm_roll=roll,
                        palm_pitch=pitch,
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
                # Add status overlay
                status = "Hands: " + (
                    ", ".join(new_gestures.keys()) if new_gestures
                    else "none detected"
                )
                cv2.putText(
                    frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Hand Gesture Control", frame)
                cv2.waitKey(1)

        # Cleanup when loop exits
        if self._show_feed:
            cv2.destroyWindow("Hand Gesture Control")
