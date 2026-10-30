"""
src/vision/engagement_analyzer.py — Visual engagement scoring from video frames.

Computes per-frame engagement scores by combining head pose data,
face count changes, and motion analysis into a 0–100 scale score.
"""

import logging
import math
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# How many frames of history to keep for motion analysis
_HISTORY_LEN = 30


class EngagementAnalyzer:
    """
    Computes engagement scores from visual signals in video frames.

    Combines:
    - Head pose (pitch/yaw deviations from neutral)
    - Face count stability (sudden drops indicate distraction)
    - Optical-flow motion (excessive movement = distraction)

    All scoring is privacy-preserving: no individual identification occurs.
    """

    def __init__(self) -> None:
        """Initialise the engagement analyser with an empty history buffer."""
        self._prev_frame_gray: np.ndarray | None = None
        self._motion_history: deque[float] = deque(maxlen=_HISTORY_LEN)
        self._face_count_history: deque[int] = deque(maxlen=_HISTORY_LEN)

    # ── Public Methods ────────────────────────────────────────────────────────

    def analyze_frame(
        self, frame: np.ndarray, faces: list[dict[str, Any]]
    ) -> float:
        """
        Compute a visual engagement score (0–100) for a single frame.

        Args:
            frame: BGR numpy array from OpenCV.
            faces: List of face dicts from FaceTracker.detect_faces().

        Returns:
            Engagement score as a float in [0, 100].
        """
        if frame is None or frame.size == 0:
            return 50.0

        face_count = len(faces)
        self._face_count_history.append(face_count)

        # --- Component 1: motion score (lower motion = higher score) ---
        motion_score = self._compute_motion_score(frame)
        self._motion_history.append(motion_score)

        # --- Component 2: head pose score averaged across all faces ---
        pose_score = self._compute_pose_score(faces)

        # --- Component 3: face presence stability ---
        stability_score = self._compute_stability_score()

        # Weighted average (motion penalises excessive movement)
        raw = (
            0.35 * pose_score
            + 0.35 * stability_score
            + 0.30 * (100.0 - motion_score)  # invert: less motion = more engaged
        )

        return float(np.clip(raw, 0.0, 100.0))

    def detect_inactivity(
        self, pose_history: list[dict[str, float]], threshold_minutes: float = 2.0
    ) -> bool:
        """
        Detect whether a student appears inactive for a sustained period.

        Args:
            pose_history: List of head-pose dicts (pitch, yaw, roll) over time.
            threshold_minutes: Duration in minutes of near-zero movement to
                               classify as inactive.

        Returns:
            True if inactivity is detected.
        """
        if not pose_history:
            return False

        # Expect pose_history to represent ~1-second samples
        required_samples = int(threshold_minutes * 60)
        if len(pose_history) < required_samples:
            return False

        recent = pose_history[-required_samples:]
        pitches = [p.get("pitch", 0) for p in recent]
        yaws = [p.get("yaw", 0) for p in recent]

        pitch_var = float(np.var(pitches))
        yaw_var = float(np.var(yaws))

        # Near-zero variance in both axes for sustained period = inactive
        return pitch_var < 1.0 and yaw_var < 1.0

    def compute_session_trend(
        self, scores_timeline: list[float]
    ) -> dict[str, float]:
        """
        Compute aggregate statistics across a session's score timeline.

        Args:
            scores_timeline: Ordered list of engagement scores over a session.

        Returns:
            dict with keys: ``peak``, ``valley``, ``average``,
            ``std_dev``, ``trend`` (positive = improving, negative = declining).
        """
        if not scores_timeline:
            return {"peak": 0.0, "valley": 0.0, "average": 0.0, "std_dev": 0.0, "trend": 0.0}

        arr = np.array(scores_timeline, dtype=float)
        n = len(arr)

        # Simple linear regression for trend
        x = np.arange(n)
        if n > 1:
            slope = float(np.polyfit(x, arr, 1)[0])
        else:
            slope = 0.0

        return {
            "peak": float(arr.max()),
            "valley": float(arr.min()),
            "average": float(arr.mean()),
            "std_dev": float(arr.std()),
            "trend": round(slope * n, 2),  # total projected change over session
        }

    def classify_engagement_level(self, score: float) -> str:
        """
        Map a numeric score to a categorical engagement level.

        Args:
            score: Engagement score in [0, 100].

        Returns:
            ``"high"``, ``"medium"``, or ``"low"``.
        """
        if score >= 70:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _compute_motion_score(self, frame: np.ndarray) -> float:
        """
        Estimate overall scene motion via frame differencing (0–100).

        High motion score means lots of movement in the scene.
        """
        try:
            import cv2 as cv

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (5, 5), 0)

            if self._prev_frame_gray is None:
                self._prev_frame_gray = gray
                return 0.0

            diff = cv.absdiff(self._prev_frame_gray, gray)
            self._prev_frame_gray = gray

            # Normalise mean pixel difference to 0–100
            mean_diff = float(np.mean(diff))
            return float(np.clip(mean_diff * 5.0, 0.0, 100.0))
        except Exception:
            return 0.0

    def _compute_pose_score(self, faces: list[dict[str, Any]]) -> float:
        """
        Derive a head-pose-based attention score averaged across all faces.

        Looking straight ahead (low yaw deviation) → higher score.
        """
        if not faces:
            return 50.0  # Neutral when no faces detected

        scores: list[float] = []
        for face in faces:
            yaw = abs(face.get("yaw", 0.0))
            pitch = abs(face.get("pitch", 0.0))

            # Penalise large deviations from forward-facing pose
            yaw_penalty = min(yaw / 45.0, 1.0) * 40
            pitch_penalty = min(pitch / 30.0, 1.0) * 20
            score = 100.0 - yaw_penalty - pitch_penalty
            scores.append(max(score, 0.0))

        return float(np.mean(scores))

    def _compute_stability_score(self) -> float:
        """
        Score based on how stable the face count has been recently.

        Sudden drops or spikes in face count reduce the score.
        """
        if len(self._face_count_history) < 2:
            return 75.0

        counts = list(self._face_count_history)
        diffs = [abs(counts[i] - counts[i - 1]) for i in range(1, len(counts))]
        avg_diff = float(np.mean(diffs))

        # An avg_diff of 0 → 100 score; avg_diff ≥ 3 → 40 score
        score = 100.0 - (avg_diff / 3.0) * 60.0
        return float(np.clip(score, 40.0, 100.0))
