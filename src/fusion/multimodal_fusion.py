"""
src/fusion/multimodal_fusion.py — Multimodal signal fusion for engagement scoring.

Combines visual engagement scores and audio interaction levels into a
single unified engagement metric using configurable weights, then provides
windowed aggregation and attention-curve generation for charting.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    Fuses visual and audio engagement signals into a unified score.

    The final score is a weighted linear combination:
        fused = visual_weight * visual_score + audio_weight * (audio_level * 100)

    where the weights sum to 1.0.
    """

    def __init__(
        self, visual_weight: float = 0.6, audio_weight: float = 0.4
    ) -> None:
        """
        Initialise the fusion engine.

        Args:
            visual_weight: Weight for the visual engagement score (0–1).
            audio_weight: Weight for the audio interaction level (0–1).

        Raises:
            ValueError: If weights do not sum to 1.0 (within tolerance).
        """
        total = visual_weight + audio_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"visual_weight + audio_weight must equal 1.0 (got {total:.3f})"
            )
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self._signal_buffer: list[dict[str, Any]] = []

    # ── Public Methods ────────────────────────────────────────────────────────

    def fuse_signals(
        self,
        visual_score: float,
        audio_level: float,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Fuse a visual score and audio level into a unified engagement signal.

        Args:
            visual_score: Visual engagement score in [0, 100].
            audio_level: Audio interaction level in [0.0, 1.0].
            timestamp: UTC datetime of the measurement. Defaults to now.

        Returns:
            dict with keys:
                - ``visual_score`` (float)
                - ``audio_level`` (float)
                - ``audio_score`` (float): audio_level normalised to [0, 100]
                - ``fused_score`` (float): weighted combination in [0, 100]
                - ``level`` (str): "high" | "medium" | "low"
                - ``timestamp`` (str): ISO 8601 string
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        visual_score = float(np.clip(visual_score, 0.0, 100.0))
        audio_level = float(np.clip(audio_level, 0.0, 1.0))
        audio_score = audio_level * 100.0

        fused = self.visual_weight * visual_score + self.audio_weight * audio_score
        fused = float(np.clip(fused, 0.0, 100.0))

        level = self._classify(fused)

        signal: dict[str, Any] = {
            "visual_score": round(visual_score, 2),
            "audio_level": round(audio_level, 4),
            "audio_score": round(audio_score, 2),
            "fused_score": round(fused, 2),
            "level": level,
            "timestamp": timestamp.isoformat(),
        }
        self._signal_buffer.append(signal)
        return signal

    def aggregate_window(
        self,
        signals: list[dict[str, Any]],
        window_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Aggregate a list of signals into a statistical summary over a time window.

        Args:
            signals: List of signal dicts from ``fuse_signals()``.
            window_seconds: Time window size in seconds (for labelling only).

        Returns:
            dict with aggregate statistics:
                - ``count`` (int): Number of data points.
                - ``visual_mean``, ``audio_mean``, ``fused_mean`` (float)
                - ``fused_max``, ``fused_min``, ``fused_std`` (float)
                - ``dominant_level`` (str): Most common engagement level.
                - ``window_seconds`` (int)
        """
        if not signals:
            return {
                "count": 0,
                "visual_mean": 0.0,
                "audio_mean": 0.0,
                "fused_mean": 0.0,
                "fused_max": 0.0,
                "fused_min": 0.0,
                "fused_std": 0.0,
                "dominant_level": "unknown",
                "window_seconds": window_seconds,
            }

        visual_scores = np.array([s["visual_score"] for s in signals], dtype=float)
        audio_scores = np.array([s["audio_score"] for s in signals], dtype=float)
        fused_scores = np.array([s["fused_score"] for s in signals], dtype=float)
        levels = [s["level"] for s in signals]

        dominant_level = max(set(levels), key=levels.count)

        return {
            "count": len(signals),
            "visual_mean": round(float(visual_scores.mean()), 2),
            "audio_mean": round(float(audio_scores.mean()), 2),
            "fused_mean": round(float(fused_scores.mean()), 2),
            "fused_max": round(float(fused_scores.max()), 2),
            "fused_min": round(float(fused_scores.min()), 2),
            "fused_std": round(float(fused_scores.std()), 2),
            "dominant_level": dominant_level,
            "window_seconds": window_seconds,
        }

    def generate_attention_curve(
        self, timeline: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Transform a list of fused signals into chart-ready attention curve data.

        Args:
            timeline: Ordered list of signal dicts from ``fuse_signals()``.

        Returns:
            List of dicts with keys: ``x`` (timestamp string), ``y`` (fused_score),
            ``level`` (engagement level), ``visual`` (visual score),
            ``audio`` (audio score).
        """
        curve: list[dict[str, Any]] = []
        for point in timeline:
            curve.append(
                {
                    "x": point.get("timestamp", ""),
                    "y": point.get("fused_score", 0.0),
                    "level": point.get("level", "unknown"),
                    "visual": point.get("visual_score", 0.0),
                    "audio": point.get("audio_score", 0.0),
                }
            )
        return curve

    def clear_buffer(self) -> None:
        """Clear the internal signal buffer."""
        self._signal_buffer.clear()

    @property
    def buffered_signals(self) -> list[dict[str, Any]]:
        """Return all signals accumulated in the internal buffer."""
        return list(self._signal_buffer)

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> str:
        """
        Classify a fused score into a categorical engagement level.

        Args:
            score: Fused engagement score in [0, 100].

        Returns:
            ``"high"``, ``"medium"``, or ``"low"``.
        """
        if score >= 70:
            return "high"
        if score >= 40:
            return "medium"
        return "low"
