"""
tests/test_fusion.py — Unit tests for MultimodalFusion.
"""

import unittest
from src.fusion.multimodal_fusion import MultimodalFusion


class TestMultimodalFusionInit(unittest.TestCase):
    """Tests for MultimodalFusion.__init__."""

    def test_default_weights(self):
        """Default weights should be 0.6 visual + 0.4 audio."""
        fusion = MultimodalFusion()
        self.assertAlmostEqual(fusion.visual_weight, 0.6)
        self.assertAlmostEqual(fusion.audio_weight, 0.4)

    def test_custom_weights(self):
        """Custom valid weights should be stored correctly."""
        fusion = MultimodalFusion(visual_weight=0.7, audio_weight=0.3)
        self.assertAlmostEqual(fusion.visual_weight, 0.7)
        self.assertAlmostEqual(fusion.audio_weight, 0.3)

    def test_invalid_weights_raise_value_error(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        with self.assertRaises(ValueError):
            MultimodalFusion(visual_weight=0.5, audio_weight=0.3)  # sum=0.8

    def test_weight_sum_tolerance(self):
        """Weights summing to 1.001 should pass tolerance check."""
        fusion = MultimodalFusion(visual_weight=0.601, audio_weight=0.4)
        self.assertIsNotNone(fusion)


class TestFuseSignals(unittest.TestCase):
    """Tests for MultimodalFusion.fuse_signals."""

    def setUp(self):
        self.fusion = MultimodalFusion()

    def test_fuse_returns_dict(self):
        """fuse_signals should return a dict."""
        result = self.fusion.fuse_signals(70.0, 0.8)
        self.assertIsInstance(result, dict)

    def test_fuse_required_keys(self):
        """Result dict should contain all required keys."""
        result = self.fusion.fuse_signals(60.0, 0.5)
        for key in ("visual_score", "audio_level", "audio_score", "fused_score", "level", "timestamp"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_fuse_high_engagement(self):
        """High visual + high audio should produce high fused score."""
        result = self.fusion.fuse_signals(90.0, 1.0)
        self.assertGreaterEqual(result["fused_score"], 70.0)
        self.assertEqual(result["level"], "high")

    def test_fuse_low_engagement(self):
        """Low visual + low audio should produce low fused score."""
        result = self.fusion.fuse_signals(10.0, 0.0)
        self.assertLess(result["fused_score"], 40.0)
        self.assertEqual(result["level"], "low")

    def test_fuse_medium_engagement(self):
        """Moderate values should produce medium engagement."""
        result = self.fusion.fuse_signals(50.0, 0.5)
        self.assertGreaterEqual(result["fused_score"], 40.0)
        self.assertLess(result["fused_score"], 70.0)
        self.assertEqual(result["level"], "medium")

    def test_fuse_clamps_values(self):
        """Out-of-range inputs should be clamped (not raise)."""
        result = self.fusion.fuse_signals(150.0, 2.0)
        self.assertLessEqual(result["fused_score"], 100.0)
        self.assertGreaterEqual(result["fused_score"], 0.0)

    def test_fuse_correct_math(self):
        """Verify weighted formula: 0.6*visual + 0.4*(audio*100)."""
        result = self.fusion.fuse_signals(80.0, 0.5)
        expected = 0.6 * 80.0 + 0.4 * (0.5 * 100.0)  # = 48 + 20 = 68
        self.assertAlmostEqual(result["fused_score"], expected, places=1)

    def test_fuse_accumulates_in_buffer(self):
        """Signals should accumulate in the internal buffer."""
        self.fusion.clear_buffer()
        self.fusion.fuse_signals(60.0, 0.5)
        self.fusion.fuse_signals(70.0, 0.6)
        self.assertEqual(len(self.fusion.buffered_signals), 2)


class TestAggregateWindow(unittest.TestCase):
    """Tests for MultimodalFusion.aggregate_window."""

    def setUp(self):
        self.fusion = MultimodalFusion()

    def _make_signals(self, n=10):
        """Generate n synthetic signals."""
        signals = []
        for i in range(n):
            s = self.fusion.fuse_signals(50.0 + i, 0.5)
            signals.append(s)
        return signals

    def test_aggregate_returns_dict(self):
        """aggregate_window should return a dict."""
        signals = self._make_signals(5)
        result = self.fusion.aggregate_window(signals)
        self.assertIsInstance(result, dict)

    def test_aggregate_empty_returns_zeros(self):
        """Empty signal list should return zero stats."""
        result = self.fusion.aggregate_window([])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["fused_mean"], 0.0)

    def test_aggregate_required_keys(self):
        """Result must contain all summary keys."""
        signals = self._make_signals(5)
        result = self.fusion.aggregate_window(signals)
        for key in ("count", "visual_mean", "audio_mean", "fused_mean", "fused_max", "fused_min", "dominant_level"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_aggregate_count_matches(self):
        """count should equal the length of the input signals list."""
        signals = self._make_signals(7)
        result = self.fusion.aggregate_window(signals)
        self.assertEqual(result["count"], 7)

    def test_aggregate_max_gte_mean_gte_min(self):
        """max ≥ mean ≥ min should always hold."""
        signals = self._make_signals(10)
        result = self.fusion.aggregate_window(signals)
        self.assertGreaterEqual(result["fused_max"], result["fused_mean"])
        self.assertGreaterEqual(result["fused_mean"], result["fused_min"])


class TestGenerateAttentionCurve(unittest.TestCase):
    """Tests for MultimodalFusion.generate_attention_curve."""

    def setUp(self):
        self.fusion = MultimodalFusion()

    def test_curve_has_same_length_as_input(self):
        """Attention curve should have one point per input signal."""
        self.fusion.clear_buffer()
        for _ in range(5):
            self.fusion.fuse_signals(60.0, 0.5)
        curve = self.fusion.generate_attention_curve(self.fusion.buffered_signals)
        self.assertEqual(len(curve), 5)

    def test_curve_points_have_required_keys(self):
        """Each curve point should have x, y, level, visual, audio keys."""
        self.fusion.clear_buffer()
        self.fusion.fuse_signals(60.0, 0.5)
        curve = self.fusion.generate_attention_curve(self.fusion.buffered_signals)
        for point in curve:
            for key in ("x", "y", "level", "visual", "audio"):
                self.assertIn(key, point)

    def test_empty_timeline_returns_empty_curve(self):
        """Empty timeline should produce an empty curve."""
        curve = self.fusion.generate_attention_curve([])
        self.assertEqual(curve, [])


class TestClassify(unittest.TestCase):
    """Tests for MultimodalFusion._classify."""

    def test_classify_high(self):
        self.assertEqual(MultimodalFusion._classify(75.0), "high")

    def test_classify_medium(self):
        self.assertEqual(MultimodalFusion._classify(55.0), "medium")

    def test_classify_low(self):
        self.assertEqual(MultimodalFusion._classify(20.0), "low")

    def test_classify_boundary_70_is_high(self):
        self.assertEqual(MultimodalFusion._classify(70.0), "high")

    def test_classify_boundary_40_is_medium(self):
        self.assertEqual(MultimodalFusion._classify(40.0), "medium")

    def test_classify_boundary_39_is_low(self):
        self.assertEqual(MultimodalFusion._classify(39.9), "low")


if __name__ == "__main__":
    unittest.main()
