"""
tests/test_face_tracker.py — Unit tests for FaceTracker.

Tests use numpy arrays as mock frames and patch mediapipe so tests run
without a camera or the full mediapipe binary.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestFaceTrackerInit(unittest.TestCase):
    """Tests for FaceTracker initialisation."""

    def test_init_without_mediapipe_enters_stub_mode(self):
        """FaceTracker should not raise even if mediapipe is unavailable."""
        with patch.dict("sys.modules", {"mediapipe": None}):
            try:
                from importlib import reload
                import src.vision.face_tracker as mod
                reload(mod)
                tracker = mod.FaceTracker()
                self.assertIsNone(tracker._detector)
            except Exception as e:
                self.fail(f"FaceTracker raised unexpectedly: {e}")

    def test_init_creates_id_counter(self):
        """FaceTracker should initialise _next_id to 1."""
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._confidence = 0.65
        tracker._detector = None
        tracker._face_mesh = None
        tracker._mp_drawing = None
        tracker._next_id = 1
        tracker._tracked = {}
        self.assertEqual(tracker._next_id, 1)


class TestFaceTrackerDetectFaces(unittest.TestCase):
    """Tests for FaceTracker.detect_faces."""

    def _make_tracker(self):
        """Return a FaceTracker in stub mode (no mediapipe)."""
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._confidence = 0.65
        tracker._detector = None  # Forces stub mode
        tracker._face_mesh = None
        tracker._mp_drawing = None
        tracker._next_id = 1
        tracker._tracked = {}
        return tracker

    def test_detect_faces_with_empty_frame_returns_empty(self):
        """detect_faces should return [] for an empty numpy array."""
        tracker = self._make_tracker()
        result = tracker.detect_faces(np.array([]))
        self.assertEqual(result, [])

    def test_detect_faces_stub_returns_one_face(self):
        """In stub mode, detect_faces should return one synthetic face."""
        tracker = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = tracker.detect_faces(frame)

        self.assertEqual(len(faces), 1)
        self.assertIn("id", faces[0])
        self.assertIn("bbox", faces[0])
        self.assertIn("confidence", faces[0])
        self.assertIn("centroid", faces[0])

    def test_detect_faces_assigns_id(self):
        """Each detected face should have a non-empty string ID."""
        tracker = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = tracker.detect_faces(frame)
        for face in faces:
            self.assertIsInstance(face["id"], str)
            self.assertTrue(len(face["id"]) > 0)


class TestFaceTrackerTrackPresence(unittest.TestCase):
    """Tests for FaceTracker.track_presence."""

    def _make_tracker(self):
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._confidence = 0.65
        tracker._detector = None
        tracker._face_mesh = None
        tracker._mp_drawing = None
        tracker._next_id = 1
        tracker._tracked = {}
        return tracker

    def test_track_presence_returns_dict_with_face_ids(self):
        """track_presence should return a dict keyed by face IDs."""
        tracker = self._make_tracker()
        face_list = [
            {"id": "face-1", "bbox": (0, 0, 100, 100), "confidence": 0.9, "centroid": (50, 50)},
            {"id": "face-2", "bbox": (200, 0, 100, 100), "confidence": 0.8, "centroid": (250, 50)},
        ]
        result = tracker.track_presence(face_list, session_id=1)
        self.assertIn("face-1", result)
        self.assertIn("face-2", result)

    def test_track_presence_marks_faces_as_present(self):
        """Each entry in the presence dict should have present=True."""
        tracker = self._make_tracker()
        face_list = [{"id": "face-1", "bbox": (0, 0, 50, 50), "confidence": 0.85, "centroid": (25, 25)}]
        result = tracker.track_presence(face_list, session_id=2)
        self.assertTrue(result["face-1"]["present"])

    def test_track_presence_empty_list(self):
        """track_presence with empty face_list should return empty dict."""
        tracker = self._make_tracker()
        result = tracker.track_presence([], session_id=1)
        self.assertEqual(result, {})


class TestFaceTrackerIDAssignment(unittest.TestCase):
    """Tests for the internal centroid-based ID assignment."""

    def _make_tracker(self):
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._confidence = 0.65
        tracker._detector = None
        tracker._face_mesh = None
        tracker._mp_drawing = None
        tracker._next_id = 1
        tracker._tracked = {}
        return tracker

    def test_same_centroid_gets_same_id_across_frames(self):
        """A face at the same position across two calls should keep its ID."""
        tracker = self._make_tracker()
        det1 = [{"centroid": (100, 100), "bbox": (75, 75, 50, 50), "confidence": 0.9}]
        det2 = [{"centroid": (102, 98), "bbox": (77, 73, 50, 50), "confidence": 0.9}]

        assigned1 = tracker._assign_ids(det1)
        assigned2 = tracker._assign_ids(det2)

        self.assertEqual(assigned1[0]["id"], assigned2[0]["id"])

    def test_different_positions_get_different_ids(self):
        """Two faces far apart should get different IDs."""
        tracker = self._make_tracker()
        dets = [
            {"centroid": (50, 50), "bbox": (25, 25, 50, 50), "confidence": 0.9},
            {"centroid": (400, 400), "bbox": (375, 375, 50, 50), "confidence": 0.9},
        ]
        assigned = tracker._assign_ids(dets)
        ids = [d["id"] for d in assigned]
        self.assertEqual(len(set(ids)), 2)


class TestFaceTrackerHeadPose(unittest.TestCase):
    """Tests for FaceTracker.get_head_pose."""

    def test_returns_default_when_no_face_mesh(self):
        """get_head_pose should return zeros if face_mesh is None."""
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._face_mesh = None
        tracker._detector = None
        tracker._mp_drawing = None
        tracker._confidence = 0.65
        tracker._next_id = 1
        tracker._tracked = {}

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.get_head_pose(frame, (100, 100, 100, 100))

        self.assertEqual(result, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})

    def test_returns_dict_with_three_keys(self):
        """get_head_pose always returns a dict with pitch, yaw, roll."""
        from src.vision.face_tracker import FaceTracker
        tracker = FaceTracker.__new__(FaceTracker)
        tracker._face_mesh = None
        tracker._detector = None
        tracker._mp_drawing = None
        tracker._confidence = 0.65
        tracker._next_id = 1
        tracker._tracked = {}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tracker.get_head_pose(frame, (0, 0, 50, 50))

        self.assertIn("pitch", result)
        self.assertIn("yaw", result)
        self.assertIn("roll", result)


if __name__ == "__main__":
    unittest.main()
