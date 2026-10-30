"""
src/vision/face_tracker.py — Face detection and multi-person presence tracking.

Uses MediaPipe Face Detection for robust, lightweight face localisation
and a simple centroid-based tracker to assign stable IDs across frames.
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum pixel distance between centroids to consider them the same person
_MAX_CENTROID_DISTANCE = 100
# Number of frames before a tracked person is considered lost
_MAX_MISSING_FRAMES = 30


class FaceTracker:
    """
    Multi-person face detection and frame-to-frame tracking.

    Uses MediaPipe Face Detection for per-frame detection and a
    centroid-based approach to maintain consistent face IDs across frames,
    enabling reliable attendance tracking without biometric identification.
    """

    def __init__(self, min_detection_confidence: float = 0.65) -> None:
        """
        Initialise the face tracker.

        Args:
            min_detection_confidence: Minimum detection confidence threshold
                                      (0.0 – 1.0). Lower values detect more
                                      faces but may increase false positives.
        """
        self._confidence = min_detection_confidence
        self._detector = None
        self._mp_drawing = None
        self._face_mesh = None
        self._next_id = 1
        # Mapping: face_id -> {"centroid": (cx, cy), "missing_frames": int}
        self._tracked: dict[str, dict[str, Any]] = {}

        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=min_detection_confidence,
            )
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=30,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
            )
            self._mp_drawing = mp.solutions.drawing_utils
            logger.info("FaceTracker: MediaPipe initialised.")
        except ImportError:
            logger.warning("FaceTracker: MediaPipe not installed. Falling back to stub mode.")
        except Exception as exc:
            logger.error("FaceTracker: Init failed — %s", exc)

    # ── Public Methods ────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect all faces in a video frame.

        Args:
            frame: BGR numpy array (OpenCV frame).

        Returns:
            List of face dicts, each with keys:
                - ``id`` (str): Stable tracker ID.
                - ``bbox`` (tuple): (x, y, w, h) bounding box in pixels.
                - ``confidence`` (float): Detection confidence (0–1).
                - ``centroid`` (tuple): (cx, cy) centre of the bounding box.
        """
        if frame is None or frame.size == 0:
            return []

        if self._detector is None:
            # Stub: return a synthetic single face for testing
            h, w = frame.shape[:2]
            return [
                {
                    "id": "stub-1",
                    "bbox": (w // 4, h // 4, w // 2, h // 2),
                    "confidence": 0.9,
                    "centroid": (w // 2, h // 2),
                }
            ]

        import cv2 as cv

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        raw_detections: list[dict[str, Any]] = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                cx = x + bw // 2
                cy = y + bh // 2
                score = detection.score[0] if detection.score else 0.0
                raw_detections.append(
                    {
                        "bbox": (x, y, bw, bh),
                        "confidence": float(score),
                        "centroid": (cx, cy),
                    }
                )

        return self._assign_ids(raw_detections)

    def track_presence(
        self, face_list: list[dict[str, Any]], session_id: int
    ) -> dict[str, Any]:
        """
        Update the presence record for each detected face.

        Args:
            face_list: Output of ``detect_faces()``.
            session_id: Current session identifier (for logging).

        Returns:
            Dict mapping face_id -> presence info dict with keys:
                ``present`` (bool), ``confidence`` (float).
        """
        presence: dict[str, Any] = {}
        for face in face_list:
            fid = face["id"]
            presence[fid] = {
                "present": True,
                "confidence": face["confidence"],
                "session_id": session_id,
            }
        return presence

    def get_head_pose(
        self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]
    ) -> dict[str, float]:
        """
        Estimate head pose (pitch, yaw, roll) for a single face region.

        Uses MediaPipe Face Mesh landmarks and solvePnP for 3-D pose
        estimation. Returns zeros when estimation is not possible.

        Args:
            frame: Full BGR frame.
            face_bbox: (x, y, w, h) bounding box of the face.

        Returns:
            dict with keys: ``pitch``, ``yaw``, ``roll`` (all in degrees).
        """
        default = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        if self._face_mesh is None:
            return default

        try:
            import cv2 as cv

            x, y, w, h = face_bbox
            face_roi = frame[y : y + h, x : x + w]
            if face_roi.size == 0:
                return default

            rgb_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_roi)

            if not results.multi_face_landmarks:
                return default

            landmarks = results.multi_face_landmarks[0]
            ih, iw = face_roi.shape[:2]

            # Key landmark indices for pose estimation (nose tip, chin, etc.)
            key_points_2d = np.array(
                [
                    [landmarks.landmark[1].x * iw, landmarks.landmark[1].y * ih],   # Nose tip
                    [landmarks.landmark[152].x * iw, landmarks.landmark[152].y * ih],  # Chin
                    [landmarks.landmark[263].x * iw, landmarks.landmark[263].y * ih],  # Left eye
                    [landmarks.landmark[33].x * iw, landmarks.landmark[33].y * ih],   # Right eye
                    [landmarks.landmark[287].x * iw, landmarks.landmark[287].y * ih], # Left mouth
                    [landmarks.landmark[57].x * iw, landmarks.landmark[57].y * ih],   # Right mouth
                ],
                dtype=np.float64,
            )

            # Approximate 3D model points
            model_points_3d = np.array(
                [
                    [0.0, 0.0, 0.0],       # Nose tip
                    [0.0, -63.6, -12.5],   # Chin
                    [-43.3, 32.7, -26.0],  # Left eye
                    [43.3, 32.7, -26.0],   # Right eye
                    [-28.9, -28.9, -24.1], # Left mouth
                    [28.9, -28.9, -24.1],  # Right mouth
                ],
                dtype=np.float64,
            )

            focal = iw
            cam_matrix = np.array(
                [[focal, 0, iw / 2], [0, focal, ih / 2], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv.solvePnP(
                model_points_3d, key_points_2d, cam_matrix, dist_coeffs,
                flags=cv.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return default

            rmat, _ = cv.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv.RQDecomp3x3(rmat)

            return {
                "pitch": float(angles[0]),
                "yaw": float(angles[1]),
                "roll": float(angles[2]),
            }
        except Exception as exc:
            logger.debug("FaceTracker.get_head_pose: %s", exc)
            return default

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _assign_ids(
        self, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Assign stable IDs to detections using centroid matching.

        Args:
            detections: Raw detection dicts with ``centroid`` but no ``id``.

        Returns:
            Detection dicts with ``id`` added.
        """
        # Mark all existing tracked faces as potentially missing
        for fid in self._tracked:
            self._tracked[fid]["missing_frames"] += 1

        matched_ids: set[str] = set()
        result: list[dict[str, Any]] = []

        for det in detections:
            cx, cy = det["centroid"]
            best_id: str | None = None
            best_dist = float("inf")

            for fid, info in self._tracked.items():
                if fid in matched_ids:
                    continue
                tx, ty = info["centroid"]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist and dist < _MAX_CENTROID_DISTANCE:
                    best_dist = dist
                    best_id = fid

            if best_id:
                self._tracked[best_id]["centroid"] = (cx, cy)
                self._tracked[best_id]["missing_frames"] = 0
                matched_ids.add(best_id)
                det["id"] = best_id
            else:
                new_id = f"face-{self._next_id}"
                self._next_id += 1
                self._tracked[new_id] = {"centroid": (cx, cy), "missing_frames": 0}
                matched_ids.add(new_id)
                det["id"] = new_id

            result.append(det)

        # Prune stale tracks
        stale = [
            fid
            for fid, info in self._tracked.items()
            if info["missing_frames"] > _MAX_MISSING_FRAMES
        ]
        for fid in stale:
            del self._tracked[fid]

        return result
