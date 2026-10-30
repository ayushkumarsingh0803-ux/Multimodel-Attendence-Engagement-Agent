"""
src/audio/audio_processor.py — Audio feature extraction and voice activity detection.

Provides methods to detect voice activity, measure interaction levels,
measure silence, and extract feature vectors from audio files using librosa.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Energy threshold (RMS) below which audio is considered silence
_SILENCE_THRESHOLD = 0.01
# Frame length and hop for RMS computation (in samples at sr=22050)
_FRAME_LENGTH = 2048
_HOP_LENGTH = 512


class AudioProcessor:
    """
    Audio analysis engine for classroom engagement signals.

    Provides:
    - Voice Activity Detection (VAD) using short-time energy
    - Interaction level measurement (speaking proportion)
    - Silence duration estimation
    - Rich feature extraction (MFCCs, ZCR, spectral centroid, etc.)

    Uses librosa for all signal processing. Degrades gracefully when
    librosa is not installed.
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        """
        Initialise the audio processor.

        Args:
            sample_rate: Target sample rate for all audio analysis (Hz).
        """
        self.sample_rate = sample_rate
        self._librosa_available = False

        try:
            import librosa  # type: ignore  # noqa: F401

            self._librosa_available = True
            logger.info("AudioProcessor: librosa available.")
        except ImportError:
            logger.warning("AudioProcessor: librosa not installed. Using fallback.")

    # ── Public Methods ────────────────────────────────────────────────────────

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Determine whether voice activity is present in an audio chunk.

        Uses short-time Root Mean Square (RMS) energy. Frames with energy
        above ``_SILENCE_THRESHOLD`` are counted as active.

        Args:
            audio_chunk: 1-D numpy array of audio samples (float32, [-1, 1]).

        Returns:
            True if voice activity is detected.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        if self._librosa_available:
            try:
                import librosa  # type: ignore

                rms = librosa.feature.rms(
                    y=audio_chunk,
                    frame_length=_FRAME_LENGTH,
                    hop_length=_HOP_LENGTH,
                )[0]
                return bool(np.mean(rms) > _SILENCE_THRESHOLD)
            except Exception as exc:
                logger.debug("AudioProcessor.detect_voice_activity (librosa): %s", exc)

        # Fallback: plain RMS
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
        return rms > _SILENCE_THRESHOLD

    def measure_interaction_level(self, audio_window: np.ndarray) -> float:
        """
        Measure the proportion of an audio window that contains voice activity.

        Args:
            audio_window: 1-D numpy array of audio samples covering a time window.

        Returns:
            Interaction level in [0.0, 1.0] where 1.0 = fully active speech.
        """
        if audio_window is None or len(audio_window) == 0:
            return 0.0

        chunk_size = _FRAME_LENGTH
        n_chunks = max(1, len(audio_window) // chunk_size)
        active_chunks = 0

        for i in range(n_chunks):
            chunk = audio_window[i * chunk_size : (i + 1) * chunk_size]
            if self.detect_voice_activity(chunk):
                active_chunks += 1

        return active_chunks / n_chunks

    def get_silence_duration(self, audio_stream: np.ndarray) -> float:
        """
        Estimate total silence duration in an audio stream.

        Args:
            audio_stream: 1-D numpy array of audio samples.

        Returns:
            Silence duration in seconds.
        """
        if audio_stream is None or len(audio_stream) == 0:
            return 0.0

        if self._librosa_available:
            try:
                import librosa  # type: ignore

                rms = librosa.feature.rms(
                    y=audio_stream,
                    frame_length=_FRAME_LENGTH,
                    hop_length=_HOP_LENGTH,
                )[0]
                silent_frames = np.sum(rms < _SILENCE_THRESHOLD)
                # Convert frames to seconds
                return float(silent_frames * _HOP_LENGTH / self.sample_rate)
            except Exception as exc:
                logger.debug("AudioProcessor.get_silence_duration: %s", exc)

        # Fallback: split into 0.1-second chunks and count silent ones
        chunk_size = int(self.sample_rate * 0.1)
        silent_seconds = 0.0
        for i in range(0, len(audio_stream), chunk_size):
            chunk = audio_stream[i : i + chunk_size]
            if not self.detect_voice_activity(chunk):
                silent_seconds += 0.1

        return silent_seconds

    def extract_audio_features(self, audio_path: str) -> dict[str, Any]:
        """
        Extract a comprehensive feature vector from an audio file.

        Features extracted (when librosa is available):
        - ``mfcc_mean``: Mean of the first 13 MFCCs
        - ``mfcc_std``: Std dev of the first 13 MFCCs
        - ``zcr_mean``: Mean Zero Crossing Rate
        - ``spectral_centroid_mean``: Mean spectral centroid (Hz)
        - ``spectral_bandwidth_mean``: Mean spectral bandwidth (Hz)
        - ``rms_mean``: Mean RMS energy
        - ``tempo``: Estimated tempo (BPM)
        - ``voice_activity_ratio``: Proportion of voiced frames
        - ``silence_duration_s``: Total silence in seconds
        - ``duration_s``: Total audio duration in seconds

        Args:
            audio_path: Path to an audio file (WAV, MP3, etc.).

        Returns:
            Dict of feature name -> value. Returns default values on error.
        """
        default_features: dict[str, Any] = {
            "mfcc_mean": [],
            "mfcc_std": [],
            "zcr_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "rms_mean": 0.0,
            "tempo": 0.0,
            "voice_activity_ratio": 0.0,
            "silence_duration_s": 0.0,
            "duration_s": 0.0,
            "error": None,
        }

        if not Path(audio_path).exists():
            default_features["error"] = f"File not found: {audio_path}"
            return default_features

        if not self._librosa_available:
            default_features["error"] = "librosa not available"
            return default_features

        try:
            import librosa  # type: ignore

            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            rms = librosa.feature.rms(y=y, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]
            tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") else float(tempo_arr)

            silence_s = self.get_silence_duration(y)
            vad_ratio = self.measure_interaction_level(y)

            return {
                "mfcc_mean": mfccs.mean(axis=1).tolist(),
                "mfcc_std": mfccs.std(axis=1).tolist(),
                "zcr_mean": float(zcr.mean()),
                "spectral_centroid_mean": float(spectral_centroid.mean()),
                "spectral_bandwidth_mean": float(spectral_bw.mean()),
                "rms_mean": float(rms.mean()),
                "tempo": tempo,
                "voice_activity_ratio": vad_ratio,
                "silence_duration_s": silence_s,
                "duration_s": duration,
                "error": None,
            }
        except Exception as exc:
            logger.error("AudioProcessor.extract_audio_features: %s", exc)
            default_features["error"] = str(exc)
            return default_features
