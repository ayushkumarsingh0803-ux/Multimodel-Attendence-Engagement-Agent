"""
src/ai_services/tts_service.py — Text-to-Speech service using gTTS.

Converts text announcements to MP3 audio files and saves them to
the configured output directory.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service backed by Google Text-to-Speech (gTTS).

    Audio files are written to ``output_dir`` as MP3 files.  The service
    degrades gracefully when gTTS is not installed — it logs a warning and
    returns a placeholder path rather than raising.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialise the TTS service.

        Args:
            output_dir: Directory where generated MP3 files are saved.
                        Created automatically if it does not exist.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify gTTS availability at import time
        try:
            from gtts import gTTS  # type: ignore  # noqa: F401

            self._gtts_available = True
        except ImportError:
            logger.warning("TTSService: gTTS is not installed. TTS features disabled.")
            self._gtts_available = False

    def synthesize(self, text: str, filename: str, lang: str = "en") -> str:
        """
        Synthesise ``text`` to speech and save as an MP3 file.

        Args:
            text: The text to convert to audio.
            filename: Output filename (without extension).
            lang: BCP-47 language code passed to gTTS (default: "en").

        Returns:
            Absolute path to the generated MP3 file.
        """
        if not filename.endswith(".mp3"):
            filename = f"{filename}.mp3"

        output_path = self.output_dir / filename

        if not self._gtts_available:
            logger.warning("TTSService.synthesize: gTTS unavailable, skipping.")
            return str(output_path)

        try:
            from gtts import gTTS  # type: ignore

            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(output_path))
            logger.info("TTSService: Saved audio to '%s'.", output_path)
        except Exception as exc:
            logger.error("TTSService.synthesize failed: %s", exc)

        return str(output_path)

    def list_audio_files(self) -> list[dict[str, str]]:
        """
        List all MP3 files in the output directory.

        Returns:
            List of dicts, each with keys: ``filename``, ``path``, ``size_kb``.
        """
        files: list[dict[str, str]] = []
        for fp in sorted(self.output_dir.glob("*.mp3"), key=os.path.getmtime, reverse=True):
            size_kb = fp.stat().st_size / 1024
            files.append(
                {
                    "filename": fp.name,
                    "path": str(fp),
                    "size_kb": f"{size_kb:.1f}",
                }
            )
        return files

    def delete_audio(self, filename: str) -> bool:
        """
        Delete a specific audio file.

        Args:
            filename: Filename (with or without path) to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        target = self.output_dir / Path(filename).name
        if target.exists() and target.parent == self.output_dir:
            target.unlink()
            logger.info("TTSService: Deleted '%s'.", target)
            return True
        logger.warning("TTSService.delete_audio: File '%s' not found.", filename)
        return False
