"""
src/storage/file_manager.py — File management utilities.

Provides a unified interface for saving, listing, and managing uploaded
files, generated audio, and generated code scripts across the three
storage directories.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages three storage directories: uploads, audio, and code.

    Provides consistent methods to save, list, inspect, and delete files
    across all three directories, with path traversal protection.
    """

    def __init__(self, upload_dir: str, audio_dir: str, code_dir: str) -> None:
        """
        Initialise the file manager.

        Args:
            upload_dir: Directory for uploaded classroom videos/audio.
            audio_dir: Directory for TTS-generated MP3 files.
            code_dir: Directory for AI-generated Python scripts.
        """
        self.upload_dir = Path(upload_dir)
        self.audio_dir = Path(audio_dir)
        self.code_dir = Path(code_dir)

        for directory in (self.upload_dir, self.audio_dir, self.code_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ── Upload Management ─────────────────────────────────────────────────────

    def save_upload(self, file_obj: FileStorage, filename: str) -> str:
        """
        Save an uploaded file to the uploads directory.

        Args:
            file_obj: Werkzeug FileStorage object from a Flask request.
            filename: Sanitised filename (use werkzeug.utils.secure_filename).

        Returns:
            Absolute path of the saved file.
        """
        target = self.upload_dir / filename
        file_obj.save(str(target))
        logger.info("FileManager: Saved upload '%s'.", target)
        return str(target)

    def list_uploads(self) -> list[dict[str, Any]]:
        """
        List all files in the uploads directory.

        Returns:
            List of file-info dicts (see ``get_file_info``).
        """
        return self._list_directory(self.upload_dir)

    # ── Audio Management ──────────────────────────────────────────────────────

    def list_audio(self) -> list[dict[str, Any]]:
        """
        List all files in the generated audio directory.

        Returns:
            List of file-info dicts.
        """
        return self._list_directory(self.audio_dir, extensions={".mp3", ".wav", ".ogg"})

    # ── Code Management ───────────────────────────────────────────────────────

    def list_code(self) -> list[dict[str, Any]]:
        """
        List all Python scripts in the generated code directory.

        Returns:
            List of file-info dicts.
        """
        return self._list_directory(self.code_dir, extensions={".py"})

    # ── Shared Utilities ──────────────────────────────────────────────────────

    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file by its absolute path.

        Refuses to delete files outside the managed directories for safety.

        Args:
            filepath: Absolute path of the file to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        target = Path(filepath).resolve()

        # Security: only allow deletion within managed directories
        allowed_parents = {
            self.upload_dir.resolve(),
            self.audio_dir.resolve(),
            self.code_dir.resolve(),
        }
        if not any(str(target).startswith(str(p)) for p in allowed_parents):
            logger.warning("FileManager.delete_file: Rejected path '%s' (outside managed dirs).", filepath)
            return False

        if target.exists():
            target.unlink()
            logger.info("FileManager: Deleted '%s'.", target)
            return True

        logger.warning("FileManager.delete_file: File not found — '%s'.", filepath)
        return False

    def get_file_info(self, filepath: str) -> dict[str, Any]:
        """
        Return metadata about a file.

        Args:
            filepath: Absolute path of the file.

        Returns:
            Dict with keys: ``filename``, ``path``, ``size_bytes``,
            ``size_kb``, ``extension``, ``modified_at``, ``exists``.
        """
        p = Path(filepath)
        if not p.exists():
            return {
                "filename": p.name,
                "path": filepath,
                "size_bytes": 0,
                "size_kb": "0.0",
                "extension": p.suffix,
                "modified_at": None,
                "exists": False,
            }

        stat = p.stat()
        return {
            "filename": p.name,
            "path": str(p),
            "size_bytes": stat.st_size,
            "size_kb": f"{stat.st_size / 1024:.1f}",
            "extension": p.suffix,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "exists": True,
        }

    def clear_directory(self, directory: str) -> int:
        """
        Delete all files in a managed directory.

        Args:
            directory: One of "uploads", "audio", "code".

        Returns:
            Number of files deleted.
        """
        dir_map = {
            "uploads": self.upload_dir,
            "audio": self.audio_dir,
            "code": self.code_dir,
        }
        target_dir = dir_map.get(directory)
        if not target_dir:
            logger.warning("FileManager.clear_directory: Unknown directory '%s'.", directory)
            return 0

        count = 0
        for fp in target_dir.iterdir():
            if fp.is_file():
                fp.unlink()
                count += 1

        logger.info("FileManager: Cleared %d files from '%s'.", count, target_dir)
        return count

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _list_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        List files in a directory, optionally filtered by extension.

        Args:
            directory: Path object of the directory to list.
            extensions: Set of allowed extensions (e.g. {".mp3", ".wav"}).
                        None means all files.

        Returns:
            List of file-info dicts, sorted by modification time descending.
        """
        results: list[dict[str, Any]] = []
        if not directory.exists():
            return results

        for fp in directory.iterdir():
            if not fp.is_file():
                continue
            if extensions and fp.suffix.lower() not in extensions:
                continue
            results.append(self.get_file_info(str(fp)))

        results.sort(key=lambda x: x.get("modified_at") or "", reverse=True)
        return results
