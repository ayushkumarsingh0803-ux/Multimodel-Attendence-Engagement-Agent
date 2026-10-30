"""
config.py — Application configuration for Multimodal Attendance & Engagement Agent.
"""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """
    Central configuration class.

    All application settings are defined here and read from environment
    variables (with sensible defaults). Import this class wherever
    configuration is needed.
    """

    # ── Gemini AI ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = "gemini-2.0-flash"

    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
    DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")

    # ── File System Paths ─────────────────────────────────────────────────────
    UPLOAD_FOLDER: str = str(BASE_DIR / "uploads")
    AUDIO_FOLDER: str = str(BASE_DIR / "generated_audio")
    CODE_FOLDER: str = str(BASE_DIR / "generated_code")
    DB_PATH: str = str(BASE_DIR / "attendance.db")

    # SQLAlchemy URI (SQLite)
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # ── Upload Limits ─────────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_VIDEO_EXTENSIONS: set = {"mp4", "avi", "mov", "mkv", "webm"}
    ALLOWED_AUDIO_EXTENSIONS: set = {"mp3", "wav", "ogg", "flac"}

    # ── Engagement Thresholds ─────────────────────────────────────────────────
    # Score above which a student is considered "engaged"
    ENGAGEMENT_THRESHOLD: float = float(os.getenv("ENGAGEMENT_THRESHOLD", "60.0"))
    # Confidence above which a face detection is considered valid
    PRESENCE_THRESHOLD: float = float(os.getenv("PRESENCE_THRESHOLD", "0.65"))

    # ── Multimodal Fusion Weights ─────────────────────────────────────────────
    VISUAL_WEIGHT: float = float(os.getenv("VISUAL_WEIGHT", "0.6"))
    AUDIO_WEIGHT: float = float(os.getenv("AUDIO_WEIGHT", "0.4"))

    # ── Session Settings ──────────────────────────────────────────────────────
    SESSION_COOKIE_SECURE: bool = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY: bool = True
    PERMANENT_SESSION_LIFETIME: int = 86400  # 24 hours in seconds

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required directories if they do not exist."""
        for directory in (cls.UPLOAD_FOLDER, cls.AUDIO_FOLDER, cls.CODE_FOLDER):
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate critical configuration values.

        Returns:
            list[str]: A list of warning/error messages. Empty list = OK.
        """
        warnings: list[str] = []
        if not cls.GEMINI_API_KEY:
            warnings.append(
                "GEMINI_API_KEY is not set. AI features will be disabled."
            )
        if cls.SECRET_KEY == "":
            warnings.append("FLASK_SECRET_KEY is empty. Using auto-generated key.")
        return warnings
