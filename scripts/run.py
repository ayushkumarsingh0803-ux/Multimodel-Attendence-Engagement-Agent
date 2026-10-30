"""
scripts/run.py — Application runner for Multimodal Attendance Agent.

Reads configuration from config.py (which reads from .env) and launches
the Flask development server. For production, use a WSGI server such as
gunicorn or waitress instead.
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from app import create_app

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Validate configuration and start the Flask application."""
    warnings = Config.validate()
    for warning in warnings:
        logger.warning(warning)

    app = create_app(Config)

    logger.info(
        "Starting Multimodal Attendance Agent on http://%s:%s",
        Config.HOST,
        Config.PORT,
    )

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=Config.DEBUG,
    )


if __name__ == "__main__":
    main()
