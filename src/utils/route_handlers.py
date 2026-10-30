"""
src/utils/route_handlers.py — Flask route handler utility functions.

Provides validation, response formatting, and file-upload helpers used
by the route functions in app.py to keep them concise and testable.
"""

import logging
import os
from pathlib import Path
from typing import Any

from flask import Request
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

_REQUIRED_SESSION_FIELDS = ["name"]


def validate_session_request(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate the JSON body of a start-session request.

    Args:
        data: Parsed JSON dict from the Flask request.

    Returns:
        (True, "") on success, or (False, error_message) on failure.
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    name = data.get("name", "").strip()
    if not name:
        # Auto-generate a name rather than hard-failing
        data["name"] = "Unnamed Session"

    if len(data.get("name", "")) > 255:
        return False, "Session name must be 255 characters or fewer."

    return True, ""


def format_attendance_response(
    records: list[Any],
    sessions: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Format a list of Attendee ORM objects into a JSON-safe response dict.

    Args:
        records: List of Attendee model instances.
        sessions: Optional list of Session instances for the sessions array.

    Returns:
        Dict with keys: ``count``, ``records``, and optionally ``sessions``.
    """
    formatted_records = []
    for rec in records:
        formatted_records.append(
            {
                "id": rec.id,
                "session_id": rec.session_id,
                "face_id": rec.face_id,
                "first_seen": rec.first_seen.isoformat() if rec.first_seen else None,
                "last_seen": rec.last_seen.isoformat() if rec.last_seen else None,
                "total_duration_s": round(rec.total_duration or 0, 1),
                "total_duration_min": round((rec.total_duration or 0) / 60, 2),
                "engagement_avg": round(rec.engagement_avg or 0, 1),
                "status": "present" if (rec.total_duration or 0) > 0 else "absent",
            }
        )

    response: dict[str, Any] = {
        "count": len(formatted_records),
        "records": formatted_records,
    }

    if sessions is not None:
        response["sessions"] = [
            {
                "id": s.id,
                "name": s.name,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "status": s.status,
            }
            for s in sessions
        ]

    return response


def format_engagement_response(records: list[Any]) -> dict[str, Any]:
    """
    Format a list of EngagementRecord ORM objects into a JSON-safe response.

    Args:
        records: List of EngagementRecord model instances.

    Returns:
        Dict with keys: ``count``, ``records``, ``timeline``, ``summary``.
    """
    formatted: list[dict[str, Any]] = []
    fused_scores: list[float] = []
    visual_scores: list[float] = []
    audio_levels: list[float] = []
    level_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    for rec in records:
        fs = rec.fused_score or 0.0
        vs = rec.visual_score or 0.0
        al = rec.audio_level or 0.0
        lvl = rec.level or "medium"

        fused_scores.append(fs)
        visual_scores.append(vs)
        audio_levels.append(al)
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

        formatted.append(
            {
                "id": rec.id,
                "session_id": rec.session_id,
                "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
                "visual_score": round(vs, 2),
                "audio_level": round(al, 4),
                "fused_score": round(fs, 2),
                "level": lvl,
            }
        )

    summary: dict[str, Any] = {}
    if fused_scores:
        summary = {
            "avg_fused": round(sum(fused_scores) / len(fused_scores), 2),
            "max_fused": round(max(fused_scores), 2),
            "min_fused": round(min(fused_scores), 2),
            "avg_visual": round(sum(visual_scores) / len(visual_scores), 2),
            "avg_audio": round(sum(audio_levels) / len(audio_levels), 4),
            "level_counts": level_counts,
        }

    # Timeline: just timestamps + fused_score for chart rendering
    timeline = [
        {"t": r["timestamp"], "y": r["fused_score"]}
        for r in formatted
    ]

    return {
        "count": len(formatted),
        "records": formatted,
        "timeline": timeline,
        "summary": summary,
    }


def handle_file_upload(
    request: Request,
    allowed_extensions: set[str],
) -> tuple[str | None, str | None]:
    """
    Extract, validate, and prepare an uploaded file from a Flask request.

    Args:
        request: The Flask Request object.
        allowed_extensions: Set of permitted file extensions (without dots,
                            e.g. {"mp4", "avi", "mov"}).

    Returns:
        (filepath, None) on success, or (None, error_message) on failure.
        Note: ``filepath`` is the *original* filename (not yet saved) —
        the caller is responsible for actually saving via FileManager.
    """
    if "file" not in request.files:
        return None, "No file part in request."

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return None, "No file selected."

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lstrip(".").lower()

    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        return None, f"File type '.{ext}' not allowed. Permitted: {allowed}."

    return filename, None
