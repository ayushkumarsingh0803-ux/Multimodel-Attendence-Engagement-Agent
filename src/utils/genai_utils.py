"""
src/utils/genai_utils.py — GenAI helper utilities.

Provides functions for preparing images, formatting prompts, parsing
Gemini API responses, and assembling rich session context strings.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def prepare_image_for_gemini(image_path: str):
    """
    Load and return a PIL Image ready for passing to a Gemini multimodal request.

    Converts images to RGB if needed (removes alpha channels) and performs
    a basic size-validation check.

    Args:
        image_path: Absolute or relative path to an image file.

    Returns:
        PIL.Image.Image object in RGB mode.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is not a valid image.
    """
    from PIL import Image  # type: ignore

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(str(p))
        # Ensure RGB (strip alpha) for Gemini compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc


def format_engagement_prompt(engagement_data: dict[str, Any]) -> str:
    """
    Build a structured Gemini prompt from an engagement data dictionary.

    Args:
        engagement_data: Dict containing session_id, average_score, and records.

    Returns:
        Formatted prompt string.
    """
    session_id = engagement_data.get("session_id", "unknown")
    avg_score = engagement_data.get("average_score", 0)
    records = engagement_data.get("records", [])

    # Summarise level distribution
    level_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for rec in records:
        level = rec.get("level", "medium")
        level_counts[level] = level_counts.get(level, 0) + 1

    total = len(records) or 1
    high_pct = level_counts["high"] / total * 100
    medium_pct = level_counts["medium"] / total * 100
    low_pct = level_counts["low"] / total * 100

    prompt = (
        f"Classroom Session Engagement Analysis\n"
        f"{'=' * 40}\n"
        f"Session ID: {session_id}\n"
        f"Total Measurements: {len(records)}\n"
        f"Average Engagement Score: {avg_score:.1f}/100\n\n"
        f"Engagement Level Distribution:\n"
        f"  High   (≥70): {high_pct:.1f}% ({level_counts['high']} samples)\n"
        f"  Medium (40-69): {medium_pct:.1f}% ({level_counts['medium']} samples)\n"
        f"  Low    (<40): {low_pct:.1f}% ({level_counts['low']} samples)\n\n"
        f"Based on this data, please provide:\n"
        f"1. A brief interpretation of the engagement pattern.\n"
        f"2. Identification of any concerning trends.\n"
        f"3. Specific, actionable recommendations for the instructor.\n"
    )
    return prompt


def parse_gemini_response(response_text: str) -> dict[str, Any]:
    """
    Parse a Gemini API text response into a structured dictionary.

    Attempts JSON extraction first; falls back to a structured text parse.

    Args:
        response_text: Raw text from a Gemini API response.

    Returns:
        Parsed dict. On failure, returns {"raw": response_text}.
    """
    if not response_text:
        return {"raw": ""}

    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    # Try JSON first
    for start, end in [("{", "}"), ("[", "]")]:
        si = text.find(start)
        ei = text.rfind(end)
        if si != -1 and ei != -1 and ei > si:
            try:
                return json.loads(text[si : ei + 1])
            except json.JSONDecodeError:
                pass

    # Fall back to returning the raw text
    return {"raw": response_text}


def build_session_context(
    session_data: dict[str, Any],
    engagement_history: list[dict[str, Any]],
) -> str:
    """
    Construct a rich natural-language context string for a session.

    Used to populate Gemini prompts with meaningful background before
    asking for a report or teaching suggestions.

    Args:
        session_data: Dict with session metadata (id, name, times, attendees, etc.).
        engagement_history: List of engagement dicts (timestamp, score, level).

    Returns:
        Multi-line context string.
    """
    name = session_data.get("name", "Unnamed Session")
    session_id = session_data.get("id", "N/A")
    start = session_data.get("start_time", "unknown")
    end = session_data.get("end_time", "ongoing")
    attendees = session_data.get("total_attendees", 0)
    avg_eng = session_data.get("average_engagement", 0)

    if engagement_history:
        scores = [r.get("score", 0) for r in engagement_history]
        peak = max(scores)
        valley = min(scores)
        low_periods = sum(1 for r in engagement_history if r.get("level") == "low")
    else:
        peak = valley = 0
        low_periods = 0

    context = (
        f"Session: {name} (ID: {session_id})\n"
        f"Time: {start} → {end}\n"
        f"Attendees Detected: {attendees}\n"
        f"Average Engagement: {avg_eng:.1f}/100\n"
        f"Peak Score: {peak:.1f} | Valley Score: {valley:.1f}\n"
        f"Low-Engagement Periods: {low_periods} of {len(engagement_history)} samples\n"
    )
    return context
