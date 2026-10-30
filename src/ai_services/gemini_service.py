"""
src/ai_services/gemini_service.py — Google Gemini 2.0 Flash integration.

Provides vision analysis, text generation, teaching suggestions,
session reports, and image generation via the google-generativeai SDK.
"""

import json
import logging
import base64
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Wrapper around the Google Generative AI SDK for Gemini 2.0 Flash.

    Handles all interactions with the Gemini API including multimodal
    vision analysis, text generation, and image generation. All methods
    degrade gracefully when the API key is absent or a request fails.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        """
        Initialise the Gemini service.

        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model identifier (default: gemini-2.0-flash).
        """
        self.model_name = model_name
        self._available = False
        self._model = None
        self._image_model = None

        if not api_key:
            logger.warning("GeminiService: No API key provided. AI features disabled.")
            return

        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name)
            self._image_model = genai.GenerativeModel("gemini-2.0-flash")
            self._genai = genai
            self._available = True
            logger.info("GeminiService: Initialised with model '%s'.", model_name)
        except ImportError:
            logger.error("GeminiService: google-generativeai package not installed.")
        except Exception as exc:
            logger.error("GeminiService: Initialisation failed — %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_frame(self, image_path: str) -> dict[str, Any]:
        """
        Analyse a single classroom frame for engagement metrics.

        Sends the image to Gemini Vision and asks it to estimate:
        - attention_score (0–100)
        - distraction_indicators (list of strings)
        - engagement_level ("high" | "medium" | "low")
        - notes (str)

        Args:
            image_path: Absolute path to the image file (JPEG/PNG).

        Returns:
            dict with keys: attention_score, distraction_indicators,
                            engagement_level, notes.
        """
        default = {
            "attention_score": 50,
            "distraction_indicators": [],
            "engagement_level": "medium",
            "notes": "Analysis unavailable.",
        }

        if not self._available:
            return default

        try:
            from PIL import Image  # type: ignore

            img = Image.open(image_path)
            prompt = (
                "You are an educational AI assistant analysing a classroom frame. "
                "Estimate student engagement WITHOUT identifying individuals. "
                "Respond ONLY with valid JSON matching this schema:\n"
                '{"attention_score": <int 0-100>, '
                '"distraction_indicators": [<str>, ...], '
                '"engagement_level": "<high|medium|low>", '
                '"notes": "<brief observation>"}'
            )
            response = self._model.generate_content([prompt, img])
            return self._parse_json_response(response.text, default)
        except Exception as exc:
            logger.error("GeminiService.analyze_frame: %s", exc)
            return default

    def generate_teaching_suggestions(
        self, engagement_data: dict[str, Any]
    ) -> list[str]:
        """
        Generate actionable teaching suggestions based on engagement data.

        Args:
            engagement_data: Dict containing session_id, records list,
                             and average_score.

        Returns:
            List of suggestion strings (3–6 items).
        """
        fallback = [
            "Consider introducing a short interactive Q&A to re-engage students.",
            "A 5-minute break may help restore attention.",
            "Try varying your presentation format (video, demo, group work).",
        ]

        if not self._available:
            return fallback

        try:
            avg = engagement_data.get("average_score", 50)
            records = engagement_data.get("records", [])
            low_periods = [r for r in records if r.get("level") == "low"]

            prompt = (
                f"You are an expert educational coach. A classroom session had:\n"
                f"- Average engagement score: {avg:.1f}/100\n"
                f"- Total data points: {len(records)}\n"
                f"- Low-engagement periods: {len(low_periods)}\n\n"
                "Generate 4-5 specific, actionable teaching suggestions to improve "
                "engagement. Format your response as a JSON array of strings:\n"
                '["suggestion 1", "suggestion 2", ...]'
            )
            response = self._model.generate_content(prompt)
            parsed = self._parse_json_response(response.text, fallback)
            if isinstance(parsed, list):
                return parsed
            return fallback
        except Exception as exc:
            logger.error("GeminiService.generate_teaching_suggestions: %s", exc)
            return fallback

    def generate_session_report(self, session_data: dict[str, Any]) -> str:
        """
        Generate a Markdown session report using Gemini.

        Args:
            session_data: Dict with session metadata and aggregate metrics.

        Returns:
            Markdown-formatted report string.
        """
        fallback = (
            f"# Session Report: {session_data.get('name', 'Unknown')}\n\n"
            f"**Total Attendees**: {session_data.get('total_attendees', 0)}\n"
            f"**Average Engagement**: {session_data.get('average_engagement', 0):.1f}%\n\n"
            "_AI report generation was unavailable. Showing summary statistics only._"
        )

        if not self._available:
            return fallback

        try:
            prompt = (
                "You are an educational analytics assistant. Generate a professional "
                "Markdown session report based on the following data:\n\n"
                f"{json.dumps(session_data, indent=2, default=str)}\n\n"
                "Include sections: Overview, Attendance Summary, Engagement Analysis, "
                "Key Observations, and Recommendations. Use Markdown headers and bullet points."
            )
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as exc:
            logger.error("GeminiService.generate_session_report: %s", exc)
            return fallback

    def generate_image(self, prompt: str) -> bytes | None:
        """
        Generate an image using Gemini's image generation capability.

        Args:
            prompt: Text description of the image to generate.

        Returns:
            Raw image bytes (PNG) or None on failure.
        """
        if not self._available:
            return None

        try:
            response = self._image_model.generate_content(
                [f"Generate an image: {prompt}"]
            )
            # Extract image bytes if available in the response parts
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return base64.b64decode(part.inline_data.data)
            logger.warning("GeminiService.generate_image: No image data in response.")
            return None
        except Exception as exc:
            logger.error("GeminiService.generate_image: %s", exc)
            return None

    def analyze_audio_context(self, audio_summary: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse an audio summary for classroom interaction context.

        Args:
            audio_summary: Dict with keys like voice_activity_ratio,
                           silence_duration, interaction_level.

        Returns:
            dict with interpretation and engagement_hint.
        """
        default = {
            "interpretation": "Normal classroom audio detected.",
            "engagement_hint": "medium",
            "recommended_action": None,
        }

        if not self._available:
            return default

        try:
            prompt = (
                "Analyse this classroom audio summary and provide context:\n"
                f"{json.dumps(audio_summary, indent=2)}\n\n"
                "Respond as JSON: "
                '{"interpretation": "<str>", "engagement_hint": "<high|medium|low>", '
                '"recommended_action": "<str or null>"}'
            )
            response = self._model.generate_content(prompt)
            return self._parse_json_response(response.text, default)
        except Exception as exc:
            logger.error("GeminiService.analyze_audio_context: %s", exc)
            return default

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(text: str, fallback: Any) -> Any:
        """
        Attempt to extract and parse JSON from a Gemini response string.

        Args:
            text: Raw response text that may contain JSON.
            fallback: Value to return if parsing fails.

        Returns:
            Parsed Python object or fallback.
        """
        if not text:
            return fallback

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        # Try to find JSON boundaries
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue

        # Last resort: try parsing the whole string
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("GeminiService: Could not parse JSON from response.")
            return fallback
