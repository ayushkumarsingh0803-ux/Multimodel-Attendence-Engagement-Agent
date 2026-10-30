"""
tests/test_gemini_service.py — Unit tests for GeminiService.

Uses unittest.mock to patch google.generativeai so tests run without
a real API key or network connection.
"""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_mock_genai():
    """Build a minimal mock of the google.generativeai module."""
    mock_genai = MagicMock(name="google.generativeai")

    # Mock GenerativeModel
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    return mock_genai, mock_model


class TestGeminiServiceInit(unittest.TestCase):
    """Tests for GeminiService.__init__."""

    def test_init_without_api_key_marks_unavailable(self):
        """GeminiService should be unavailable when no API key is given."""
        from src.ai_services.gemini_service import GeminiService

        svc = GeminiService("", "gemini-2.0-flash")
        self.assertFalse(svc._available)

    def test_init_with_mock_genai_marks_available(self):
        """GeminiService should be available when genai initialises OK."""
        mock_genai, _ = _make_mock_genai()

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            from importlib import reload
            import src.ai_services.gemini_service as mod
            reload(mod)
            svc = mod.GeminiService("fake-key", "gemini-2.0-flash")

        self.assertTrue(svc._available)


class TestGeminiServiceAnalyzeFrame(unittest.TestCase):
    """Tests for GeminiService.analyze_frame."""

    def _make_service(self):
        mock_genai, mock_model = _make_mock_genai()

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            from importlib import reload
            import src.ai_services.gemini_service as mod
            reload(mod)
            svc = mod.GeminiService("fake-key", "gemini-2.0-flash")

        svc._model = mock_model
        svc._available = True
        return svc, mock_model

    def test_analyze_frame_returns_default_when_unavailable(self):
        """analyze_frame should return default dict when service is unavailable."""
        from src.ai_services.gemini_service import GeminiService

        svc = GeminiService("", "gemini-2.0-flash")
        result = svc.analyze_frame("/nonexistent/image.jpg")

        self.assertIn("attention_score", result)
        self.assertIn("engagement_level", result)
        self.assertEqual(result["engagement_level"], "medium")

    @patch("src.ai_services.gemini_service.GeminiService.analyze_frame")
    def test_analyze_frame_returns_parsed_json(self, mock_method):
        """analyze_frame should parse Gemini's JSON response correctly."""
        expected = {
            "attention_score": 82,
            "distraction_indicators": [],
            "engagement_level": "high",
            "notes": "Students appear attentive.",
        }
        mock_method.return_value = expected

        from src.ai_services.gemini_service import GeminiService
        svc = GeminiService("fake", "gemini-2.0-flash")
        result = svc.analyze_frame("/fake/image.jpg")

        self.assertEqual(result["attention_score"], 82)
        self.assertEqual(result["engagement_level"], "high")

    def test_analyze_frame_handles_exception_gracefully(self):
        """analyze_frame should return defaults when an exception occurs."""
        svc, mock_model = self._make_service()

        # Patch PIL import to raise so the frame-loading fails
        with patch("builtins.open", side_effect=Exception("IO error")):
            result = svc.analyze_frame("/nonexistent/path.jpg")

        self.assertIn("attention_score", result)
        self.assertEqual(result["attention_score"], 50)


class TestGeminiServiceGenerateSuggestions(unittest.TestCase):
    """Tests for GeminiService.generate_teaching_suggestions."""

    def test_returns_fallback_when_unavailable(self):
        """generate_teaching_suggestions returns fallback list when unavailable."""
        from src.ai_services.gemini_service import GeminiService

        svc = GeminiService("", "gemini-2.0-flash")
        suggestions = svc.generate_teaching_suggestions({"session_id": 1, "records": [], "average_score": 50})

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    @patch("src.ai_services.gemini_service.GeminiService.generate_teaching_suggestions")
    def test_returns_list_of_strings(self, mock_method):
        """generate_teaching_suggestions should return a list of strings."""
        mock_method.return_value = [
            "Use interactive polls.",
            "Add a short quiz.",
            "Take a 5-minute break.",
        ]

        from src.ai_services.gemini_service import GeminiService
        svc = GeminiService("fake", "gemini-2.0-flash")
        data = {"session_id": 1, "records": [], "average_score": 45.0}
        result = svc.generate_teaching_suggestions(data)

        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)


class TestGeminiServiceGenerateReport(unittest.TestCase):
    """Tests for GeminiService.generate_session_report."""

    def test_returns_fallback_markdown_when_unavailable(self):
        """generate_session_report should return markdown string when unavailable."""
        from src.ai_services.gemini_service import GeminiService

        svc = GeminiService("", "gemini-2.0-flash")
        session_data = {
            "id": 1,
            "name": "Test Session",
            "total_attendees": 5,
            "average_engagement": 72.5,
        }
        report = svc.generate_session_report(session_data)

        self.assertIsInstance(report, str)
        self.assertIn("Test Session", report)
        self.assertIn("Total Attendees", report)

    @patch("src.ai_services.gemini_service.GeminiService.generate_session_report")
    def test_report_contains_session_name(self, mock_method):
        """generate_session_report should include the session name in its output."""
        mock_method.return_value = "# Session Report: My Lecture\n\n**Attendees**: 20"

        from src.ai_services.gemini_service import GeminiService
        svc = GeminiService("fake", "gemini-2.0-flash")
        report = svc.generate_session_report({"name": "My Lecture", "total_attendees": 20})

        self.assertIn("My Lecture", report)


class TestParseJsonResponse(unittest.TestCase):
    """Tests for GeminiService._parse_json_response (static method)."""

    def setUp(self):
        from src.ai_services.gemini_service import GeminiService
        self.parse = GeminiService._parse_json_response

    def test_parse_plain_json_object(self):
        result = self.parse('{"key": "value"}', {})
        self.assertEqual(result, {"key": "value"})

    def test_parse_json_with_markdown_fence(self):
        result = self.parse('```json\n{"score": 75}\n```', {})
        self.assertEqual(result["score"], 75)

    def test_parse_json_array(self):
        result = self.parse('["a", "b", "c"]', [])
        self.assertEqual(result, ["a", "b", "c"])

    def test_returns_fallback_on_invalid_json(self):
        result = self.parse("This is not JSON.", {"default": True})
        self.assertEqual(result, {"default": True})

    def test_returns_fallback_on_empty_string(self):
        result = self.parse("", "fallback")
        self.assertEqual(result, "fallback")


if __name__ == "__main__":
    unittest.main()
