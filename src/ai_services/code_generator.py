"""
src/ai_services/code_generator.py — AI-driven code generation and safe execution.

Uses Gemini to generate Python analysis scripts, saves them to disk,
and executes them in an isolated subprocess.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ai_services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

_SCRIPT_HEADER = '''\
"""Auto-generated analysis script — Multimodal Attendance Agent."""
import json, sys, pathlib
'''

_EXECUTION_TIMEOUT = 30  # seconds


class CodeGenerator:
    """
    Generates and safely executes Python analysis scripts using Gemini.

    Generated scripts are saved to ``output_dir`` for auditability.
    Execution runs in a subprocess with a timeout to prevent runaway code.
    """

    def __init__(self, gemini_service: "GeminiService", output_dir: str) -> None:
        """
        Initialise the code generator.

        Args:
            gemini_service: An initialised GeminiService instance.
            output_dir: Directory to store generated scripts.
        """
        self.gemini = gemini_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_analysis_script(self, requirements: str) -> str:
        """
        Ask Gemini to generate a Python analysis script.

        Args:
            requirements: Natural-language description of what the script
                          should do.

        Returns:
            Generated Python source code as a string.
        """
        prompt = (
            "You are a Python expert. Write a complete, standalone Python script "
            "that satisfies the following requirements:\n\n"
            f"{requirements}\n\n"
            "Rules:\n"
            "1. Use only the Python standard library or numpy/pandas if needed.\n"
            "2. Add a main() function and call it at the bottom.\n"
            "3. Print results as JSON to stdout.\n"
            "4. Include docstrings and inline comments.\n"
            "5. Return ONLY the Python source code, no markdown fences."
        )

        if not self.gemini._available:  # type: ignore[attr-defined]
            # Return a minimal placeholder script when AI is unavailable
            return (
                f"{_SCRIPT_HEADER}\n"
                "def main():\n"
                f'    """Script for: {requirements[:80]}"""\n'
                '    print(json.dumps({"status": "AI unavailable", '
                '"requirements": '
                f'"{requirements[:80]}"'
                "}))\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )

        try:
            response = self.gemini._model.generate_content(prompt)  # type: ignore[attr-defined]
            code = response.text.strip()
            # Strip markdown code fences if present
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()
            return code
        except Exception as exc:
            logger.error("CodeGenerator.generate_analysis_script: %s", exc)
            return f'# Error generating script\nprint("Error: {exc}")\n'

    def save_script(self, code: str, filename: str) -> str:
        """
        Save a generated script to the output directory.

        Args:
            code: Python source code.
            filename: Desired filename (should end in .py).

        Returns:
            Absolute path to the saved file.
        """
        if not filename.endswith(".py"):
            filename = f"{filename}.py"

        filepath = self.output_dir / filename
        filepath.write_text(code, encoding="utf-8")
        logger.info("CodeGenerator: Script saved to '%s'.", filepath)
        return str(filepath)

    def execute_script(self, filepath: str) -> dict[str, object]:
        """
        Execute a Python script in an isolated subprocess.

        Args:
            filepath: Absolute path to the Python script to run.

        Returns:
            dict with keys:
                - ``stdout`` (str): Standard output from the script.
                - ``stderr`` (str): Standard error from the script.
                - ``returncode`` (int): Process exit code.
                - ``timed_out`` (bool): True if execution exceeded timeout.
        """
        result: dict[str, object] = {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "timed_out": False,
        }

        script_path = Path(filepath)
        if not script_path.exists():
            result["stderr"] = f"Script not found: {filepath}"
            return result

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT,
                cwd=str(script_path.parent),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Execution timed out after {_EXECUTION_TIMEOUT}s."
            result["timed_out"] = True
        except Exception as exc:
            result["stderr"] = str(exc)

        return result

    def list_scripts(self) -> list[dict[str, str]]:
        """
        List all Python scripts in the output directory.

        Returns:
            List of dicts with keys: ``filename``, ``path``, ``size_kb``.
        """
        scripts = []
        for fp in sorted(self.output_dir.glob("*.py"), key=os.path.getmtime, reverse=True):
            scripts.append(
                {
                    "filename": fp.name,
                    "path": str(fp),
                    "size_kb": f"{fp.stat().st_size / 1024:.1f}",
                }
            )
        return scripts
