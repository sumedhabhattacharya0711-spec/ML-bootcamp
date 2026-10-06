"""Project folders, found from this file's location (src/meeting_assistant/paths.py)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = PROJECT_DIR / "prompts"
RUNS_DIR = PROJECT_DIR / "runs"
