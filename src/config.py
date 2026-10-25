"""Global configuration and paths for ZentroMart VoiceOps 2026."""
import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Capstone_Project_Dataset_FTTT_2026"
if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR

KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"

# Fixed reference date per capstone specification: 20 October 2026
REFERENCE_DATE = date(2026, 10, 20)
REFERENCE_DATE_STR = "2026-10-20"

# Document precedence hierarchy (lower integer = higher precedence)
DOC_PRECEDENCE = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
    "meta": 6,
}

# Gnani Voice configuration
GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")
GNANI_TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_DEFAULT_VOICE = os.environ.get("GNANI_VOICE", "Kaveri")
GNANI_DEFAULT_LANGUAGE = "en-IN"

# LLM Provider configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Output directories
OUTPUTS_DIR = BASE_DIR / "sample_outputs"
try:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

