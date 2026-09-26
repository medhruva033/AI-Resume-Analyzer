import os
from pathlib import Path

from dotenv import load_dotenv


# Project root:
# AI-Resume-Analyzer/
BASE_DIR = Path(__file__).resolve().parents[3]

# Load .env from project root
load_dotenv(BASE_DIR / ".env")


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres@2026")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ai_resume_analyzer")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key-in-production"
)


UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = 10

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]