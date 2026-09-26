from pathlib import Path

from fastapi import UploadFile, HTTPException
from pypdf import PdfReader

from app.core.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
)


async def save_resume(file: UploadFile) -> tuple[str, str]:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read()

    max_size = MAX_FILE_SIZE_MB * 1024 * 1024

    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File must be smaller than {MAX_FILE_SIZE_MB} MB.",
        )

    safe_name = (
        f"{abs(hash(file.filename))}_"
        f"{Path(file.filename).name}"
    )

    file_path = UPLOAD_DIR / safe_name

    file_path.write_bytes(content)

    return file.filename, str(file_path)


def extract_text(file_path: str) -> str:

    path = Path(file_path)

    if path.suffix.lower() == ".txt":

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if path.suffix.lower() == ".pdf":

        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    return ""