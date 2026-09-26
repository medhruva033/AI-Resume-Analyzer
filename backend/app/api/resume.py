from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.security import verify_token
from app.database.connection import get_db
from app.models.resume import Resume
from app.schemas.resume import ResumeResponse
from app.services.resume_service import (
    extract_text,
    save_resume,
)


router = APIRouter(
    prefix="/api/resumes",
    tags=["Resumes"],
)


def get_current_user_id(
    authorization: str | None = Header(
        default=None
    ),
) -> int:

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authorization header required.",
        )

    if not authorization.startswith("Bearer "):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format.",
        )

    token = authorization.replace(
        "Bearer ",
        "",
        1,
    )

    payload = verify_token(token)

    if not payload:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    return int(payload["user_id"])


@router.post(
    "/upload",
    response_model=ResumeResponse,
)
async def upload_resume(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):

    user_id = get_current_user_id(
        authorization
    )

    filename, file_path = await save_resume(
        file
    )

    text = extract_text(
        file_path
    )

    resume = Resume(
        user_id=user_id,
        filename=filename,
        file_path=file_path,
        extracted_text=text,
    )

    db.add(resume)
    db.commit()
    db.refresh(resume)

    return resume


@router.get(
    "",
    response_model=list[ResumeResponse],
)
def get_resumes(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):

    user_id = get_current_user_id(
        authorization
    )

    return (
        db.query(Resume)
        .filter(Resume.user_id == user_id)
        .order_by(Resume.created_at.desc())
        .all()
    )