from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import verify_token
from app.database.connection import get_db
from app.models.resume import Resume
from app.schemas.analysis import AnalysisResponse
from app.services.analysis_service import create_analysis


router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis"],
)


class AnalyzeRequest(BaseModel):
    resume_id: int
    job_description: str = ""


def get_current_user_id(
    authorization: str | None = Header(default=None),
) -> int:

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required.",
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
    "/analyze",
    response_model=AnalysisResponse,
)
def analyze(
    request: AnalyzeRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):

    user_id = get_current_user_id(
        authorization
    )

    resume = (
        db.query(Resume)
        .filter(
            Resume.id == request.resume_id,
            Resume.user_id == user_id,
        )
        .first()
    )

    if not resume:
        raise HTTPException(
            status_code=404,
            detail="Resume not found.",
        )

    return create_analysis(
        db=db,
        user_id=user_id,
        resume=resume,
        job_description=request.job_description,
    )


@router.get(
    "/history",
    response_model=list[AnalysisResponse],
)
def history(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):

    user_id = get_current_user_id(
        authorization
    )

    from app.models.analysis import Analysis

    return (
        db.query(Analysis)
        .filter(
            Analysis.user_id == user_id
        )
        .order_by(
            Analysis.created_at.desc()
        )
        .all()
    )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
)
def get_analysis(
    analysis_id: int,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):

    user_id = get_current_user_id(
        authorization
    )

    from app.models.analysis import Analysis

    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
        .first()
    )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return analysis