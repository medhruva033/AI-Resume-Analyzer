from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ai_service import (
    generate_ai_analysis,
)


router = APIRouter(
    prefix="/api/ai",
    tags=["AI"],
)


class ChatRequest(BaseModel):

    message: str

    ats_score: float = 0

    match_score: float = 0


@router.post("/chat")
def chat(request: ChatRequest):

    message = request.message.lower()

    if "score" in message:

        response = (
            f"Your ATS score is "
            f"{request.ats_score}/100 and "
            f"your job match score is "
            f"{request.match_score}/100."
        )

    elif "improve" in message:

        response = (
            "Improve your resume by tailoring "
            "skills and project descriptions to "
            "the target job description."
        )

    else:

        response = (
            "I can help you understand your ATS "
            "score, job match, missing skills, "
            "and resume improvements."
        )

    return {
        "response": response
    }