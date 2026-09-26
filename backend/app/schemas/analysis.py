from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AnalysisCreate(BaseModel):

    resume_id: int

    job_description: str = ""


class AnalysisResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    resume_id: int

    ats_score: float

    match_score: float

    skills_found: list[str] = Field(default_factory=list)

    missing_skills: list[str] = Field(default_factory=list)

    recommendations: list[str] = Field(default_factory=list)

    ai_analysis: str | None = None

    created_at: datetime