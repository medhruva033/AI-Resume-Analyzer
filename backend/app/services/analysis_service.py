from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.resume import Resume
from app.services.ai_service import (
    calculate_ats_score,
    calculate_match_score,
    generate_ai_analysis,
    generate_recommendations,
)


def create_analysis(
    db: Session,
    user_id: int,
    resume: Resume,
    job_description: str = "",
) -> Analysis:

    resume_text = resume.extracted_text or ""
    job_description = job_description or ""

    # Calculate ATS score
    ats_score = calculate_ats_score(
        resume_text
    )

    # Calculate job match score and skills
    (
        match_score,
        skills_found,
        missing_skills,
    ) = calculate_match_score(
        resume_text,
        job_description,
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        missing_skills,
        ats_score,
    )

    # Generate AI analysis summary
    ai_analysis = generate_ai_analysis(
        ats_score,
        match_score,
        missing_skills,
    )

    # Create database record
    analysis = Analysis(
        user_id=user_id,
        resume_id=resume.id,
        job_description=job_description,
        ats_score=ats_score,
        match_score=match_score,
        skills_found=skills_found,
        missing_skills=missing_skills,
        recommendations=recommendations,
        ai_analysis=ai_analysis,
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return analysis