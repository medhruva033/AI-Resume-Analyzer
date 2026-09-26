from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    Text,
)
from sqlalchemy.orm import relationship

from app.database.connection import Base


class Analysis(Base):

    __tablename__ = "analyses"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    resume_id = Column(
        Integer,
        ForeignKey("resumes.id"),
        nullable=False,
        index=True,
    )

    job_description = Column(
        Text,
        nullable=True,
    )

    ats_score = Column(
        Float,
        default=0,
    )

    match_score = Column(
        Float,
        default=0,
    )

    skills_found = Column(
        JSON,
        default=list,
    )

    missing_skills = Column(
        JSON,
        default=list,
    )

    recommendations = Column(
        JSON,
        default=list,
    )

    ai_analysis = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="analyses",
    )

    resume = relationship(
        "Resume",
        back_populates="analyses",
    )