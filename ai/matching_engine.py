"""
AI Resume Analyzer
Matching Engine

Responsible for comparing resume skills with job-description skills.

The engine receives raw resume/JD text through compare_skills()
and delegates skill extraction to skill_extractor.py.

It also provides lower-level functions for comparing already
extracted skill lists.
"""

from typing import Dict, List, Set

from ai.skill_extractor import (
    compare_skills as extract_and_compare_skills
)


# =========================================================
# SKILL NORMALIZATION
# =========================================================

def normalize_skill(skill: str) -> str:
    """
    Normalize a skill for reliable comparison.
    """

    if not skill:
        return ""

    return (
        skill.lower()
        .strip()
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
    )


def normalize_skills(skills: List[str]) -> Set[str]:
    """
    Convert a list of skills into a normalized set.
    """

    return {
        normalize_skill(skill)
        for skill in skills
        if skill and skill.strip()
    }


# =========================================================
# LIST-BASED SKILL MATCHING
# =========================================================

def match_skills(
    resume_skills: List[str],
    job_skills: List[str]
) -> Dict:
    """
    Compare two lists of extracted skills.

    Returns:
        matched_skills
        missing_skills
        resume_only_skills
        skill_match_percentage
    """

    resume_set = normalize_skills(resume_skills)
    job_set = normalize_skills(job_skills)

    matched = sorted(
        resume_set.intersection(job_set)
    )

    missing = sorted(
        job_set.difference(resume_set)
    )

    resume_only = sorted(
        resume_set.difference(job_set)
    )

    if job_set:
        match_percentage = (
            len(matched) / len(job_set)
        ) * 100
    else:
        match_percentage = 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_only_skills": resume_only,
        "skill_match_percentage": round(
            match_percentage,
            2
        )
    }


# =========================================================
# RAW TEXT SKILL COMPARISON
# =========================================================

def compare_skills(
    resume_text: str,
    job_description: str
) -> Dict:
    """
    Compare skills directly from raw resume text
    and job-description text.

    Skill extraction is handled by skill_extractor.py.
    """

    if not resume_text or not resume_text.strip():
        raise ValueError(
            "Resume text cannot be empty."
        )

    if not job_description or not job_description.strip():
        raise ValueError(
            "Job description cannot be empty."
        )

    # -----------------------------------------------------
    # Extract skills using the existing skill extractor
    # -----------------------------------------------------

    extracted = extract_and_compare_skills(
        resume_text,
        job_description
    )

    resume_skills = extracted.get(
        "resume_skills",
        []
    )

    job_skills = extracted.get(
        "job_skills",
        []
    )

    # -----------------------------------------------------
    # Run normalized matching
    # -----------------------------------------------------

    matching = match_skills(
        resume_skills,
        job_skills
    )

    # -----------------------------------------------------
    # Return one consistent structure
    # -----------------------------------------------------

    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,

        "matched_skills": matching[
            "matched_skills"
        ],

        "missing_skills": matching[
            "missing_skills"
        ],

        "resume_only_skills": matching[
            "resume_only_skills"
        ],

        "skill_match_percentage": matching[
            "skill_match_percentage"
        ],

        # Compatibility with analysis_engine.py
        "skill_match_score": matching[
            "skill_match_percentage"
        ]
    }


# =========================================================
# MATCH SCORE
# =========================================================

def calculate_match_score(
    resume_skills: List[str],
    job_skills: List[str]
) -> float:
    """
    Calculate the skill-based match score.
    """

    result = match_skills(
        resume_skills,
        job_skills
    )

    return result[
        "skill_match_percentage"
    ]


# =========================================================
# SKILL GAP
# =========================================================

def generate_skill_gap(
    resume_skills: List[str],
    job_skills: List[str]
) -> Dict:
    """
    Generate a structured skill-gap analysis.
    """

    result = match_skills(
        resume_skills,
        job_skills
    )

    return {
        "match_score": result[
            "skill_match_percentage"
        ],

        "strengths": result[
            "matched_skills"
        ],

        "skill_gaps": result[
            "missing_skills"
        ],

        "additional_skills": result[
            "resume_only_skills"
        ]
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample_resume = """
    Software Engineer with experience in Python,
    SQL, Docker and machine learning.

    Built backend applications using Python.
    Developed machine learning projects.
    Worked with SQL databases and Git.
    """

    sample_job = """
    Software Engineer

    Requirements:

    Strong Python programming skills.
    Experience with REST APIs.
    Knowledge of SQL.
    Experience with FastAPI.
    Knowledge of AWS.
    Experience with Docker.
    Machine learning knowledge.
    """

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("MATCHING ENGINE TEST")
    print("=" * 60)

    try:

        result = compare_skills(
            sample_resume,
            sample_job
        )

        print("\nRESUME SKILLS:")
        print(result["resume_skills"])

        print("\nJOB SKILLS:")
        print(result["job_skills"])

        print("\nMATCHED SKILLS:")
        print(result["matched_skills"])

        print("\nMISSING SKILLS:")
        print(result["missing_skills"])

        print("\nRESUME-ONLY SKILLS:")
        print(result["resume_only_skills"])

        print("\nSKILL MATCH:")
        print(
            f'{result["skill_match_score"]}%'
        )

        print("\nTest completed successfully.")

    except Exception as error:

        print("\nERROR:")
        print(
            type(error).__name__,
            ":",
            error
        )