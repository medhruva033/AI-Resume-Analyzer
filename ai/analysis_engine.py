"""
AI Resume Analyzer
Analysis Engine

Combines:
1. Resume Parsing
2. JD Parsing
3. Skill Matching
4. ATS Analysis
5. Semantic Matching
6. Experience Analysis
7. Overall Scoring
8. Recommendations
"""

from typing import Dict, List

from ai.resume_parser import parse_resume
from ai.jd_parser import parse_job_description
from ai.ats_analyzer import analyze_ats
from ai.embedding_engine import compare_resume_with_job
from ai.matching_engine import compare_skills
from ai.scoring_engine import calculate_score


# =========================================================
# HELPERS
# =========================================================

def get_resume_section(
    sections: Dict[str, str],
    names: List[str]
) -> List[str]:
    """Return available resume section contents."""

    results = []

    for name in names:

        content = sections.get(name)

        if content and content.strip():
            results.append(content.strip())

    return results


def get_job_section(
    sections: Dict[str, str],
    names: List[str]
) -> List[str]:
    """Return available JD section contents."""

    results = []

    for name in names:

        content = sections.get(name)

        if content and content.strip():
            results.append(content.strip())

    return results


# =========================================================
# SEMANTIC ANALYSIS
# =========================================================

def calculate_section_semantic_score(
    resume_text: str,
    job_description: str
) -> Dict:
    """
    Calculate semantic similarity using relevant
    resume and JD sections.

    Falls back to full-document comparison only when
    structured content is genuinely unavailable.
    """

    resume_data = parse_resume(resume_text)
    job_data = parse_job_description(job_description)

    resume_sections = resume_data.get(
        "sections",
        {}
    )

    job_sections = job_data.get(
        "sections",
        {}
    )

    # -----------------------------------------------------
    # Resume content relevant to job matching
    # -----------------------------------------------------

    resume_parts = get_resume_section(
        resume_sections,
        [
            "PROFESSIONAL SUMMARY",
            "SUMMARY",
            "PROFILE",
            "ABOUT ME",
            "OBJECTIVE",
            "CAREER OBJECTIVE",
            "EXPERIENCE",
            "WORK EXPERIENCE",
            "EMPLOYMENT",
            "INTERNSHIP",
            "PROJECTS",
            "SKILLS",
            "TECHNICAL SKILLS",
            "CORE SKILLS",
            "EDUCATION",
        ]
    )

    # -----------------------------------------------------
    # JD content relevant to job matching
    # -----------------------------------------------------

    job_parts = get_job_section(
        job_sections,
        [
            "requirements",
            "responsibilities",
            "qualifications",
        ]
    )

    # -----------------------------------------------------
    # Structured comparison
    # -----------------------------------------------------

    if resume_parts and job_parts:

        structured_resume = "\n".join(
            resume_parts
        )

        structured_job = "\n".join(
            job_parts
        )

        result = compare_resume_with_job(
            structured_resume,
            structured_job
        )

        score = float(
            result.get(
                "percentage",
                0
            )
        )

        similarity = float(
            result.get(
                "semantic_similarity",
                score / 100
            )
        )

        return {
            "score": round(score, 2),
            "similarity": round(
                similarity,
                4
            ),
            "method": "section_based",
        }

    # -----------------------------------------------------
    # Full-document fallback
    # -----------------------------------------------------

    result = compare_resume_with_job(
        resume_text,
        job_description
    )

    score = float(
        result.get(
            "percentage",
            0
        )
    )

    similarity = float(
        result.get(
            "semantic_similarity",
            score / 100
        )
    )

    return {
        "score": round(score, 2),
        "similarity": round(
            similarity,
            4
        ),
        "method": "full_document",
    }


# =========================================================
# EXPERIENCE ANALYSIS
# =========================================================

def analyze_experience(
    resume_text: str,
    job_description: str
) -> float:
    """
    Estimate experience relevance by comparing resume
    experience/project content against JD requirements
    and responsibilities.
    """

    resume_data = parse_resume(
        resume_text
    )

    job_data = parse_job_description(
        job_description
    )

    resume_sections = resume_data.get(
        "sections",
        {}
    )

    job_sections = job_data.get(
        "sections",
        {}
    )

    # -----------------------------------------------------
    # Resume experience
    # -----------------------------------------------------

    experience_parts = get_resume_section(
        resume_sections,
        [
            "EXPERIENCE",
            "WORK EXPERIENCE",
            "EMPLOYMENT",
            "INTERNSHIP",
            "PROJECTS",
        ]
    )

    # -----------------------------------------------------
    # Job requirements
    # -----------------------------------------------------

    job_parts = get_job_section(
        job_sections,
        [
            "requirements",
            "responsibilities",
        ]
    )

    # -----------------------------------------------------
    # Semantic experience comparison
    # -----------------------------------------------------

    if experience_parts and job_parts:

        experience_text = "\n".join(
            experience_parts
        )

        job_text = "\n".join(
            job_parts
        )

        result = compare_resume_with_job(
            experience_text,
            job_text
        )

        return round(
            float(
                result.get(
                    "percentage",
                    0
                )
            ),
            2
        )

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    resume = resume_text.lower()
    job = job_description.lower()

    experience_terms = [
        "experience",
        "worked",
        "developed",
        "built",
        "created",
        "designed",
        "implemented",
        "managed",
        "led",
        "internship",
        "project",
        "years",
    ]

    job_terms = [
        term
        for term in experience_terms
        if term in job
    ]

    if not job_terms:
        return 70.0

    matched = sum(
        1
        for term in job_terms
        if term in resume
    )

    return round(
        min(
            (matched / len(job_terms)) * 100,
            100
        ),
        2
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    ats_result: Dict,
    skill_result: Dict,
    semantic_score: float,
    experience_score: float
) -> List[str]:

    recommendations = []

    # -----------------------------------------------------
    # Missing skills
    # -----------------------------------------------------

    missing_skills = skill_result.get(
        "missing_skills",
        []
    )

    if missing_skills:

        skills_text = ", ".join(
            str(skill)
            for skill in missing_skills[:8]
        )

        recommendations.append(
            f"Review the missing job skills: "
            f"{skills_text}. Add them only if you "
            f"genuinely have relevant knowledge or experience."
        )

    # -----------------------------------------------------
    # ATS issues
    # -----------------------------------------------------

    ats_issues = ats_result.get(
        "issues",
        []
    )

    for issue in ats_issues[:3]:

        recommendations.append(
            str(issue)
        )

    # -----------------------------------------------------
    # Semantic score
    # -----------------------------------------------------

    if semantic_score < 50:

        recommendations.append(
            "The resume has low semantic alignment "
            "with the job description. Strengthen "
            "relevant experience and project descriptions."
        )

    elif semantic_score < 70:

        recommendations.append(
            "The resume has moderate semantic alignment. "
            "Strengthen descriptions that are most relevant "
            "to the target role."
        )

    # -----------------------------------------------------
    # Experience
    # -----------------------------------------------------

    if experience_score < 50:

        recommendations.append(
            "The resume does not clearly demonstrate "
            "enough relevant experience for this role."
        )

    # -----------------------------------------------------
    # Default
    # -----------------------------------------------------

    if not recommendations:

        recommendations.append(
            "The resume shows good alignment. Continue "
            "improving measurable achievements and "
            "job-specific terminology."
        )

    return recommendations


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_resume(
    resume_text: str,
    job_description: str
) -> Dict:

    if not resume_text or not resume_text.strip():

        raise ValueError(
            "Resume text cannot be empty."
        )

    if not job_description or not job_description.strip():

        raise ValueError(
            "Job description cannot be empty."
        )

    # -----------------------------------------------------
    # Parse resume and JD
    # -----------------------------------------------------

    resume_data = parse_resume(
        resume_text
    )

    job_data = parse_job_description(
        job_description
    )

    # -----------------------------------------------------
    # Skill matching
    # -----------------------------------------------------

    skill_result = compare_skills(
        resume_text,
        job_description
    )

    skill_match_score = float(
        skill_result.get(
            "skill_match_score",
            skill_result.get(
                "skill_match",
                0
            )
        )
    )

    # -----------------------------------------------------
    # ATS
    # -----------------------------------------------------

    job_skills = skill_result.get(
        "job_skills",
        []
    )

    ats_result = analyze_ats(
        resume_text,
        job_skills
    )

    ats_score = float(
        ats_result.get(
            "ats_score",
            0
        )
    )

    # -----------------------------------------------------
    # Semantic analysis
    # -----------------------------------------------------

    semantic_result = calculate_section_semantic_score(
        resume_text,
        job_description
    )

    semantic_score = float(
        semantic_result.get(
            "score",
            0
        )
    )

    # -----------------------------------------------------
    # Experience
    # -----------------------------------------------------

    experience_score = analyze_experience(
        resume_text,
        job_description
    )

    # -----------------------------------------------------
    # Overall score
    # -----------------------------------------------------

    score_result = calculate_score(
        ats_score=ats_score,
        skill_match_score=skill_match_score,
        semantic_score=semantic_score,
        experience_score=experience_score
    )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = generate_recommendations(
        ats_result=ats_result,
        skill_result=skill_result,
        semantic_score=semantic_score,
        experience_score=experience_score
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {

        "summary": (
            f"The resume shows "
            f"{score_result.get('category', 'moderate').lower()} "
            f"alignment with the job description."
        ),

        "score": score_result,

        "candidate": {
            "name": resume_data.get("name"),
            "email": resume_data.get("email"),
            "phone": resume_data.get("phone"),
        },

        "job_analysis": {
            "experience_requirements":
                job_data.get(
                    "experience_requirements",
                    []
                ),

            "education_requirements":
                job_data.get(
                    "education_requirements",
                    []
                ),
        },

        "skill_analysis": {

            "resume_skills":
                skill_result.get(
                    "resume_skills",
                    []
                ),

            "job_skills":
                skill_result.get(
                    "job_skills",
                    []
                ),

            "matched_skills":
                skill_result.get(
                    "matched_skills",
                    []
                ),

            "missing_skills":
                skill_result.get(
                    "missing_skills",
                    []
                ),

            "resume_only_skills":
                skill_result.get(
                    "resume_only_skills",
                    []
                ),

            "skill_match_score":
                skill_match_score,
        },

        "ats_analysis":
            ats_result,

        "semantic_analysis": {

            "score":
                semantic_score,

            "similarity":
                semantic_result.get(
                    "similarity",
                    0
                ),

            "percentage":
                semantic_score,

            "method":
                semantic_result.get(
                    "method",
                    "unknown"
                ),
        },

        "experience_analysis": {

            "score":
                experience_score
        },

        "resume_sections":
            resume_data.get(
                "sections",
                {}
            ),

        "job_sections":
            job_data.get(
                "sections",
                {}
            ),

        "recommendations":
            recommendations,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("ANALYSIS ENGINE TEST")
    print("=" * 60)

    sample_resume = """
    Software Engineer with experience in Python,
    SQL, Docker and machine learning.

    EXPERIENCE
    Built backend applications using Python.
    Developed machine learning projects.
    Worked with SQL databases and Git.

    SKILLS
    Python
    SQL
    Docker
    Machine Learning

    EDUCATION
    Bachelor's degree in Computer Science.
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

    Responsibilities:

    Build scalable backend applications.
    Work with engineering teams.

    Qualifications:

    Bachelor's degree in Computer Science.
    """

    try:

        result = analyze_resume(
            sample_resume,
            sample_job
        )

        print("\nDETECTED RESUME SECTIONS:")

        for section in result[
            "resume_sections"
        ]:

            print(
                "-",
                section
            )

        print("\nDETECTED JOB SECTIONS:")

        for section in result[
            "job_sections"
        ]:

            print(
                "-",
                section
            )

        print("\nSUMMARY:")

        print(
            result["summary"]
        )

        print("\nOVERALL SCORE:")

        print(
            result["score"]["overall_score"],
            "%"
        )

        print("\nCATEGORY:")

        print(
            result["score"]["category"]
        )

        print("\nSCORE BREAKDOWN:")

        for key, value in result[
            "score"
        ]["breakdown"].items():

            print(
                f"{key}: {value}%"
            )

        print("\nSEMANTIC METHOD:")

        print(
            result[
                "semantic_analysis"
            ]["method"]
        )

        print("\nSEMANTIC SCORE:")

        print(
            result[
                "semantic_analysis"
            ]["percentage"],
            "%"
        )

        print("\nEXPERIENCE SCORE:")

        print(
            result[
                "experience_analysis"
            ]["score"],
            "%"
        )

        print("\nMATCHED SKILLS:")

        print(
            result[
                "skill_analysis"
            ]["matched_skills"]
        )

        print("\nMISSING SKILLS:")

        print(
            result[
                "skill_analysis"
            ]["missing_skills"]
        )

        print("\nRECOMMENDATIONS:")

        for recommendation in result[
            "recommendations"
        ]:

            print(
                "-",
                recommendation
            )

        print(
            "\nTest completed successfully."
        )

    except Exception as error:

        print("\nERROR:")

        print(
            type(error).__name__,
            ":",
            error
        )