"""
AI Resume Analyzer
Resume Rewriter

Generates structured resume improvement suggestions.

This module is designed to work across different career fields.
It does not assume a specific degree, industry, or job role.
"""

from typing import Dict, List, Any


def improve_summary(
    current_summary: str,
    target_role: str = ""
) -> str:
    """
    Generate a basic improved version of a professional summary.

    This is the baseline version.
    LLM-powered rewriting will be integrated later.
    """

    if not current_summary or not current_summary.strip():
        return (
            "Add a concise professional summary describing your "
            "experience, key strengths, and career focus."
        )

    summary = current_summary.strip()

    # Remove excessive whitespace
    summary = " ".join(summary.split())

    if target_role:
        return (
            f"{summary} "
            f"Targeting opportunities as a {target_role}."
        )

    return summary


def improve_bullet(
    bullet: str,
    action: str = "improve"
) -> str:
    """
    Improve a resume bullet at a basic structural level.

    Later, an LLM will transform this into a stronger,
    evidence-based achievement statement.
    """

    if not bullet or not bullet.strip():
        return ""

    bullet = " ".join(bullet.strip().split())

    if action == "improve":
        return f"• {bullet}"

    return bullet


def generate_missing_skill_suggestions(
    missing_skills: List[str]
) -> List[str]:
    """
    Generate suggestions for skills missing from the resume.

    Important:
    The system does NOT claim the candidate has these skills.
    It only recommends considering them when relevant.
    """

    suggestions = []

    for skill in missing_skills:

        if not skill or not skill.strip():
            continue

        clean_skill = skill.strip()

        suggestions.append(
            f"Consider adding or developing '{clean_skill}' "
            f"if it is relevant to your actual experience."
        )

    return suggestions


def generate_resume_improvements(
    resume_data: Dict[str, Any],
    missing_skills: List[str] | None = None,
    target_role: str = ""
) -> Dict[str, Any]:
    """
    Generate structured resume improvement recommendations.

    Args:
        resume_data:
            Structured resume information.

        missing_skills:
            Skills detected in the job description but not
            demonstrated in the resume.

        target_role:
            Optional target job role.

    Returns:
        Structured improvement suggestions.
    """

    if missing_skills is None:
        missing_skills = []

    summary = resume_data.get("summary", "")

    improvements = []

    # Summary improvement
    if summary:
        improvements.append({
            "section": "Summary",
            "suggestion": improve_summary(
                summary,
                target_role
            )
        })
    else:
        improvements.append({
            "section": "Summary",
            "suggestion":
                "Add a concise professional summary highlighting "
                "your strongest relevant qualifications."
        })

    # Missing skills
    skill_suggestions = generate_missing_skill_suggestions(
        missing_skills
    )

    for suggestion in skill_suggestions:
        improvements.append({
            "section": "Skills",
            "suggestion": suggestion
        })

    # General evidence-based recommendations
    improvements.extend([
        {
            "section": "Experience",
            "suggestion":
                "Use specific responsibilities, achievements, "
                "and measurable outcomes where supported by your "
                "actual experience."
        },
        {
            "section": "Projects",
            "suggestion":
                "Describe relevant projects using the problem, "
                "your contribution, tools or methods used, and "
                "the resulting outcome."
        },
        {
            "section": "ATS",
            "suggestion":
                "Use clear section headings, consistent formatting, "
                "and terminology that accurately reflects the job "
                "description."
        }
    ])

    return {
        "target_role": target_role,
        "improvements": improvements
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("RESUME REWRITER TEST")
    print("=" * 60)

    sample_resume = {
        "summary":
            "Diploma graduate with basic knowledge of "
            "programming and computer applications."
    }

    missing_skills = [
        "AWS",
        "FastAPI"
    ]

    result = generate_resume_improvements(
        resume_data=sample_resume,
        missing_skills=missing_skills,
        target_role="Software Engineer"
    )

    print("\nTARGET ROLE:")
    print(result["target_role"])

    print("\nIMPROVEMENTS:")

    for item in result["improvements"]:
        print(f"\n[{item['section']}]")
        print(item["suggestion"])

    print("\nTest completed successfully.")