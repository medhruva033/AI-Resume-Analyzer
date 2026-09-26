"""
AI Resume Analyzer
Recommendation Engine

Generates personalized recommendations based on the
actual resume and job description analysis.

Designed to work across different career fields.
"""


from typing import Dict, List


def recommend_missing_skills(
    missing_skills: List[str]
) -> List[str]:
    """
    Generate recommendations for skills missing
    from the resume.
    """

    recommendations = []

    for skill in missing_skills:
        recommendations.append(
            f"Consider adding '{skill}' to your resume "
            f"only if you genuinely have relevant experience "
            f"with this skill."
        )

    return recommendations


def recommend_resume_improvements(
    resume_data: Dict
) -> List[str]:
    """
    Analyze the resume structure and suggest improvements.
    """

    recommendations = []

    sections = resume_data.get("sections", {})

    if not sections:
        recommendations.append(
            "Your resume does not appear to have clearly "
            "structured sections."
        )
        return recommendations

    section_names = {
        section.lower()
        for section in sections.keys()
    }

    if not any(
        section in section_names
        for section in [
            "experience",
            "work experience",
            "employment"
        ]
    ):
        recommendations.append(
            "Consider adding a clearly structured "
            "work experience section."
        )

    if not any(
        section in section_names
        for section in [
            "education",
            "academic"
        ]
    ):
        recommendations.append(
            "Consider adding an education section."
        )

    if not any(
        section in section_names
        for section in [
            "skills",
            "technical skills",
            "core skills"
        ]
    ):
        recommendations.append(
            "Consider adding a dedicated skills section."
        )

    return recommendations


def generate_recommendations(
    resume_data: Dict,
    analysis: Dict
) -> List[str]:
    """
    Generate complete personalized recommendations.
    """

    recommendations = []

    # -------------------------------------------------
    # 1. Missing Skills
    # -------------------------------------------------

    missing_skills = analysis.get(
        "skill_analysis",
        {}
    ).get(
        "missing_skills",
        []
    )

    recommendations.extend(
        recommend_missing_skills(
            missing_skills
        )
    )

    # -------------------------------------------------
    # 2. Resume Structure
    # -------------------------------------------------

    recommendations.extend(
        recommend_resume_improvements(
            resume_data
        )
    )

    # -------------------------------------------------
    # 3. Match Score
    # -------------------------------------------------

    score = analysis.get(
        "skill_analysis",
        {}
    ).get(
        "match_score",
        0
    )

    if score < 40:

        recommendations.append(
            "The resume has relatively low alignment "
            "with the provided job description. "
            "Review the job requirements and highlight "
            "relevant experience that you genuinely possess."
        )

    elif score < 70:

        recommendations.append(
            "The resume has moderate alignment. "
            "Improve the visibility of relevant skills, "
            "experience, projects, and achievements."
        )

    else:

        recommendations.append(
            "The resume shows strong skill alignment. "
            "Focus on measurable achievements and "
            "clear evidence of impact."
        )

    # -------------------------------------------------
    # Remove duplicate recommendations
    # -------------------------------------------------

    unique_recommendations = []

    for recommendation in recommendations:

        if recommendation not in unique_recommendations:
            unique_recommendations.append(
                recommendation
            )

    return unique_recommendations


def get_top_recommendations(
    recommendations: List[str],
    limit: int = 5
) -> List[str]:
    """
    Return the most important recommendations.
    """

    return recommendations[:limit]


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_resume = {

        "name": "John Doe",

        "email": "john@example.com",

        "phone": "9876543210",

        "skills": [
            "Python",
            "SQL",
            "Docker"
        ],

        "sections": {

            "experience":
                "Software Engineer",

            "education":
                "Bachelor's Degree",

            "skills":
                "Python, SQL, Docker"

        }
    }


    sample_analysis = {

        "skill_analysis": {

            "match_score": 60.0,

            "matched_skills": [
                "docker",
                "python",
                "sql"
            ],

            "missing_skills": [
                "aws",
                "fastapi"
            ],

            "additional_skills": [
                "git"
            ]
        }
    }


    recommendations = generate_recommendations(
        sample_resume,
        sample_analysis
    )


    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("RECOMMENDATION ENGINE TEST")
    print("=" * 60)


    print("\nRECOMMENDATIONS:")

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )


    print("\nTOP RECOMMENDATIONS:")

    top = get_top_recommendations(
        recommendations
    )

    for recommendation in top:

        print(
            "-",
            recommendation
        )


    print("\nTest completed successfully.")