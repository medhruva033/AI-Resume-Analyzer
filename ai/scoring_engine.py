"""
AI Resume Analyzer
Scoring Engine

Combines ATS, skill matching, semantic similarity,
and experience scores into one overall score.
"""


def calculate_score(
    ats_score: float,
    skill_match_score: float,
    semantic_score: float,
    experience_score: float,
) -> dict:
    """
    Calculate the overall resume-job match score.

    All input scores should be between 0 and 100.
    """

    # Keep every score within valid range
    ats_score = max(0, min(100, ats_score))
    skill_match_score = max(0, min(100, skill_match_score))
    semantic_score = max(0, min(100, semantic_score))
    experience_score = max(0, min(100, experience_score))

    # Weighted scoring
    #
    # Skills and semantic similarity are more important
    # for determining how well the resume matches the job.
    overall_score = (
        ats_score * 0.20
        + skill_match_score * 0.30
        + semantic_score * 0.30
        + experience_score * 0.20
    )

    overall_score = round(overall_score, 2)

    # Determine category
    if overall_score >= 85:
        category = "Excellent Match"
    elif overall_score >= 70:
        category = "Strong Match"
    elif overall_score >= 55:
        category = "Moderate Match"
    elif overall_score >= 40:
        category = "Weak Match"
    else:
        category = "Low Match"

    # Generate message
    if overall_score >= 85:
        message = (
            "The resume strongly aligns with the job requirements. "
            "Only minor improvements may be needed."
        )

    elif overall_score >= 70:
        message = (
            "The resume shows strong alignment with the job requirements. "
            "A few areas could still be improved."
        )

    elif overall_score >= 55:
        message = (
            "The resume shows moderate alignment with the job requirements. "
            "Several areas may need improvement."
        )

    elif overall_score >= 40:
        message = (
            "The resume shows limited alignment with the job requirements. "
            "Important skills and experience may be missing."
        )

    else:
        message = (
            "The resume has low alignment with the job requirements. "
            "Significant improvements may be required."
        )

    return {
        "overall_score": overall_score,
        "category": category,
        "message": message,
        "breakdown": {
            "ats_score": round(ats_score, 2),
            "skill_match_score": round(skill_match_score, 2),
            "semantic_score": round(semantic_score, 2),
            "experience_score": round(experience_score, 2),
        },
    }


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("SCORING ENGINE TEST")
    print("=" * 60)

    result = calculate_score(
        ats_score=80,
        skill_match_score=66.67,
        semantic_score=63.29,
        experience_score=70,
    )

    print("\nOVERALL SCORE:")
    print(result["overall_score"], "%")

    print("\nCATEGORY:")
    print(result["category"])

    print("\nMESSAGE:")
    print(result["message"])

    print("\nSCORE BREAKDOWN:")

    for key, value in result["breakdown"].items():
        print(f"{key}: {value}%")

    print("\nTest completed successfully.")