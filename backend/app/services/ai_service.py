import re


COMMON_SKILLS = {
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "react",
    "node.js",
    "fastapi",
    "flask",
    "django",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "html",
    "css",
}


def extract_skills(text: str) -> list[str]:

    text_lower = text.lower()

    found = []

    for skill in COMMON_SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found.append(skill)

    return sorted(found)


def calculate_match_score(
    resume_text: str,
    job_description: str,
) -> tuple[float, list[str], list[str]]:

    resume_skills = set(
        extract_skills(resume_text)
    )

    job_skills = set(
        extract_skills(job_description)
    )

    if not job_skills:

        return 0.0, sorted(resume_skills), []

    matched = resume_skills.intersection(job_skills)

    missing = job_skills - resume_skills

    score = (
        len(matched) / len(job_skills)
    ) * 100

    return (
        round(score, 2),
        sorted(matched),
        sorted(missing),
    )


def calculate_ats_score(
    resume_text: str,
) -> float:

    if not resume_text.strip():
        return 0.0

    score = 50.0

    text_lower = resume_text.lower()

    sections = [
        "education",
        "experience",
        "skills",
        "projects",
        "contact",
    ]

    for section in sections:

        if section in text_lower:
            score += 8

    if len(resume_text) > 1000:
        score += 5

    if len(resume_text) > 2000:
        score += 5

    return min(round(score, 2), 100.0)


def generate_recommendations(
    missing_skills: list[str],
    ats_score: float,
) -> list[str]:

    recommendations = []

    if ats_score < 70:

        recommendations.append(
            "Improve resume structure and add clear ATS-friendly sections."
        )

    if missing_skills:

        recommendations.append(
            "Add relevant missing skills only if you genuinely have those skills."
        )

        recommendations.append(
            "Consider building projects demonstrating: "
            + ", ".join(missing_skills[:5])
        )

    if not recommendations:

        recommendations.append(
            "Your resume has a reasonable structure. Continue tailoring it to each job description."
        )

    return recommendations


def generate_ai_analysis(
    ats_score: float,
    match_score: float,
    missing_skills: list[str],
) -> str:

    return (
        f"Resume ATS score: {ats_score}/100. "
        f"Job match score: {match_score}/100. "
        f"Missing skills identified: "
        f"{', '.join(missing_skills) if missing_skills else 'None identified'}."
    )