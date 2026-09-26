"""
AI Resume Analyzer
ATS Analyzer

Analyzes resume compatibility with Applicant Tracking Systems.
"""

import re
from typing import Dict, List


# ---------------------------------------------------------
# CONTACT INFORMATION
# ---------------------------------------------------------

def check_contact_information(text: str) -> Dict[str, bool]:
    """
    Check whether the resume contains basic contact information.
    """

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    phone_pattern = r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"

    email_found = bool(re.search(email_pattern, text))

    phone_found = bool(re.search(phone_pattern, text))

    return {
        "email": email_found,
        "phone": phone_found
    }


# ---------------------------------------------------------
# SECTION DETECTION
# ---------------------------------------------------------

def detect_sections(text: str) -> Dict[str, bool]:
    """
    Detect common resume sections.
    """

    text_lower = text.lower()

    section_patterns = {
        "experience": [
            "experience",
            "work experience",
            "professional experience",
            "employment"
        ],

        "education": [
            "education",
            "academic background",
            "qualifications"
        ],

        "skills": [
            "skills",
            "technical skills",
            "core skills"
        ],

        "projects": [
            "projects",
            "academic projects",
            "personal projects"
        ],

        "certifications": [
            "certifications",
            "certificates",
            "licenses"
        ],

        "summary": [
            "summary",
            "professional summary",
            "profile",
            "objective"
        ]
    }

    result = {}

    for section, patterns in section_patterns.items():

        result[section] = any(
            pattern in text_lower
            for pattern in patterns
        )

    return result


# ---------------------------------------------------------
# KEYWORD MATCHING
# ---------------------------------------------------------

def calculate_keyword_score(
    resume_text: str,
    job_keywords: List[str]
) -> Dict:
    """
    Compare job-description keywords with resume text.
    """

    resume_lower = resume_text.lower()

    matched_keywords = []
    missing_keywords = []

    for keyword in job_keywords:

        keyword_clean = keyword.strip()

        if not keyword_clean:
            continue

        if keyword_clean.lower() in resume_lower:
            matched_keywords.append(keyword_clean)
        else:
            missing_keywords.append(keyword_clean)

    total_keywords = len(matched_keywords) + len(missing_keywords)

    if total_keywords == 0:
        score = 0.0
    else:
        score = (
            len(matched_keywords) / total_keywords
        ) * 100

    return {
        "score": round(score, 2),
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords
    }


# ---------------------------------------------------------
# RESUME LENGTH
# ---------------------------------------------------------

def analyze_resume_length(text: str) -> Dict:
    """
    Analyze approximate resume length.
    """

    words = text.split()

    word_count = len(words)

    if word_count < 150:
        category = "very_short"

    elif word_count < 300:
        category = "short"

    elif word_count <= 1000:
        category = "good"

    else:
        category = "long"

    return {
        "word_count": word_count,
        "category": category
    }


# ---------------------------------------------------------
# ATS ISSUES
# ---------------------------------------------------------

def detect_ats_issues(
    text: str,
    contact_info: Dict[str, bool],
    sections: Dict[str, bool],
    keyword_analysis: Dict
) -> List[str]:
    """
    Detect potential ATS problems.
    """

    issues = []

    # Contact information

    if not contact_info["email"]:
        issues.append(
            "Email address was not detected."
        )

    if not contact_info["phone"]:
        issues.append(
            "Phone number was not detected."
        )

    # Important sections

    required_sections = [
        "experience",
        "education",
        "skills"
    ]

    for section in required_sections:

        if not sections.get(section, False):

            issues.append(
                f"{section.title()} section was not detected."
            )

    # Resume length

    length_analysis = analyze_resume_length(text)

    if length_analysis["category"] == "very_short":

        issues.append(
            "Resume appears unusually short."
        )

    elif length_analysis["category"] == "long":

        issues.append(
            "Resume may be longer than necessary."
        )

    # Missing keywords

    if keyword_analysis["missing_keywords"]:

        issues.append(
            "Some important job-description keywords "
            "were not found in the resume."
        )

    return issues


# ---------------------------------------------------------
# ATS SCORE
# ---------------------------------------------------------

def calculate_ats_score(
    contact_info: Dict[str, bool],
    sections: Dict[str, bool],
    keyword_score: float
) -> float:
    """
    Calculate an ATS compatibility score.
    """

    # Contact score
    contact_score = (
        sum(contact_info.values()) / 2
    ) * 100

    # Important sections
    required_sections = [
        "experience",
        "education",
        "skills"
    ]

    section_score = (
        sum(
            sections.get(section, False)
            for section in required_sections
        )
        / len(required_sections)
    ) * 100

    # Weighted ATS score
    ats_score = (
        contact_score * 0.20
        + section_score * 0.30
        + keyword_score * 0.50
    )

    return round(ats_score, 2)


# ---------------------------------------------------------
# MAIN ATS ANALYSIS
# ---------------------------------------------------------

def analyze_ats(
    resume_text: str,
    job_keywords: List[str]
) -> Dict:
    """
    Perform complete ATS analysis.
    """

    contact_info = check_contact_information(
        resume_text
    )

    sections = detect_sections(
        resume_text
    )

    keyword_analysis = calculate_keyword_score(
        resume_text,
        job_keywords
    )

    length_analysis = analyze_resume_length(
        resume_text
    )

    issues = detect_ats_issues(
        resume_text,
        contact_info,
        sections,
        keyword_analysis
    )

    ats_score = calculate_ats_score(
        contact_info,
        sections,
        keyword_analysis["score"]
    )

    return {
        "ats_score": ats_score,

        "contact": contact_info,

        "sections": sections,

        "keyword_score": keyword_analysis["score"],

        "matched_keywords":
            keyword_analysis["matched_keywords"],

        "missing_keywords":
            keyword_analysis["missing_keywords"],

        "resume_length":
            length_analysis,

        "issues": issues
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_resume = """
    John Doe
    john@example.com
    +919876543210

    Professional Summary

    Software Engineer with experience building
    backend applications.

    Experience

    Software Developer
    Developed applications using Python and SQL.
    Worked with Docker.

    Education

    Bachelor's degree in Computer Science.

    Skills

    Python
    SQL
    Docker

    Projects

    Resume Analyzer
    """

    sample_job_keywords = [
        "Python",
        "SQL",
        "Docker",
        "AWS",
        "FastAPI"
    ]

    result = analyze_ats(
        sample_resume,
        sample_job_keywords
    )

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("ATS ANALYZER TEST")
    print("=" * 60)

    print("\nATS SCORE:")
    print(result["ats_score"], "%")

    print("\nCONTACT:")
    print(result["contact"])

    print("\nSECTIONS:")
    print(result["sections"])

    print("\nKEYWORD SCORE:")
    print(result["keyword_score"], "%")

    print("\nMATCHED KEYWORDS:")
    print(result["matched_keywords"])

    print("\nMISSING KEYWORDS:")
    print(result["missing_keywords"])

    print("\nRESUME LENGTH:")
    print(result["resume_length"])

    print("\nATS ISSUES:")

    if result["issues"]:
        for issue in result["issues"]:
            print("-", issue)
    else:
        print("No major ATS issues detected.")

    print("\nTest completed successfully.")