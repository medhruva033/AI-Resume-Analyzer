"""
AI Resume Analyzer
Job Description Parser

Generic parser designed to work across different career fields.

It extracts:
- Job title / general information
- Requirements
- Responsibilities
- Qualifications
- Experience requirements
- Education requirements
- Keywords
"""

import re
from typing import Dict, List


# ---------------------------------------------------------
# SECTION HEADINGS
# ---------------------------------------------------------

SECTION_PATTERNS = {
    "requirements": [
        "requirements",
        "required skills",
        "required qualifications",
        "minimum qualifications",
        "what you need",
        "skills required",
        "technical requirements",
    ],
    "responsibilities": [
        "responsibilities",
        "roles and responsibilities",
        "role and responsibilities",
        "job responsibilities",
        "duties",
        "what you'll do",
        "what you will do",
        "key responsibilities",
    ],
    "qualifications": [
        "qualifications",
        "preferred qualifications",
        "academic qualifications",
        "education",
        "educational requirements",
    ],
}


# ---------------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize job description text."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


# ---------------------------------------------------------
# DETECT SECTION
# ---------------------------------------------------------

def detect_section(line: str) -> str | None:
    """
    Detect a known JD section heading.

    Supports both:

        Requirements:

    and:

        Job Title: Software Engineer Requirements: ...
    """

    cleaned = line.strip().lower()

    if not cleaned:
        return None

    # Remove heading punctuation from the end.
    cleaned = re.sub(r"[:\-]+$", "", cleaned).strip()

    # Exact heading match.
    for section, headings in SECTION_PATTERNS.items():
        for heading in headings:
            if cleaned == heading:
                return section

    # Heading appearing inside a line.
    for section, headings in SECTION_PATTERNS.items():
        for heading in headings:
            pattern = rf"\b{re.escape(heading)}\s*:"

            if re.search(
                pattern,
                cleaned,
                flags=re.IGNORECASE
            ):
                return section

    return None


# ---------------------------------------------------------
# EXTRACT SECTIONS
# ---------------------------------------------------------

def extract_sections(text: str) -> Dict[str, str]:
    """
    Split a job description into logical sections.

    Handles both:

    1. Normal multiline JD:

       Requirements:
       - Python
       - SQL

       Responsibilities:
       - Build applications

    2. Single-line JD produced by form submission:

       Job Title: Software Engineer Requirements:
       - Python - SQL Responsibilities:
       - Build applications
    """

    text = normalize_text(text)

    sections = {
        "general": [],
        "requirements": [],
        "responsibilities": [],
        "qualifications": [],
    }

    # -----------------------------------------------------
    # Insert line breaks before recognized section headings.
    # This is important because Swagger/form submission may
    # collapse the entire JD into one line.
    # -----------------------------------------------------

    for section, headings in SECTION_PATTERNS.items():

        for heading in sorted(
            headings,
            key=len,
            reverse=True
        ):

            pattern = rf"\b{re.escape(heading)}\s*:"

            text = re.sub(
                pattern,
                f"\n{heading.upper()}:\n",
                text,
                flags=re.IGNORECASE
            )

    current_section = "general"

    # -----------------------------------------------------
    # Process normalized lines.
    # -----------------------------------------------------

    for line in text.split("\n"):

        stripped = line.strip()

        if not stripped:
            continue

        detected = detect_section(stripped)

        if detected:

            current_section = detected

            # If anything exists after the heading,
            # preserve it.
            parts = stripped.split(":", 1)

            if len(parts) == 2:

                remaining = parts[1].strip()

                if remaining:
                    sections[current_section].append(
                        remaining
                    )

            continue

        sections[current_section].append(stripped)

    # -----------------------------------------------------
    # Remove empty sections.
    # -----------------------------------------------------

    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
    }


# ---------------------------------------------------------
# EXTRACT EXPERIENCE
# ---------------------------------------------------------

def extract_experience(text: str) -> List[str]:
    """
    Extract experience requirements.

    Examples:
    - 2+ years of experience
    - 3 years experience
    - 5-7 years of experience
    - minimum 2 years
    - at least 2 years
    """

    if not text:
        return []

    patterns = [
        r"\b\d+\s*\+?\s*(?:-\s*\d+)?\s*years?\s+of\s+experience\b",
        r"\b\d+\s*\+?\s*(?:-\s*\d+)?\s*years?\s+experience\b",
        r"\bminimum\s+\d+\s*years?\b",
        r"\bat\s+least\s+\d+\s*years?\b",
    ]

    results = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            cleaned = match.strip()

            if cleaned not in results:
                results.append(cleaned)

    return results


# ---------------------------------------------------------
# EXTRACT EDUCATION
# ---------------------------------------------------------

def extract_education(text: str) -> List[str]:
    """
    Extract education requirements.

    Searches qualification/education content only.
    """

    if not text:
        return []

    education_patterns = [
        r"\b(?:bachelor'?s?|b\.?s\.?|b\.?e\.?|b\.?tech)\b.*",
        r"\b(?:master'?s?|m\.?s\.?|m\.?e\.?|m\.?tech)\b.*",
        r"\b(?:ph\.?d\.?|doctorate)\b.*",
        r"\b(?:diploma)\b.*",
        r"\b(?:degree)\b.*",
        r"\b(?:college|university)\b.*",
        r"\b(?:graduate|postgraduate)\b.*",
    ]

    results = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        for pattern in education_patterns:

            match = re.search(
                pattern,
                line,
                flags=re.IGNORECASE
            )

            if match:

                cleaned = line.lstrip(
                    "-•* "
                ).strip()

                if cleaned not in results:
                    results.append(cleaned)

                break

    return results


# ---------------------------------------------------------
# EXTRACT KEYWORDS
# ---------------------------------------------------------

def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from a job description.

    This is a baseline keyword extractor.
    Later the AI engine can combine this with
    semantic embeddings and LLM reasoning.
    """

    if not text:
        return []

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "with",
        "we",
        "you",
        "your",
        "will",
        "this",
        "that",
        "their",
        "they",
        "should",
        "can",
        "must",
        "job",
        "role",
        "work",
        "working",
        "team",
        "teams",
        "candidate",
        "candidates",
        "experience",
        "requirements",
        "responsibilities",
        "qualifications",
    }

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.\-]*\b",
        text.lower()
    )

    keywords = []

    for word in words:

        word = word.strip(
            ".,:;()[]{}"
        )

        if not word:
            continue

        if word in stop_words:
            continue

        if len(word) < 2:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


# ---------------------------------------------------------
# PARSE JOB DESCRIPTION
# ---------------------------------------------------------

def parse_job_description(text: str) -> Dict:
    """
    Parse a complete job description into structured data.
    """

    if not text or not text.strip():

        return {
            "sections": {},
            "requirements": [],
            "responsibilities": [],
            "experience_requirements": [],
            "education_requirements": [],
            "keywords": [],
        }

    text = normalize_text(text)

    sections = extract_sections(text)

    requirements = sections["requirements"]

    responsibilities = sections["responsibilities"]

    qualifications = sections["qualifications"]

    # -----------------------------------------------------
    # Experience requirements
    # -----------------------------------------------------

    experience_requirements = extract_experience(
        requirements + "\n" + qualifications
    )

    # -----------------------------------------------------
    # Education requirements
    # -----------------------------------------------------

    education_requirements = extract_education(
       requirements + "\n" + qualifications
    )

    # -----------------------------------------------------
    # Keywords
    # -----------------------------------------------------

    keywords = extract_keywords(text)

    return {
        "sections": sections,

        "requirements": (
            requirements.split("\n")
            if requirements
            else []
        ),

        "responsibilities": (
            responsibilities.split("\n")
            if responsibilities
            else []
        ),

        "experience_requirements": (
            experience_requirements
        ),

        "education_requirements": (
            education_requirements
        ),

        "keywords": keywords,
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("JOB DESCRIPTION PARSER TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Normal multiline JD
    # -----------------------------------------------------

    sample_job = """
Software Engineer

Requirements:
- Strong Python programming skills
- Experience with REST APIs
- Knowledge of SQL
- 2+ years of experience

Responsibilities:
- Build scalable backend applications
- Work with engineering teams

Qualifications:
- Bachelor's degree in Computer Science or related field.
"""

    result = parse_job_description(sample_job)

    print("\nSECTIONS:")
    print(result["sections"])

    print("\nREQUIREMENTS:")
    print(result["requirements"])

    print("\nRESPONSIBILITIES:")
    print(result["responsibilities"])

    print("\nEXPERIENCE:")
    print(result["experience_requirements"])

    print("\nEDUCATION:")
    print(result["education_requirements"])

    print("\nKEYWORDS:")
    print(result["keywords"])

    # -----------------------------------------------------
    # Single-line JD test
    # -----------------------------------------------------

    single_line_job = (
        "Job Title: Junior Software Developer "
        "Requirements: "
        "- Diploma or Bachelor's degree in Computer Science, "
        "Information Technology, or a related field "
        "- Good knowledge of C and C++ "
        "- Basic knowledge of programming and computer applications "
        "- Problem-solving and logical thinking skills "
        "- Good communication and teamwork skills "
        "- Ability to learn new technologies quickly "
        "- Basic understanding of software development and debugging "
        "- Leadership skills are an advantage "
        "Responsibilities: "
        "- Develop and maintain software applications "
        "- Write, test, and debug C/C++ programs "
        "- Troubleshoot basic software issues "
        "- Work with the development team to complete projects "
        "- Participate in software testing and documentation "
        "- Learn and apply new technologies"
    )

    single_result = parse_job_description(
        single_line_job
    )

    print("\n" + "=" * 60)
    print("SINGLE-LINE JD TEST")
    print("=" * 60)

    print("\nSECTIONS:")
    print(single_result["sections"])

    print("\nREQUIREMENTS:")
    print(single_result["requirements"])

    print("\nRESPONSIBILITIES:")
    print(single_result["responsibilities"])

    print("\nEXPERIENCE:")
    print(single_result["experience_requirements"])

    print("\nEDUCATION:")
    print(single_result["education_requirements"])

    print("\nKEYWORDS:")
    print(single_result["keywords"])

    print("\nTest completed successfully.")