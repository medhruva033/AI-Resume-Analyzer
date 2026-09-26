"""
AI Resume Analyzer
Resume Parser

Generic parser designed to work across different
career fields.

It extracts:
- Candidate name
- Email
- Phone
- Resume sections
"""

import re


# =========================================================
# EMAIL EXTRACTION
# =========================================================

def extract_email(text: str) -> str | None:
    """
    Extract an email address from resume text.

    Handles PDF extraction where text immediately after
    the email may be attached to the domain.

    Example:
        hachadadpraveen@gmail.comVijayapura

    Returns:
        hachadadpraveen@gmail.com
    """

    if not text:
        return None

    email_pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+"
        r"\."
        r"(?:"
        r"com|org|net|edu|gov|mil|"
        r"in|co\.in|ac\.in|org\.in|net\.in|"
        r"io|ai|dev|app|tech|me|info|biz|xyz"
        r")"
    )

    match = re.search(
        email_pattern,
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return match.group(0)


# =========================================================
# PHONE EXTRACTION
# =========================================================

def extract_phone(text: str) -> str | None:
    """
    Extract an Indian mobile phone number.

    Supports:

    9876543210
    +919876543210
    +91 9876543210
    +91-9876543210
    """

    if not text:
        return None

    phone_pattern = (
        r"(?:\+91[\s-]?)?"
        r"[6-9]\d{9}"
    )

    match = re.search(
        phone_pattern,
        text
    )

    return match.group(0) if match else None


# =========================================================
# RESUME SECTION EXTRACTION
# =========================================================

def extract_sections(text: str) -> dict:
    """
    Detect common resume sections dynamically.

    Works across different career fields.
    """

    section_names = [
        "SUMMARY",
        "PROFESSIONAL SUMMARY",
        "PROFILE",
        "ABOUT ME",
        "OBJECTIVE",
        "CAREER OBJECTIVE",
        "EDUCATION",
        "EXPERIENCE",
        "WORK EXPERIENCE",
        "EMPLOYMENT",
        "INTERNSHIP",
        "SKILLS",
        "TECHNICAL SKILLS",
        "CORE SKILLS",
        "PROJECTS",
        "CERTIFICATIONS",
        "ACHIEVEMENTS",
        "AWARDS",
        "LANGUAGES",
        "PUBLICATIONS",
        "INTERESTS",
        "REFERENCES",
    ]

    sections = {}

    current_section = "GENERAL"

    sections[current_section] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        upper_line = line.upper()

        # -------------------------------------------------
        # Remove common punctuation from headings
        # -------------------------------------------------

        cleaned_heading = re.sub(
            r"[:\-]+$",
            "",
            upper_line
        ).strip()

        # -------------------------------------------------
        # Detect section heading
        # -------------------------------------------------

        if cleaned_heading in section_names:

            current_section = cleaned_heading

            sections[current_section] = []

        else:

            sections[current_section].append(
                line
            )

    # -----------------------------------------------------
    # Convert lists to text
    # -----------------------------------------------------

    return {
        section: "\n".join(content).strip()
        for section, content in sections.items()
        if content
    }


# =========================================================
# NAME EXTRACTION
# =========================================================

def extract_name(text: str) -> str | None:
    """
    Try to identify the candidate name.

    Usually the candidate name appears near the beginning
    of the resume.
    """

    if not text:
        return None

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    # -----------------------------------------------------
    # Look at the first few lines
    # -----------------------------------------------------

    for line in lines[:5]:

        # Ignore obvious headings
        if line.upper() in {
            "RESUME",
            "CURRICULUM VITAE",
            "CV",
            "PROFILE",
        }:
            continue

        # Ignore lines containing email
        if "@" in line:
            continue

        # Ignore lines containing numbers
        if re.search(r"\d", line):
            continue

        words = line.split()

        # Basic name-like pattern
        if 2 <= len(words) <= 5:

            if all(
                re.match(
                    r"^[A-Za-z.'-]+$",
                    word
                )
                for word in words
            ):

                return line

    return None


# =========================================================
# MAIN RESUME PARSER
# =========================================================

def parse_resume(text: str) -> dict:
    """
    Parse resume text into structured information.
    """

    if not text or not text.strip():

        return {
            "name": None,
            "email": None,
            "phone": None,
            "sections": {},
        }

    sections = extract_sections(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "sections": sections,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("RESUME PARSER TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Normal resume test
    # -----------------------------------------------------

    sample_resume = """
    Dhruva M K
    dhruva@example.com
    +919876543210

    PROFESSIONAL SUMMARY
    Computer Science student interested in AI
    and software development.

    EDUCATION
    Bachelor's Degree in Computer Science

    SKILLS
    Python, C++, SQL, Machine Learning

    PROJECTS
    AI Resume Analyzer

    EXPERIENCE
    Software Development Intern
    """

    result = parse_resume(
        sample_resume
    )

    print("\nNAME:")
    print(
        result["name"]
    )

    print("\nEMAIL:")
    print(
        result["email"]
    )

    print("\nPHONE:")
    print(
        result["phone"]
    )

    print("\nSECTIONS:")

    for section, content in result[
        "sections"
    ].items():

        print(
            f"\n[{section}]"
        )

        print(
            content
        )

    # -----------------------------------------------------
    # PDF-style email test
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PDF EMAIL TEST")
    print("=" * 60)

    pdf_style_text = (
        "PRAVEEN HACHADAD\n"
        "Vijayapura,Karnataka,India : "
        "8951090239 "
        "hachadadpraveen@gmail.com"
        "Vijayapura,Karnataka,India"
    )

    print("\nEXTRACTED EMAIL:")

    print(
        extract_email(
            pdf_style_text
        )
    )

    print("\nEXTRACTED PHONE:")

    print(
        extract_phone(
            pdf_style_text
        )
    )

    print("\nTest completed successfully.")