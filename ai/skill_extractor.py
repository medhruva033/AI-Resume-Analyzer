"""
AI Resume Analyzer
Skill Extraction Engine

Extracts meaningful skills from resumes and job descriptions.

The engine is designed to work across different career fields
without depending on a fixed resume dataset.
"""

import re
from typing import Dict, List, Set


# ---------------------------------------------------------
# Skill vocabulary
# ---------------------------------------------------------
# This is an ontology of common professional skills.
# It is NOT a dataset of resumes.
#
# Later, this can be extended with an embedding/LLM-based
# discovery layer for completely unknown skills.

SKILL_ALIASES = {

    # Programming
    "python": "python",
    "java": "java",
    "c++": "c++",
    "c#": "c#",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "php": "php",
    "ruby": "ruby",
    "kotlin": "kotlin",
    "swift": "swift",

    # Web / Backend
    "html": "html",
    "html5": "html",
    "css": "css",
    "css3": "css",
    "react": "react",
    "reactjs": "react",
    "react.js": "react",
    "angular": "angular",
    "vue": "vue",
    "node.js": "node.js",
    "nodejs": "node.js",
    "express": "express",
    "express.js": "express",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "spring boot": "spring boot",
    "rest api": "rest api",
    "rest apis": "rest api",
    "graphql": "graphql",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "oracle": "oracle",
    "sqlite": "sqlite",

    # Cloud
    "aws": "aws",
    "amazon web services": "aws",
    "azure": "azure",
    "microsoft azure": "azure",
    "google cloud": "google cloud",
    "gcp": "google cloud",

    # DevOps
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "jenkins": "jenkins",
    "github actions": "github actions",
    "git": "git",
    "github": "github",
    "gitlab": "gitlab",
    "terraform": "terraform",

    # AI / ML
    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "dl": "deep learning",
    "natural language processing": "natural language processing",
    "nlp": "natural language processing",
    "computer vision": "computer vision",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "opencv": "opencv",
    "transformers": "transformers",
    "llm": "llm",
    "large language model": "llm",
    "generative ai": "generative ai",
    "rag": "rag",

    # Data
    "data analysis": "data analysis",
    "data analytics": "data analytics",
    "data science": "data science",
    "statistics": "statistics",
    "power bi": "power bi",
    "tableau": "tableau",
    "excel": "excel",

    # Software Engineering
    "data structures": "data structures",
    "algorithms": "algorithms",
    "dsa": "data structures and algorithms",
    "object oriented programming": "object oriented programming",
    "oop": "object oriented programming",
    "system design": "system design",
    "microservices": "microservices",
    "software testing": "software testing",
    "unit testing": "unit testing",
    "api development": "api development",

    # Marketing
    "digital marketing": "digital marketing",
    "seo": "seo",
    "search engine optimization": "seo",
    "google analytics": "google analytics",
    "content marketing": "content marketing",
    "content strategy": "content strategy",
    "social media marketing": "social media marketing",
    "social media": "social media",
    "campaign management": "campaign management",
    "market research": "market research",
    "advertising": "advertising",
    "email marketing": "email marketing",
    "brand management": "brand management",

    # Business
    "project management": "project management",
    "product management": "product management",
    "business analysis": "business analysis",
    "leadership": "leadership",
    "communication": "communication",
    "teamwork": "teamwork",
}


# ---------------------------------------------------------
# Normalize text
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for reliable skill matching.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalize common separators
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("•", " ")

    # Remove unnecessary punctuation while keeping
    # characters useful for technical skills.
    text = re.sub(r"[()\[\]{},;:|]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Find skills
# ---------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """
    Extract meaningful professional skills from text.

    Uses phrase matching against a normalized skill ontology.
    """

    if not text:
        return []

    normalized = normalize_text(text)

    found: Set[str] = set()

    # Sort longest phrases first.
    # This prevents:
    #
    # "machine learning"
    #
    # from being reduced to:
    #
    # "machine"
    #
    # or
    #
    # "learning"
    #
    aliases = sorted(
        SKILL_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for alias, canonical_skill in aliases:

        # Escape special characters such as C++, C#
        escaped_alias = re.escape(alias)

        pattern = rf"(?<!\w){escaped_alias}(?!\w)"

        if re.search(pattern, normalized):
            found.add(canonical_skill)

    return sorted(found)


# ---------------------------------------------------------
# Compare resume and job skills
# ---------------------------------------------------------

def compare_skills(
    resume_text: str,
    job_text: str
) -> Dict[str, List[str]]:
    """
    Compare skills between resume and job description.
    """

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    matched_skills = sorted(
        resume_skills.intersection(job_skills)
    )

    missing_skills = sorted(
        job_skills.difference(resume_skills)
    )

    resume_only_skills = sorted(
        resume_skills.difference(job_skills)
    )

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_only_skills": resume_only_skills,
    }


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_resume = """
    Marketing professional with experience in SEO,
    Google Analytics, content strategy, digital marketing,
    campaign management, social media and market research.
    """

    sample_job = """
    We are looking for a Marketing Specialist with experience
    in SEO, advertising, campaign management, content marketing,
    Google Analytics and market research.
    """

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("SKILL EXTRACTION ENGINE TEST")
    print("=" * 60)

    resume_skills = extract_skills(sample_resume)
    job_skills = extract_skills(sample_job)

    print("\nRESUME SKILLS:")
    print(resume_skills)

    print("\nJOB SKILLS:")
    print(job_skills)

    print("\nSKILL COMPARISON:")

    comparison = compare_skills(
        sample_resume,
        sample_job
    )

    print(comparison)

    print("\nTest completed successfully.")