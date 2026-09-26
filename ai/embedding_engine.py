"""
AI Resume Analyzer
Embedding Engine

This module converts resume and job-description text into
numerical vectors (embeddings).

Embeddings allow the system to understand semantic meaning
rather than relying only on exact keyword matching.

Examples:

"Machine Learning" <-> "ML"
"financial analysis" <-> "analyzing financial statements"
"customer acquisition" <-> "growing customer base"
"software development" <-> "building applications"

The engine is designed to work across different career fields.
"""

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# GENERATE SINGLE EMBEDDING
# =========================================================

def generate_embedding(text: str) -> np.ndarray:
    """
    Convert a single piece of text into an embedding vector.

    Parameters
    ----------
    text:
        Text to convert into an embedding.

    Returns
    -------
    np.ndarray
        Numerical embedding vector.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding = model.encode(
        text.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


# =========================================================
# GENERATE MULTIPLE EMBEDDINGS
# =========================================================

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for multiple text inputs.

    Parameters
    ----------
    texts:
        List of text strings.

    Returns
    -------
    np.ndarray
        Matrix containing one embedding per text.
    """

    if not texts:
        raise ValueError("Text list cannot be empty.")

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        raise ValueError("No valid text was provided.")

    embeddings = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# =========================================================
# CALCULATE SEMANTIC SIMILARITY
# =========================================================

def calculate_semantic_similarity(
    text1: str,
    text2: str
) -> float:
    """
    Calculate semantic similarity between two text strings.

    Returns
    -------
    float
        Similarity score between 0 and 1.

    1.0 = highly similar
    0.0 = very different
    """

    if not text1 or not text1.strip():
        return 0.0

    if not text2 or not text2.strip():
        return 0.0

    embeddings = model.encode(
        [text1.strip(), text2.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    # Keep score safely between 0 and 1
    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return round(similarity, 4)


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def semantic_similarity(
    text_a: str,
    text_b: str
) -> float:
    """
    Backward-compatible wrapper.

    Existing code using semantic_similarity()
    will continue to work.
    """

    return calculate_semantic_similarity(
        text_a,
        text_b
    )


# =========================================================
# COMPARE RESUME WITH JOB DESCRIPTION
# =========================================================

def compare_resume_with_job(
    resume_text: str,
    job_description: str
) -> Dict:
    """
    Calculate semantic similarity between a resume
    and a job description.

    Returns a structured result containing:
    - semantic similarity
    - percentage score
    """

    score = calculate_semantic_similarity(
        resume_text,
        job_description
    )

    return {
        "semantic_similarity": score,
        "percentage": round(score * 100, 2)
    }


# =========================================================
# COMPARE INDIVIDUAL SKILLS
# =========================================================

def compare_skill_similarity(
    resume_skills: List[str],
    job_skills: List[str],
    threshold: float = 0.65
) -> Dict:
    """
    Compare resume skills against job-description skills
    using semantic similarity.

    This allows related skills to be recognized even when
    wording is different.

    Example:

    Resume:
        machine learning

    Job:
        ML

    These can be considered semantically related.
    """

    # -----------------------------------------------------
    # Empty resume skills
    # -----------------------------------------------------

    if not resume_skills:

        return {
            "matched": [],
            "missing": [
                {
                    "skill": skill,
                    "best_resume_match": None,
                    "similarity": 0.0
                }
                for skill in job_skills
            ],
            "resume_only": []
        }

    # -----------------------------------------------------
    # Empty job skills
    # -----------------------------------------------------

    if not job_skills:

        return {
            "matched": [],
            "missing": [],
            "resume_only": resume_skills
        }

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    resume_embeddings = generate_embeddings(
        resume_skills
    )

    job_embeddings = generate_embeddings(
        job_skills
    )

    # -----------------------------------------------------
    # Calculate similarity matrix
    # -----------------------------------------------------

    similarity_matrix = cosine_similarity(
        job_embeddings,
        resume_embeddings
    )

    matched = []
    missing = []

    matched_resume_skills = set()

    # -----------------------------------------------------
    # Compare every job skill
    # -----------------------------------------------------

    for job_index, job_skill in enumerate(job_skills):

        best_resume_index = int(
            np.argmax(
                similarity_matrix[job_index]
            )
        )

        best_score = float(
            similarity_matrix[
                job_index,
                best_resume_index
            ]
        )

        best_resume_skill = resume_skills[
            best_resume_index
        ]

        # -------------------------------------------------
        # Skill matched
        # -------------------------------------------------

        if best_score >= threshold:

            matched.append(
                {
                    "job_skill": job_skill,
                    "resume_skill": best_resume_skill,
                    "similarity": round(
                        best_score,
                        4
                    )
                }
            )

            matched_resume_skills.add(
                best_resume_skill
            )

        # -------------------------------------------------
        # Skill missing
        # -------------------------------------------------

        else:

            missing.append(
                {
                    "skill": job_skill,
                    "best_resume_match": best_resume_skill,
                    "similarity": round(
                        best_score,
                        4
                    )
                }
            )

    # -----------------------------------------------------
    # Resume-only skills
    # -----------------------------------------------------

    resume_only = [
        skill
        for skill in resume_skills
        if skill not in matched_resume_skills
    ]

    return {
        "matched": matched,
        "missing": missing,
        "resume_only": resume_only
    }


# =========================================================
# FIND MOST SIMILAR TEXT
# =========================================================

def find_most_similar(
    query: str,
    candidates: List[str],
    top_k: int = 5
) -> List[Dict]:
    """
    Find the most semantically similar candidates
    to a query.

    Useful for:

    - Matching projects to job requirements
    - Matching experience to requirements
    - Finding relevant resume sections
    - Career recommendations
    """

    if not query or not query.strip():
        return []

    if not candidates:
        return []

    # -----------------------------------------------------
    # Clean candidates
    # -----------------------------------------------------

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate and candidate.strip()
    ]

    if not valid_candidates:
        return []

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    query_embedding = generate_embedding(
        query
    )

    candidate_embeddings = generate_embeddings(
        valid_candidates
    )

    # -----------------------------------------------------
    # Calculate similarity
    # -----------------------------------------------------

    scores = cosine_similarity(
        [query_embedding],
        candidate_embeddings
    )[0]

    results = []

    for index, score in enumerate(scores):

        results.append(
            {
                "text": valid_candidates[index],
                "similarity": round(
                    float(score),
                    4
                )
            }
        )

    # -----------------------------------------------------
    # Sort highest similarity first
    # -----------------------------------------------------

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# RANK RESUME SECTIONS
# =========================================================

def rank_resume_sections(
    sections: Dict[str, str],
    job_description: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Rank resume sections according to their semantic
    relevance to a job description.

    Example:

    {
        "summary": "...",
        "experience": "...",
        "projects": "...",
        "education": "..."
    }
    """

    if not sections:
        return []

    if not job_description or not job_description.strip():
        return []

    # -----------------------------------------------------
    # Keep only valid sections
    # -----------------------------------------------------

    valid_sections = {
        name: content
        for name, content in sections.items()
        if content and content.strip()
    }

    if not valid_sections:
        return []

    section_names = list(
        valid_sections.keys()
    )

    section_texts = list(
        valid_sections.values()
    )

    # -----------------------------------------------------
    # Find most similar sections
    # -----------------------------------------------------

    results = find_most_similar(
        job_description,
        section_texts,
        top_k=top_k
    )

    # -----------------------------------------------------
    # Add section name
    # -----------------------------------------------------

    for result in results:

        section_text = result["text"]

        try:
            index = section_texts.index(
                section_text
            )

            result["section"] = section_names[
                index
            ]

        except ValueError:

            result["section"] = "unknown"

        del result["text"]

    return results


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    resume_text = """
    I developed machine learning models using Python
    and worked on predictive analytics projects.
    """

    job_description = """
    Looking for a candidate with experience in ML,
    Python and predictive modeling.
    """

    print("\n" + "-" * 50)
    print("AI RESUME ANALYZER")
    print("EMBEDDING ENGINE TEST")
    print("-" * 50)

    # -----------------------------------------------------
    # Semantic similarity test
    # -----------------------------------------------------

    score = calculate_semantic_similarity(
        resume_text,
        job_description
    )

    print("\nSEMANTIC SIMILARITY:")
    print(score)

    print("\nSEMANTIC MATCH:")
    print(f"{score * 100:.2f}%")

    # -----------------------------------------------------
    # Resume vs Job test
    # -----------------------------------------------------

    comparison = compare_resume_with_job(
        resume_text,
        job_description
    )

    print("\nRESUME vs JOB:")
    print(comparison)

    # -----------------------------------------------------
    # Skill similarity test
    # -----------------------------------------------------

    resume_skills = [
        "Python",
        "machine learning",
        "SQL"
    ]

    job_skills = [
        "Python",
        "ML",
        "SQL",
        "Docker"
    ]

    skill_result = compare_skill_similarity(
        resume_skills,
        job_skills
    )

    print("\nSKILL SEMANTIC MATCHING:")
    print(skill_result)

    # -----------------------------------------------------
    # Most similar text test
    # -----------------------------------------------------

    candidates = [
        "Built machine learning models using Python.",
        "Created a responsive frontend website.",
        "Analyzed customer data using SQL."
    ]

    similar = find_most_similar(
        "machine learning development",
        candidates,
        top_k=3
    )

    print("\nMOST SIMILAR TEXT:")
    for item in similar:
        print(item)

    print("\nTest completed successfully.")