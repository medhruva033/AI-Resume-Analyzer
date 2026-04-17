import streamlit as st
from pypdf import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher   # 🔥 important

# Load NLP model
nlp = spacy.load("en_core_web_sm")

st.title("AI Resume Analyzer 🚀")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")


# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


# -------- NLP PREPROCESSING --------
def preprocess_text(text):
    doc = nlp(text)
    clean_text = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            clean_text.append(token.lemma_)

    return " ".join(clean_text).lower()


# -------- KEYWORD EXPANSION --------
def expand_keywords(text):
    synonyms = {
        "sde": "software engineer",
        "software": "developer programmer",
        "dsa": "data structures algorithms",
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "c++": "programming",
        "python": "programming"
    }

    words = text.split()
    expanded = words.copy()

    for word in words:
        if word in synonyms:
            expanded.extend(synonyms[word].split())

    return " ".join(expanded)


# -------- 🔥 FIXED SIMILARITY FUNCTION --------
def calculate_similarity(resume, job_desc):
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    vectors = vectorizer.fit_transform([resume, job_desc])
    tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Sequence similarity (important fix)
    seq_score = SequenceMatcher(None, resume, job_desc).ratio()

    # Final combined score
    final_score = (tfidf_score * 0.7) + (seq_score * 0.3)

    return final_score


# -------- MISSING SKILLS --------
def get_missing_skills(resume, job_desc):
    resume_words = set(resume.split())
    jd_words = set(job_desc.split())

    missing = jd_words - resume_words
    return list(missing)[:10]


# -------- MAIN LOGIC --------
if uploaded_file is not None and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text.strip():
        st.error("❌ Could not extract text from PDF")
    else:
        st.success("Resume Text Extracted Successfully ✅")

        # Clean text
        clean_resume = preprocess_text(resume_text)
        clean_jd = preprocess_text(job_description)

        # Expand keywords
        expanded_resume = expand_keywords(clean_resume)
        expanded_jd = expand_keywords(clean_jd)

        st.success("Text Processed Successfully ✅")

        # 🔥 USE COMBINED SIMILARITY (FIXED)
        score = calculate_similarity(resume_text.lower(), job_description.lower())

        # -------- 🔥 UI IMPROVEMENT --------
        st.subheader("Match Score")

        st.progress(min(int(score * 100), 100))

        if score < 0.4:
            st.error(f"{round(score*100,2)}% - Low Match ❌")
        elif score < 0.7:
            st.warning(f"{round(score*100,2)}% - Moderate Match ⚠️")
        else:
            st.success(f"{round(score*100,2)}% - Good Match ✅")

        # Missing skills
        missing_skills = get_missing_skills(expanded_resume, expanded_jd)

        st.subheader("Missing Skills:")
        st.write(missing_skills)