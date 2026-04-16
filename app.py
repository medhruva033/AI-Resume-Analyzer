import streamlit as st
from pypdf import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# -------- SIMILARITY FUNCTION --------
def calculate_similarity(resume, job_desc):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    vectors = vectorizer.fit_transform([resume, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


# -------- MISSING SKILLS --------
def get_missing_skills(resume, job_desc):
    resume_words = set(resume.split())
    jd_words = set(job_desc.split())

    missing = jd_words - resume_words
    return list(missing)[:10]


# -------- MAIN LOGIC --------
if uploaded_file is not None and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    # Debug check
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

        # 🔥 IMPORTANT FIX → Use RAW TEXT for similarity
        score = calculate_similarity(resume_text.lower(), job_description.lower())

        st.subheader(f"Match Score: {round(score * 100, 2)}%")

        # Missing skills (use cleaned text)
        missing_skills = get_missing_skills(expanded_resume, expanded_jd)

        st.subheader("Missing Skills:")
        st.write(missing_skills)