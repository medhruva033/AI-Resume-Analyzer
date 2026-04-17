# AI Resume Analyzer 🌍

Analyze any resume against a job description and get a match score + missing skills.

## Features
- PDF text extraction (pdfplumber)
- NLP preprocessing (spaCy)
- Hybrid similarity (TF-IDF + SequenceMatcher)
- Missing skills detection
- Streamlit UI

## Tech Stack
Python, Streamlit, spaCy, scikit-learn, pdfplumber

## Run Locally
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py

## Output
- Match Score (%)
- Missing Skills
- Suggestions
