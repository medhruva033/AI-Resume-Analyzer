# AI Resume Analyzer

An AI-powered resume analysis platform that helps job seekers evaluate their resumes against job descriptions and identify areas for improvement.

## Features

- User registration and login
- Secure authentication
- Resume PDF upload
- Resume text extraction
- ATS score calculation
- Job description matching
- Skill detection
- Missing skill identification
- Resume recommendations
- AI-generated resume analysis
- AI resume assistant
- Analysis history
- Clean and responsive dashboard

## Tech Stack

### Frontend
- React
- Vite
- React Router
- JavaScript
- CSS

### Backend
- FastAPI
- Python
- SQLAlchemy
- PostgreSQL
- JWT Authentication

### AI / Resume Processing
- Resume text extraction
- Skill extraction
- ATS analysis
- Job matching
- Recommendation generation
- AI analysis

## Project Structure

```text
AI-Resume-Analyzer/
│
├── ai/
│   ├── analysis_engine.py
│   ├── ats_analyzer.py
│   ├── embedding_engine.py
│   ├── jd_parser.py
│   ├── matching_engine.py
│   ├── pdf_parser.py
│   ├── recommendation_engine.py
│   ├── resume_parser.py
│   ├── resume_rewriter.py
│   ├── scoring_engine.py
│   └── skill_extractor.py
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── tests/
├── docs/
├── .env.example
├── docker-compose.yml
└── README.md
