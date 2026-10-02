# AI Resume Analyzer

An AI-powered resume analysis platform that helps job seekers evaluate their resumes against job descriptions, identify skill gaps, understand ATS compatibility, and improve their resumes through actionable recommendations.

---

## 🚀 Features

### 🔐 Authentication
- User registration and login
- JWT-based authentication
- Secure user sessions

### 📄 Resume Analysis
- Upload resumes in PDF format
- Extract text from resumes
- Parse resume content
- Detect technical and professional skills
- Identify missing skills
- Generate resume improvement recommendations

### 🎯 ATS & Job Matching
- Calculate ATS compatibility score
- Parse job descriptions
- Compare resume content with job requirements
- Identify matching and missing skills
- Analyze resume-job compatibility

### 🤖 AI-Powered Features
- AI-generated resume analysis
- AI resume assistant
- Resume improvement suggestions
- Resume rewriting support
- Recommendation generation

### 📊 Dashboard
- Clean and responsive user interface
- Resume analysis results
- Analysis history
- Job description analysis
- Resume improvement insights

---

## 🧠 AI & Resume Processing

The project contains dedicated modules for different stages of resume analysis:

- **PDF Parsing** — Extract text from uploaded resumes
- **Resume Parsing** — Process and structure resume information
- **Job Description Parsing** — Extract relevant requirements from job descriptions
- **Skill Extraction** — Identify relevant skills from resumes and job descriptions
- **ATS Analysis** — Evaluate resume compatibility
- **Scoring Engine** — Calculate resume/job matching scores
- **Embedding Engine** — Support semantic representation and matching
- **Matching Engine** — Compare resumes with job descriptions
- **Recommendation Engine** — Generate improvement recommendations
- **Resume Rewriter** — Assist with resume content improvement
- **AI Analysis Engine** — Generate deeper resume insights

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      Frontend        │
                    │   React + Vite       │
                    └──────────┬───────────┘
                               │
                               │ REST API
                               ▼
                    ┌──────────────────────┐
                    │       Backend        │
                    │   FastAPI + Python   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
       │ AI Engine   │  │  PostgreSQL  │  │ Authentication│
       │             │  │   Database   │  │     JWT       │
       └─────────────┘  └──────────────┘  └──────────────┘
              │
              ▼
       Resume Analysis
              │
       ┌──────┴────────┐
       ▼               ▼
   Resume Data     Job Description
       │               │
       └───────┬───────┘
               ▼
        Matching & Scoring
               │
               ▼
        Recommendations
```

---

## 🛠️ Tech Stack

### Frontend

- React
- Vite
- React Router
- JavaScript
- CSS

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- JWT Authentication

### AI / Resume Processing

- Resume text extraction
- Resume parsing
- Skill extraction
- ATS analysis
- Job description parsing
- Job matching
- Embedding-based processing
- Recommendation generation
- AI-powered analysis
- Resume rewriting

### Development & Deployment

- Git
- GitHub
- Docker
- Docker Compose

---

## 📁 Project Structure

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
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── tests/
├── docs/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── LICENSE
└── README.md
```

---

## ⚙️ How It Works

```text
1. User uploads resume
          ↓
2. PDF text extraction
          ↓
3. Resume parsing
          ↓
4. Skill extraction
          ↓
5. User provides job description
          ↓
6. Job description parsing
          ↓
7. Resume ↔ Job matching
          ↓
8. ATS scoring
          ↓
9. Missing skill detection
          ↓
10. AI analysis
          ↓
11. Recommendations
          ↓
12. Resume improvement
```

---

## 🔑 Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
AI_API_KEY=your_ai_api_key
```

> Never commit real API keys, passwords, database credentials, or other secrets to GitHub.

---

## 💻 Local Development

### 1. Clone the repository

```bash
git clone https://github.com/medhruva033/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Backend

```bash
cd backend
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

### 3. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will provide the frontend URL in the terminal.

---

## 🐳 Docker

The project includes Docker Compose configuration for running project services together.

```bash
docker compose up --build
```

To stop the services:

```bash
docker compose down
```

---

## 🧪 Testing

Run the project's test suite according to the test configuration in the repository.

Example:

```bash
pytest
```

---

## 🔒 Security

- JWT-based authentication
- Environment variables for sensitive configuration
- `.gitignore` for local secrets and generated files
- Backend API validation
- Database-backed user management

---

## 📌 Current Project Status

| Component | Status |
|---|---|
| Frontend | ✅ Implemented |
| Backend | ✅ Implemented |
| AI Modules | ✅ Implemented |
| Authentication | ✅ Implemented |
| Resume Processing | ✅ Implemented |
| ATS Analysis | ✅ Implemented |
| Job Matching | ✅ Implemented |
| Database Layer | ✅ Implemented |
| Tests | 🧪 Available |
| Documentation | ✅ Included |
| Docker Configuration | ✅ Included |
| Production Deployment | ⏳ Pending |

---

## 🔮 Future Improvements

Potential future improvements include:

- Advanced semantic matching
- More detailed ATS analysis
- Improved skill taxonomy
- Resume version management
- Job recommendation system
- Analytics and progress tracking
- Cloud deployment
- Automated CI/CD
- More comprehensive test coverage

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

## 👨‍💻 Author

**Dhruv Kolekar**

Computer Science Engineering Student  
AI & Software Development

---

## ⭐ Project

If you find this project useful, consider giving the repository a star on GitHub.
