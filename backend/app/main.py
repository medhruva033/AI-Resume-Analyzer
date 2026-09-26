from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.database.database import create_tables

from app.api.ai import router as ai_router
from app.api.analysis import router as analysis_router
from app.api.auth import router as auth_router
from app.api.resume import router as resume_router


app = FastAPI(
    title="AI Resume Analyzer API",
    description="Backend API for AI-powered resume analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()


# Register all API routers
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(analysis_router)
app.include_router(ai_router)


@app.get("/")
def root():
    return {
        "message": "AI Resume Analyzer API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }