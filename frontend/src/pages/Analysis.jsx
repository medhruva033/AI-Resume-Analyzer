import {
  useEffect,
  useState,
} from "react";

import {
  Link,
  useParams,
} from "react-router-dom";

import JobDescription
  from "../components/JobDescription";

import ScoreCard
  from "../components/ScoreCard";

import SkillMatch
  from "../components/SkillMatch";

import SkillGap
  from "../components/SkillGap";

import Recommendation
  from "../components/Recommendation";

import AIChat
  from "../components/AIChat";

import Loading
  from "../components/Loading";

import { api } from "../services/api";


export default function Analysis() {

  const { resumeId } = useParams();

  const [jobDescription, setJobDescription] =
    useState("");

  const [resume, setResume] =
    useState(null);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [pageLoading, setPageLoading] =
    useState(true);

  const [error, setError] =
    useState("");


  useEffect(() => {

    async function load() {

      try {

        const resumes =
          await api.getResumes();

        const found =
          resumes.find(
            (item) =>
              String(item.id) ===
              String(resumeId)
          );

        setResume(found);

      } catch (error) {

        setError(
          error.message
        );

      } finally {

        setPageLoading(false);

      }
    }

    load();

  }, [resumeId]);


  async function analyze() {

    setError("");
    setLoading(true);

    try {

      const data =
        await api.analyze(
          resumeId,
          jobDescription
        );

      setResult(data);

    } catch (error) {

      setError(
        error.message
      );

    } finally {

      setLoading(false);

    }
  }


  if (pageLoading) {
    return (
      <Loading
        text="Loading resume..."
      />
    );
  }


  return (
    <section className="analysis-page">

      {/* HEADER */}

      <div className="analysis-header">

        <div>

          <p className="analysis-label">
            RESUME ANALYSIS
          </p>

          <h1>
            {resume?.filename ||
              `Resume #${resumeId}`}
          </h1>

          <p>
            Compare your resume against
            a target job description.
          </p>

        </div>


        <Link
          to="/dashboard"
          className="analysis-back"
        >
          ← Dashboard
        </Link>

      </div>


      {/* ERROR */}

      {error && (
        <div className="error analysis-error">
          {error}
        </div>
      )}


      {/* JOB DESCRIPTION */}

      <div className="analysis-panel">

        <div className="analysis-panel-header">

          <div>

            <p className="analysis-label">
              JOB DESCRIPTION
            </p>

            <h2>
              Target Job
            </h2>

            <p>
              Paste the job description you
              want to compare your resume with.
            </p>

          </div>

          <div className="analysis-blue-icon">
            JD
          </div>

        </div>


        <JobDescription
          value={jobDescription}
          onChange={setJobDescription}
        />


        <div className="analysis-action">

          <button
            className="analysis-primary-button"
            onClick={analyze}
            disabled={loading}
          >
            {loading
              ? "Analyzing..."
              : "Analyze Resume"}
          </button>

        </div>

      </div>


      {/* RESULTS */}

      {result && (

        <div className="analysis-results">

          {/* SCORE CARDS */}

          <div className="analysis-section-title">

            <div>

              <p className="analysis-label">
                RESULTS
              </p>

              <h2>
                Resume Performance
              </h2>

            </div>

          </div>


          <div className="analysis-score-grid">

            <div className="analysis-score-wrapper">

              <ScoreCard
                title="ATS Score"
                score={result.ats_score}
              />

            </div>


            <div className="analysis-score-wrapper">

              <ScoreCard
                title="Job Match"
                score={result.match_score}
              />

            </div>

          </div>


          {/* SKILLS */}

          <div className="analysis-two-column">

            <div className="analysis-panel">

              <SkillMatch
                skills={result.skills_found}
              />

            </div>


            <div className="analysis-panel">

              <SkillGap
                skills={result.missing_skills}
              />

            </div>

          </div>


          {/* RECOMMENDATIONS */}

          <div className="analysis-panel">

            <Recommendation
              recommendations={
                result.recommendations
              }
            />

          </div>


          {/* AI ANALYSIS */}

          <div className="analysis-panel">

            <div className="analysis-panel-header">

              <div>

                <p className="analysis-label">
                  AI ANALYSIS
                </p>

                <h2>
                  Analysis Summary
                </h2>

              </div>

              <div className="analysis-ai-icon">
                AI
              </div>

            </div>


            <p className="analysis-summary">
              {result.ai_analysis ||
                "No analysis available."}
            </p>

          </div>


          {/* AI CHAT */}

          <div className="analysis-chat">

            <AIChat
              atsScore={
                result.ats_score
              }
              matchScore={
                result.match_score
              }
            />

          </div>

        </div>

      )}

    </section>
  );
}