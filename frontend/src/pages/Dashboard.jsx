import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import ResumeUpload from "../components/ResumeUpload";
import Loading from "../components/Loading";
import { api } from "../services/api";


export default function Dashboard() {
  const [resumes, setResumes] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const user = JSON.parse(
    localStorage.getItem("user") || "{}"
  );


  async function loadDashboard() {
    try {
      setError("");

      const [resumeData, historyData] =
        await Promise.all([
          api.getResumes(),
          api.getHistory(),
        ]);

      setResumes(
        Array.isArray(resumeData)
          ? resumeData
          : []
      );

      setHistory(
        Array.isArray(historyData)
          ? historyData
          : []
      );

    } catch (error) {
      setError(
        error.message ||
        "Unable to load dashboard."
      );
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    loadDashboard();
  }, []);


  const latestAnalysis =
    history.length > 0
      ? history[0]
      : null;


  return (
    <section className="clean-dashboard">

      {/* HEADER */}

      <div className="clean-dashboard-header">

        <div>
          <p className="clean-label">
            DASHBOARD
          </p>

          <h1>
            Welcome back,{" "}
            <span>
              {user.name || "User"}
            </span>
          </h1>

          <p className="clean-subtitle">
            Manage your resumes and analyze
            your job compatibility.
          </p>
        </div>

        <Link
          to="#upload"
          className="clean-primary-button"
        >
          + Upload Resume
        </Link>

      </div>


      {/* STATS */}

      <div className="clean-stats">

        <div className="clean-box">
          <span>Total Resumes</span>
          <strong>{resumes.length}</strong>
          <small>Uploaded resumes</small>
        </div>

        <div className="clean-box">
          <span>ATS Score</span>
          <strong>
            {latestAnalysis
              ? latestAnalysis.ats_score
              : "--"}
          </strong>
          <small>Latest analysis</small>
        </div>

        <div className="clean-box">
          <span>Job Match</span>
          <strong>
            {latestAnalysis
              ? `${latestAnalysis.match_score}%`
              : "--"}
          </strong>
          <small>Latest match</small>
        </div>

        <div className="clean-box">
          <span>Analyses</span>
          <strong>{history.length}</strong>
          <small>Completed</small>
        </div>

      </div>


      {/* UPLOAD */}

      <div
        id="upload"
        className="clean-panel"
      >

        <div className="clean-panel-header">

          <div>
            <p className="clean-label">
              RESUME
            </p>

            <h2>
              Upload Resume
            </h2>

            <p>
              Upload your PDF resume to start
              the analysis.
            </p>
          </div>

          <div className="clean-blue-icon">
            PDF
          </div>

        </div>

        <ResumeUpload
          onUploaded={loadDashboard}
        />

      </div>


      {/* LATEST ANALYSIS */}

      <div className="clean-panel">

        <div className="clean-panel-header">

          <div>
            <p className="clean-label">
              ANALYSIS
            </p>

            <h2>
              Latest Analysis
            </h2>
          </div>

          <Link
            to="/history"
            className="clean-text-link"
          >
            View History →
          </Link>

        </div>


        {latestAnalysis ? (

          <div className="clean-analysis">

            <div className="clean-score-box">

              <strong>
                {latestAnalysis.ats_score}
              </strong>

              <span>
                /100
              </span>

              <small>
                ATS Score
              </small>

            </div>


            <div className="clean-analysis-info">

              <h3>
                Resume Analysis
              </h3>

              <p>
                Your latest resume has been
                analyzed successfully.
              </p>

              <div className="clean-match">

                <span>
                  Job Match
                </span>

                <strong>
                  {latestAnalysis.match_score}%
                </strong>

              </div>

            </div>


            <Link
              to={`/analysis/${latestAnalysis.resume_id}`}
              className="clean-secondary-button"
            >
              View Analysis
            </Link>

          </div>

        ) : (

          <div className="clean-empty">
            <h3>No analysis yet</h3>
            <p>
              Upload a resume and run your
              first analysis.
            </p>
          </div>

        )}

      </div>


      {/* RESUMES */}

      <div className="clean-resumes">

        <div className="clean-section-header">

          <div>
            <p className="clean-label">
              YOUR FILES
            </p>

            <h2>
              Uploaded Resumes
            </h2>
          </div>

          <span>
            {resumes.length}{" "}
            {resumes.length === 1
              ? "Resume"
              : "Resumes"}
          </span>

        </div>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        {loading ? (

          <Loading />

        ) : resumes.length === 0 ? (

          <div className="clean-empty">
            <h3>No resumes uploaded</h3>
            <p>
              Upload your first resume above.
            </p>
          </div>

        ) : (

          <div className="clean-resume-grid">

            {resumes.map((resume) => (

              <div
                className="clean-resume-card"
                key={resume.id}
              >

                <div className="clean-resume-icon">
                  PDF
                </div>

                <div className="clean-resume-info">

                  <h3 title={resume.filename}>
                    {resume.filename}
                  </h3>

                  <p>
                    Uploaded{" "}
                    {new Date(
                      resume.created_at
                    ).toLocaleDateString(
                      undefined,
                      {
                        day: "numeric",
                        month: "short",
                        year: "numeric",
                      }
                    )}
                  </p>

                </div>

                <Link
                  to={`/analysis/${resume.id}`}
                  className="clean-analyze"
                >
                  Analyze →
                </Link>

              </div>

            ))}

          </div>

        )}

      </div>

    </section>
  );
}