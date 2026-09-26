import {
  Link,
} from "react-router-dom";


export default function Home() {

  return (
    <section className="hero">

      <div className="hero-content">

        <p className="eyebrow">
          AI-POWERED RESUME ANALYZER
        </p>

        <h1>
          Build a resume
          that matches the job.
        </h1>

        <p className="hero-text">
          Analyze your resume,
          check ATS readiness,
          compare skills with a
          job description and
          get actionable
          recommendations.
        </p>


        <div className="hero-buttons">

          <Link
            to="/register"
            className="btn"
          >
            Start Analyzing
          </Link>

          <Link
            to="/login"
            className="btn secondary"
          >
            Login
          </Link>

        </div>

      </div>


      <div className="hero-card">

        <div className="hero-score">
          <span>
            ATS SCORE
          </span>

          <strong>
            85
          </strong>

          <small>
            / 100
          </small>
        </div>


        <div className="hero-row">
          <span>
            Skills matched
          </span>

          <strong>
            12
          </strong>
        </div>


        <div className="hero-row">
          <span>
            Missing skills
          </span>

          <strong>
            4
          </strong>
        </div>


        <div className="hero-row">
          <span>
            Recommendations
          </span>

          <strong>
            ✓
          </strong>
        </div>

      </div>

    </section>
  );
}