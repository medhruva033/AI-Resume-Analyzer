export default function JobDescription({
  value,
  onChange,
}) {

  return (
    <div className="card">

      <p className="eyebrow">
        TARGET JOB
      </p>

      <h2>
        Job Description
      </h2>

      <p className="muted">
        Paste the job description
        you want to compare against.
      </p>

      <textarea
        className="textarea"
        rows="12"
        value={value}
        onChange={(event) =>
          onChange(
            event.target.value
          )
        }
        placeholder="Paste the job description here..."
      />

    </div>
  );
}