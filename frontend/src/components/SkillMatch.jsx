export default function SkillMatch({
  skills = [],
}) {

  return (
    <div className="card">

      <p className="eyebrow">
        MATCHED SKILLS
      </p>

      <h2>
        Skills Found
      </h2>


      {skills.length === 0 ? (

        <p className="muted">
          No matching skills found.
        </p>

      ) : (

        <div className="tags">

          {skills.map(
            (skill) => (
              <span
                className="tag green"
                key={skill}
              >
                {skill}
              </span>
            )
          )}

        </div>

      )}

    </div>
  );
}