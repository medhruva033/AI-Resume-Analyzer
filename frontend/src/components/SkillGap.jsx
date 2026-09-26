export default function SkillGap({
  skills = [],
}) {

  return (
    <div className="card">

      <p className="eyebrow">
        SKILL GAP
      </p>

      <h2>
        Missing Skills
      </h2>


      {skills.length === 0 ? (

        <p className="muted">
          No missing skills identified.
        </p>

      ) : (

        <div className="tags">

          {skills.map(
            (skill) => (
              <span
                className="tag orange"
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