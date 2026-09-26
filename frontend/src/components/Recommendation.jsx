export default function Recommendation({
  recommendations = [],
}) {

  return (
    <div className="card">

      <p className="eyebrow">
        IMPROVEMENTS
      </p>

      <h2>
        Recommendations
      </h2>


      {recommendations.length === 0 ? (

        <p className="muted">
          No recommendations available.
        </p>

      ) : (

        <ul className="recommendations">

          {recommendations.map(
            (item, index) => (
              <li
                key={index}
              >
                {item}
              </li>
            )
          )}

        </ul>

      )}

    </div>
  );
}