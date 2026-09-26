export default function ScoreCard({
  title,
  score,
}) {

  const value =
    Number(score || 0);


  return (
    <div className="score-card">

      <span>
        {title}
      </span>

      <strong>
        {value.toFixed(1)}
      </strong>

      <small>
        / 100
      </small>

    </div>
  );
}