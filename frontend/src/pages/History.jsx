import {
  useEffect,
  useState,
} from "react";

import {
  Link,
} from "react-router-dom";

import Loading
  from "../components/Loading";

import { api } from "../services/api";


export default function History() {

  const [history,
    setHistory] =
    useState([]);

  const [loading,
    setLoading] =
    useState(true);

  const [error,
    setError] =
    useState("");


  useEffect(() => {

    async function load() {

      try {

        const result =
          await api.getHistory();

        setHistory(result);

      } catch (error) {

        setError(
          error.message
        );

      } finally {

        setLoading(false);

      }
    }

    load();

  }, []);


  return (
    <section>

      <div className="page-header">

        <p className="eyebrow">
          HISTORY
        </p>

        <h1>
          Analysis History
        </h1>

        <p className="muted">
          Review your previous
          resume analyses.
        </p>

      </div>


      {error && (
        <div className="error">
          {error}
        </div>
      )}


      {loading ? (

        <Loading />

      ) : history.length === 0 ? (

        <div className="empty">
          No analyses yet.
        </div>

      ) : (

        <div className="history-list">

          {history.map(
            (item) => (

              <div
                className="history-card"
                key={item.id}
              >

                <div>

                  <h3>
                    Analysis #
                    {item.id}
                  </h3>

                  <p>
                    {new Date(
                      item.created_at
                    ).toLocaleString()}
                  </p>

                </div>


                <div className="history-score">

                  <span>
                    ATS

                    <strong>
                      {Number(
                        item.ats_score
                      ).toFixed(0)}
                    </strong>
                  </span>


                  <span>
                    Match

                    <strong>
                      {Number(
                        item.match_score
                      ).toFixed(0)}
                    </strong>
                  </span>

                </div>


                <Link
                  className="btn small secondary"
                  to={`/analysis/${item.resume_id}`}
                >
                  Open
                </Link>

              </div>

            )
          )}

        </div>

      )}

    </section>
  );
}