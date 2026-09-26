import {
  useState,
} from "react";

import { api } from "../services/api";


export default function AIChat({
  atsScore,
  matchScore,
}) {

  const [message, setMessage] =
    useState("");

  const [response, setResponse] =
    useState("");

  const [loading, setLoading] =
    useState(false);


  async function sendMessage() {

    if (!message.trim()) {
      return;
    }

    setLoading(true);

    try {

      const result =
        await api.chat(
          message,
          atsScore,
          matchScore
        );

      setResponse(
        result.response
      );

      setMessage("");

    } catch (error) {

      setResponse(
        error.message
      );

    } finally {

      setLoading(false);

    }
  }


  return (
    <div className="card">

      <p className="eyebrow">
        AI ASSISTANT
      </p>

      <h2>
        Ask About Your Resume
      </h2>


      {response && (
        <div className="ai-response">
          {response}
        </div>
      )}


      <div className="chat-row">

        <input
          className="input"
          value={message}
          onChange={(event) =>
            setMessage(
              event.target.value
            )
          }
          onKeyDown={(event) => {

            if (
              event.key ===
              "Enter"
            ) {
              sendMessage();
            }

          }}
          placeholder="How can I improve my resume?"
        />


        <button
          className="btn"
          onClick={sendMessage}
          disabled={loading}
        >
          {loading
            ? "..."
            : "Ask"}
        </button>

      </div>

    </div>
  );
}