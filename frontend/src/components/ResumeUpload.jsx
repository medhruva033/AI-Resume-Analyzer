import {
  useRef,
  useState,
} from "react";

import { api } from "../services/api";


export default function ResumeUpload({
  onUploaded,
}) {

  const fileInput =
    useRef(null);

  const [file, setFile] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  function selectFile(event) {

    const selected =
      event.target.files?.[0];

    setFile(selected || null);

    setError("");
  }


  async function upload() {

    if (!file) {
      setError(
        "Please select a resume."
      );

      return;
    }

    setLoading(true);
    setError("");

    try {

      const result =
        await api.uploadResume(
          file
        );

      setFile(null);

      if (fileInput.current) {
        fileInput.current.value =
          "";
      }

      onUploaded?.(result);

    } catch (error) {

      setError(
        error.message
      );

    } finally {

      setLoading(false);

    }
  }


  return (
    <div className="card">

      <div className="section-title">

        <div>
          <p className="eyebrow">
            RESUME
          </p>

          <h2>
            Upload Resume
          </h2>

          <p className="muted">
            Upload your PDF resume
            for analysis.
          </p>
        </div>

      </div>


      <input
        ref={fileInput}
        type="file"
        accept=".pdf,.txt,.docx"
        onChange={selectFile}
        className="file-input"
      />


      {file && (
        <div className="file-name">
          Selected: {file.name}
        </div>
      )}


      {error && (
        <div className="error">
          {error}
        </div>
      )}


      <button
        className="btn"
        onClick={upload}
        disabled={loading}
      >
        {loading
          ? "Uploading..."
          : "Upload Resume"}
      </button>

    </div>
  );
}