const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8001";


async function request(
  endpoint,
  options = {}
) {
  const token =
    localStorage.getItem("token");

  const headers = {
    ...(options.headers || {}),
  };

  if (token) {
    headers["Authorization"] =
      `Bearer ${token}`;
  }

  let body = options.body;

  if (
    body &&
    !(body instanceof FormData)
  ) {
    headers["Content-Type"] =
      "application/json";

    body = JSON.stringify(body);
  }

  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      ...options,
      headers,
      body,
    }
  );

  const contentType =
    response.headers.get(
      "content-type"
    ) || "";

  let data;

  if (
    contentType.includes(
      "application/json"
    )
  ) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem(
        "token"
      );

      localStorage.removeItem(
        "user"
      );
    }

    let message =
      `Request failed: ${response.status}`;

    if (
      data &&
      typeof data === "object" &&
      data.detail
    ) {
      if (
        Array.isArray(data.detail)
      ) {
        message = data.detail
          .map(
            (item) =>
              item.msg || "Validation error"
          )
          .join(", ");
      } else {
        message = data.detail;
      }
    }

    throw new Error(message);
  }

  return data;
}


export const api = {

  health() {
    return request("/health");
  },


  register(data) {
    return request(
      "/api/auth/register",
      {
        method: "POST",
        body: data,
      }
    );
  },


  login(data) {
    return request(
      "/api/auth/login",
      {
        method: "POST",
        body: data,
      }
    );
  },


  getResumes() {
    return request(
      "/api/resumes"
    );
  },


  uploadResume(file) {

    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );

    return request(
      "/api/resumes/upload",
      {
        method: "POST",
        body: formData,
      }
    );
  },


  analyze(
    resumeId,
    jobDescription
  ) {

    return request(
      "/api/analysis/analyze",
      {
        method: "POST",
        body: {
          resume_id:
            Number(resumeId),

          job_description:
            jobDescription || "",
        },
      }
    );
  },


  getHistory() {
    return request(
      "/api/analysis/history"
    );
  },


  getAnalysis(id) {
    return request(
      `/api/analysis/${id}`
    );
  },


  chat(
    message,
    atsScore = 0,
    matchScore = 0
  ) {

    return request(
      "/api/ai/chat",
      {
        method: "POST",
        body: {
          message,
          ats_score: atsScore,
          match_score: matchScore,
        },
      }
    );
  },
};