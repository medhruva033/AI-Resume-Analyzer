import {
  useState,
} from "react";

import {
  Link,
  useNavigate,
} from "react-router-dom";

import { api } from "../services/api";


export default function Login() {

  const navigate =
    useNavigate();


  const [email, setEmail] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);


  async function submit(
    event
  ) {

    event.preventDefault();

    setError("");
    setLoading(true);


    try {

      const result =
        await api.login({
          email,
          password,
        });


      localStorage.setItem(
        "token",
        result.token
      );


      localStorage.setItem(
        "user",
        JSON.stringify(
          result.user
        )
      );


      navigate(
        "/dashboard"
      );

    } catch (error) {

      setError(
        error.message
      );

    } finally {

      setLoading(false);

    }
  }


  return (
    <section className="auth">

      <form
        className="auth-card"
        onSubmit={submit}
      >

        <p className="eyebrow">
          WELCOME BACK
        </p>

        <h1>
          Login
        </h1>

        <p className="muted">
          Access your resume
          dashboard.
        </p>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        <label>
          Email

          <input
            className="input"
            type="email"
            value={email}
            onChange={(event) =>
              setEmail(
                event.target.value
              )
            }
            required
          />

        </label>


        <label>
          Password

          <input
            className="input"
            type="password"
            value={password}
            onChange={(event) =>
              setPassword(
                event.target.value
              )
            }
            required
          />

        </label>


        <button
          className="btn full"
          disabled={loading}
        >
          {loading
            ? "Logging in..."
            : "Login"}
        </button>


        <p className="auth-link">
          Don't have an account?

          {" "}

          <Link to="/register">
            Create account
          </Link>
        </p>

      </form>

    </section>
  );
}