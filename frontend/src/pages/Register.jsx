import {
  useState,
} from "react";

import {
  Link,
  useNavigate,
} from "react-router-dom";

import { api } from "../services/api";


export default function Register() {

  const navigate =
    useNavigate();


  const [name, setName] =
    useState("");

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
        await api.register({
          name,
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
          GET STARTED
        </p>

        <h1>
          Create Account
        </h1>

        <p className="muted">
          Start analyzing your
          resume.
        </p>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        <label>
          Name

          <input
            className="input"
            value={name}
            onChange={(event) =>
              setName(
                event.target.value
              )
            }
            required
          />

        </label>


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
            minLength="6"
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
            ? "Creating..."
            : "Create Account"}
        </button>


        <p className="auth-link">

          Already have an account?

          {" "}

          <Link to="/login">
            Login
          </Link>

        </p>

      </form>

    </section>
  );
}