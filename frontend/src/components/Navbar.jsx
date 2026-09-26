import {
  Link,
  NavLink,
  useNavigate,
} from "react-router-dom";


export default function Navbar() {

  const navigate =
    useNavigate();

  const token =
    localStorage.getItem(
      "token"
    );

  const user =
    JSON.parse(
      localStorage.getItem(
        "user"
      ) || "null"
    );


  function logout() {

    localStorage.removeItem(
      "token"
    );

    localStorage.removeItem(
      "user"
    );

    navigate("/login");
  }


  return (
    <header className="navbar">

      <Link
        to="/"
        className="logo"
      >
        <span className="logo-box">
          AI
        </span>

        Resume Analyzer
      </Link>


      <nav>

        {token ? (
          <>

            <NavLink to="/dashboard">
              Dashboard
            </NavLink>

            <NavLink to="/history">
              History
            </NavLink>

            <span className="user-name">
              {user?.name}
            </span>

            <button
              className="btn small"
              onClick={logout}
            >
              Logout
            </button>

          </>
        ) : (
          <>

            <NavLink to="/login">
              Login
            </NavLink>

            <Link
              to="/register"
              className="btn small"
            >
              Get Started
            </Link>

          </>
        )}

      </nav>

    </header>
  );
}