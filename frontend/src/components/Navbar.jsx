import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../auth.jsx";

export default function Navbar() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  return (
    <header className="navbar">
      <div className="container nav-inner">
        <NavLink to="/" className="brand">
          <img src="/favicon.svg" alt="" width="26" height="26" />
          PixFix
        </NavLink>
        {user && (
          <nav className="nav-links">
            <NavLink to="/" end>Editor</NavLink>
            <NavLink to="/history">History</NavLink>
            {user.is_admin && <NavLink to="/admin">Admin</NavLink>}
          </nav>
        )}
        <div className="nav-right">
          {user ? (
            <>
              <span className="muted small">{user.username}</span>
              <button className="btn ghost sm" onClick={() => { logout(); nav("/login"); }}>
                Log out
              </button>
            </>
          ) : (
            <>
              <NavLink to="/login" className="btn ghost sm">Log in</NavLink>
              <NavLink to="/register" className="btn sm">Sign up</NavLink>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
