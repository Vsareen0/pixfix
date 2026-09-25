import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth.jsx";

export default function Register() {
  const { user, register } = useAuth();
  const nav = useNavigate();
  const [form, setForm] = useState({ username: "", email: "", password: "", confirm: "" });
  const [fields, setFields] = useState({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setFields({});
    if (form.password !== form.confirm) {
      setFields({ confirm: "Passwords do not match" });
      return;
    }
    setBusy(true);
    try {
      await register(form.username, form.email, form.password);
      nav("/", { replace: true });
    } catch (err) {
      setError(err.message);
      setFields(err.fields || {});
    } finally {
      setBusy(false);
    }
  };

  const field = (key, label, type = "text", extra = {}) => (
    <label>
      {label}
      <input type={type} value={form[key]} onChange={set(key)} required
             className={fields[key] ? "invalid" : ""} {...extra} />
      {fields[key] && <span className="field-error">{fields[key]}</span>}
    </label>
  );

  return (
    <div className="auth-card card">
      <h1>Create your account</h1>
      <p className="muted">Free, and your images stay on this server.</p>
      <form onSubmit={submit} className="form">
        {field("username", "Username", "text", { autoFocus: true, minLength: 3, maxLength: 50 })}
        {field("email", "Email", "email")}
        {field("password", "Password", "password", { minLength: 8 })}
        {field("confirm", "Confirm password", "password")}
        <p className="muted small">At least 8 characters with a letter and a number.</p>
        {error && !Object.keys(fields).length && <div className="alert error">{error}</div>}
        <button className="btn block" disabled={busy}>{busy ? "Creating…" : "Sign up"}</button>
      </form>
      <p className="muted small center">
        Already registered? <Link to="/login">Log in</Link>
      </p>
    </div>
  );
}
