import { FormEvent, useState } from "react";

interface LoginFormProps {
  onSubmit: (email: string, password: string) => Promise<void> | void;
  onSwitchToSignup: () => void;
  loading: boolean;
  error?: string | null;
}

export function LoginForm({ onSubmit, onSwitchToSignup, loading, error }: LoginFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setValidationError(null);

    if (!email || !password) {
      setValidationError("Email and password are required.");
      return;
    }

    await onSubmit(email, password);
  };

  return (
    <div className="card auth-card">
      <h1>Welcome back</h1>
      <p style={{ textAlign: "center", marginTop: "0.3rem", color: "rgba(226,232,240,0.7)" }}>
        Sign in to resume your trading simulations.
      </p>

      {error && <div className="error-banner">{error}</div>}
      {validationError && <div className="error-banner">{validationError}</div>}

      <form onSubmit={handleSubmit} noValidate>
        <label>
          <span style={{ display: "block", marginBottom: "0.35rem" }}>Email</span>
          <input
            autoComplete="email"
            inputMode="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="jane.doe@example.com"
          />
        </label>

        <label>
          <span style={{ display: "block", marginBottom: "0.35rem" }}>Password</span>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="••••••••"
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>

      <footer>
        <span>Need an account? </span>
        <button
          type="button"
          onClick={onSwitchToSignup}
          style={{ background: "transparent", border: "none", color: "#93c5fd" }}
        >
          Create one
        </button>
      </footer>
    </div>
  );
}
