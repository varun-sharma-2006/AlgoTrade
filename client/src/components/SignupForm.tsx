import { FormEvent, useState } from "react";

interface SignupFormProps {
  onSubmit: (name: string, email: string, password: string) => Promise<void> | void;
  onSwitchToLogin: () => void;
  loading: boolean;
  error?: string | null;
}

const MAX_PASSWORD_LENGTH = 72;

export function SignupForm({ onSubmit, onSwitchToLogin, loading, error }: SignupFormProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setValidationError(null);

    if (!name.trim()) {
      setValidationError("Please provide your name.");
      return;
    }

    if (!email || !password) {
      setValidationError("Email and password are required.");
      return;
    }

    if (password.length < 8) {
      setValidationError("Password must be at least 8 characters long.");
      return;
    }

    if (password.length > MAX_PASSWORD_LENGTH) {
      setValidationError(`Password must be ${MAX_PASSWORD_LENGTH} characters or fewer.`);
      return;
    }

    await onSubmit(name.trim(), email, password);
  };

  return (
    <div className="card auth-card">
      <h1>Create your account</h1>
      <p style={{ textAlign: "center", marginTop: "0.3rem", color: "rgba(226,232,240,0.7)" }}>
        Build and monitor personalised trading simulations.
      </p>

      {error && <div className="error-banner">{error}</div>}
      {validationError && <div className="error-banner">{validationError}</div>}

      <form onSubmit={handleSubmit} noValidate>
        <label>
          <span style={{ display: "block", marginBottom: "0.35rem" }}>Name</span>
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Jane Doe"
            autoComplete="name"
          />
        </label>

        <label>
          <span style={{ display: "block", marginBottom: "0.35rem" }}>Email</span>
          <input
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="jane.doe@example.com"
            autoComplete="email"
            inputMode="email"
          />
        </label>

        <label>
          <span style={{ display: "block", marginBottom: "0.35rem" }}>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            maxLength={MAX_PASSWORD_LENGTH}
            placeholder="Create a strong password"
            autoComplete="new-password"
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Creating account..." : "Sign up"}
        </button>
      </form>

      <footer>
        <span>Already registered? </span>
        <button
          type="button"
          onClick={onSwitchToLogin}
          style={{ background: "transparent", border: "none", color: "#93c5fd" }}
        >
          Sign in instead
        </button>
      </footer>
    </div>
  );
}
