import { FormEvent, useState } from "react";
import { AuthBrand, AuthLayout } from "./AuthLayout";

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
    <AuthLayout>
      <AuthBrand />
      <h2>Welcome back</h2>
      <p className="auth-lede">Sign in to pick up your research and paper-trading simulations.</p>

      {error && <div className="error-banner">{error}</div>}
      {validationError && <div className="error-banner">{validationError}</div>}

      <form className="auth-form" onSubmit={handleSubmit} noValidate>
        <label>
          <span>Email</span>
          <input
            type="email"
            autoComplete="email"
            inputMode="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </label>

        <label>
          <span>Password</span>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>

        <button type="submit" className="auth-submit" disabled={loading}>
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="auth-switch">
        New here?{" "}
        <button type="button" onClick={onSwitchToSignup} className="auth-link">
          Create an account
        </button>
      </p>
    </AuthLayout>
  );
}
