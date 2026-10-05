import { FormEvent, useState } from "react";
import { AuthBrand, AuthLayout } from "./AuthLayout";

interface SignupFormProps {
  onSubmit: (name: string, email: string, password: string) => Promise<void> | void;
  onSwitchToLogin: () => void;
  loading: boolean;
  error?: string | null;
}

const MIN_PASSWORD_LENGTH = 8;
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

    if (password.length < MIN_PASSWORD_LENGTH) {
      setValidationError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters long.`);
      return;
    }

    if (password.length > MAX_PASSWORD_LENGTH) {
      setValidationError(`Password must be ${MAX_PASSWORD_LENGTH} characters or fewer.`);
      return;
    }

    await onSubmit(name.trim(), email, password);
  };

  return (
    <AuthLayout>
      <AuthBrand />
      <h2>Create your account</h2>
      <p className="auth-lede">Research strategies and run paper-trading simulations — no real money involved.</p>

      {error && <div className="error-banner">{error}</div>}
      {validationError && <div className="error-banner">{validationError}</div>}

      <form className="auth-form" onSubmit={handleSubmit} noValidate>
        <label>
          <span>Name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} autoComplete="name" />
        </label>

        <label>
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            inputMode="email"
          />
        </label>

        <label>
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            maxLength={MAX_PASSWORD_LENGTH}
            autoComplete="new-password"
            aria-describedby="password-hint"
          />
        </label>
        <small id="password-hint" className="auth-hint password-hint">
          At least {MIN_PASSWORD_LENGTH} characters. Very common passwords are rejected.
        </small>

        <button type="submit" className="auth-submit" disabled={loading}>
          {loading ? "Creating account…" : "Create account"}
        </button>
      </form>

      <p className="auth-switch">
        Already have an account?{" "}
        <button type="button" onClick={onSwitchToLogin} className="auth-link">
          Sign in instead
        </button>
      </p>
    </AuthLayout>
  );
}
