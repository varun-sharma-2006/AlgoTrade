import { GoogleSignIn } from "./GoogleSignIn";

interface GoogleLoginPageProps {
  clientId: string;
  onCredential: (credential: string) => void;
  loading: boolean;
  error?: string | null;
}

const FEATURES = [
  "Backtest 3 strategies on 2 years of real data, fees included",
  "Compare every strategy against buy & hold",
  "Live quotes, candlestick charts and an AI trading copilot",
];

export function GoogleLoginPage({ clientId, onCredential, loading, error }: GoogleLoginPageProps) {
  return (
    <div className="card auth-card google-login">
      <h1>Algo Trade Simulator</h1>
      <p className="subtle" style={{ textAlign: "center", marginTop: "0.3rem" }}>
        Test trading strategies honestly before risking real money.
      </p>

      <ul className="login-features">
        {FEATURES.map((feature) => (
          <li key={feature}>{feature}</li>
        ))}
      </ul>

      {error ? <div className="error-banner">{error}</div> : null}

      <GoogleSignIn clientId={clientId} onCredential={onCredential} disabled={loading} />
      {loading ? <p className="subtle" style={{ textAlign: "center" }}>Signing you in...</p> : null}

      <p className="privacy-note">
        Sign-in is verified by Google. When you sign in, your name, email address and profile photo are shared
        with the owner of this app, who can see when you visited. Nothing else in your Google account is accessed.
      </p>
    </div>
  );
}
