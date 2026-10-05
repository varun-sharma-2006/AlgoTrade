import { AuthBrand, AuthLayout } from "./AuthLayout";
import { GoogleSignIn } from "./GoogleSignIn";

interface GoogleLoginPageProps {
  clientId: string;
  onCredential: (credential: string) => void;
  loading: boolean;
  error?: string | null;
}

export function GoogleLoginPage({ clientId, onCredential, loading, error }: GoogleLoginPageProps) {
  return (
    <AuthLayout>
      <AuthBrand />
      <h2>Sign in to your workspace</h2>
      <p className="auth-lede">Research strategies and run paper-trading simulations — no real money involved.</p>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="auth-google">
        <GoogleSignIn clientId={clientId} onCredential={onCredential} disabled={loading} />
      </div>
      {loading ? <p className="auth-hint">Signing you in…</p> : null}

      <p className="auth-hint">
        One click with your Google account, no new password needed. Sign-in is verified by Google; your name, email
        address and profile photo are shared with the owner of this app, who can see when you visited. Nothing else in
        your Google account is accessed. <a href="/privacy.html">Privacy policy</a>
      </p>
      <p className="auth-switch">For education only · not financial advice</p>
    </AuthLayout>
  );
}
