import type React from "react";
import { GoogleSignIn } from "./GoogleSignIn";
import { KeyIcon, LockIcon, LogoMark, ShieldIcon } from "./Icons";

interface GoogleLoginPageProps {
  clientId: string;
  onCredential: (credential: string) => void;
  loading: boolean;
  error?: string | null;
}

const HIGHLIGHTS = [
  { value: "3", label: "Strategies" },
  { value: "2 yrs", label: "Of daily data" },
  { value: "10 bps", label: "Fees modelled" },
];

const TRUST = [
  { icon: ShieldIcon, text: "Identity verified by Google" },
  { icon: KeyIcon, text: "No passwords to create or leak" },
  { icon: LockIcon, text: "Your workspace stays private to you" },
];

// Decorative chart for the brand panel: candles plus a gold trend line that draws itself in.
// Closes form a rising, realistically choppy path; each candle opens at the previous close.
const CLOSES = [100, 97, 103, 101, 108, 104, 111, 116, 112, 121, 118, 127, 133, 129, 138, 146];
const ART = { width: 420, height: 160, pad: 14 };

function buildArt() {
  const lows = CLOSES.map((close, i) => Math.min(close, CLOSES[i - 1] ?? close) - 3 - (i % 3));
  const highs = CLOSES.map((close, i) => Math.max(close, CLOSES[i - 1] ?? close) + 3 + ((i + 1) % 3));
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const step = (ART.width - ART.pad * 2) / (CLOSES.length - 1);
  const x = (i: number) => ART.pad + i * step;
  const y = (price: number) => ART.pad + ((max - price) / (max - min)) * (ART.height - ART.pad * 2);
  const candles = CLOSES.map((close, i) => {
    const open = CLOSES[i - 1] ?? close - 2;
    return { x: x(i), open: y(open), close: y(close), high: y(highs[i]), low: y(lows[i]), up: close >= open };
  });
  const line = CLOSES.map((close, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(close).toFixed(1)}`).join(" ");
  const area = `${line} L${x(CLOSES.length - 1)},${ART.height} L${x(0)},${ART.height} Z`;
  return { candles, line, area, width: Math.max(step * 0.42, 6) };
}

const art = buildArt();

function MarketArt() {
  return (
    <svg className="market-art" viewBox={`0 0 ${ART.width} ${ART.height}`} aria-hidden="true">
      <defs>
        <linearGradient id="art-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#9c7535" />
          <stop offset="100%" stopColor="#f6dfa8" />
        </linearGradient>
        <linearGradient id="art-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(212,175,106,0.26)" />
          <stop offset="100%" stopColor="rgba(212,175,106,0)" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} className="art-grid" x1={0} x2={ART.width} y1={ART.height * f} y2={ART.height * f} />
      ))}
      <path className="art-area" d={art.area} fill="url(#art-fill)" />
      {art.candles.map((c, index) => (
        <g key={c.x} className={c.up ? "candle up" : "candle down"} style={{ "--i": index } as React.CSSProperties}>
          <line x1={c.x} x2={c.x} y1={c.high} y2={c.low} />
          <rect
            x={c.x - art.width / 2}
            y={Math.min(c.open, c.close)}
            width={art.width}
            height={Math.max(Math.abs(c.open - c.close), 2)}
            rx="2"
          />
        </g>
      ))}
      <path className="art-line" d={art.line} fill="none" stroke="url(#art-line)" strokeWidth="2.2" />
    </svg>
  );
}

export function GoogleLoginPage({ clientId, onCredential, loading, error }: GoogleLoginPageProps) {
  return (
    <div className="login-page">
      <section className="login-brand">
        <div className="login-brand-top">
          <LogoMark size={44} />
          <span className="wordmark">Algo Trade Simulator</span>
        </div>

        <div className="login-hero">
          <span className="eyebrow">Private trading laboratory</span>
          <h1>
            Test every idea <em>before</em> the market does.
          </h1>
          <p>
            Backtest real strategies on real prices, fees included, and see honestly how they compare with simply
            buying and holding.
          </p>
        </div>

        <MarketArt />

        <ul className="login-highlights">
          {HIGHLIGHTS.map((item) => (
            <li key={item.label}>
              <strong>{item.value}</strong>
              <span>{item.label}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="login-panel">
        <div className="login-card">
          <div className="login-card-logo">
            <LogoMark size={52} />
          </div>
          <span className="eyebrow">Welcome</span>
          <h2>Sign in to your workspace</h2>
          <p className="subtle login-lede">One click with your Google account. No new password needed.</p>

          {error ? <div className="error-banner">{error}</div> : null}

          <GoogleSignIn clientId={clientId} onCredential={onCredential} disabled={loading} />
          {loading ? <p className="subtle signing-in">Signing you in…</p> : null}

          <div className="divider" role="presentation">
            <span>Why Google sign-in</span>
          </div>

          <ul className="trust-list">
            {TRUST.map(({ icon: TrustIcon, text }) => (
              <li key={text}>
                <TrustIcon />
                {text}
              </li>
            ))}
          </ul>

          <p className="privacy-note">
            Sign-in is verified by Google. When you sign in, your name, email address and profile photo are shared
            with the owner of this app, who can see when you visited. Nothing else in your Google account is
            accessed. <a href="/privacy.html">Privacy policy</a>
          </p>
        </div>
        <p className="login-footnote">For education only · not financial advice</p>
      </section>
    </div>
  );
}
