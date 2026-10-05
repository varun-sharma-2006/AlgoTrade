import type { ReactNode } from "react";
import { ChatIcon, LayersIcon, ShieldIcon, SimulationsIcon } from "./Icons";

const FEATURES = [
  { icon: SimulationsIcon, text: "Backtest strategies", featured: true },
  { icon: LayersIcon, text: "Evaluate models" },
  { icon: ChatIcon, text: "Research with copilot" },
];

/** Split sign-in screen: product pitch on the left, the form card on the right. */
export function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="auth-split">
      <section className="auth-hero">
        <span className="auth-eyebrow">Algo Trade Simulator / Research workspace</span>
        <h1>
          Turn market ideas into measured <span className="auth-accent">decisions.</span>
        </h1>
        <p>A focused workspace for strategy research, backtesting, and paper trading.</p>
        <ul className="auth-features">
          {FEATURES.map(({ icon: Icon, text, featured }) => (
            <li key={text} className={featured ? "featured" : undefined}>
              <span className="auth-feature-icon">
                <Icon />
              </span>
              {text}
            </li>
          ))}
        </ul>
        <p className="auth-footnote">
          <ShieldIcon />
          Simulated capital. Real learning.
        </p>
      </section>
      <section className="auth-card-wrap">
        <div className="auth-panel">{children}</div>
      </section>
    </div>
  );
}

/** Logo and name at the top of the sign-in card. */
export function AuthBrand() {
  return (
    <div className="auth-brand">
      <span className="auth-brand-mark">
        <SimulationsIcon width={20} height={20} />
      </span>
      <strong>Algo Trade Simulator</strong>
    </div>
  );
}
