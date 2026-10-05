import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { fetchReport, saveCustomStrategy } from "./api";
import { ReportPage } from "./components/ReportPage";
import type { StrategyRules } from "./types";
import "@fontsource-variable/inter";
import "@fontsource-variable/playfair-display";
import "@fontsource-variable/playfair-display/wght-italic.css";
import "./index.css";

/** The signed-in user's token, if any (shared reports are public; saving their rules needs an account). */
function storedToken(): string | null {
  try {
    const raw = window.localStorage.getItem("algo-trade-session");
    return raw ? (JSON.parse(raw) as { token?: string }).token ?? null : null;
  } catch {
    return null;
  }
}

const reportMatch = window.location.pathname.match(/^\/r\/([\w-]{4,40})\/?$/);
const token = storedToken();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {reportMatch ? (
      <ReportPage
        reportId={reportMatch[1]}
        onLoad={fetchReport}
        onSaveRules={
          token
            ? async (name: string, rules: StrategyRules) => {
                await saveCustomStrategy(token, { name, rules });
              }
            : undefined
        }
      />
    ) : (
      <App />
    )}
  </React.StrictMode>,
);

// Installable app (PWA): cache the app shell so it opens instantly and offline. Production builds only.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => undefined);
  });
}
