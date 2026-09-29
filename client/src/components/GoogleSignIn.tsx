import { useEffect, useRef, useState } from "react";

// Minimal typing for Google Identity Services (https://developers.google.com/identity/gsi/web).
interface GoogleId {
  initialize(options: {
    client_id: string;
    callback: (response: { credential: string }) => void;
    auto_select?: boolean;
    ux_mode?: "popup" | "redirect";
  }): void;
  renderButton(element: HTMLElement, options: Record<string, unknown>): void;
  disableAutoSelect(): void;
}

declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

const SCRIPT_SRC = "https://accounts.google.com/gsi/client";
let scriptPromise: Promise<void> | null = null;

function loadGoogleScript(): Promise<void> {
  if (window.google?.accounts?.id) return Promise.resolve();
  scriptPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = SCRIPT_SRC;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => {
      scriptPromise = null;
      reject(new Error("Couldn't load Google sign-in. Check your connection or ad blocker."));
    };
    document.head.appendChild(script);
  });
  return scriptPromise;
}

/** Stops Google from silently signing the same account back in after an explicit logout. */
export function googleSignOut() {
  window.google?.accounts?.id?.disableAutoSelect();
}

interface GoogleSignInProps {
  clientId: string;
  onCredential: (credential: string) => void;
  disabled?: boolean;
}

export function GoogleSignIn({ clientId, onCredential, disabled }: GoogleSignInProps) {
  const container = useRef<HTMLDivElement | null>(null);
  const callback = useRef(onCredential);
  callback.current = onCredential;
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadGoogleScript()
      .then(() => {
        const google = window.google?.accounts?.id;
        if (cancelled || !google || !container.current) return;
        google.initialize({
          client_id: clientId,
          callback: (response) => callback.current(response.credential),
          auto_select: false,
          ux_mode: "popup",
        });
        google.renderButton(container.current, {
          theme: "outline",
          size: "large",
          shape: "pill",
          text: "continue_with",
          logo_alignment: "left",
          width: 320,
        });
      })
      .catch((error: Error) => !cancelled && setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  return (
    <div className="google-signin" aria-busy={disabled} style={disabled ? { opacity: 0.6, pointerEvents: "none" } : undefined}>
      <div ref={container} />
      {loadError ? <div className="error-banner">{loadError}</div> : null}
    </div>
  );
}
