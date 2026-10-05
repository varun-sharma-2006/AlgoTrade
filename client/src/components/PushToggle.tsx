import { useEffect, useState } from "react";

export interface PushApi {
  getKey: () => Promise<{ publicKey: string | null; available: boolean }>;
  subscribe: (subscription: PushSubscriptionJSON) => Promise<unknown>;
  unsubscribe: (subscription: PushSubscriptionJSON) => Promise<unknown>;
}

/** The server's base64url VAPID key as the bytes the Push API expects. */
export function keyBytes(base64url: string): Uint8Array {
  const padded = base64url.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (base64url.length % 4)) % 4);
  const raw = atob(padded);
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

const supported = () =>
  typeof window !== "undefined" && "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;

/** Turns phone / desktop notifications for trade signals on or off for this device. */
export function PushToggle({ api }: { api: PushApi }) {
  const [enabled, setEnabled] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!supported()) {
      setAvailable(false);
      return;
    }
    api
      .getKey()
      .then((key) => setAvailable(key.available))
      .catch(() => setAvailable(false));
    navigator.serviceWorker
      .getRegistration()
      .then((registration) => registration?.pushManager.getSubscription())
      .then((subscription) => setEnabled(Boolean(subscription)))
      .catch(() => setEnabled(false));
  }, [api]);

  const toggle = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const registration = await navigator.serviceWorker.register("/sw.js");
      const existing = await registration.pushManager.getSubscription();
      if (enabled && existing) {
        await api.unsubscribe(existing.toJSON());
        await existing.unsubscribe();
        setEnabled(false);
        setMessage("Notifications are off on this device.");
        return;
      }
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        setMessage("Notifications are blocked for this site; allow them in the browser's site settings.");
        return;
      }
      const { publicKey } = await api.getKey();
      if (!publicKey) throw new Error("Push notifications aren't configured on the server yet.");
      await navigator.serviceWorker.ready;
      const subscription =
        existing ??
        (await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(publicKey) }));
      await api.subscribe(subscription.toJSON());
      setEnabled(true);
      setMessage("Notifications are on: this device will get trade signals and alerts.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Couldn't change notifications.");
    } finally {
      setBusy(false);
    }
  };

  if (available === false) {
    return (
      <p className="hint">
        {supported()
          ? "Push notifications aren't configured on this server yet (VAPID keys)."
          : "This browser doesn't support push notifications. On iPhone, add the app to your home screen first."}
      </p>
    );
  }
  return (
    <div className="push-toggle">
      <button type="button" className="button-ghost" disabled={busy || available === null} onClick={() => void toggle()}>
        {busy ? "Working…" : enabled ? "Turn off notifications on this device" : "Notify me on this device"}
      </button>
      {message ? <span className="subtle">{message}</span> : null}
    </div>
  );
}
