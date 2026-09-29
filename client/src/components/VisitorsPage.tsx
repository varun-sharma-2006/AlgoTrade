import { useCallback, useEffect, useMemo, useState } from "react";
import type { SignInEvent, Visitor, VisitorsResponse } from "../types";

interface VisitorsPageProps {
  onLoad: () => Promise<VisitorsResponse>;
}

const DAY = 24 * 60 * 60 * 1000;

export function timeAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "–";
  const seconds = Math.round((now - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days < 30 ? `${days} d ago` : new Date(iso).toLocaleDateString();
}

export function describeDevice(userAgent?: string | null): string {
  if (!userAgent) return "Unknown device";
  const ua = userAgent;
  const os = /iPhone|iPad/.test(ua)
    ? "iOS"
    : /Android/.test(ua)
      ? "Android"
      : /Windows/.test(ua)
        ? "Windows"
        : /Mac OS X|Macintosh/.test(ua)
          ? "macOS"
          : /Linux/.test(ua)
            ? "Linux"
            : "Other";
  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /Brave/.test(ua)
      ? "Brave"
      : /Firefox\//.test(ua)
        ? "Firefox"
        : /Chrome\//.test(ua)
          ? "Chrome"
          : /Safari\//.test(ua)
            ? "Safari"
            : "Browser";
  return `${browser} on ${os}`;
}

function toCsv(users: Visitor[]): string {
  const escape = (value: unknown) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const rows = users.map((u) => [u.name, u.email, u.createdAt, u.lastLoginAt, u.loginCount].map(escape).join(","));
  return ["Name,Email,First seen,Last sign-in,Sign-ins", ...rows].join("\n");
}

function Avatar({ person }: { person: { name: string; picture?: string | null } }) {
  return person.picture ? (
    <img className="avatar" src={person.picture} alt="" referrerPolicy="no-referrer" />
  ) : (
    <span className="avatar placeholder">{person.name.slice(0, 1).toUpperCase()}</span>
  );
}

export function VisitorsPage({ onLoad }: VisitorsPageProps) {
  const [data, setData] = useState<VisitorsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await onLoad());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load visitors.");
    } finally {
      setLoading(false);
    }
  }, [onLoad]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const stats = useMemo(() => {
    const now = Date.now();
    const logins: SignInEvent[] = data?.logins ?? [];
    const within = (ms: number) => logins.filter((entry) => now - new Date(entry.at).getTime() < ms);
    return {
      total: data?.totalUsers ?? 0,
      today: new Set(within(DAY).map((entry) => entry.email)).size,
      week: new Set(within(7 * DAY).map((entry) => entry.email)).size,
    };
  }, [data]);

  const downloadCsv = () => {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([toCsv(data.users)], { type: "text/csv" }));
    const link = Object.assign(document.createElement("a"), { href: url, download: "visitors.csv" });
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="visitors">
      <header className="header">
        <div>
          <h1>Visitors</h1>
          <p className="subtle">Everyone who has signed in with Google. Only admins can see this page.</p>
        </div>
        <div className="actions">
          <button type="button" onClick={downloadCsv} disabled={!data?.users.length}>
            Export CSV
          </button>
          <button type="button" onClick={() => void refresh()} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="stats-grid">
        <div className="stat-card">
          <span className="label">Total visitors</span>
          <strong className="value">{stats.total}</strong>
        </div>
        <div className="stat-card">
          <span className="label">Visitors today</span>
          <strong className="value">{stats.today}</strong>
        </div>
        <div className="stat-card">
          <span className="label">Visitors this week</span>
          <strong className="value">{stats.week}</strong>
        </div>
      </div>

      <div className="panel">
        <header>
          <h2>People</h2>
          <span className="hint">Most recent first</span>
        </header>
        {data?.users.length ? (
          <table className="visitors-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>First seen</th>
                <th>Last sign-in</th>
                <th>Sign-ins</th>
              </tr>
            </thead>
            <tbody>
              {data.users.map((visitor) => (
                <tr key={visitor.id}>
                  <td className="person">
                    <Avatar person={visitor} />
                    {visitor.name}
                  </td>
                  <td>{visitor.email}</td>
                  <td>{visitor.createdAt ? new Date(visitor.createdAt).toLocaleDateString() : "–"}</td>
                  <td>{timeAgo(visitor.lastLoginAt)}</td>
                  <td>{visitor.loginCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="empty">{loading ? "Loading..." : "Nobody has signed in yet."}</p>
        )}
      </div>

      <div className="panel">
        <header>
          <h2>Recent sign-ins</h2>
          <span className="hint">Last 100</span>
        </header>
        {data?.logins.length ? (
          <ul className="signin-log">
            {data.logins.map((entry) => (
              <li key={entry.id}>
                <Avatar person={entry} />
                <div>
                  <strong>{entry.name}</strong> <span className="subtle">{entry.email}</span>
                  <div className="subtle">{describeDevice(entry.userAgent)}</div>
                </div>
                <span className="subtle when" title={new Date(entry.at).toLocaleString()}>
                  {timeAgo(entry.at)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty">No sign-ins yet.</p>
        )}
      </div>
    </section>
  );
}
