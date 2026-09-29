import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GoogleLoginPage } from "../components/GoogleLoginPage";
import { VisitorsPage, describeDevice, timeAgo } from "../components/VisitorsPage";
import type { VisitorsResponse } from "../types";

afterEach(() => {
  cleanup();
  delete window.google;
});

describe("GoogleLoginPage", () => {
  it("renders Google's button and forwards the credential", async () => {
    let callback: ((response: { credential: string }) => void) | undefined;
    const renderButton = vi.fn();
    window.google = {
      accounts: {
        id: {
          initialize: vi.fn((options) => {
            callback = options.callback;
          }),
          renderButton,
          disableAutoSelect: vi.fn(),
        },
      },
    };
    const onCredential = vi.fn();
    render(<GoogleLoginPage clientId="abc.apps.googleusercontent.com" onCredential={onCredential} loading={false} />);

    await waitFor(() => expect(renderButton).toHaveBeenCalled());
    expect(window.google.accounts.id.initialize).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: "abc.apps.googleusercontent.com", auto_select: false }),
    );
    callback?.({ credential: "signed-id-token" });
    expect(onCredential).toHaveBeenCalledWith("signed-id-token");
    expect(screen.getByText(/shared with the owner of this app/)).toBeInTheDocument();
  });
});

describe("VisitorsPage", () => {
  it("shows visitors, stats and recent sign-ins", async () => {
    const now = new Date().toISOString();
    const data: VisitorsResponse = {
      totalUsers: 2,
      users: [
        { id: "1", email: "alice@gmail.com", name: "Alice", createdAt: now, lastLoginAt: now, loginCount: 3 },
        { id: "2", email: "bob@gmail.com", name: "Bob", createdAt: now, lastLoginAt: null, loginCount: 0 },
      ],
      logins: [
        {
          id: "e1",
          userId: "1",
          email: "alice@gmail.com",
          name: "Alice",
          provider: "google",
          userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0) AppleWebKit Version/18.0 Mobile Safari/605.1",
          at: now,
        },
      ],
    };
    render(<VisitorsPage onLoad={() => Promise.resolve(data)} />);

    expect(await screen.findByText("bob@gmail.com")).toBeInTheDocument();
    expect(screen.getByText("Total visitors").nextSibling?.textContent).toBe("2");
    expect(screen.getByText("Visitors today").nextSibling?.textContent).toBe("1");
    expect(screen.getByText("Safari on iOS")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export CSV" })).toBeEnabled();
  });

  it("shows the server error for non-admins", async () => {
    render(<VisitorsPage onLoad={() => Promise.reject(new Error("Admins only"))} />);
    expect(await screen.findByText("Admins only")).toBeInTheDocument();
  });
});

describe("helpers", () => {
  it("describes devices", () => {
    expect(describeDevice("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0 Safari/537.36")).toBe("Chrome on Windows");
    expect(describeDevice("Mozilla/5.0 (Windows NT 10.0) Chrome/140 Safari/537.36 Edg/140")).toBe("Edge on Windows");
    expect(describeDevice(null)).toBe("Unknown device");
  });

  it("formats relative times", () => {
    const now = Date.parse("2026-09-30T12:00:00Z");
    expect(timeAgo("2026-09-30T11:59:30Z", now)).toBe("just now");
    expect(timeAgo("2026-09-30T11:15:00Z", now)).toBe("45 min ago");
    expect(timeAgo("2026-09-29T12:00:00Z", now)).toBe("1 d ago");
    expect(timeAgo(null, now)).toBe("–");
  });
});
