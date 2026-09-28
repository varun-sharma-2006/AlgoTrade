import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "../api";

function mockFetch(status: number, body: string) {
  const fetchMock = vi.fn().mockResolvedValue(new Response(body || null, { status }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

describe("request", () => {
  it("sends JSON with the bearer token and parses the response", async () => {
    const fetchMock = mockFetch(200, JSON.stringify({ ok: true }));
    await expect(request("/thing", { method: "POST", body: { a: 1 }, token: "abc" })).resolves.toEqual({ ok: true });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer abc");
    expect(init.body).toBe('{"a":1}');
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch(204, "");
    await expect(request("/thing")).resolves.toBeUndefined();
  });

  it("raises ApiError with the server detail", async () => {
    mockFetch(422, JSON.stringify({ detail: "shortWindow must be less than longWindow" }));
    await expect(request("/train")).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "shortWindow must be less than longWindow",
    });
  });

  it("handles non-JSON error bodies", async () => {
    mockFetch(500, "Internal Server Error");
    const error = (await request("/boom").catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toBe("Internal Server Error");
  });

  it("clears the stored session on 401", async () => {
    window.localStorage.setItem("algo-trade-session", "{}");
    mockFetch(401, JSON.stringify({ detail: "Invalid or expired session" }));
    await expect(request("/me")).rejects.toMatchObject({ status: 401 });
    expect(window.localStorage.getItem("algo-trade-session")).toBeNull();
  });
});
