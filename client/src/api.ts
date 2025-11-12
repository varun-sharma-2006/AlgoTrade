import type {
  MarketQuote,
  Simulation,
  SimulationInput,
  SimulationUpdate,
  User,
  OverviewResponse,
  TrainingPayload,
  TrainingResult,
  PredictionResult,
  StrategyDefinition,
  ChatRequestPayload,
  ChatResponsePayload,
  ChatAction,
  SparklineSeries,
  SearchResult,
  ChartResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, token } = options;
  const headers: HeadersInit = {
    Accept: "application/json",
  };

  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  const data = text ? JSON.parse(text) : undefined;


if (response.status === 401) {
  const unauthorizedDetail = data?.detail ?? response.statusText;
  if (typeof window !== "undefined") {
    window.localStorage.removeItem("algo-trade-session");
  }
  const unauthorizedError = new Error(
    typeof unauthorizedDetail === "string" ? unauthorizedDetail : JSON.stringify(unauthorizedDetail),
  );
  (unauthorizedError as Error & { status?: number }).status = 401;
  throw unauthorizedError;
}

  if (!response.ok) {
    const detail = data?.detail ?? response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data as T;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface SignupPayload {
  email: string;
  password: string;
  name: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface DevAuthBypassPayload {
  email?: string;
  name?: string;
}

export function signup(payload: SignupPayload) {
  return request<AuthResponse>("/auth/signup", { method: "POST", body: payload });
}

export function login(payload: LoginPayload) {
  return request<AuthResponse>("/auth/login", { method: "POST", body: payload });
}

export function devAuthBypass(payload?: DevAuthBypassPayload) {
  return request<AuthResponse>("/dev/auth/bypass", { method: "POST", body: payload });
}

export function fetchWatchlist(symbols?: string[]) {
  const query = symbols?.length ? `?symbols=${symbols.join(",")}` : "";
  return request<MarketQuote[]>(`/market/watchlist${query}`);
}

export function fetchQuote(symbol: string) {
  return request<MarketQuote>(`/market/quote/${encodeURIComponent(symbol)}`);
}

export function fetchSimulations(token: string) {
  return request<Simulation[]>("/simulations", { token });
}

export function createSimulation(token: string, payload: SimulationInput) {
  return request<Simulation>("/simulations", { method: "POST", body: payload, token });
}

export function updateSimulation(token: string, id: string, payload: SimulationUpdate) {
  return request<Simulation>(`/simulations/${id}`, {
    method: "PATCH",
    body: payload,
    token,
  });
}

export function deleteSimulation(token: string, id: string) {
  return request<void>(`/simulations/${id}`, {
    method: "DELETE",
    token,
  });
}

export function fetchOverview(token: string) {
  return request<OverviewResponse>("/analytics/overview", { token });
}

export function fetchStrategies() {
  return request<StrategyDefinition[]>("/analytics/strategies");
}

export function trainStrategy(token: string, payload: TrainingPayload) {
  return request<TrainingResult>("/analytics/train", { method: "POST", body: payload, token });
}

export function predictStrategy(token: string, symbol: string) {
  return request<PredictionResult>("/analytics/predict", {
    method: "POST",
    body: { symbol },
    token,
  });
}

export function askChat(token: string, payload: ChatRequestPayload) {
  return request<ChatResponsePayload>("/chat", { method: "POST", body: payload, token });
}

export function fetchSparkline(token: string, symbols?: string[]) {
  const params = symbols?.length ? `?symbols=${symbols.join(",")}` : "";
  return request<SparklineSeries[]>(`/analytics/sparkline${params}`, { token });
}

export function searchSymbols(token: string, query: string) {
  const encoded = encodeURIComponent(query);
  return request<SearchResult[]>(`/market/search?q=${encoded}`, { token });
}

export function fetchChart(token: string, symbol: string, options?: { range?: string; interval?: string }) {
  const params = new URLSearchParams();
  if (options?.range) {
    params.set("range", options.range);
  }
  if (options?.interval) {
    params.set("interval", options.interval);
  }
  const query = params.toString();
  const path = `/market/chart/${encodeURIComponent(symbol)}${query ? `?${query}` : ""}`;
  return request<ChartResponse>(path, { token });
}
