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
  AuthConfig,
  VisitorsResponse,
  CustomStrategy,
  PortfolioResponse,
  StrategyRules,
  WalkForwardPayload,
  WalkForwardResult,
  RobustnessPayload,
  RobustnessResult,
  BasketPayload,
  BasketResult,
  AlertSettings,
  AlertSettingsResponse,
  DailySignal,
  OptimizeResult,
  ReviewResult,
  LeaderboardResult,
  SipPayload,
  SipResult,
  OptionsPayload,
  OptionsResult,
  ReportSummary,
  SharedReport,
  TextRulesResult,
  WatchAlert,
  Condition,
  SurvivorshipResult,
  Sp500Sample,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
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
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : undefined;
  } catch {
    // Non-JSON body, e.g. a plain-text "Internal Server Error" from a proxy or crashed handler.
    data = { detail: text };
  }

  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      window.localStorage.removeItem("algo-trade-session");
    }
    const detail = (data as { detail?: unknown } | undefined)?.detail ?? response.statusText;
    throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
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

export function fetchAuthConfig() {
  return request<AuthConfig>("/auth/config");
}

export function googleLogin(credential: string) {
  return request<AuthResponse>("/auth/google", { method: "POST", body: { credential } });
}

export function fetchMe(token: string) {
  return request<User>("/auth/me", { token });
}

export function fetchVisitors(token: string) {
  return request<VisitorsResponse>("/admin/visitors", { token });
}

export function fetchPortfolio(token: string) {
  return request<PortfolioResponse>("/portfolio", { token });
}

export function fetchCustomStrategies(token: string) {
  return request<CustomStrategy[]>("/strategies/custom", { token });
}

export function saveCustomStrategy(
  token: string,
  payload: { name: string; description?: string; rules: StrategyRules },
) {
  return request<CustomStrategy>("/strategies/custom", { method: "POST", body: payload, token });
}

export function deleteCustomStrategy(token: string, id: string) {
  return request<void>(`/strategies/custom/${encodeURIComponent(id)}`, { method: "DELETE", token });
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

export function runWalkForward(token: string, payload: WalkForwardPayload) {
  return request<WalkForwardResult>("/analytics/walk-forward", { method: "POST", body: payload, token });
}

export function runRobustness(token: string, payload: RobustnessPayload) {
  return request<RobustnessResult>("/analytics/robustness", { method: "POST", body: payload, token });
}

export function runBasket(token: string, payload: BasketPayload) {
  return request<BasketResult>("/analytics/basket", { method: "POST", body: payload, token });
}

export function exportPine(token: string, name: string, rules: StrategyRules) {
  return request<{ script: string }>("/strategies/pine", { method: "POST", body: { name, rules }, token });
}

export function fetchAlertSettings(token: string) {
  return request<AlertSettingsResponse>("/alerts/settings", { token });
}

export function saveAlertSettings(token: string, settings: AlertSettings) {
  return request<AlertSettingsResponse>("/alerts/settings", { method: "PUT", body: settings, token });
}

export function sendTestAlert(token: string) {
  return request<{ sent: Record<string, boolean> }>("/alerts/test", { method: "POST", token });
}

export function fetchDailySignals(token: string) {
  return request<DailySignal[]>("/alerts/signals", { token });
}

export function predictStrategy(
  token: string,
  symbol: string,
  strategy?: {
    strategyId: string;
    parameters: Record<string, number>;
    rules?: StrategyRules | null;
    allowShort?: boolean;
  },
) {
  // Sending the strategy lets any server instance answer, even one that didn't run the backtest.
  return request<PredictionResult>("/analytics/predict", {
    method: "POST",
    body: { symbol, ...strategy },
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

export function optimizeStrategy(token: string, payload: TrainingPayload) {
  return request<OptimizeResult>("/analytics/optimize", { method: "POST", body: payload, token });
}

export function reviewBacktest(token: string, payload: RobustnessPayload) {
  return request<ReviewResult>("/analytics/review", { method: "POST", body: payload, token });
}

export function runLeaderboard(token: string, symbols: string[], custom: Array<{ name: string; rules: StrategyRules }>) {
  return request<LeaderboardResult>("/analytics/leaderboard", { method: "POST", body: { symbols, custom }, token });
}

export function runSip(token: string, payload: SipPayload) {
  return request<SipResult>("/analytics/sip", { method: "POST", body: payload, token });
}

export function runOptions(token: string, payload: OptionsPayload) {
  return request<OptionsResult>("/analytics/options", { method: "POST", body: payload, token });
}

export function rulesFromText(token: string, text: string) {
  return request<TextRulesResult>("/strategies/from-text", { method: "POST", body: { text }, token });
}

export function exportNotebook(token: string, payload: TrainingPayload) {
  return request<Record<string, unknown>>("/strategies/notebook", { method: "POST", body: payload, token });
}

export function createReport(token: string, payload: RobustnessPayload & { title?: string; includeRobustness?: boolean }) {
  return request<ReportSummary>("/reports", { method: "POST", body: payload, token });
}

export function fetchMyReports(token: string) {
  return request<ReportSummary[]>("/reports", { token });
}

export function fetchReport(id: string) {
  return request<SharedReport>("/reports/" + encodeURIComponent(id));
}

export function deleteReport(token: string, id: string) {
  return request<void>("/reports/" + encodeURIComponent(id), { method: "DELETE", token });
}

export function fetchWatchAlerts(token: string) {
  return request<WatchAlert[]>("/alerts/watch", { token });
}

export function addWatchAlert(token: string, payload: { symbol: string; condition: Condition; note?: string }) {
  return request<WatchAlert>("/alerts/watch", { method: "POST", body: payload, token });
}

export function deleteWatchAlert(token: string, id: string) {
  return request<void>("/alerts/watch/" + encodeURIComponent(id), { method: "DELETE", token });
}

/** Save JSON or text as a file download in the browser. */
export function downloadFile(name: string, contents: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function checkSurvivorship(token: string, symbols: string[], years = 2) {
  return request<SurvivorshipResult>("/analytics/survivorship", { method: "POST", body: { symbols, years }, token });
}

export function fetchSp500Sample(token: string, size = 20, years = 2) {
  return request<Sp500Sample>(`/analytics/sp500-sample?size=${size}&years=${years}`, { token });
}

export function fetchPushKey() {
  return request<{ publicKey: string | null; available: boolean }>("/push/key");
}

export function subscribePush(token: string, subscription: PushSubscriptionJSON) {
  return request<{ subscribed: boolean; devices: number }>("/push/subscribe", { method: "POST", body: subscription, token });
}

export function unsubscribePush(token: string, subscription: PushSubscriptionJSON) {
  return request<{ subscribed: boolean }>("/push/unsubscribe", { method: "POST", body: subscription, token });
}
