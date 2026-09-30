export interface User {
  id: string;
  email: string;
  name: string;
  picture?: string;
  isAdmin?: boolean;
}

export interface AuthConfig {
  googleClientId: string | null;
  passwordLogin: boolean;
  devBypass: boolean;
}

export interface Visitor extends User {
  createdAt: string | null;
  lastLoginAt: string | null;
  loginCount: number;
}

export interface SignInEvent {
  id: string;
  userId: string;
  email: string;
  name: string;
  picture?: string | null;
  provider: string;
  userAgent?: string | null;
  at: string;
}

export interface VisitorsResponse {
  users: Visitor[];
  logins: SignInEvent[];
  totalUsers: number;
}

export interface MarketQuote {
  symbol: string;
  price: number | null;
  change: number | null;
  changePercent: number | null;
  previousClose?: number | null;
  currency?: string | null;
  updated: string;
}

export interface Simulation {
  id: string;
  symbol: string;
  strategy: string;
  startingCapital: number;
  status: string;
  createdAt: string;
  notes?: string | null;
  strategyId?: string;
  parameters?: Record<string, number>;
  rules?: StrategyRules | null;
  startDate?: string;
}

export interface SimulationInput {
  symbol: string;
  strategy: string;
  startingCapital: number;
  notes?: string;
  strategyId?: string;
  parameters?: Record<string, number>;
  rules?: StrategyRules | null;
  startDate?: string;
}

/* ---------- Strategy Builder ---------- */

export type OperandKind = "price" | "sma" | "ema" | "rsi" | "value";

export interface Operand {
  kind: OperandKind;
  period?: number;
  value?: number;
}

export type ConditionOp = ">" | "<" | "crosses_above" | "crosses_below";

export interface Condition {
  left: Operand;
  op: ConditionOp;
  right: Operand;
}

export interface StrategyRules {
  entry: Condition[];
  exit: Condition[];
  stopLoss?: number | null;
  takeProfit?: number | null;
}

export interface CustomStrategy {
  id: string;
  name: string;
  description?: string | null;
  rules: StrategyRules;
  createdAt: string;
}

/* ---------- Portfolio ---------- */

export interface PortfolioPosition {
  id: string;
  symbol: string;
  strategy: string;
  strategyId: string;
  status: string;
  startingCapital: number;
  currency: string;
  value: number;
  pnl: number;
  pnlPct: number;
  dayChange?: number;
  dayChangePct?: number;
  inMarket?: boolean;
  shares?: number;
  lastPrice?: number;
  startDate?: string;
  startPrice?: number;
  buyHoldValue?: number;
  trades?: number;
  history?: Array<{ date: string; value: number }>;
  error?: string;
}

export interface PortfolioResponse {
  summary: {
    totalValue: number;
    totalCapital: number;
    pnl: number;
    pnlPct: number;
    dayChange: number;
    dayChangePct: number;
    positions: number;
    inMarket: number;
  };
  history: Array<{ date: string; value: number }>;
  allocation: Array<{ id: string; symbol: string; value: number; weight: number }>;
  positions: PortfolioPosition[];
}

export interface SimulationUpdate {
  status?: string;
  notes?: string | null;
}

export interface OverviewTotals {
  totalSimulations: number;
  activeSimulations: number;
  completedSimulations: number;
  totalStartingCapital: number;
  averageStartingCapital: number;
  trainedModels: number;
}

export interface OverviewResponse {
  totals: OverviewTotals;
  watchlist: string[];
  recentSimulations: Simulation[];
  strategiesTrained: string[];
}

/** Return and risk statistics for one equity curve (annualised, risk-free rate 0). */
export interface RiskStats {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  calmar: number;
}

export interface BenchmarkStats extends RiskStats {
  symbol: string;
  name: string;
  /** The strategy's sensitivity to the index, its excess annual return and its correlation. */
  beta: number;
  alpha: number;
  correlation: number;
}

export interface StrategyMetrics {
  totalReturn: number;
  annualizedReturn: number;
  volatility?: number;
  sortino?: number;
  calmar?: number;
  slippageBps?: number;
  buyHoldReturn: number;
  excessReturn: number;
  winRate: number;
  trades: number;
  closedTrades: number;
  avgTradeReturn: number;
  sharpe: number;
  maxDrawdown: number;
  exposure: number;
  feeBps: number;
}

export type StrategyId = "sma-crossover" | "mean-reversion" | "trend-follow" | "ml-logistic" | "buy-hold" | "custom";

export interface TrainingPayload {
  symbol: string;
  strategyId: StrategyId;
  shortWindow?: number;
  longWindow?: number;
  lookback?: number;
  deviation?: number;
  channel?: number;
  threshold?: number;
  trainWindow?: number;
  rules?: StrategyRules;
  slippageBps?: number;
}

export interface BacktestTrade {
  entryDate: string;
  entryPrice: number;
  exitDate: string;
  exitPrice: number;
  return: number;
}

export interface TrainingResult {
  symbol: string;
  strategyId: StrategyId;
  parameters: Record<string, number>;
  rules?: StrategyRules | null;
  metrics: StrategyMetrics;
  buyHold?: RiskStats;
  benchmark?: BenchmarkStats | null;
  model?: ModelReport;
  trades: BacktestTrade[];
  openTrade: BacktestTrade | null;
  sample: Array<
    {
      timestamp: string;
      close: number;
      equity: number;
      position: number;
      buyHold?: number;
      drawdown?: number;
      buyHoldDrawdown?: number;
    } & Record<string, number | string | null | undefined>
  >;
  period: { start: string; end: string; days: number };
  trainedAt: string;
}

/** Out-of-sample quality of the machine-learning strategy's predictions. */
export interface ModelReport {
  model: string;
  predictions: number;
  accuracy: number;
  baselineAccuracy: number;
  upDays: number;
  auc: number | null;
  precisionWhenLong: number | null;
  daysLong: number;
  threshold: number;
  trainWindow: number;
  retrainEvery: number;
  refits: number;
  latestProbability: number | null;
  featureWeights: Array<{ feature: string; weight: number }>;
}

export type WalkForwardStrategyId = "sma-crossover" | "mean-reversion" | "trend-follow" | "ml-logistic";

export interface WalkForwardPayload {
  symbol: string;
  strategyId: WalkForwardStrategyId;
  slippageBps?: number;
}

export interface WalkForwardFold {
  trainStart: string;
  testStart: string;
  testEnd: string;
  params: Record<string, number>;
  trainSharpe: number;
  trainReturn: number;
  testReturn: number;
  buyHoldReturn: number;
  trades: number;
}

export interface WalkForwardResult {
  symbol: string;
  strategyId: WalkForwardStrategyId;
  trainDays: number;
  testDays: number;
  gridSize: number;
  folds: WalkForwardFold[];
  curve: Array<{ timestamp: string; equity: number; buyHold: number; drawdown: number }>;
  metrics: {
    outOfSampleReturn: number;
    outOfSampleAnnualized: number;
    outOfSampleSharpe: number;
    outOfSampleMaxDrawdown: number;
    inSampleAnnualized: number;
    buyHoldReturn: number;
    buyHoldAnnualized: number;
    buyHoldSharpe: number;
    foldsBeatBuyHold: number;
    mostChosenParams: Record<string, number>;
    mostChosenCount: number;
    costBps: number;
  };
  period: { start: string; end: string; days: number };
  feeBps: number;
  slippageBps: number;
}

export interface PredictionResult {
  symbol: string;
  strategyId: string;
  signal: string;
  confidence: number;
  summary: string;
  metadata: Record<string, unknown>;
  generatedAt: string;
}

export interface StrategyDefinition {
  id: string;
  name: string;
  description: string;
  recommendedFor: string[];
  parameters: Array<Record<string, string>>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  actions?: ChatAction[];
  citations?: string[];
}

export interface ChatAction {
  type: string;
  label: string;
  data: Record<string, unknown>;
}

export interface ChatRequestPayload {
  message: string;
  history: Array<{ role: "user" | "assistant"; content: string }>;
}

export interface ChatResponsePayload {
  reply: string;
  citations: string[];
  actions: ChatAction[];
}

export interface SparklinePoint {
  timestamp: string;
  close: number;
}

export interface SparklineSeries {
  symbol: string;
  points: SparklinePoint[];
}

export interface SearchResult {
  symbol: string;
  shortName?: string;
  longName?: string;
  exchange?: string;
  type?: string;
}

export interface ChartPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
}

export interface ChartResponse {
  symbol: string;
  points: ChartPoint[];
  timezone?: string | null;
  currency?: string | null;
  range: string;
  interval: string;
  previousClose?: number | null;
}
