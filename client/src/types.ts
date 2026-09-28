export interface User {
  id: string;
  email: string;
  name: string;
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
}

export interface SimulationInput {
  symbol: string;
  strategy: string;
  startingCapital: number;
  notes?: string;
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

export interface StrategyMetrics {
  totalReturn: number;
  annualizedReturn: number;
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

export type StrategyId = "sma-crossover" | "mean-reversion" | "trend-follow";

export interface TrainingPayload {
  symbol: string;
  strategyId: StrategyId;
  shortWindow?: number;
  longWindow?: number;
  lookback?: number;
  deviation?: number;
  channel?: number;
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
  metrics: StrategyMetrics;
  trades: BacktestTrade[];
  openTrade: BacktestTrade | null;
  sample: Array<{ timestamp: string; close: number; equity: number; position: number } & Record<string, number | string>>;
  period: { start: string; end: string; days: number };
  trainedAt: string;
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
