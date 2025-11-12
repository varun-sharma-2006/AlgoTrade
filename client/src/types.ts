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
  winRate: number;
  trades: number;
  sharpe: number;
  maxDrawdown: number;
}

export interface TrainingPayload {
  symbol: string;
  shortWindow: number;
  longWindow: number;
  strategyId?: string;
}

export interface TrainingResult {
  symbol: string;
  strategyId: string;
  shortWindow: number;
  longWindow: number;
  metrics: StrategyMetrics;
  sample: Array<{ timestamp: string; close: number; shortSma: number; longSma: number }>;
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
