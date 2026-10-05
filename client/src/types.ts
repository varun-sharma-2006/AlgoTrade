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

export type OperandKind =
  | "price"
  | "sma"
  | "ema"
  | "rsi"
  | "macd"
  | "macd_signal"
  | "macd_hist"
  | "atr"
  | "volume"
  | "volume_sma"
  | "highest"
  | "lowest"
  | "roc"
  | "value";

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
  /** "all": every entry rule must hold (default); "any": one is enough. */
  entryMode?: "all" | "any";
  side?: "long" | "short";
  stopLoss?: number | null;
  takeProfit?: number | null;
  /** Fraction from the best close since entry. */
  trailingStop?: number | null;
  maxHoldDays?: number | null;
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
  ledger?: LedgerEntry[];
  /** Exchange-rate gain or loss since the start, when the stock trades in another currency. */
  fxReturn?: number;
  fxConverted?: boolean;
  fxMissing?: boolean;
  baseCurrency?: string;
  /** An order decided at the last close that fills at the next open. */
  pendingOrder?: "buy" | "sell" | null;
  brokerMirror?: boolean;
  error?: string;
}

export interface LedgerEntry {
  date: string;
  side: "buy" | "sell";
  price: number;
  shares: number;
  notional: number;
  fee: number;
  slippage: number;
  stop: boolean;
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
    baseCurrency?: string;
    execution?: ExecutionMode;
  };
  history: Array<{ date: string; value: number }>;
  allocation: Array<{ id: string; symbol: string; value: number; weight: number }>;
  positions: PortfolioPosition[];
}

export interface SimulationUpdate {
  status?: string;
  notes?: string | null;
  brokerMirror?: boolean;
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

/** Return and risk statistics for one equity curve (annualised; Sharpe and Sortino net of the risk-free rate). */
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
  sellFeeBps?: number;
  borrowBps?: number;
  riskFreeRate?: number;
  var95?: number;
  cvar95?: number;
  profitFactor?: number | null;
  turnover?: number;
  avgGrossExposure?: number;
  longTrades?: number;
  shortTrades?: number;
  execution?: ExecutionMode;
  sizing?: SizingMode;
  allowShort?: boolean;
  periodsPerYear?: number;
  overnightReturn?: number | null;
  intradayReturn?: number | null;
  impactCost?: number | null;
  maxParticipation?: number | null;
  earningsReturn?: number | null;
  buyHoldEarningsReturn?: number | null;
  earningsDays?: number | null;
  avoidedEarnings?: boolean;
}

export type ExecutionMode = "close" | "next_open";
export type SizingMode = "full" | "fixed" | "vol-target";

export type StrategyId =
  | "sma-crossover"
  | "mean-reversion"
  | "trend-follow"
  | "regime-switch"
  | "ml-logistic"
  | "buy-hold"
  | "custom";

export interface CostModel {
  model: string;
  description: string;
  buyFeeBps: number;
  sellFeeBps: number;
  slippageBps: number;
  buyBps: number;
  sellBps: number;
}

export interface MonthlyReturn {
  month: string;
  year: number;
  return: number;
}

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
  horizon?: number;
  modelType?: 0 | 1;
  erWindow?: number;
  erThreshold?: number;
  rules?: StrategyRules;
  slippageBps?: number;
  execution?: ExecutionMode;
  sizing?: SizingMode;
  sizeFraction?: number;
  targetVol?: number;
  maxLeverage?: number;
  allowShort?: boolean;
  borrowBps?: number;
  riskFreeRate?: number;
  interval?: BarInterval;
  capital?: number;
  marketImpact?: boolean;
  avoidEarnings?: boolean;
  newsFeatures?: boolean;
}

export type BarInterval = "1d" | "1h";

export interface TaxReport {
  region: "IN" | "US";
  crypto: boolean;
  currency: string;
  capital: number;
  rules: string;
  totalTax: number;
  preTaxReturn: number;
  afterTaxReturn: number;
  carryForwardLoss: number;
  years: Array<{ year: string; shortTermGain: number; longTermGain: number; taxableLongTerm: number; tax: number }>;
}

export interface BacktestTrade {
  entryDate: string;
  entryPrice: number;
  exitDate: string;
  exitPrice: number;
  return: number;
  side?: "long" | "short";
  bars?: number;
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
  costs?: CostModel;
  currency?: string;
  interval?: BarInterval;
  tax?: TaxReport;
  earnings?: { available: boolean; recent?: string[]; reason?: string };
  news?: { used: boolean; detail: string | null };
  monthly?: MonthlyReturn[];
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
      rollingSharpe?: number | null;
      rollingBeta?: number | null;
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
  modelType?: 0 | 1;
  /** "coefficient" (logistic regression, signed) or "importance" (boosted trees, share of total gain). */
  weightKind?: "coefficient" | "importance";
  features?: string[];
  horizon?: number;
  calibration?: Array<{ predicted: number; actual: number; count: number }>;
  thresholdScan?: Array<{ threshold: number; return: number; trades: number }>;
  suggestedThreshold?: number | null;
}

export type WalkForwardStrategyId =
  | "sma-crossover"
  | "mean-reversion"
  | "trend-follow"
  | "regime-switch"
  | "ml-logistic"
  | "custom";

export interface WalkForwardPayload {
  symbol: string;
  strategyId: WalkForwardStrategyId;
  slippageBps?: number;
  execution?: ExecutionMode;
  allowShort?: boolean;
  rules?: StrategyRules;
}

/* ---------- Robustness ---------- */

export interface Percentiles {
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export interface RobustnessPayload extends TrainingPayload {
  paths?: number;
}

export interface SensitivityCell {
  x: number;
  y: number | null;
  valid: boolean;
  sharpe?: number;
  totalReturn?: number;
  maxDrawdown?: number;
  trades?: number;
}

export interface RobustnessResult {
  symbol: string;
  strategyId: StrategyId;
  parameters: Record<string, number>;
  metrics: { totalReturn: number; sharpe: number; maxDrawdown: number; buyHoldReturn: number; trades: number };
  monteCarlo: {
    paths: number;
    block: number;
    finalReturn: Percentiles;
    maxDrawdown: Percentiles;
    sharpe: Percentiles;
    probLoss: number;
    probBeatBuyHold: number | null;
    fan: Array<Percentiles & { step: number; timestamp: string }>;
  } | null;
  sensitivity: {
    xParam: string;
    xValues: number[];
    yParam: string | null;
    yValues: number[];
    cells: SensitivityCell[];
    current: Record<string, number | null>;
    positiveShare: number;
    medianSharpe: number;
    bestSharpe: number;
  } | null;
  deflatedSharpe: {
    trials: number;
    sharpe: number;
    expectedMaxSharpe: number;
    deflatedSharpe: number;
    probabilisticSharpe: number;
    skew: number;
    kurtosis: number;
  } | null;
  pbo: {
    pbo: number;
    combinations: number;
    trials: number;
    slices: number;
    medianLogit: number;
    winnerOutOfSampleSharpe: number;
  } | null;
  randomEntries?: { paths: number; trades: number; percentile: number; median: number; p95: number } | null;
  sixtyForty?: RiskStats | null;
  factors?: FactorAttribution | null;
  period: { start: string; end: string; days: number };
}

export interface FactorAttribution {
  days: number;
  start: string;
  end: string;
  alpha: number;
  alphaT: number;
  alphaSignificant: boolean;
  rSquared: number;
  loadings: Array<{ factor: string; label: string; beta: number; t: number }>;
}

/* ---------- Basket (portfolio) backtests ---------- */

export type Weighting = "equal" | "inverse-vol" | "risk-parity";
export type Rebalance = "weekly" | "monthly" | "quarterly";

export interface BasketPayload {
  symbols: string[];
  strategyId: string;
  parameters?: Record<string, number>;
  rules?: StrategyRules | null;
  weighting: Weighting;
  rebalance: Rebalance;
  topN: number;
  allowShort?: boolean;
  slippageBps?: number;
  riskFreeRate?: number;
}

export interface CrossSectionRow {
  symbol: string;
  sharpe: number;
  totalReturn: number;
  buyHoldReturn: number;
  excessReturn: number;
  buyHoldSharpe: number;
  maxDrawdown: number;
  trades: number;
}

export interface BasketResult {
  symbols: string[];
  strategyId: string;
  parameters: Record<string, number>;
  weighting: Weighting;
  rebalance: Rebalance;
  topN: number;
  metrics: RiskStats & {
    var95: number;
    cvar95: number;
    equalWeightReturn: number;
    excessReturn: number;
    turnover: number;
    costPaid: number;
    rebalances: number;
  };
  equalWeight: RiskStats;
  benchmark: BenchmarkStats | null;
  holdings: Array<{
    symbol: string;
    avgWeight: number;
    finalWeight: number;
    contribution: number;
    timeInMarket: number;
    buyHoldReturn: number;
  }>;
  curve: Array<{ timestamp: string; equity: number; equalWeight: number; drawdown: number }>;
  monthly: MonthlyReturn[];
  period: { start: string; end: string; days: number };
  missing: string[];
  crossSection: {
    rows: CrossSectionRow[];
    summary: {
      count: number;
      medianSharpe?: number;
      sharpeP25?: number;
      sharpeP75?: number;
      shareProfitable?: number;
      shareBeatBuyHold?: number;
      shareSharpeAboveBuyHold?: number;
    };
  };
  riskFreeRate: number;
}

/* ---------- Alerts ---------- */

export interface AlertSettings {
  email: boolean;
  telegramChatId: string | null;
}

export interface AlertSettingsResponse {
  settings: AlertSettings;
  available: { email: boolean; telegram: boolean; broker?: boolean; push?: boolean };
}

export interface DailySignal {
  simulationId: string;
  symbol: string;
  strategy: string;
  signal: string;
  summary: string;
  date: string;
  price: number;
  currency: string;
  actionable: boolean;
  brokerMirror?: boolean;
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
  position?: number;
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

/* ---------- Research tools ---------- */

export interface OptimizeResult {
  symbol: string;
  strategyId: string;
  current: { parameters: Record<string, number>; sharpe: number; totalReturn: number };
  best: { parameters: Record<string, number>; sharpe: number; totalReturn: number; maxDrawdown: number; trades: number };
  trials: number;
  deflatedSharpe: RobustnessResult["deflatedSharpe"];
  pbo: RobustnessResult["pbo"];
  walkForward: { inSampleAnnualized: number; outOfSampleAnnualized: number; buyHoldAnnualized: number } | null;
  warnings: string[];
  trustworthy: boolean;
}

export interface ReviewFinding {
  level: "good" | "warn" | "bad";
  title: string;
  detail: string;
}

export interface ReviewVerdict {
  label: string;
  text: string;
  good: number;
  bad: number;
  warnings: number;
}

export interface ReviewResult {
  symbol: string;
  strategyId: string;
  findings: ReviewFinding[];
  verdict: ReviewVerdict;
  summary: string | null;
}

export interface LeaderboardRow {
  strategyId: string;
  name: string;
  symbols: number;
  medianSharpe: number;
  worstSharpe: number;
  medianReturn: number;
  medianMaxDrawdown: number;
  shareBeatBuyHold: number;
  shareBetterSharpe: number;
  score: number;
  results: Array<{ symbol: string; sharpe: number; totalReturn: number; excessReturn: number; maxDrawdown: number }>;
}

export interface LeaderboardResult {
  symbols: string[];
  missing: string[];
  rows: LeaderboardRow[];
}

export interface SipPayload {
  symbol: string;
  monthly: number;
  stepUp: number;
  years: number;
  timingStrategy?: string | null;
  timingRules?: StrategyRules | null;
}

export interface SipOutcome {
  value: number;
  gain: number;
  xirr?: number | null;
  cagr?: number | null;
  taxIfRedeemed: number;
  cashWaiting?: number;
}

export interface SipResult {
  symbol: string;
  currency: string;
  region: "IN" | "US";
  timingStrategy: string | null;
  months: number;
  years: number;
  invested: number;
  lastInstalment: number;
  sip: SipOutcome;
  lumpSum: SipOutcome;
  timed?: SipOutcome;
  series: Array<{ date: string; invested: number; sip: number; lumpSum: number; timed: number | null }>;
  period: { start: string; end: string };
}

export interface OptionsPayload {
  symbol: string;
  strategy: "covered-call" | "cash-secured-put";
  otm: number;
  days: number;
  volPremium: number;
  costBps: number;
}

export interface OptionsResult {
  symbol: string;
  currency: string;
  lastPrice: number;
  strategy: OptionsPayload["strategy"];
  otm: number;
  days: number;
  volPremium: number;
  metrics: RiskStats & {
    buyHoldReturn: number;
    excessReturn: number;
    premiumYield: number;
    rolls: number;
    assigned: number;
    assignedShare: number;
  };
  buyHold: RiskStats;
  curve: Array<{ timestamp: string; equity: number; buyHold: number; drawdown: number }>;
  period: { start: string; end: string; days: number };
}

export interface ReportSummary {
  id: string;
  title: string;
  author: string;
  createdAt: string;
  symbol?: string | null;
  path?: string;
}

export interface SharedReport {
  id: string;
  title: string;
  author: string;
  createdAt: string;
  content: {
    backtest: TrainingResult;
    robustness: RobustnessResult | null;
    review: { findings: ReviewFinding[]; verdict: ReviewVerdict };
    settings: TrainingPayload;
  };
}

export interface TextRulesResult {
  rules: StrategyRules;
  source: "gemini" | "parser";
  notes: string;
}

export interface WatchAlert {
  id: string;
  symbol: string;
  condition: Condition;
  note?: string | null;
  status: {
    triggered: boolean;
    left: number | null;
    right: number | null;
    date: string;
    price: number;
    currency: string;
    description: string;
  } | null;
}

export interface SurvivorshipResult {
  asOf: string;
  membersThen: number;
  inIndexThen: string[];
  joinedLater: Array<{ symbol: string; date: string | null }>;
  notInIndex: string[];
  removedSince: Array<{ symbol: string; date: string }>;
  survivorShare: number;
}

export interface Sp500Sample {
  asOf: string;
  symbols: string[];
  leftSince: string[];
  membersThen: number;
}
