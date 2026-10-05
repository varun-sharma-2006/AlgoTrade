import type {
  CustomStrategy,
  MarketQuote,
  Simulation,
  SimulationInput,
  SimulationUpdate,
  TrainingPayload,
  TrainingResult,
  PredictionResult,
  User,
  WalkForwardPayload,
  WalkForwardResult,
  RobustnessPayload,
  RobustnessResult,
  OptimizeResult,
  ReviewResult,
} from "../types";
import { SimulationForm } from "./SimulationForm";
import { SimulationList } from "./SimulationList";
import { Watchlist } from "./Watchlist";
import { StrategyTrainer } from "./StrategyTrainer";

interface DashboardProps {
  user: User;
  watchlist: MarketQuote[];
  simulations: Simulation[];
  onRefreshWatchlist: () => void;
  onCreateSimulation: (payload: SimulationInput) => Promise<void> | void;
  onUpdateSimulation: (id: string, payload: SimulationUpdate) => Promise<void> | void;
  onDeleteSimulation: (id: string) => Promise<void> | void;
  onTrainStrategy: (payload: TrainingPayload) => Promise<void> | void;
  onPredictStrategy: (symbol: string) => Promise<void> | void;
  onWalkForward?: (payload: WalkForwardPayload) => Promise<WalkForwardResult>;
  onRobustness?: (payload: RobustnessPayload) => Promise<RobustnessResult>;
  onOptimize?: (payload: TrainingPayload) => Promise<OptimizeResult>;
  onReview?: (payload: RobustnessPayload) => Promise<ReviewResult>;
  onShare?: (payload: TrainingPayload) => Promise<string>;
  onNotebook?: (payload: TrainingPayload) => Promise<Record<string, unknown>>;
  recentTraining: TrainingResult | null;
  recentPrediction: PredictionResult | null;
  onLogout: () => void;
  loading: boolean;
  error?: string | null;
  customStrategies?: CustomStrategy[];
}

export function Dashboard({
  user,
  watchlist,
  simulations,
  onRefreshWatchlist,
  onCreateSimulation,
  onUpdateSimulation,
  onDeleteSimulation,
  onTrainStrategy,
  onPredictStrategy,
  onWalkForward,
  onRobustness,
  onOptimize,
  onReview,
  onShare,
  onNotebook,
  recentTraining,
  recentPrediction,
  onLogout,
  loading,
  error,
  customStrategies,
}: DashboardProps) {
  return (
    <main>
      <div className="header">
        <div>
          <span className="eyebrow">Workspace</span>
          <h1>Simulations</h1>
          <p>Track live market moves, backtest strategies, and manage your paper-trading experiments.</p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="section-grid">
        <Watchlist quotes={watchlist} onRefresh={onRefreshWatchlist} loading={loading} />
        <SimulationForm onSubmit={onCreateSimulation} loading={loading} customStrategies={customStrategies} />
        <StrategyTrainer
          onTrain={onTrainStrategy}
          onPredict={onPredictStrategy}
          onWalkForward={onWalkForward}
          onRobustness={onRobustness}
          onOptimize={onOptimize}
          onReview={onReview}
          onShare={onShare}
          onNotebook={onNotebook}
          training={recentTraining}
          prediction={recentPrediction}
          loading={loading}
        />
      </div>

      <SimulationList
        simulations={simulations}
        onUpdate={onUpdateSimulation}
        onDelete={onDeleteSimulation}
        loading={loading}
      />
    </main>
  );
}
