import type {
  MarketQuote,
  Simulation,
  SimulationInput,
  SimulationUpdate,
  TrainingPayload,
  TrainingResult,
  PredictionResult,
  User,
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
  recentTraining: TrainingResult | null;
  recentPrediction: PredictionResult | null;
  onLogout: () => void;
  loading: boolean;
  error?: string | null;
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
  recentTraining,
  recentPrediction,
  onLogout,
  loading,
  error,
}: DashboardProps) {
  return (
    <main>
      <div className="header">
        <div>
          <h1>Simulation workspace</h1>
          <p style={{ color: "rgba(226,232,240,0.7)", margin: "0.4rem 0 0" }}>
            Track real-time market moves, iterate on strategies, and manage experiments.
          </p>
        </div>
        <button type="button" onClick={onLogout} style={{ background: "#ef4444" }}>
          Log out
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="section-grid">
        <Watchlist quotes={watchlist} onRefresh={onRefreshWatchlist} loading={loading} />
        <SimulationForm onSubmit={onCreateSimulation} loading={loading} />
        <StrategyTrainer
          onTrain={onTrainStrategy}
          onPredict={onPredictStrategy}
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
