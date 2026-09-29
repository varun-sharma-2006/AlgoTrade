import type { Simulation, SimulationUpdate } from "../types";

interface SimulationListProps {
  simulations: Simulation[];
  onUpdate: (id: string, payload: SimulationUpdate) => Promise<void> | void;
  onDelete: (id: string) => Promise<void> | void;
  loading: boolean;
}

const statusLabels: Record<string, string> = {
  active: "Active",
  completed: "Completed",
  paused: "Paused",
};

export function SimulationList({ simulations, onUpdate, onDelete, loading }: SimulationListProps) {
  const handleToggleStatus = (simulation: Simulation) => {
    const nextStatus = simulation.status === "completed" ? "active" : "completed";
    void onUpdate(simulation.id, { status: nextStatus });
  };

  return (
    <div className="card">
      <h2>Saved simulations</h2>
      <p className="hint">Private to your account and saved between visits.</p>

      {simulations.length === 0 ? (
        <div className="empty-state">Launch your first simulation to see it listed here.</div>
      ) : (
        <table className="simulation-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Strategy</th>
              <th>Capital</th>
              <th>Status</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {simulations.map((simulation) => (
              <tr key={simulation.id}>
                <td>
                  <span className="badge">{simulation.symbol}</span>
                </td>
                <td>{simulation.strategy}</td>
                <td>
                  {simulation.startingCapital.toLocaleString(undefined, {
                    style: "currency",
                    currency: "USD",
                  })}
                </td>
                <td>
                  <span className={`status-pill ${simulation.status === "completed" ? "done" : "live"}`}>
                    {statusLabels[simulation.status] ?? simulation.status}
                  </span>
                </td>
                <td>{new Date(simulation.createdAt).toLocaleString()}</td>
                <td>
                  <div className="flex-row">
                    <button
                      type="button"
                      disabled={loading}
                      className="button-ghost"
                      onClick={() => handleToggleStatus(simulation)}
                    >
                      {simulation.status === "completed" ? "Reopen" : "Mark complete"}
                    </button>
                    <button
                      type="button"
                      disabled={loading}
                      className="button-danger"
                      onClick={() => onDelete(simulation.id)}
                    >
                      Remove
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
