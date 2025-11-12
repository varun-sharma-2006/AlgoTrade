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
      <p style={{ marginTop: "-0.5rem", color: "rgba(226,232,240,0.7)" }}>
        All simulations are stored securely per user in MongoDB.
      </p>

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
                <td>{statusLabels[simulation.status] ?? simulation.status}</td>
                <td>{new Date(simulation.createdAt).toLocaleString()}</td>
                <td>
                  <div className="flex-row">
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => handleToggleStatus(simulation)}
                      style={{
                        background: simulation.status === "completed" ? "#0ea5e9" : "#22c55e",
                      }}
                    >
                      {simulation.status === "completed" ? "Reopen" : "Mark complete"}
                    </button>
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => onDelete(simulation.id)}
                      style={{ background: "#ef4444" }}
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
