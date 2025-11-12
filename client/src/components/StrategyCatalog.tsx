import type { StrategyDefinition } from "../types";

interface StrategyCatalogProps {
  strategies: StrategyDefinition[];
}

export function StrategyCatalog({ strategies }: StrategyCatalogProps) {
  return (
    <section className="panel">
      <header>
        <h2>Strategy catalogue</h2>
        <span className="hint">Reference playbooks powering the simulator</span>
      </header>
      {strategies.length ? (
        <ul className="strategy-list">
          {strategies.map((strategy) => (
            <li key={strategy.id}>
              <div className="heading">
                <strong>{strategy.name}</strong>
                <span className="badge">{strategy.id}</span>
              </div>
              <p>{strategy.description}</p>
              <div className="meta">
                <span>
                  <strong>Best for:</strong> {strategy.recommendedFor.join(", ")}
                </span>
                <span>
                  <strong>Parameters:</strong>{" "}
                  {strategy.parameters.map((param) => `${param.name ?? Object.keys(param)[0]}=${param.value ?? Object.values(param)[0]}`).join(", ")}
                </span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="empty">No strategies available.</p>
      )}
    </section>
  );
}
