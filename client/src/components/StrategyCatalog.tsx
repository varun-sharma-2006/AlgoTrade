import type { StrategyDefinition } from "../types";

interface StrategyCatalogProps {
  strategies: StrategyDefinition[];
}

/** "shortWindow" -> "Short window" */
export function humanize(name: string): string {
  const words = name.replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function StrategyCatalog({ strategies }: StrategyCatalogProps) {
  return (
    <section>
      <header className="header">
        <div>
          <span className="eyebrow">Playbooks</span>
          <h1>Strategies</h1>
          <p>The rules the strategy lab backtests. Each one is long-only and pays a fee on every entry and exit.</p>
        </div>
      </header>
      {strategies.length ? (
        <ul className="strategy-grid">
          {strategies.map((strategy, index) => (
            <li key={strategy.id} className="panel strategy-card">
              <span className="strategy-number">{String(index + 1).padStart(2, "0")}</span>
              <h2>{strategy.name}</h2>
              <p>{strategy.description}</p>
              <div className="strategy-tags">
                {strategy.recommendedFor.map((tag) => (
                  <span key={tag} className="badge">
                    {tag}
                  </span>
                ))}
              </div>
              <dl className="strategy-params">
                {strategy.parameters.map((param) => {
                  const name = param.name ?? Object.keys(param)[0];
                  const value = param.value ?? Object.values(param)[0];
                  return (
                    <div key={name}>
                      <dt>{humanize(name)}</dt>
                      <dd>{value}</dd>
                    </div>
                  );
                })}
              </dl>
            </li>
          ))}
        </ul>
      ) : (
        <p className="empty">No strategies available.</p>
      )}
    </section>
  );
}
