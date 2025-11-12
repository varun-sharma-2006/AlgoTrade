import { useEffect, useMemo, useState } from "react";
import type { ChartPoint, ChartResponse, SearchResult } from "../types";

interface LiveMarketPageProps {
  defaultSymbols: string[];
  onSearch: (query: string) => Promise<SearchResult[]>;
  onFetchChart: (symbol: string, options: { range: string; interval: string }) => Promise<ChartResponse>;
}

const RANGE_OPTIONS = [
  { label: "5 days", value: "5d", interval: "1h" },
  { label: "1 month", value: "1mo", interval: "1d" },
  { label: "3 months", value: "3mo", interval: "1d" },
  { label: "6 months", value: "6mo", interval: "1d" },
  { label: "1 year", value: "1y", interval: "1d" },
];

function CandlestickChart({ points }: { points: ChartPoint[] }) {
  if (!points.length) {
    return <div className="empty">No chart data available.</div>;
  }

  const width = 760;
  const height = 320;
  const candles = points.slice(-160);
  const highs = candles.map((point) => point.high);
  const lows = candles.map((point) => point.low);
  const max = Math.max(...highs);
  const min = Math.min(...lows);
  const range = max - min || 1;
  const candleWidth = Math.max(4, Math.floor(width / candles.length) - 4);
  const projectX = (index: number) => index * (width / candles.length) + candleWidth / 2;
  const projectY = (value: number) => height - ((value - min) / range) * height;

  const closePath = candles
    .map((point, index) => `${index === 0 ? "M" : "L"}${projectX(index)},${projectY(point.close)}`)
    .join(" ");

  const gridLines = Array.from({ length: 6 }, (_, index) => {
    const ratio = index / 5;
    const price = max - ratio * range;
    return { key: index, y: projectY(price), label: price.toFixed(2) };
  });

  return (
    <svg className="candlestick" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id="close-gradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(250,204,21,0.25)" />
          <stop offset="100%" stopColor="rgba(250,204,21,0)" />
        </linearGradient>
      </defs>
      {gridLines.map((line) => (
        <g key={line.key} className="grid-line">
          <line x1={0} x2={width} y1={line.y} y2={line.y} stroke="rgba(148,163,184,0.25)" strokeDasharray="6 6" />
          <text x={4} y={line.y - 4} fill="rgba(148,163,184,0.8)" fontSize={10}>
            {line.label}
          </text>
        </g>
      ))}
      <path d={`${closePath} L${width},${height} L0,${height} Z`} fill="url(#close-gradient)" opacity={0.7} />
      <path d={closePath} fill="none" stroke="#facc15" strokeWidth={1.6} strokeLinecap="round" />
      {candles.map((point, index) => {
        const x = projectX(index);
        const openY = projectY(point.open);
        const closeY = projectY(point.close);
        const highY = projectY(point.high);
        const lowY = projectY(point.low);
        const bullish = point.close >= point.open;
        const color = bullish ? "#22c55e" : "#ef4444";
        const rectY = bullish ? closeY : openY;
        const rectHeight = Math.max(Math.abs(closeY - openY), 1.8);
        return (
          <g key={point.timestamp}>
            <line x1={x + candleWidth / 2} x2={x + candleWidth / 2} y1={highY} y2={lowY} stroke={color} strokeWidth={1.2} />
            <rect
              x={x}
              y={rectY}
              width={candleWidth}
              height={rectHeight}
              fill={color}
              rx={1.5}
              opacity={0.85}
            />
          </g>
        );
      })}
    </svg>
  );
}

export function LiveMarketPage({ defaultSymbols, onSearch, onFetchChart }: LiveMarketPageProps) {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(defaultSymbols[0] ?? "AAPL");
  const [chart, setChart] = useState<ChartResponse | null>(null);
  const [range, setRange] = useState(RANGE_OPTIONS[1]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latestPoint = useMemo(() => chart?.points.at(-1) ?? null, [chart]);
  const previousPoint = useMemo(() => chart?.points.at(-2) ?? null, [chart]);
  const delta = useMemo(() => {
    if (!latestPoint || !previousPoint) {
      return null;
    }
    const change = latestPoint.close - previousPoint.close;
    const changePercent = previousPoint.close ? (change / previousPoint.close) * 100 : null;
    return { change, changePercent };
  }, [latestPoint, previousPoint]);

  useEffect(() => {
    let ignore = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await onFetchChart(selectedSymbol, { range: range.value, interval: range.interval });
        if (!ignore) {
          setChart(response);
        }
      } catch (fetchError) {
        if (!ignore) {
          console.error(fetchError);
          setError(fetchError instanceof Error ? fetchError.message : "Unable to load chart data.");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    };
    void load();
    return () => {
      ignore = true;
    };
  }, [selectedSymbol, range, onFetchChart]);

  const handleSearch = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setSearchResults([]);
      return;
    }
    try {
      const results = await onSearch(trimmed);
      setSearchResults(results.slice(0, 8));
    } catch (searchError) {
      console.error(searchError);
      setError(searchError instanceof Error ? searchError.message : "Unable to search symbols.");
    }
  };

  return (
    <section className="live-market">
      <header className="header">
        <div>
          <h1>Live market data</h1>
          <p style={{ color: "rgba(226,232,240,0.7)", marginTop: "0.35rem" }}>
            Search any ticker and inspect intraday trends with detailed candles and overlays.
          </p>
        </div>
      </header>

      <form className="search-bar" onSubmit={handleSearch}>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search symbols or companies (e.g. Reliance, MSFT, NVDA)"
        />
        <button type="submit">Search</button>
      </form>

      {searchResults.length ? (
        <ul className="search-results">
          {searchResults.map((result) => (
            <li key={`${result.symbol}-${result.type}`}>
              <button
                type="button"
                onClick={() => {
                  setSelectedSymbol(result.symbol);
                  setSearchResults([]);
                  setQuery("");
                }}
              >
                <strong>{result.symbol}</strong> {result.shortName ?? result.longName ?? ""}
                <span className="meta">{result.exchange ?? result.type ?? ""}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="range-selector">
        {RANGE_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={range.value === option.value ? "active" : ""}
            onClick={() => setRange(option)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <div className="chart-panel">
        <div className="chart-header">
          <div>
            <h2>{selectedSymbol}</h2>
            <span className="subtle">
              {chart?.currency ? `Prices in ${chart.currency}` : ""}
              {chart?.timezone ? ` · ${chart.timezone}` : ""}
            </span>
          </div>
          {latestPoint ? (
            <div className="last-quote">
              <strong>{latestPoint.close.toFixed(2)}</strong>
              {delta ? (
                <span className={delta.change >= 0 ? "positive" : "negative"}>
                  {delta.change >= 0 ? "+" : ""}
                  {delta.change.toFixed(2)} ({delta.changePercent ? delta.changePercent.toFixed(2) : "0.00"}%)
                </span>
              ) : null}
              <span className="subtle">Close</span>
            </div>
          ) : null}
        </div>

        {error && <div className="error-banner">{error}</div>}
        {loading ? <div className="splash">Loading chart...</div> : <CandlestickChart points={chart?.points ?? []} />}
      </div>

      <aside className="chart-summary">
        <h3>Latest candles</h3>
        {chart?.points?.length ? (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Open</th>
                <th>High</th>
                <th>Low</th>
                <th>Close</th>
                <th>Volume</th>
              </tr>
            </thead>
            <tbody>
              {chart.points.slice(-12).reverse().map((point) => (
                <tr key={point.timestamp}>
                  <td>{new Date(point.timestamp).toLocaleDateString()}</td>
                  <td>{point.open.toFixed(2)}</td>
                  <td>{point.high.toFixed(2)}</td>
                  <td>{point.low.toFixed(2)}</td>
                  <td>{point.close.toFixed(2)}</td>
                  <td>{point.volume ? point.volume.toLocaleString() : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="empty">No recent data.</p>
        )}
      </aside>
    </section>
  );
}
