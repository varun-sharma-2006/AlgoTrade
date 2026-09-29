import { useMemo } from "react";

interface SparklineChartProps {
  points: Array<{ timestamp: string; close: number }>;
  /** Fill the container's width instead of the fixed 180px thumbnail size. */
  stretch?: boolean;
}

export function SparklineChart({ points, stretch = false }: SparklineChartProps) {
  // Hooks must run on every render, so this stays above the early return.
  const gradientId = useMemo(() => `sparkline-gradient-${Math.random().toString(36).slice(2)}`, []);
  if (!points.length) {
    return null;
  }

  const width = 180;
  const height = 72;
  const values = points.map((point) => point.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / Math.max(points.length - 1, 1);

  const projectY = (value: number) => height - ((value - min) / range) * height;
  const linePath = points
    .map((point, index) => {
      const x = index * step;
      const y = projectY(point.close);
      return `${index === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");

  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  const baselineY = projectY(points[0].close);
  const lastPoint = points[points.length - 1];
  const lastX = (points.length - 1) * step;
  const lastY = projectY(lastPoint.close);

  const gridLines = Array.from({ length: 4 }, (_, index) => {
    const y = (height / 4) * (index + 1);
    return { key: index, y };
  });

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={stretch ? "sparkline stretch" : "sparkline"}
      preserveAspectRatio={stretch ? "none" : undefined}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(212,175,106,0.35)" />
          <stop offset="100%" stopColor="rgba(212,175,106,0)" />
        </linearGradient>
      </defs>
      {gridLines.map((line) => (
        <line key={line.key} x1={0} x2={width} y1={line.y} y2={line.y} stroke="rgba(238,232,220,0.06)" strokeWidth={0.8} />
      ))}
      <path d={areaPath} fill={`url(#${gradientId})`} opacity={0.75} />
      <path
        d={linePath}
        fill="none"
        stroke="#e3c27f"
        strokeWidth={1.8}
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <line x1={0} x2={width} y1={baselineY} y2={baselineY} stroke="rgba(212,175,106,0.22)" strokeDasharray="3 5" />
      {/* A stretched (non-uniformly scaled) chart would squash the dot into an oval, so skip it there. */}
      {stretch ? null : <circle cx={lastX} cy={lastY} r={3.6} fill="#f6dfa8" stroke="#0b0d14" strokeWidth={1.5} />}
    </svg>
  );
}
