import { useMemo } from "react";

interface SparklineChartProps {
  points: Array<{ timestamp: string; close: number }>;
}

export function SparklineChart({ points }: SparklineChartProps) {
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
  const gradientId = useMemo(() => `sparkline-gradient-${Math.random().toString(36).slice(2)}`, []);

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
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="sparkline">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(96,165,250,0.4)" />
          <stop offset="100%" stopColor="rgba(96,165,250,0)" />
        </linearGradient>
      </defs>
      {gridLines.map((line) => (
        <line key={line.key} x1={0} x2={width} y1={line.y} y2={line.y} stroke="rgba(148,163,184,0.2)" strokeWidth={0.8} />
      ))}
      <path d={areaPath} fill={`url(#${gradientId})`} opacity={0.75} />
      <path d={linePath} fill="none" stroke="#60a5fa" strokeWidth={2} strokeLinecap="round" />
      <line x1={0} x2={width} y1={baselineY} y2={baselineY} stroke="rgba(96,165,250,0.25)" strokeDasharray="4 4" />
      <circle cx={lastX} cy={lastY} r={3.6} fill="#f97316" stroke="#ffffff" strokeWidth={1.2} />
    </svg>
  );
}
