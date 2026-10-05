"""Factor attribution: how much of a strategy's return is just exposure to well-known market factors?

Regresses the strategy's daily excess returns on the Fama-French five factors (market, size, value,
profitability, investment) plus momentum, from Kenneth French's data library (US stocks, published with a
lag of a month or two). A large, significant intercept (alpha) is return the factors don't explain; high
loadings with no alpha mean the "edge" is a factor you could buy cheaply in an index fund.
"""

from __future__ import annotations

import csv
import io
import math
import time
import zipfile
from typing import Any

import requests

from backend import risk
from backend.config import logger

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "five": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
}
FACTOR_NAMES = {
    "Mkt-RF": "Market",
    "SMB": "Size (small minus big)",
    "HML": "Value (high minus low book/price)",
    "RMW": "Profitability (robust minus weak)",
    "CMA": "Investment (conservative minus aggressive)",
    "Mom": "Momentum",
}
CACHE_SECONDS = 24 * 3600
_cache: dict[str, tuple[float, dict[str, dict[str, float]]]] = {}


def _parse(raw: bytes) -> dict[str, dict[str, float]]:
    """Rows of a French-library CSV as {YYYY-MM-DD: {column: decimal return}}."""
    archive = zipfile.ZipFile(io.BytesIO(raw))
    text = archive.read(archive.namelist()[0]).decode("latin-1")
    rows: dict[str, dict[str, float]] = {}
    header: list[str] | None = None
    known = set(FACTOR_NAMES) | {"RF"}
    for row in csv.reader(io.StringIO(text)):
        cells = [c.strip() for c in row]
        if len(cells) > 1 and known & set(cells[1:]):
            header = cells[1:]  # the column row starts with an empty cell: ",Mkt-RF,SMB,..."
            continue
        if not cells or not (cells[0].isdigit() and len(cells[0]) == 8):
            if rows:
                break  # the daily table ends where the dated rows stop
            continue
        if header is None:
            continue
        day = f"{cells[0][:4]}-{cells[0][4:6]}-{cells[0][6:]}"
        try:
            rows[day] = {name: float(value) / 100 for name, value in zip(header, cells[1:], strict=False)}
        except ValueError:
            continue
    return rows


def load_factors() -> dict[str, dict[str, float]] | None:
    """Daily factor returns by date (cached for a day), or None if the library can't be reached."""
    cached = _cache.get("factors")
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]
    try:
        tables = {}
        for key, name in FILES.items():
            response = requests.get(BASE + name, timeout=20)
            response.raise_for_status()
            tables[key] = _parse(response.content)
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
        logger.warning("Fama-French factors unavailable: %s", exc)
        return None
    merged = {
        day: row | {"Mom": tables["momentum"][day].get("Mom", next(iter(tables["momentum"][day].values())))}
        for day, row in tables["five"].items()
        if day in tables["momentum"]
    }
    _cache["factors"] = (time.time(), merged)
    return merged


def regress(y: list[float], x: list[list[float]]) -> tuple[list[float], list[float], float]:
    """OLS with an intercept: coefficients (intercept first), their standard errors, and R-squared."""
    from backend.ml import _solve

    n, k = len(y), len(x[0]) + 1
    rows = [[1.0, *r] for r in x]
    xtx = [[sum(r[a] * r[b] for r in rows) for b in range(k)] for a in range(k)]
    xty = [sum(r[a] * v for r, v in zip(rows, y, strict=True)) for a in range(k)]
    beta = _solve(xtx, xty)
    fitted = [sum(b * v for b, v in zip(beta, r, strict=True)) for r in rows]
    residuals = [a - b for a, b in zip(y, fitted, strict=True)]
    sse = sum(e * e for e in residuals)
    mean_y = sum(y) / n
    sst = sum((v - mean_y) ** 2 for v in y)
    sigma2 = sse / max(n - k, 1)
    # Standard errors from the diagonal of sigma^2 (X'X)^-1, one column at a time.
    errors = []
    for a in range(k):
        unit = [1.0 if i == a else 0.0 for i in range(k)]
        column = _solve(xtx, unit)
        errors.append(math.sqrt(max(sigma2 * column[a], 0.0)))
    return beta, errors, 1 - sse / sst if sst > 0 else 0.0


def attribution(
    timestamps: list[str], daily_returns: list[float], factors: dict[str, dict[str, float]] | None = None
) -> dict[str, Any] | None:
    """Factor loadings, t-statistics and annualised alpha for a daily strategy return series.

    `daily_returns[k]` is the return from timestamps[k] to timestamps[k + 1].
    """
    factors = factors if factors is not None else load_factors()
    if not factors:
        return None
    names = list(FACTOR_NAMES)
    y, x = [], []
    for k, r in enumerate(daily_returns):
        row = factors.get(timestamps[k + 1][:10])
        if row and all(name in row for name in names):
            y.append(r - row.get("RF", 0.0))
            x.append([row[name] for name in names])
    if len(y) < 60:
        return None
    beta, errors, r2 = regress(y, x)
    t_alpha = beta[0] / errors[0] if errors[0] else 0.0
    return {
        "days": len(y),
        "start": next(timestamps[k + 1][:10] for k in range(len(daily_returns)) if timestamps[k + 1][:10] in factors),
        "end": max(day for day in (ts[:10] for ts in timestamps) if day in factors),
        "alpha": beta[0] * risk.TRADING_DAYS,
        "alphaT": t_alpha,
        "alphaSignificant": abs(t_alpha) >= 2,
        "rSquared": r2,
        "loadings": [
            {
                "factor": name,
                "label": FACTOR_NAMES[name],
                "beta": beta[i + 1],
                "t": beta[i + 1] / errors[i + 1] if errors[i + 1] else 0.0,
            }
            for i, name in enumerate(names)
        ],
    }
