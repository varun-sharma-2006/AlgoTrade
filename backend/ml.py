"""Machine-learning strategy: predicts whether the close `horizon` days ahead will be higher than today's.

Two models, both written in plain Python so the backend needs no numpy or scikit-learn:

* Logistic regression (L2-regularised, fitted with Newton's method), refitted every 21 trading days.
* Gradient-boosted decision stumps (logistic loss, histogram splits), refitted every 63 trading days. Stumps
  pick up thresholds ("RSI below 30 matters, above doesn't") that a linear model can't.

Features: 8 from closing prices (returns, moving-average gaps, RSI, volatility, Bollinger z-score), plus,
when available, volume vs its 20-day average, the average true range relative to price, and the market
index's 20-day return and volatility (the market regime).

Everything is time-aware, so there is no look-ahead:

* The features for day t use data up to day t only.
* The label for day i (is close[i + horizon] > close[i]?) is known only at the close of day i + horizon, so
  the model used on day t is trained on days i <= t - horizon.
* Feature scaling and tree split points are fitted on the training window alone.

The strategy is long while the predicted probability of a rise is at least `threshold` (and, with shorting
on, short while it is at most 1 - threshold).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any

from backend.strategies import Series, moving_average, rolling_std

BASE_FEATURES = [
    ("r1", "1-day return"),
    ("r5", "5-day return"),
    ("r20", "20-day return"),
    ("sma10", "Price vs 10-day average"),
    ("sma50", "Price vs 50-day average"),
    ("rsi14", "RSI (14)"),
    ("vol20", "20-day volatility"),
    ("zscore", "Bollinger z-score (20)"),
]
FEATURES = BASE_FEATURES  # kept for callers that list the price-only features
VOLUME_FEATURE = ("volRatio", "Volume vs 20-day average")
RANGE_FEATURE = ("atr14", "Average true range / price (14)")
MARKET_FEATURES = [("mkt20", "Market 20-day return"), ("mktVol20", "Market 20-day volatility")]
NEWS_FEATURE = ("news7", "News tone, previous 7 days (GDELT)")

FEATURE_WARMUP = 51  # the 50-day average and 20-day return volatility need this many closes
MIN_TRAIN = 252  # at least a year of labelled days before the first prediction
RETRAIN_EVERY = 21  # logistic regression: refit roughly monthly
RETRAIN_EVERY_BOOSTED = 63  # boosted trees: refit quarterly (slower to fit)
L2 = 1.0  # ridge penalty on the standardised coefficients (not the intercept)
BOOST_ROUNDS = 40
BOOST_LEARNING_RATE = 0.1
BOOST_BINS = 12
BOOST_MIN_LEAF = 20
DEFAULTS = {"threshold": 0.52, "trainWindow": 504, "horizon": 1, "modelType": 0}
MODEL_NAMES = {0: "Logistic regression", 1: "Gradient-boosted trees"}

_cache: dict[tuple, dict[str, Any]] = {}


def warmup() -> int:
    return FEATURE_WARMUP + MIN_TRAIN


def _forward_fill(values: Series | None) -> Series | None:
    if values is None:
        return None
    out: Series = []
    last = None
    for v in values:
        if v is not None:
            last = v
        out.append(last)
    return out


def features(
    closes: list[float], bars: dict[str, list[Any]] | None = None, market: Series | None = None
) -> list[list[float] | None]:
    """One feature row per day (None during the warm-up)."""
    return feature_table(closes, bars, market)[0]


def feature_table(
    closes: list[float], bars: dict[str, list[Any]] | None = None, market: Series | None = None
) -> tuple[list[list[float] | None], list[tuple[str, str]]]:
    from backend.rules import atr, rsi

    n = len(closes)
    bars = bars or {}
    names = list(BASE_FEATURES)
    sma10, sma20, sma50 = (moving_average(closes, w) for w in (10, 20, 50))
    std20 = rolling_std(closes, 20)
    rsi14 = rsi(closes, 14)
    returns = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, n)]
    vol20 = rolling_std(returns, 20)

    volume = bars.get("volume")
    volume_avg: Series | None = None
    if volume is not None and len(volume) == n and all(v for v in volume[-60:]):
        filled = [float(v or 0.0) for v in volume]
        volume_avg = moving_average(filled, 20)
        names.append(VOLUME_FEATURE)
    highs, lows = bars.get("high"), bars.get("low")
    atr14: Series | None = None
    if highs is not None and lows is not None and None not in highs and None not in lows and len(highs) == n:
        atr14 = atr(closes, 14, highs, lows)
        names.append(RANGE_FEATURE)
    market = _forward_fill(market) if market is not None and len(market) == n else None
    market_vol: Series | None = None
    if market is not None and sum(1 for v in market if v is not None) > n // 2:
        market_returns = [market[i] / market[i - 1] - 1 if i and market[i] and market[i - 1] else 0.0 for i in range(n)]
        market_vol = rolling_std(market_returns, 20)
        names.extend(MARKET_FEATURES)
    else:
        market = None
    news = bars.get("news")
    if news is not None and (len(news) != n or sum(1 for v in news if v is not None) < n // 3):
        news = None
    if news is not None:
        names.append(NEWS_FEATURE)

    rows: list[list[float] | None] = []
    for i in range(n):
        if i < FEATURE_WARMUP - 1 or rsi14[i] is None or vol20[i] is None:
            rows.append(None)
            continue
        c = closes[i]
        row = [
            returns[i],
            c / closes[i - 5] - 1,
            c / closes[i - 20] - 1,
            c / sma10[i] - 1,
            c / sma50[i] - 1,
            rsi14[i] / 100 - 0.5,
            vol20[i],
            (c - sma20[i]) / std20[i] if std20[i] else 0.0,
        ]
        if volume_avg is not None:
            row.append(float(volume[i] or 0.0) / volume_avg[i] - 1 if volume_avg[i] else 0.0)
        if atr14 is not None:
            row.append(atr14[i] / c if atr14[i] is not None else 0.0)
        if market is not None and market_vol is not None:
            now, then = market[i], market[i - 20]
            row.append(now / then - 1 if now and then else 0.0)
            row.append(market_vol[i] or 0.0)
        if news is not None:
            row.append(news[i] if news[i] is not None else 0.0)
        rows.append(row)
    return rows, names


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems here are at most 13x13)."""
    size = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            continue
        for r in range(col + 1, size):
            factor = a[r][col] / a[col][col]
            if factor:
                for k in range(col, size + 1):
                    a[r][k] -= factor * a[col][k]
    x = [0.0] * size
    for r in range(size - 1, -1, -1):
        if abs(a[r][r]) < 1e-12:
            continue
        x[r] = (a[r][size] - sum(a[r][k] * x[k] for k in range(r + 1, size))) / a[r][r]
    return x


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def fit(rows: list[list[float]], labels: list[int], iterations: int = 8) -> dict[str, Any]:
    """Standardise the features, then fit L2-regularised logistic regression with Newton's method."""
    k = len(rows[0])
    means = [sum(r[j] for r in rows) / len(rows) for j in range(k)]
    scales = []
    for j in range(k):
        var = sum((r[j] - means[j]) ** 2 for r in rows) / len(rows)
        scales.append(math.sqrt(var) or 1.0)
    xs = [[1.0] + [(r[j] - means[j]) / scales[j] for j in range(k)] for r in rows]
    w = [0.0] * (k + 1)
    for _ in range(iterations):
        grad = [0.0] * (k + 1)
        hess = [[0.0] * (k + 1) for _ in range(k + 1)]
        for x, y in zip(xs, labels, strict=True):
            p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x, strict=True)))
            err, weight = p - y, p * (1 - p)
            for a in range(k + 1):
                grad[a] += err * x[a]
                wx = weight * x[a]
                row = hess[a]
                for b in range(a, k + 1):
                    row[b] += wx * x[b]
        for a in range(1, k + 1):
            grad[a] += L2 * w[a]
            hess[a][a] += L2
        for a in range(k + 1):
            for b in range(a):
                hess[a][b] = hess[b][a]
        step = _solve(hess, grad)
        w = [wi - si for wi, si in zip(w, step, strict=True)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return {"type": "logistic", "weights": w, "means": means, "scales": scales}


def fit_boosted(
    rows: list[list[float]],
    labels: list[int],
    rounds: int = BOOST_ROUNDS,
    learning_rate: float = BOOST_LEARNING_RATE,
    bins: int = BOOST_BINS,
    min_leaf: int = BOOST_MIN_LEAF,
    lam: float = 1.0,
) -> dict[str, Any]:
    """Gradient boosting with depth-1 trees (stumps) on logistic loss, using quantile-binned split points."""
    n, k = len(rows), len(rows[0])
    cuts: list[list[float]] = []
    codes: list[list[int]] = []
    for j in range(k):
        column = sorted(r[j] for r in rows)
        edges = sorted({column[q * n // bins] for q in range(1, bins)})
        cuts.append(edges)
        codes.append([bisect_right(edges, r[j]) for r in rows])
    rate = min(max(sum(labels) / n, 1e-3), 1 - 1e-3)
    base = math.log(rate / (1 - rate))
    scores = [base] * n
    trees: list[tuple[int, float, float, float]] = []
    importance = [0.0] * k
    for _ in range(rounds):
        probs = [_sigmoid(s) for s in scores]
        grads = [p - y for p, y in zip(probs, labels, strict=True)]
        hess = [p * (1 - p) for p in probs]
        g_total, h_total = sum(grads), sum(hess)
        parent = g_total * g_total / (h_total + lam)
        best_gain, best = 0.0, None
        for j in range(k):
            size = len(cuts[j]) + 1
            gs, hs, cs = [0.0] * size, [0.0] * size, [0] * size
            for idx, b in enumerate(codes[j]):
                gs[b] += grads[idx]
                hs[b] += hess[idx]
                cs[b] += 1
            gl = hl = 0.0
            cl = 0
            for b in range(size - 1):
                gl += gs[b]
                hl += hs[b]
                cl += cs[b]
                if cl < min_leaf or n - cl < min_leaf:
                    continue
                gr, hr = g_total - gl, h_total - hl
                gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
                if gain > best_gain:
                    best_gain, best = gain, (j, b, cuts[j][b], -gl / (hl + lam), -gr / (hr + lam))
        if best is None:
            break
        j, b, cut, left, right = best
        importance[j] += best_gain
        trees.append((j, cut, learning_rate * left, learning_rate * right))
        column = codes[j]
        for idx in range(n):
            scores[idx] += learning_rate * (left if column[idx] <= b else right)
    return {"type": "boosted", "base": base, "trees": trees, "importance": importance}


def predict(model: dict[str, Any], row: list[float]) -> float:
    if model.get("type") == "boosted":
        z = model["base"] + sum(left if row[j] < cut else right for j, cut, left, right in model["trees"])
        return _sigmoid(z)
    w, means, scales = model["weights"], model["means"], model["scales"]
    z = w[0] + sum(w[j + 1] * (row[j] - means[j]) / scales[j] for j in range(len(row)))
    return _sigmoid(z)


def _extra_key(bars: dict[str, list[Any]] | None, market: Series | None) -> tuple:
    bars = bars or {}
    return tuple(
        hash(tuple(bars[k])) if bars.get(k) is not None else None for k in ("volume", "high", "low", "news")
    ) + (hash(tuple(market)) if market is not None else None,)


def probabilities(
    closes: list[float],
    train_window: int,
    *,
    horizon: int = 1,
    model_type: int = 0,
    bars: dict[str, list[Any]] | None = None,
    market: Series | None = None,
) -> dict[str, Any]:
    """Out-of-sample probability of a rise for every day that has a model trained strictly on its past."""
    horizon = max(1, int(horizon))
    key = (len(closes), hash(tuple(closes)), int(train_window), horizon, int(model_type), _extra_key(bars, market))
    if key in _cache:
        return _cache[key]
    rows, names = feature_table(closes, bars, market)
    n = len(closes)
    labels = [int(closes[i + horizon] > closes[i]) if i + horizon < n else 0 for i in range(n)]
    first = next((i for i, row in enumerate(rows) if row is not None), n)
    probs: Series = [None] * n
    model: dict[str, Any] | None = None
    fits = 0
    every = RETRAIN_EVERY_BOOSTED if model_type == 1 else RETRAIN_EVERY
    last_fit = -every
    for t in range(first, n):
        # Days whose label is known at the close of t: i + horizon <= t.
        lo, hi = max(first, t - train_window), t - horizon
        if hi - lo + 1 < MIN_TRAIN or rows[t] is None:
            continue
        if model is None or t - last_fit >= every:
            known = range(lo, hi + 1)
            train_rows = [rows[i] for i in known]
            train_labels = [labels[i] for i in known]
            model = fit_boosted(train_rows, train_labels) if model_type == 1 else fit(train_rows, train_labels)
            last_fit, fits = t, fits + 1
        probs[t] = predict(model, rows[t])
    result = {"probs": probs, "labels": labels, "model": model, "fits": fits, "features": names, "horizon": horizon}
    if len(_cache) > 16:
        _cache.clear()
    _cache[key] = result
    return result


def _run(
    closes: list[float], params: dict[str, float], bars: Any = None, market: Series | None = None
) -> dict[str, Any]:
    return probabilities(
        closes,
        int(params.get("trainWindow", DEFAULTS["trainWindow"])),
        horizon=int(params.get("horizon", DEFAULTS["horizon"])),
        model_type=int(params.get("modelType", DEFAULTS["modelType"])),
        bars=bars,
        market=market,
    )


def signals(
    closes: list[float],
    params: dict[str, float],
    *,
    allow_short: bool = False,
    bars: dict[str, list[Any]] | None = None,
    market: Series | None = None,
) -> tuple[list[int], dict[str, Series]]:
    threshold = float(params.get("threshold", DEFAULTS["threshold"]))
    probs = _run(closes, params, bars, market)["probs"]
    target = [
        0 if p is None else (1 if p >= threshold else (-1 if allow_short and p <= 1 - threshold else 0)) for p in probs
    ]
    return target, {"probUp": probs}


def _auc(scores: list[float], labels: list[int]) -> float | None:
    """ROC-AUC via the rank-sum (Mann-Whitney) statistic, with ties averaged."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def calibration(scores: list[float], labels: list[int], buckets: int = 10) -> list[dict[str, float]]:
    """Equal-count buckets of predicted probability: average prediction vs how often the price really rose."""
    if len(scores) < buckets * 10:
        buckets = max(1, len(scores) // 20)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = []
    for b in range(buckets):
        chunk = order[b * len(order) // buckets : (b + 1) * len(order) // buckets]
        if chunk:
            out.append(
                {
                    "predicted": sum(scores[i] for i in chunk) / len(chunk),
                    "actual": sum(labels[i] for i in chunk) / len(chunk),
                    "count": len(chunk),
                }
            )
    return out


def threshold_scan(
    closes: list[float], probs: Series, start: int, cost_bps: float
) -> tuple[list[dict[str, float]], float | None]:
    """Net-of-cost return of each buy threshold on the days *before* the evaluation window.

    Choosing the threshold here, rather than on the backtest window, keeps the choice out of sample.
    """
    from backend.strategies import simulate

    first = next((i for i, p in enumerate(probs) if p is not None), None)
    if first is None or start - first < 60:
        return [], None
    scan = []
    for step in range(21):
        threshold = round(0.45 + step * 0.01, 2)
        target = [int(p is not None and p >= threshold) for p in probs]
        run = simulate(closes, target, first, start - 1, cost_bps)
        scan.append({"threshold": threshold, "return": run["equity"][-1] - 1, "trades": len(run["trades"])})
    best = max(scan, key=lambda row: row["return"])
    return scan, best["threshold"]


def report(
    closes: list[float],
    params: dict[str, float],
    start: int = 0,
    *,
    cost_bps: float = 0.0,
    bars: dict[str, list[Any]] | None = None,
    market: Series | None = None,
) -> dict[str, Any]:
    """How good the out-of-sample predictions were over the evaluation window [start, end)."""
    threshold = float(params.get("threshold", DEFAULTS["threshold"]))
    model_type = int(params.get("modelType", DEFAULTS["modelType"]))
    run = _run(closes, params, bars, market)
    probs, labels, horizon = run["probs"], run["labels"], run["horizon"]
    days = [t for t in range(start, len(closes) - horizon) if probs[t] is not None]
    scores = [probs[t] for t in days]
    actual = [labels[t] for t in days]
    hits = sum(int((p >= 0.5) == bool(y)) for p, y in zip(scores, actual, strict=True))
    long_days = [y for p, y in zip(scores, actual, strict=True) if p >= threshold]
    model = run["model"]
    weights = []
    if model and model.get("type") == "boosted":
        total = sum(model["importance"]) or 1.0
        weights = [
            {"feature": label, "weight": g / total}
            for g, (_, label) in zip(model["importance"], run["features"], strict=True)
        ]
    elif model:
        weights = [
            {"feature": label, "weight": model["weights"][j + 1]} for j, (_, label) in enumerate(run["features"])
        ]
    weights.sort(key=lambda item: abs(item["weight"]), reverse=True)
    scan, suggested = threshold_scan(closes, probs, start, cost_bps)
    return {
        "model": MODEL_NAMES.get(model_type, "Logistic regression"),
        "modelType": model_type,
        "weightKind": "importance" if model_type == 1 else "coefficient",
        "features": [label for _, label in run["features"]],
        "horizon": horizon,
        "predictions": len(days),
        "accuracy": hits / len(days) if days else 0.0,
        # Accuracy of always predicting the majority direction: the bar a direction model has to clear.
        "baselineAccuracy": max(sum(actual), len(actual) - sum(actual)) / len(actual) if actual else 0.0,
        "upDays": sum(actual) / len(actual) if actual else 0.0,
        "auc": _auc(scores, actual),
        "precisionWhenLong": sum(long_days) / len(long_days) if long_days else None,
        "daysLong": len(long_days),
        "threshold": threshold,
        "trainWindow": int(params.get("trainWindow", DEFAULTS["trainWindow"])),
        "retrainEvery": RETRAIN_EVERY_BOOSTED if model_type == 1 else RETRAIN_EVERY,
        "refits": run["fits"],
        "latestProbability": next((p for p in reversed(probs) if p is not None), None),
        "featureWeights": weights,
        "calibration": calibration(scores, actual),
        "thresholdScan": scan,
        "suggestedThreshold": suggested,
    }
