"""Machine-learning strategy: logistic regression that predicts whether tomorrow's close is higher.

Everything is time-aware, so there is no look-ahead:

* The features for day t use closes up to day t only.
* The label for day i (did the price rise on day i+1?) is known only at the close of day i+1, so the model
  used on day t is trained on days i <= t-1.
* The model is refitted every RETRAIN_EVERY days on a rolling window of the most recent `trainWindow`
  labelled days, and feature scaling is fitted on that window alone.

The strategy is long while the predicted probability of a rise is at least `threshold`. Written in plain
Python (Newton's method on an 8-feature model), so it needs no numpy or scikit-learn.
"""

from __future__ import annotations

import math
from typing import Any

from backend.strategies import Series, moving_average, rolling_std

FEATURES = [
    ("r1", "1-day return"),
    ("r5", "5-day return"),
    ("r20", "20-day return"),
    ("sma10", "Price vs 10-day average"),
    ("sma50", "Price vs 50-day average"),
    ("rsi14", "RSI (14)"),
    ("vol20", "20-day volatility"),
    ("zscore", "Bollinger z-score (20)"),
]
FEATURE_WARMUP = 51  # the 50-day average and 20-day return volatility need this many closes
MIN_TRAIN = 252  # at least a year of labelled days before the first prediction
RETRAIN_EVERY = 21  # refit roughly monthly
L2 = 1.0  # ridge penalty on the standardised coefficients (not the intercept)
DEFAULTS = {"threshold": 0.52, "trainWindow": 504}

_cache: dict[tuple, dict[str, Any]] = {}


def warmup() -> int:
    return FEATURE_WARMUP + MIN_TRAIN


def features(closes: list[float]) -> list[list[float] | None]:
    from backend.rules import rsi

    n = len(closes)
    sma10, sma20, sma50 = (moving_average(closes, w) for w in (10, 20, 50))
    std20 = rolling_std(closes, 20)
    rsi14 = rsi(closes, 14)
    returns = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, n)]
    vol20: Series = [None] * n
    for i in range(20, n):
        window = returns[i - 19 : i + 1]
        m = sum(window) / 20
        vol20[i] = math.sqrt(sum((r - m) ** 2 for r in window) / 19)

    rows: list[list[float] | None] = []
    for i in range(n):
        if i < FEATURE_WARMUP - 1 or rsi14[i] is None or vol20[i] is None:
            rows.append(None)
            continue
        c = closes[i]
        rows.append(
            [
                returns[i],
                c / closes[i - 5] - 1,
                c / closes[i - 20] - 1,
                c / sma10[i] - 1,
                c / sma50[i] - 1,
                rsi14[i] / 100 - 0.5,
                vol20[i],
                (c - sma20[i]) / std20[i] if std20[i] else 0.0,
            ]
        )
    return rows


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems here are 9x9)."""
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
    return {"weights": w, "means": means, "scales": scales}


def predict(model: dict[str, Any], row: list[float]) -> float:
    w, means, scales = model["weights"], model["means"], model["scales"]
    z = w[0] + sum(w[j + 1] * (row[j] - means[j]) / scales[j] for j in range(len(row)))
    return _sigmoid(z)


def probabilities(closes: list[float], train_window: int) -> dict[str, Any]:
    """Out-of-sample probability of a rise for every day that has a model trained strictly on its past."""
    key = (len(closes), hash(tuple(closes)), int(train_window))
    if key in _cache:
        return _cache[key]
    rows = features(closes)
    n = len(closes)
    labels = [int(closes[i + 1] > closes[i]) if i + 1 < n else 0 for i in range(n)]
    probs: Series = [None] * n
    model: dict[str, Any] | None = None
    fits = 0
    last_fit = -RETRAIN_EVERY
    for t in range(n):
        # Days whose label is known at the close of t: i + 1 <= t.
        known = [i for i in range(max(0, t - train_window), t) if rows[i] is not None]
        if len(known) < MIN_TRAIN or rows[t] is None:
            continue
        if model is None or t - last_fit >= RETRAIN_EVERY:
            model = fit([rows[i] for i in known], [labels[i] for i in known])
            last_fit, fits = t, fits + 1
        probs[t] = predict(model, rows[t])
    result = {"probs": probs, "labels": labels, "model": model, "fits": fits}
    if len(_cache) > 16:
        _cache.clear()
    _cache[key] = result
    return result


def signals(closes: list[float], params: dict[str, float]) -> tuple[list[int], dict[str, Series]]:
    threshold = float(params.get("threshold", DEFAULTS["threshold"]))
    probs = probabilities(closes, int(params.get("trainWindow", DEFAULTS["trainWindow"])))["probs"]
    target = [int(p is not None and p >= threshold) for p in probs]
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


def report(closes: list[float], params: dict[str, float], start: int = 0) -> dict[str, Any]:
    """How good the out-of-sample predictions were over the evaluation window [start, end)."""
    threshold = float(params.get("threshold", DEFAULTS["threshold"]))
    run = probabilities(closes, int(params.get("trainWindow", DEFAULTS["trainWindow"])))
    probs, labels = run["probs"], run["labels"]
    days = [t for t in range(start, len(closes) - 1) if probs[t] is not None]
    scores = [probs[t] for t in days]
    actual = [labels[t] for t in days]
    hits = sum(int((p >= 0.5) == bool(y)) for p, y in zip(scores, actual, strict=True))
    long_days = [y for p, y in zip(scores, actual, strict=True) if p >= threshold]
    model = run["model"]
    weights = []
    if model:
        weights = [{"feature": label, "weight": model["weights"][j + 1]} for j, (_, label) in enumerate(FEATURES)]
        weights.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return {
        "model": "Logistic regression",
        "predictions": len(days),
        "accuracy": hits / len(days) if days else 0.0,
        # Accuracy of always predicting "up": the bar a direction model has to clear.
        "baselineAccuracy": max(sum(actual), len(actual) - sum(actual)) / len(actual) if actual else 0.0,
        "upDays": sum(actual) / len(actual) if actual else 0.0,
        "auc": _auc(scores, actual),
        "precisionWhenLong": sum(long_days) / len(long_days) if long_days else None,
        "daysLong": len(long_days),
        "threshold": threshold,
        "trainWindow": int(params.get("trainWindow", DEFAULTS["trainWindow"])),
        "retrainEvery": RETRAIN_EVERY,
        "refits": run["fits"],
        "latestProbability": next((p for p in reversed(probs) if p is not None), None),
        "featureWeights": weights,
    }
