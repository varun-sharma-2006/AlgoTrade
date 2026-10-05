"""Turn a strategy described in plain English into Strategy Builder rules.

Gemini (structured JSON output) does the translation when an API key is configured; its answer is validated
against the same schema as hand-built rules. Without Gemini, a pattern parser understands the common phrasings:
RSI thresholds, price vs moving averages, moving-average crossovers (including golden/death cross), MACD,
N-day breakouts, momentum, volume, stop-loss, take-profit, trailing stop, time exits and short selling.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from backend.schemas import StrategyRules

KINDS = [
    "price", "sma", "ema", "rsi", "macd", "macd_signal", "macd_hist", "atr",
    "volume", "volume_sma", "highest", "lowest", "roc", "value",
]  # fmt: skip
_OPERAND = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "period": {"type": "integer", "nullable": True},
        "value": {"type": "number", "nullable": True},
    },
    "required": ["kind"],
}
_CONDITION = {
    "type": "object",
    "properties": {
        "left": _OPERAND,
        "op": {"type": "string", "enum": [">", "<", "crosses_above", "crosses_below"]},
        "right": _OPERAND,
    },
    "required": ["left", "op", "right"],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "entry": {"type": "array", "items": _CONDITION},
        "exit": {"type": "array", "items": _CONDITION},
        "entryMode": {"type": "string", "enum": ["all", "any"]},
        "side": {"type": "string", "enum": ["long", "short"]},
        "stopLoss": {"type": "number", "nullable": True},
        "takeProfit": {"type": "number", "nullable": True},
        "trailingStop": {"type": "number", "nullable": True},
        "maxHoldDays": {"type": "integer", "nullable": True},
        "notes": {"type": "string"},
    },
    "required": ["entry", "exit"],
}
SYSTEM = (
    "Convert the user's trading strategy into JSON rules for a daily-bar backtester. Operands: price; sma, ema, "
    "rsi, atr, roc (percent rate of change), volume_sma, highest and lowest (highest/lowest close of the previous "
    "N days, for breakouts) all need a 'period' in days; macd, macd_signal, macd_hist (12/26/9) and volume take "
    "no period; value is a fixed number in 'value'. Ops: '>', '<', 'crosses_above', 'crosses_below'. 'entry' "
    "conditions open a position (entryMode 'all' = AND, 'any' = OR); 'exit' conditions (any one) close it. "
    "stopLoss, takeProfit and trailingStop are fractions (8% = 0.08); maxHoldDays is a number of trading days. "
    "Use side 'short' only if the user wants to sell short. Use at most 5 entry and 5 exit conditions. Put "
    "anything you could not represent in 'notes'."
)

_NUM = r"(\d+(?:\.\d+)?)"
_ABOVE = {"above", "over", "greater than", ">", "crosses above", "is above", "breaks above", "rises above"}
_CROSS_UP = {"crosses above", "breaks above", "rises above"}
_CROSS_DOWN = {"crosses below", "breaks below", "falls below", "drops below"}
_COMPARE = (
    r"(?:(?:is|goes|moves|gets|stays|closes|trades)\s+)?"
    r"(crosses above|crosses below|breaks above|breaks below|rises above|falls below|drops below|"
    r"is above|is below|above|below|over|under|greater than|less than|>|<)"
)


def _op(word: str) -> str:
    word = word.strip()
    if word in _CROSS_UP:
        return "crosses_above"
    if word in _CROSS_DOWN:
        return "crosses_below"
    return ">" if word in _ABOVE else "<"


def _ma(kind: str | None, period: str) -> dict[str, Any]:
    return {"kind": "ema" if kind and "ema" in kind else "sma", "period": int(period)}


def _value(number: str) -> dict[str, Any]:
    return {"kind": "value", "value": float(number)}


def _conditions(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    conditions: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for clause in re.split(r"\band\b|\bor\b|,|;", text):
        c = clause.strip(" .")
        if not c:
            continue
        found = None
        if "golden cross" in c:
            found = {"left": _ma("sma", "50"), "op": "crosses_above", "right": _ma("sma", "200")}
        elif "death cross" in c:
            found = {"left": _ma("sma", "50"), "op": "crosses_below", "right": _ma("sma", "200")}
        elif m := re.search(rf"rsi\s*\(?(\d+)?\)?\s*{_COMPARE}\s*{_NUM}", c):
            found = {
                "left": {"kind": "rsi", "period": int(m.group(1) or 14)},
                "op": _op(m.group(2)),
                "right": _value(m.group(3)),
            }
        elif m := re.search(r"macd(?: line)?\s*(crosses above|crosses below|above|below)\s*(?:its |the )?signal", c):
            found = {"left": {"kind": "macd"}, "op": _op(m.group(1)), "right": {"kind": "macd_signal"}}
        elif m := re.search(
            r"macd(?: histogram| hist)?\s*(turns positive|turns negative|crosses above|crosses below|above|below|>|<)\s*(?:zero|0)?",
            c,
        ):
            word = {"turns positive": "crosses above", "turns negative": "crosses below"}.get(m.group(1), m.group(1))
            found = {"left": {"kind": "macd_hist"}, "op": _op(word), "right": _value("0")}
        elif m := re.search(r"(?:new|a|the)?\s*(\d+)[- ]?(?:day|session|bar)s?\s*high", c):
            found = {
                "left": {"kind": "price"},
                "op": "crosses_above",
                "right": {"kind": "highest", "period": int(m.group(1))},
            }
        elif m := re.search(r"(?:new|a|the)?\s*(\d+)[- ]?(?:day|session|bar)s?\s*low", c):
            found = {
                "left": {"kind": "price"},
                "op": "crosses_below",
                "right": {"kind": "lowest", "period": int(m.group(1))},
            }
        elif m := re.search(
            rf"(\d+)[- ]?(?:day|d|period)?\s*(sma|ema|ma|moving average|average)?\s*{_COMPARE}\s*(?:the\s*)?(\d+)[- ]?(?:day|d|period)?\s*(sma|ema|ma|moving average|average)?",
            c,
        ):
            kind = m.group(2) or m.group(5)
            found = {"left": _ma(kind, m.group(1)), "op": _op(m.group(3)), "right": _ma(m.group(5) or kind, m.group(4))}
        elif m := re.search(
            rf"(?:price|close|it|stock)\s*(?:closes\s*)?{_COMPARE}\s*(?:the\s*|its\s*)?(\d+)[- ]?(?:day|d|period)?\s*(sma|ema|ma|moving average|average)",
            c,
        ):
            found = {"left": {"kind": "price"}, "op": _op(m.group(1)), "right": _ma(m.group(3), m.group(2))}
        elif m := re.search(rf"(?:momentum|rate of change|roc)\s*\(?(\d+)?\)?\s*{_COMPARE}\s*(-?\d+(?:\.\d+)?)", c):
            found = {
                "left": {"kind": "roc", "period": int(m.group(1) or 20)},
                "op": _op(m.group(2)),
                "right": _value(m.group(3)),
            }
        elif m := re.search(rf"volume\s*{_COMPARE}\s*(?:its\s*|the\s*)?(\d+)[- ]?day\s*average", c):
            found = {
                "left": {"kind": "volume"},
                "op": _op(m.group(1)),
                "right": {"kind": "volume_sma", "period": int(m.group(2))},
            }
        elif m := re.search(rf"(?:price|close|it|stock)\s*{_COMPARE}\s*\$?{_NUM}", c):
            found = {"left": {"kind": "price"}, "op": _op(m.group(1)), "right": _value(m.group(2))}
        if found:
            conditions.append(found)
        elif not re.search(r"stop|profit|target|trail|days?|short|buy$|sell$", c):
            unparsed.append(c)
    return conditions, unparsed


def parse(text: str) -> dict[str, Any]:
    """Pattern-based translation; returns {"rules": dict | None, "unparsed": [...]}."""
    t = " " + text.lower().replace("’", "'") + " "
    rules: dict[str, Any] = {"entry": [], "exit": [], "entryMode": "all", "side": "long"}
    if re.search(r"\b(sell short|go short|short sell|short when|short it|shorting)\b", t):
        rules["side"] = "short"
    if m := re.search(rf"stop[- ]?loss(?:\s*(?:of|at))?\s*{_NUM}\s*%|{_NUM}\s*%\s*stop(?:[- ]?loss)?(?!\s*trail)", t):
        rules["stopLoss"] = float(m.group(1) or m.group(2)) / 100
    if m := re.search(rf"trailing stop(?:\s*(?:of|at))?\s*{_NUM}\s*%|{_NUM}\s*%\s*trailing", t):
        rules["trailingStop"] = float(m.group(1) or m.group(2)) / 100
        if rules.get("stopLoss") == rules["trailingStop"] and not re.search(r"stop[- ]?loss", t):
            rules.pop("stopLoss")
    if m := re.search(rf"(?:take[- ]?profit|profit target|target)(?:\s*(?:of|at))?\s*{_NUM}\s*%", t):
        rules["takeProfit"] = float(m.group(1)) / 100
    if m := re.search(
        r"(?:after|hold(?:ing)?(?: it)? for|for at most|max(?:imum)?(?: of)?)\s*(\d+)\s*(?:trading\s*)?days", t
    ):
        rules["maxHoldDays"] = int(m.group(1))

    split = re.split(
        r"\b(?:then\s+)?(?:sell|exit|cover|close(?: the position| it)?)\s+(?:it\s+)?(?:when|if|once|on)\b",
        t,
        maxsplit=1,
    )
    entry_text = re.sub(r"^\s*(?:buy|enter|go long|go short|sell short|short)\s+(?:when|if|once|on)\b", " ", split[0])
    entry_text = re.sub(r"\b(?:buy|enter|go long|sell short|go short)\s+(?:when|if|once)\b", " ", entry_text)
    if re.search(r"\bor\b", entry_text) and not re.search(r"\band\b", entry_text):
        rules["entryMode"] = "any"
    rules["entry"], unparsed = _conditions(entry_text)
    if len(split) > 1:
        rules["exit"], more = _conditions(split[1])
        unparsed += more
    rules["entry"], rules["exit"] = rules["entry"][:5], rules["exit"][:5]
    return {"rules": rules if rules["entry"] else None, "unparsed": unparsed}


def _validated(raw: dict[str, Any]) -> dict[str, Any] | None:
    cleaned = {k: v for k, v in raw.items() if k != "notes" and v is not None}
    for side in ("entry", "exit"):
        for condition in cleaned.get(side) or []:
            for key in ("left", "right"):
                condition[key] = {k: v for k, v in condition[key].items() if v is not None}
    try:
        return StrategyRules.model_validate(cleaned).model_dump(exclude_none=True)
    except ValidationError:
        return None


def to_rules(text: str) -> dict[str, Any]:
    """{"rules", "source": "gemini" | "parser", "notes"}; rules is None when nothing could be understood."""
    from backend.gemini import ask_json

    answer = ask_json(SYSTEM, text, SCHEMA)
    if answer:
        rules = _validated(answer)
        if rules:
            return {"rules": rules, "source": "gemini", "notes": answer.get("notes") or ""}
    parsed = parse(text)
    rules = _validated(parsed["rules"]) if parsed["rules"] else None
    notes = f"Not understood: {'; '.join(parsed['unparsed'])}" if parsed["unparsed"] else ""
    return {"rules": rules, "source": "parser", "notes": notes}
