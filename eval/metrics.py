"""Metric helpers: oversight detection quality, aggregate statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..models import OversightFlag


@dataclass
class DetectionMetrics:
    """Classification metrics for oversight flags in a single episode."""

    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
        }


def detection_metrics(
    flags: Sequence[OversightFlag],
    total_lies: int,
) -> DetectionMetrics:
    """Compute precision/recall/F1 for oversight flags.

    Args:
        flags: Oversight flags produced during the episode.
        total_lies: Total lies that occurred in the episode (ground truth).
    """
    tp = sum(1 for flag in flags if flag.details.get("is_correct"))
    fp = sum(1 for flag in flags if not flag.details.get("is_correct"))
    fn = max(0, total_lies - tp)
    return DetectionMetrics(true_positive=tp, false_positive=fp, false_negative=fn)


@dataclass
class AggregateStat:
    """Mean, std, and 95% CI for a list of values."""

    mean: float
    std: float
    ci95: float
    n: int

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "ci95": self.ci95, "n": self.n}


def aggregate(values: Iterable[float]) -> AggregateStat:
    values_list: List[float] = [float(v) for v in values]
    n = len(values_list)
    if n == 0:
        return AggregateStat(mean=0.0, std=0.0, ci95=0.0, n=0)
    mean = sum(values_list) / n
    if n == 1:
        return AggregateStat(mean=mean, std=0.0, ci95=0.0, n=1)
    variance = sum((v - mean) ** 2 for v in values_list) / (n - 1)
    std = math.sqrt(variance)
    ci95 = 1.96 * std / math.sqrt(n)
    return AggregateStat(mean=mean, std=std, ci95=ci95, n=n)


def summarise(rows: Sequence[dict], keys: Sequence[str]) -> dict:
    """Aggregate multiple metric dicts across seeds.

    Args:
        rows: Per-seed metric dicts (e.g. from `run_eval.run_episode`).
        keys: Numeric keys to aggregate.
    """
    out: dict = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        out[key] = aggregate(values).to_dict()
    return out
