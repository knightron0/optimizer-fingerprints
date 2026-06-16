from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def cv(self) -> float:
        if abs(self.mean) < 1e-12:
            return 0.0
        return self.std / abs(self.mean)

    def summary(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}.mean": self.mean,
            f"{prefix}.std": self.std,
            f"{prefix}.min": 0.0 if self.count == 0 else self.minimum,
            f"{prefix}.max": 0.0 if self.count == 0 else self.maximum,
            f"{prefix}.count": float(self.count),
        }


@dataclass
class FractionCounter:
    count: int = 0
    hits: int = 0

    def add(self, value: bool | None) -> None:
        if value is None:
            return
        self.count += 1
        self.hits += int(value)

    @property
    def fraction(self) -> float:
        if self.count == 0:
            return 0.0
        return self.hits / self.count


@dataclass
class NamedStats:
    stats: dict[str, RunningStats] = field(default_factory=dict)

    def add(self, name: str, value: float | None) -> None:
        self.stats.setdefault(name, RunningStats()).add(value)

    def mean(self, name: str) -> float:
        return self.stats.get(name, RunningStats()).mean

    def std(self, name: str) -> float:
        return self.stats.get(name, RunningStats()).std

    def cv(self, name: str) -> float:
        return self.stats.get(name, RunningStats()).cv
