"""Hotness computation for Market Radar articles."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, TYPE_CHECKING

from .config import HotnessConfig
from .models import Article

if TYPE_CHECKING:
    from .progress import StageHandle

class HotnessCalculator:
    """Calculate time, density and domain weighted hotness."""

    def __init__(self, config: HotnessConfig, start_time: datetime, window: timedelta) -> None:
        self.config = config
        self.start_time = start_time.astimezone(timezone.utc)
        self.window = window
        self.cutoff = self.start_time - window

    def time_coef(self, article: Article) -> float:
        ts = article.best_timestamp().astimezone(timezone.utc)
        if ts <= self.cutoff:
            return 0.0
        if ts >= self.start_time:
            return 1.0
        elapsed = (self.start_time - ts).total_seconds()
        window_seconds = self.window.total_seconds()
        if window_seconds <= 0:
            return 1.0
        ratio = min(max(elapsed / window_seconds, 0.0), 1.0)
        tail = math.exp(-self.config.time_decay)
        value = math.exp(-self.config.time_decay * ratio)
        if value <= tail:
            return 0.0
        return (value - tail) / (1.0 - tail)

    def apply(
        self,
        articles: Sequence[Article],
        stage: Optional["StageHandle"] = None,
    ) -> None:
        weights = self.config.weights
        scores = []
        if stage is not None:
            stage.set_total(len(articles))
        for art in articles:
            art.time_coef = self.time_coef(art)
            density = art.density_coef or 0.0
            domain = art.domain_coef or 0.0
            time_component = art.time_coef or 0.0
            score = (
                weights.time * time_component
                + weights.density * density
                + weights.domain * domain
            )
            art.hotness = score
            scores.append(score)
            if stage is not None:
                stage.advance(1)

        if not scores:
            return

        lo = min(scores)
        hi = max(scores)
        for art in articles:
            score = art.hotness or 0.0
            if hi > lo:
                art.hotness = (score - lo) / (hi - lo)
            else:
                art.hotness = 1.0 if score > 0 else 0.0


__all__ = ["HotnessCalculator"]
