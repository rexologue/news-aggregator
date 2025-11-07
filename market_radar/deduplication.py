"""Article deduplication helpers for the Market Radar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .models import Article

if TYPE_CHECKING:
    from .progress import StageHandle


@dataclass
class DeduplicationSettings:
    """Configuration payload used by :class:`Deduplicator`."""

    enabled: bool
    threshold: float


class Deduplicator:
    """Remove near-duplicate articles based on title embeddings."""

    def __init__(self, settings: DeduplicationSettings) -> None:
        self.settings = settings

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi > lo:
            return [(val - lo) / (hi - lo) for val in values]
        return [1.0 if val > 0 else 0.0 for val in values]

    def apply(
        self,
        articles: Sequence[Article],
        density_scores: Dict[int, float],
        title_embeddings: Optional[np.ndarray],
        stage: Optional["StageHandle"] = None,
    ) -> Tuple[List[Article], List[float]]:
        if not articles:
            return [], []

        if not self.settings.enabled or title_embeddings is None:
            if stage is not None:
                stage.set_total(len(articles))
                if len(articles):
                    stage.advance(len(articles))
            ordered = list(range(len(articles)))
            coefs = [density_scores.get(idx, 0.0) for idx in ordered]
            return list(articles), self._normalize(coefs)

        n = len(articles)
        if stage is not None:
            stage.set_total(n)

        keep_mask = np.zeros(n, dtype=bool)
        removed = np.zeros(n, dtype=bool)
        order = sorted(range(n), key=lambda idx: density_scores.get(idx, 0.0), reverse=True)

        for position, idx in enumerate(order):
            if stage is not None:
                stage.advance(1)
            if removed[idx]:
                continue

            keep_mask[idx] = True
            vector = title_embeddings[idx]
            sims = np.clip(title_embeddings @ vector, -1.0, 1.0)
            dup_idxs = np.where(sims >= self.settings.threshold)[0]
            for dup_idx in dup_idxs:
                if dup_idx == idx:
                    continue
                removed[dup_idx] = True

        keep_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        if not keep_indices:
            return [], []

        keep_indices.sort()
        coefs = [density_scores.get(idx, 0.0) for idx in keep_indices]
        normalized = self._normalize(coefs)
        deduped_articles = [articles[idx] for idx in keep_indices]
        return deduped_articles, normalized


__all__ = ["Deduplicator", "DeduplicationSettings"]
