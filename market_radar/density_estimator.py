"""Density estimation for Market Radar news articles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .config import DensityConfig
from .models import Article

if TYPE_CHECKING:
    from .progress import StageHandle


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    import re

    strip_tags = re.compile(r"<[^>]+>")
    ws_re = re.compile(r"\s+")
    text = strip_tags.sub(" ", text)
    text = ws_re.sub(" ", text).strip()
    return text


def lead(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    text = clean_text(text)
    if not text:
        return ""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    head = " ".join(parts[:2])
    if len(head) > max_chars:
        head = head[:max_chars]
    return head


def has_cuda() -> bool:
    try:
        import torch  # noqa: F401

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch optional
        return False


def get_model(model_id: str, model_cache_dir: Optional[str]):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if has_cuda() else "cpu"
    if model_cache_dir:
        from pathlib import Path

        cache_path = Path(model_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        local_files = {"config.json", "modules.json", "model.safetensors", "pytorch_model.bin"}
        if any((cache_path / f).exists() for f in local_files) or any(cache_path.glob("**/config.json")):
            return SentenceTransformer(str(cache_path), device=device)
        return SentenceTransformer(model_id, cache_folder=str(cache_path), device=device)
    return SentenceTransformer(model_id, device=device)


def encode_texts(
    model,
    titles: Sequence[str],
    contents: Sequence[str],
    title_score: float,
    content_score: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    titles_prep = [("passage: " + t) if t else "" for t in titles]
    contents_prep = [("passage: " + c) if c else "" for c in contents]

    e_title = model.encode(
        titles_prep,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    e_content = model.encode(
        contents_prep,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    w_t = np.array([title_score if bool(t) else 0.0 for t in titles], dtype=np.float32)[:, None]
    w_c = np.array([content_score if bool(c) else 0.0 for c in contents], dtype=np.float32)[:, None]
    w_sum = w_t + w_c
    w_sum = np.where(w_sum == 0.0, 1.0, w_sum)

    combined = (w_t * e_title + w_c * e_content) / w_sum
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return combined / norms, e_title


def bucket_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def group_by_window(articles: Sequence[Article], window_hours: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, art in enumerate(articles):
        dt = art.best_timestamp()
        key = bucket_key(dt) if window_hours == 24 else bucket_key(dt)
        groups.setdefault(key, []).append(idx)
    return groups


def compute_window_scores(
    idxs: List[int],
    articles: Sequence[Article],
    embeddings: np.ndarray,
) -> Dict[int, float]:
    if not idxs:
        return {}
    window_emb = embeddings[idxs, :]
    n = window_emb.shape[0]
    if n == 1:
        return {idxs[0]: 0.0}

    sim = np.clip(window_emb @ window_emb.T, -1.0, 1.0)
    dist = 1.0 - sim

    srcs = [articles[i].source_id for i in idxs]
    srcs_arr = np.array(srcs, dtype=object)
    same_src = (srcs_arr[:, None] == srcs_arr[None, :])
    mask = ~same_src
    np.fill_diagonal(mask, False)

    mean_dist = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        row_mask = mask[i]
        vals = dist[i, row_mask]
        if vals.size > 0:
            mean_dist[i] = float(vals.mean())
            valid[i] = True
        else:
            mean_dist[i] = np.nan
            valid[i] = False

    if np.any(valid):
        md_valid = mean_dist[valid]
        lo = float(np.min(md_valid))
        hi = float(np.max(md_valid))
        if hi > lo:
            norm = (mean_dist - lo) / (hi - lo)
        else:
            norm = np.zeros_like(mean_dist)
        value = 1.0 - norm
        value = np.where(np.isfinite(value), value, 0.0)
    else:
        value = np.zeros(n, dtype=np.float32)

    return {idxs[i]: float(value[i]) for i in range(n)}


class DensityEstimator:
    """Estimate density coefficients for articles."""

    def __init__(self, config: DensityConfig) -> None:
        self.config = config
        self._model = None
        self._title_embeddings: Optional[np.ndarray] = None

    def _ensure_model(self):
        if self._model is None:
            cache_dir = str(self.config.model_cache_dir) if self.config.model_cache_dir else None
            self._model = get_model(self.config.model_id, cache_dir)
        return self._model

    def estimate(
        self,
        articles: Sequence[Article],
        stage: Optional["StageHandle"] = None,
    ) -> Dict[int, float]:
        if not articles:
            self._title_embeddings = None
            return {}

        model = self._ensure_model()
        titles = [clean_text(art.title) for art in articles]
        contents = [lead(art.content, self.config.content_chars) for art in articles]

        embeddings, title_embeddings = encode_texts(
            model=model,
            titles=titles,
            contents=contents,
            title_score=self.config.title_score,
            content_score=self.config.content_score,
            batch_size=self.config.batch_size,
        )
        self._title_embeddings = title_embeddings

        groups = group_by_window(articles, self.config.window_hours)
        values: Dict[int, float] = {}
        if stage is not None:
            stage.set_total(len(articles))
            processed = 0
        for _, idxs in groups.items():
            values.update(compute_window_scores(idxs, articles, embeddings))
            if stage is not None:
                processed += len(idxs)
                stage.advance(len(idxs))
        if stage is not None and processed < len(articles):
            stage.advance(len(articles) - processed)
        return values

    def get_title_embeddings(self) -> Optional[np.ndarray]:
        return self._title_embeddings


__all__ = ["DensityEstimator"]
