"""Summarization utilities for Market Radar pipeline."""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import httpx

from .config import SummarizerConfig
from .models import Article

if TYPE_CHECKING:
    from .progress import StageHandle

CATS: Dict[str, float] = {
    "регуляторика/санкции/правовые риски": 1.0,
    "взломы/инциденты/остановки": 0.9,
    "существенные тех. прорывы/SOTA": 0.8,
    "крупные релизы/партнёрства/финансы": 0.7,
    "маркетинг/«шум»": 0.2,
}
CAT_KEYS = {k.lower(): k for k in CATS.keys()}

SYSTEM_PROMPT = (
    "Ты — финансовый аналитик. Суммаризируй новость по-русски и отнеси её к одной из 5 категорий.\n"
    "ОТВЕТЬ РОВНО ДВУМЯ СТРОКАМИ, без пояснений и кода:\n"
    "CATEGORY=<ОДНА категория из списка ниже, БЕЗ изменений формулировки>\n"
    "SUMMARY=<1–3 коротких предложения с выводом/прогнозом>\n\n"
    "Категории:\n"
    "регуляторика/санкции/правовые риски\n"
    "взломы/инциденты/остановки\n"
    "существенные тех. прорывы/SOTA\n"
    "крупные релизы/партнёрства/финансы\n"
    "маркетинг/«шум»"
)

CATEGORY_RE = re.compile(r"^\s*CATEGORY\s*=\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
SUMMARY_RE = re.compile(r"^\s*SUMMARY\s*=\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


class Summarizer:
    """LLM based summarization with graceful fallback."""

    def __init__(self, config: SummarizerConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")

    def _parse_model_output(self, text: str) -> Tuple[str, str]:
        cat_match = CATEGORY_RE.search(text)
        sum_match = SUMMARY_RE.search(text)
        raw_cat = (cat_match.group(1).strip() if cat_match else "").lower()
        summary = (sum_match.group(1).strip() if sum_match else "").strip()

        if raw_cat in CAT_KEYS:
            category = CAT_KEYS[raw_cat]
        else:
            lr = raw_cat
            if any(w in lr for w in ["регулятор", "санкц", "правов"]):
                category = "регуляторика/санкции/правовые риски"
            elif any(w in lr for w in ["взлом", "инцидент", "останов", "outage", "даунтайм"]):
                category = "взломы/инциденты/остановки"
            elif any(w in lr for w in ["sota", "прорыв", "бенчмарк"]):
                category = "существенные тех. прорывы/SOTA"
            elif any(w in lr for w in ["релиз", "партнер", "партн", "финанс", "m&a"]):
                category = "крупные релизы/партнёрства/финансы"
            else:
                category = "маркетинг/«шум»"

        if not summary:
            summary = "Саммари не извлечено: модель вернула нестандартный формат ответа."
        return category, summary

    def _call_llm(self, title: str, content: str) -> Tuple[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Заголовок: {title}\n\nТекст:\n{content or '(пусто)'}"},
            ],
            "temperature": self.config.temperature,
        }

        backoffs = [0, 1.5, 3.0]
        last_err: Optional[Exception] = None
        with httpx.Client(timeout=self.config.timeout) as client:
            for wait in backoffs:
                try:
                    if wait:
                        time.sleep(wait)
                    response = client.post(
                        "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
                    )
                    response.raise_for_status()
                    text = response.json()["choices"][0]["message"]["content"]
                    return self._parse_model_output(text)
                except Exception as exc:  # pragma: no cover - network errors
                    last_err = exc
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def _fallback(self, title: str, content: str) -> Tuple[str, str]:
        if title:
            summary = title
        elif content:
            summary = content[:280]
        else:
            summary = "Нет данных для саммари."
        return "маркетинг/«шум»", summary

    def summarize_article(self, article: Article) -> Tuple[str, float]:
        title = article.title or ""
        content = article.content or ""
        if not self.api_key:
            category, summary = self._fallback(title, content)
        else:
            try:
                category, summary = self._call_llm(title, content)
            except Exception:
                if not self.config.fallback_summary:
                    raise
                category, summary = self._fallback(title, content)
        weight = CATS.get(category, min(CATS.values()))
        article.summary = summary
        article.domain_coef = weight
        return category, weight

    def summarize(
        self,
        articles: Sequence[Article],
        stage: Optional["StageHandle"] = None,
    ) -> None:
        if stage is not None:
            stage.set_total(len(articles))
        for article in articles:
            self.summarize_article(article)
            if stage is not None:
                stage.advance(1)


__all__ = ["Summarizer"]
