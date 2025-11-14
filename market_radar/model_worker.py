"""Single-threaded access to the vLLM model for summaries and reranking."""

from __future__ import annotations

import json
import queue
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import httpx

from .models import NewsReport


@dataclass
class _ChatRequest:
    messages: List[Dict[str, Any]]
    params: Dict[str, Any]
    future: Future[str]


class ModelWorker:
    """Dispatch chat completions to the vLLM server on a dedicated thread."""

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._queue: "queue.Queue[_ChatRequest | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        with httpx.Client(timeout=120) as client:
            while not self._stop_event.is_set():
                item = self._queue.get()
                if item is None:
                    break
                try:
                    response = client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json={
                            "model": self.model_name,
                            "messages": item.messages,
                            **item.params,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    item.future.set_result(content.strip())
                except Exception as exc:  # pragma: no cover - network errors
                    item.future.set_exception(exc)
                finally:
                    self._queue.task_done()

    def _submit(self, messages: List[Dict[str, Any]], **params: Any) -> str:
        fut: Future[str] = Future()
        self._queue.put(_ChatRequest(messages=messages, params=params, future=fut))
        return fut.result()

    def summarize(
        self,
        title: str | None,
        content: str,
        max_tokens: int,
    ) -> str:
        header = title or "Без названия"
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — аналитик финансовых и бизнес-новостей. "
                    "Отвечай только на русском языке. "
                    "Для каждой статьи составляй подробную, информативную сводку: "
                    "кто участники, что произошло, когда, где, какие суммы и масштабы, "
                    "как связаны события с рынками, компаниями и отраслями, "
                    "каковы ключевые причины и возможные последствия. "
                    "Можно использовать несколько абзацев или маркированный список. "
                    "Ответ должен содержать только текст сводки: "
                    "без служебных пометок, объяснений твоих действий, тегов или рассуждений."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Сформируй подробную сводку по следующей статье.\n"
                    f"Заголовок: {header}\n\n"
                    f"Текст статьи:\n{content}\n\n"
                    "Верни только сводку, без каких-либо дополнительных комментариев."
                ),
            },
        ]
        response = self._submit(messages, max_tokens=max_tokens, temperature=0.1)
        return self._parse_summary(response)

    def rerank(
        self,
        topics: Sequence[str],
        reports: Iterable[NewsReport],
        max_tokens: int,
    ) -> Dict[str, float]:
        topic_list = ", ".join(topic.strip() for topic in topics if topic.strip())
        if not topic_list:
            raise ValueError("At least one non-empty topic must be provided")

        items = list(reports)
        formatted: List[str] = []
        for idx, report in enumerate(items, start=1):
            formatted.append(
                f"[{idx}] Title: {report.title or 'Untitled'}\nSummary: {report.summary}"
            )
        payload = "\n\n".join(formatted)

        instructions = (
            "You are ranking financial and business news articles by relevance to the given topics. "
            "Each article is labeled with a 1-based index in square brackets, like [1], [2], etc. "
            "Your task is to order ALL article indices from most relevant to least relevant. "
            "Respond ONLY with the indices of the articles in the desired order, using their numeric form "
            "(for example: '2 1 3 4' or '2, 1, 3, 4'). "
            "Do not include any additional text, explanations, reasoning, tags or comments in your answer."
        )
        user = (
            f"Topics: {topic_list}\n\n"
            "Articles:\n"
            f"{payload}\n\n"
            "Return only the ordered list of article indices."
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ]
        response = self._submit(messages, max_tokens=max_tokens, temperature=0.1)
        return self._parse_scores(response, len(items))

    def financial_advice(self, payload_json: str, max_tokens: int) -> str:
        """Generate revised spending recommendations from the LLM."""

        system_message = (
            "You are a financial-advice assistant.\n"
            "You receive JSON data describing a user’s last-month earnings, "
            "last-month expenses, and a natural-language request describing "
            "how the user wants to adjust their spending next month.\n"
            "Your task is to produce a JSON object with the exact same "
            "structure as the input, but with the “wastes” fields replaced "
            "with recommended spending amounts for next month.\n"
            "Do not add new fields. Do not modify the structure. Only adjust "
            "numeric values in “wastes”.\n"
            "Base your recommendations strictly on the provided earnings, "
            "expenses, and the user’s wishes.\n"
            "Output only valid JSON."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": payload_json},
        ]
        return self._submit(messages, max_tokens=max_tokens, temperature=0.2)

    @staticmethod
    def _parse_scores(text: str, count: int) -> Dict[str, float]:
        """
        Parse a ranking from the model response.

        The model is instructed to return only article indices (1..count) in the
        order of relevance. We extract numbers from the response, filter them by
        range, deduplicate while preserving order, and map their ranks to
        pseudo-scores in the (0, 1] interval so the caller can keep using the
        ``Dict[str, float]`` interface where higher scores mean higher
        relevance.
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Model response is empty")

        # Попробуем на всякий случай выдернуть, если кто-то вдруг обернул в JSON
        # (но мы больше этого не требуем).
        # Если это валидный JSON с полем scores, сохраним обратную совместимость.
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and "scores" in data:
            scores_raw = data.get("scores")
            scores: Dict[str, float] = {}
            if isinstance(scores_raw, list):
                for entry in scores_raw:
                    if not isinstance(entry, dict):
                        continue
                    index = entry.get("index")
                    score = entry.get("score")
                    if (
                        isinstance(index, int)
                        and isinstance(score, (int, float))
                        and 1 <= index <= count
                    ):
                        scores[str(index)] = float(score)
            if scores:
                return scores

        # Основной путь: текст содержит только индексы (например: "2 1 3 4" или "2,1,3,4").
        indices: List[int] = []
        for match in re.finditer(r"\d+", cleaned):
            idx = int(match.group(0))
            if 1 <= idx <= count and idx not in indices:
                indices.append(idx)

        if not indices:
            raise ValueError("Model response did not contain any valid indices")

        # Преобразуем ранги в псевдо-скоры (монотонно убывающие).
        # Первый (самый релевантный) получает 1.0, последний — 1/n.
        n = len(indices)
        scores: Dict[str, float] = {}
        for rank, idx in enumerate(indices):
            score = float(n - rank) / float(n)
            scores[str(idx)] = score

        return scores

    @staticmethod
    def _parse_summary(text: str) -> str:
        """Normalize the model response and return the summary text."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Model response missing summary content")

        # Поддержка старого формата <summary>...</summary>, если он вдруг встретится.
        summary_match = re.search(
            r"<summary>(.*?)</summary>",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            cleaned = summary_match.group(1).strip()

        # Нормализуем пробелы и переносы строк:
        # - несколько пробелов подряд -> один пробел
        # - лишние пробелы вокруг переводов строк убираем
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned).strip()

        if not cleaned:
            raise ValueError("Model response missing summary content")

        return cleaned


__all__ = ["ModelWorker"]
