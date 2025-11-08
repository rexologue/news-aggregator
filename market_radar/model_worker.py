"""Single-threaded access to the vLLM model for summaries and reranking."""

from __future__ import annotations

import json
import queue
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
                    "Ты — помощник, который составляет краткие сводки новостей. "
                    "Отвечай только на русском языке. "
                    "Всегда возвращай результат строго в формате JSON с ключом \"summary\", "
                    "содержащим список предложений. "
                    "Каждое предложение должно быть информативным и отражать ключевые факты. "
                    "Всего предложений должно быть не более пяти. Не добавляй других полей."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Сформируй сводку по следующей статье.\n"
                    f"Заголовок: {header}\n\n"
                    f"Текст статьи:\n{content}\n\n"
                    "Верни только JSON."
                ),
            },
        ]
        response = self._submit(messages, max_tokens=max_tokens, temperature=0.2)
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
            "You are ranking financial and business news. Given the list of topics and candidate "
            "articles, assign each article a relevance score between 0 and 1. Respond only with "
            "JSON containing an array named 'scores'. Each item must include the keys 'index' "
            "(int) and 'score' (float)."
        )
        user = (
            f"Topics: {topic_list}\n\n"
            "Articles:\n"
            f"{payload}\n\n"
            "Return the JSON now."
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ]
        response = self._submit(messages, max_tokens=max_tokens, temperature=0.1)
        return self._parse_scores(response, len(items))

    @staticmethod
    def _parse_scores(text: str, count: int) -> Dict[str, float]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model response did not contain valid JSON")
            data = json.loads(text[start : end + 1])
        scores_raw = data.get("scores")
        if not isinstance(scores_raw, list):
            raise ValueError("Model response missing 'scores' list")
        scores: Dict[str, float] = {}
        for entry in scores_raw:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            score = entry.get("score")
            if not isinstance(index, int):
                continue
            if not isinstance(score, (int, float)):
                continue
            if 1 <= index <= count:
                scores[str(index)] = float(score)
        if not scores:
            raise ValueError("No valid scores produced by the model")
        return scores

    @staticmethod
    def _parse_summary(text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model response did not contain valid JSON summary")
            data = json.loads(text[start : end + 1])
        summary_raw = data.get("summary")
        if isinstance(summary_raw, list):
            sentences = [
                sentence.strip()
                for sentence in summary_raw
                if isinstance(sentence, str) and sentence.strip()
            ]
        elif isinstance(summary_raw, str):
            sentences = [summary_raw.strip()] if summary_raw.strip() else []
        else:
            sentences = []
        if not sentences:
            raise ValueError("Model response missing summary content")
        return "\n".join(sentences)


__all__ = ["ModelWorker"]
