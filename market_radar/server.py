"""FastAPI application for serving ranked news reports."""

from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .service import NewsAggregator
from .schemas import AdvicePayload, RankingRequest, ReportResponse, RebuildResponse


def create_app(aggregator: NewsAggregator) -> FastAPI:
    app = FastAPI(title="News Aggregator", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_aggregator() -> NewsAggregator:
        return aggregator

    @app.post("/news", response_model=list[ReportResponse])
    def fetch_news(
        request: RankingRequest,
        service: NewsAggregator = Depends(get_aggregator),
    ) -> list[ReportResponse]:
        if request.top_n <= 0:
            raise HTTPException(status_code=400, detail="top_n must be positive")
        if not request.topics:
            raise HTTPException(status_code=400, detail="topics list must not be empty")
        reports = service.top_reports(request.topics, request.top_n)
        return [ReportResponse.from_report(report) for report in reports]

    @app.post("/reports/rebuild", response_model=RebuildResponse)
    def rebuild_reports(service: NewsAggregator = Depends(get_aggregator)) -> RebuildResponse:
        rebuilt = service.rebuild_reports()
        return RebuildResponse(rebuilt=rebuilt)

    @app.post("/advice", response_model=AdvicePayload)
    async def generate_advice(
        request: Request, service: NewsAggregator = Depends(get_aggregator)
    ) -> AdvicePayload:
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8")
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Request body must not be empty")
        try:
            payload_dict = json.loads(raw_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid input
            raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
        try:
            AdvicePayload.parse_obj(payload_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Invalid advice payload") from exc

        try:
            model_response = service.generate_financial_advice(raw_text)
        except Exception as exc:  # pragma: no cover - LLM/network failures
            raise HTTPException(status_code=502, detail="Failed to generate advice") from exc

        try:
            advice_json = json.loads(model_response)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Model returned invalid JSON") from exc

        try:
            validated_output = AdvicePayload.parse_obj(advice_json)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="Model returned malformed payload") from exc

        return validated_output

    return app


__all__ = ["create_app"]
