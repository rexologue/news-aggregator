"""FastAPI application for serving ranked news reports."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .service import NewsAggregator
from .schemas import RankingRequest, ReportResponse, RebuildResponse


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

    return app


__all__ = ["create_app"]
