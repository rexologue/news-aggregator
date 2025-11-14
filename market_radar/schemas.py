"""Pydantic schemas for request and response payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field, validator

from .models import NewsReport


class RankingRequest(BaseModel):
    topics: List[str] = Field(..., description="List of topics to rank news against")
    top_n: int = Field(..., gt=0, description="Number of top articles to return")

    @validator("topics")
    def validate_topics(cls, value: List[str]) -> List[str]:
        cleaned = [topic.strip() for topic in value if topic and topic.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty topic is required")
        return cleaned


class ReportResponse(BaseModel):
    agency: str
    title: str | None
    summary: str
    image_base64: str | None
    url: str
    published_at: datetime
    crawled_at: datetime

    @classmethod
    def from_report(cls, report: NewsReport) -> "ReportResponse":
        return cls(
            agency=report.agency,
            title=report.title,
            summary=report.summary,
            image_base64=report.image_base64,
            url=report.url,
            published_at=report.published_at,
            crawled_at=report.crawled_at,
        )


class RebuildResponse(BaseModel):
    rebuilt: int = Field(
        ..., description="Total number of reports available after a rebuild operation"
    )


class AdvicePayload(BaseModel):
    earnings: float = Field(..., description="Total earnings for the previous month")
    wastes: Dict[str, float] = Field(
        ..., description="Expense categories mapped to previous spending amounts"
    )
    wishes: str = Field(..., description="User request for adjusting next month's spending")

    @validator("wastes")
    def validate_wastes(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("wastes must be an object with numeric values")
        validated: Dict[str, float] = {}
        for key, amount in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("wastes keys must be non-empty strings")
            if not isinstance(amount, (int, float)):
                raise ValueError("wastes values must be numeric")
            validated[key] = float(amount)
        return validated


__all__ = ["RankingRequest", "ReportResponse", "RebuildResponse", "AdvicePayload"]
