"""差し替え口（Protocol だけ。実装は持たない）。"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from typing import Protocol

from jevsearch.models import (
    ClassifiedResult,
    JevReport,
    NormalizedQuery,
    ResultType,
    SearchResponse,
    SearchResult,
)


class SearchBackend(Protocol):
    def search(
        self, query: NormalizedQuery, engines: Sequence[str], max_results: int
    ) -> SearchResponse: ...


class Classifier(Protocol):
    def classify(
        self,
        query: str,
        results: Sequence[SearchResult],
        rule_types: Mapping[int, ResultType],
    ) -> tuple[list[ClassifiedResult], JevReport]:
        """rule_types のキーは results の位置。"""
        ...


class MemoryClient(Protocol):
    def search(
        self, query: str, *, scope: str, tags: Sequence[str], limit: int
    ) -> list[dict]: ...

    def write(
        self,
        *,
        title: str,
        content: str,
        scope: str,
        kind: str,
        provenance: str,
        source_agent: str,
        tags: Sequence[str],
        supersedes: str | None,
    ) -> str:
        """新しいメモリの id を返す。"""
        ...


class Clock(Protocol):
    def now(self) -> datetime.datetime:
        """tz 付きの現在時刻。"""
        ...
