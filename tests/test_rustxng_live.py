"""実際の RustXNG を呼ぶ結合テスト。既定では動かさない（環境変数 JEVSEARCH_RUSTXNG_URL が要る）。"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx
import pytest

from jevsearch.models import NormalizedQuery, SearchVerdict
from jevsearch.rustxng import RustxngAdapter


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@pytest.mark.integration
def test_実際の_RustXNG_に問い合わせて失敗でない():
    base_url = os.environ.get("JEVSEARCH_RUSTXNG_URL")
    if not base_url:
        pytest.skip("JEVSEARCH_RUSTXNG_URL が無い")

    with httpx.Client(timeout=30) as client:
        adapter = RustxngAdapter(
            base_url=base_url,
            rustxng_version="live",
            scrape_engines=(),
            allow_scrape=False,
            client=client,
            clock=_Clock(),
        )
        query = NormalizedQuery(original="rust", ja="rust", en=None, dictionary_version="live")
        response = adapter.search(query, ["wikipedia"], 5)

    assert response.verdict != SearchVerdict.FAILED
