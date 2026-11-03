"""検索の接続口（RustXNG アダプター）。

127.0.0.1 などで動く RustXNG（SearXNG 互換の JSON を返す）に HTTP で問い合わせ、
応答を URL・タイトル・抜粋・エンジン・公開日の5項目に写して返す。
つながらない検索語だけを1回やり直し、失敗・0件・結果ありを判定する。
API キーは扱わない（RustXNG は認証なし）。
"""

from __future__ import annotations

import datetime
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import httpx

from jevsearch.models import (
    EngineFailure,
    Lang,
    NormalizedQuery,
    SearchResponse,
    SearchResult,
    SearchVerdict,
)
from jevsearch.ports import Clock
from jevsearch.search_request import build_search_request


@dataclass
class _Merged:
    """URL でまとめている途中の1件。"""

    url: str
    title: str
    snippet: str
    engine: str
    published_date: str | None
    retrieved_at: datetime.datetime
    engines: list[str]
    matched: list[Lang]


def _text(value: object) -> str:
    """文字列でない値（None など）は空文字にする。"""
    return value if isinstance(value, str) else ""


class RustxngAdapter:
    """`ports.SearchBackend` を満たす RustXNG 用の実装。"""

    def __init__(
        self,
        base_url: str,
        rustxng_version: str,
        scrape_engines: Collection[str],
        allow_scrape: bool,
        client: httpx.Client,
        clock: Clock,
    ) -> None:
        self._base_url = base_url
        self._version = rustxng_version
        self._scrape_engines = frozenset(scrape_engines)
        self._allow_scrape = allow_scrape
        self._client = client
        self._clock = clock

    def _fetch(self, url: str, params: dict[str, str]) -> dict | None:
        """GET して JSON の辞書を返す。接続エラー・HTTP エラー・JSON でない応答は None。"""
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError:
            # TransportError のほか DecodingError（壊れた gzip など）も、つながらなかった扱いにする
            return None
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return None
        return data

    def search(
        self, query: NormalizedQuery, engines: Sequence[str], max_results: int
    ) -> SearchResponse:
        # 送るときにカンマでつなぐので、要素もカンマで分けて名前だけにする
        names = list(
            dict.fromkeys(
                name.strip() for item in engines for name in item.split(",") if name.strip()
            )
        )
        if not self._allow_scrape:
            names = [e for e in names if e not in self._scrape_engines]
        engines = names

        queries: list[tuple[Lang, str]] = [(Lang.JA, query.ja)]
        if query.en:
            queries.append((Lang.EN, query.en))

        if not engines:
            # 空の engines= だと RustXNG が既定のエンジン（画面読み取りを含む）を呼ぶので、送らない
            return SearchResponse(
                verdict=SearchVerdict.FAILED,
                results=(),
                failed_engines=(),
                unreachable_queries=tuple(lang for lang, _ in queries),
                retried_queries=(),
                rustxng_version=self._version,
            )

        merged: dict[str, _Merged] = {}
        order: dict[Lang, list[str]] = {}
        failures: list[EngineFailure] = []
        retried: list[Lang] = []
        unreachable: list[Lang] = []

        for lang, text in queries:
            url, params = build_search_request(self._base_url, text, lang, engines, max_results)
            data = self._fetch(url, params)
            if data is None:
                retried.append(lang)
                data = self._fetch(url, params)
            if data is None:
                unreachable.append(lang)
                continue
            retrieved_at = self._clock.now()
            order[lang] = self._collect_results(data, lang, retrieved_at, merged)
            self._collect_failures(data, failures)

        # 検索語の順（ja → en）に1件ずつ交互に取る。日本語版が上限まで埋まっても英語版の結果が残る
        picked: list[str] = []
        taken: set[str] = set()
        longest = max((len(urls) for urls in order.values()), default=0)
        for index in range(longest):
            for lang, _ in queries:
                urls = order.get(lang, [])
                if index >= len(urls) or urls[index] in taken:
                    continue
                if len(picked) >= max_results:
                    break
                taken.add(urls[index])
                picked.append(urls[index])

        results = tuple(
            SearchResult(
                url=m.url,
                title=m.title,
                snippet=m.snippet,
                engine=m.engine,
                published_date=m.published_date,
                retrieved_at=m.retrieved_at,
                engines=tuple(m.engines),
                matched=tuple(m.matched),
            )
            for m in (merged[url] for url in picked)
        )

        if results:
            verdict = SearchVerdict.HAS_RESULTS
        elif unreachable:
            verdict = SearchVerdict.FAILED
        else:
            verdict = SearchVerdict.NO_RESULTS

        return SearchResponse(
            verdict=verdict,
            results=results,
            failed_engines=tuple(failures),
            unreachable_queries=tuple(unreachable),
            retried_queries=tuple(retried),
            rustxng_version=self._version,
        )

    @staticmethod
    def _collect_results(
        data: dict,
        lang: Lang,
        retrieved_at: datetime.datetime,
        merged: dict[str, _Merged],
    ) -> list[str]:
        """results[] の5項目だけを読み、URL（前後の空白を落とした文字列）でまとめる。

        この検索語で出てきた URL を、出てきた順（重複なし）で返す。
        """
        seen: list[str] = []
        items = data.get("results")
        if not isinstance(items, list):
            return seen
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _text(item.get("url")).strip()
            if not url:
                continue
            if url not in seen:
                seen.append(url)
            engine = _text(item.get("engine"))
            entry = merged.get(url)
            if entry is None:
                published = item.get("publishedDate")
                entry = _Merged(
                    url=url,
                    title=_text(item.get("title")),
                    snippet=_text(item.get("content")),
                    engine=engine,
                    published_date=published if isinstance(published, str) and published else None,
                    retrieved_at=retrieved_at,
                    engines=[],
                    matched=[],
                )
                merged[url] = entry
            if engine and engine not in entry.engines:
                entry.engines.append(engine)
            if lang not in entry.matched:
                entry.matched.append(lang)
        return seen

    @staticmethod
    def _collect_failures(data: dict, failures: list[EngineFailure]) -> None:
        """unresponsive_engines（[[名前, 理由], ...]）を写す。同じ組は1つにまとめる。"""
        items = data.get("unresponsive_engines")
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, list | tuple) or not item:
                continue
            failure = EngineFailure(
                engine=_text(item[0]), reason=_text(item[1]) if len(item) > 1 else ""
            )
            if failure not in failures:
                failures.append(failure)
