"""RustXNG に送る検索リクエストの組み立て（契約）。

CLI の `--dry-run` と RustXNG のアダプターが同じ関数を使い、送る内容を食い違わせない。
"""

from __future__ import annotations

from collections.abc import Sequence

from jevsearch.models import Lang

SEARCH_PATH = "/search"
"""検索 API のパス。"""


def build_search_request(
    base_url: str,
    query: str,
    lang: Lang,
    engines: Sequence[str],
    max_results: int,
    time_range: str | None = None,
) -> tuple[str, dict[str, str]]:
    """検索 API の URL とクエリパラメータを返す。

    `categories` は入れない（エンジンは `engines` だけで指定する）。
    time_range が None のときは `time_range` も入れない。
    """
    url = base_url.rstrip("/") + SEARCH_PATH
    params = {
        "q": query,
        "format": "json",
        "engines": ",".join(engines),
        "language": lang.value,
        "max_results": str(max_results),
    }
    if time_range is not None:
        params["time_range"] = time_range
    return url, params
