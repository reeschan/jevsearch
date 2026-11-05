"""RustXNG アダプター（seam ①）のテスト。HTTP は httpx.MockTransport で差し替える。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from jevsearch.models import Lang, NormalizedQuery, SearchVerdict
from jevsearch.rustxng import RustxngAdapter

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://127.0.0.1:8888"
NOW = datetime(2026, 11, 3, 9, 0, tzinfo=timezone(timedelta(hours=9)))
JA_Q = "量子コンピュータ 入門"
EN_Q = "quantum computing introduction"
QUERY = NormalizedQuery(original="量子コンピュータ", ja=JA_Q, en=EN_Q, dictionary_version="d1")
SCRAPE = ("google_scholar", "pypi")


class FakeClock:
    def now(self) -> datetime:
        return NOW


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ja_ok() -> httpx.Response:
    return httpx.Response(200, json=load("rustxng_ja.json"))


def en_ok() -> httpx.Response:
    return httpx.Response(200, json=load("rustxng_en.json"))


def empty_ok(unresponsive: list | None = None) -> httpx.Response:
    return httpx.Response(
        200, json={"results": [], "unresponsive_engines": unresponsive or []}
    )


class Server:
    """検索語ごとに、呼ばれた順の応答（Response か例外）を返す偽の RustXNG。"""

    def __init__(self, script: dict[str, list]) -> None:
        self.script = {q: list(items) for q, items in script.items()}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script[request.url.params["q"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def calls(self, q: str) -> int:
        return sum(1 for r in self.requests if r.url.params["q"] == q)


def make(server: Server, *, allow_scrape: bool = False):
    client = httpx.Client(transport=httpx.MockTransport(server))
    adapter = RustxngAdapter(
        base_url=BASE_URL,
        rustxng_version="v0.26.0",
        scrape_engines=SCRAPE,
        allow_scrape=allow_scrape,
        client=client,
        clock=FakeClock(),
    )
    return adapter


def connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("refused")


def test_リクエストは_engines_を持ち_categories_を持たない():
    """検索語ごとに GET /search を1回。engines= で名前を指定し、categories は付けない。"""
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    make(server).search(QUERY, ["wikipedia", "arxiv"], 50)

    assert len(server.requests) == 2
    for request in server.requests:
        assert request.method == "GET"
        assert request.url.path == "/search"
        params = request.url.params
        assert params["engines"] == "wikipedia,arxiv"
        assert params["format"] == "json"
        assert params["max_results"] == "50"
        assert "categories" not in params
    assert [r.url.params["language"] for r in server.requests] == ["ja", "en"]


def test_許可がなければ画面読み取りの名前は_engines_に入らない():
    """never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない"""
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    make(server, allow_scrape=False).search(
        QUERY, ["wikipedia", "google_scholar", "arxiv", "pypi"], 50
    )

    for request in server.requests:
        names = request.url.params["engines"].split(",")
        assert names == ["wikipedia", "arxiv"]


def test_画面読み取りの名前しか残らなければ_リクエストを送らず失敗にする():
    """遷移: やり直しても検索エンジンにつながらず、結果が1件も届かなかった
    never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない
    """
    server = Server({})
    response = make(server, allow_scrape=False).search(QUERY, ["google_scholar", "pypi"], 50)

    assert server.requests == []
    assert response.verdict == SearchVerdict.FAILED
    assert response.results == ()
    assert response.failed_engines == ()
    assert response.unreachable_queries == (Lang.JA, Lang.EN)
    assert response.retried_queries == ()
    assert response.rustxng_version == "v0.26.0"


def test_カンマを含む要素は分けてから画面読み取りの名前を落とす():
    """never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない"""
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    make(server, allow_scrape=False).search(QUERY, ["wikipedia,google_scholar"], 50)

    assert len(server.requests) == 2
    for request in server.requests:
        assert request.url.params["engines"] == "wikipedia"


def test_不正な接続先_URL_も接続エラーと同じく失敗にして例外を漏らさない():
    """httpx.InvalidURL は HTTPError ではない。search() の外へ漏れず、失敗の結末になる。"""
    server = Server({})
    client = httpx.Client(transport=httpx.MockTransport(server))
    adapter = RustxngAdapter(
        base_url="http://[::1",
        rustxng_version="v0.26.0",
        scrape_engines=SCRAPE,
        allow_scrape=False,
        client=client,
        clock=FakeClock(),
    )
    response = adapter.search(QUERY, ["wikipedia"], 50)

    assert server.requests == []
    assert response.verdict == SearchVerdict.FAILED
    assert response.results == ()
    assert response.unreachable_queries == (Lang.JA, Lang.EN)
    assert response.retried_queries == (Lang.JA, Lang.EN)


def test_results_の一覧が無い_200_は壊れた応答としてやり直し_失敗にする():
    """遷移: やり直しても検索エンジンにつながらず、結果が1件も届かなかった"""
    query = NormalizedQuery(original="x", ja=JA_Q, en=None, dictionary_version="d1")
    bad = [httpx.Response(200, json={"error": "x"}), httpx.Response(200, json={"error": "x"})]
    server = Server({JA_Q: bad})
    response = make(server).search(query, ["wikipedia"], 50)

    assert len(server.requests) == 2
    assert response.verdict == SearchVerdict.FAILED
    assert response.retried_queries == (Lang.JA,)
    assert response.unreachable_queries == (Lang.JA,)


def test_許可があれば画面読み取りの名前も_engines_に入る():
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    make(server, allow_scrape=True).search(QUERY, ["wikipedia", "google_scholar"], 50)

    for request in server.requests:
        assert request.url.params["engines"] == "wikipedia,google_scholar"


def test_応答を5項目に写し_余計なキーを無視する():
    """応答の results[] から url・title・content・engine・publishedDate だけを読む。"""
    server = Server({JA_Q: [ja_ok()], EN_Q: [empty_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    first = response.results[0]
    assert first.url == "https://example.org/ja/quantum-intro"
    assert first.title == "量子コンピュータ入門"
    assert first.snippet == "量子ビットの基本を説明する。"
    assert first.engine == "wikipedia"
    assert first.published_date == "2026-05-01T00:00:00"
    assert first.retrieved_at == NOW
    # 抜粋が無ければ ""、公開日が無ければ None
    third = response.results[2]
    assert third.snippet == ""
    assert third.published_date is None
    assert response.results[1].published_date is None
    assert response.rustxng_version == "v0.26.0"
    # 余計なキー（score など）は SearchResult に持ち込まない
    assert set(vars(first)) == {
        "url", "title", "snippet", "engine", "published_date", "retrieved_at", "engines", "matched",
    }


def test_unresponsive_engines_は_failed_engines_に写り_同じ組は1つにまとまる():
    server = Server(
        {
            JA_Q: [ja_ok()],
            EN_Q: [empty_ok([["mwmbl_slow", "timeout"], ["bing", "captcha"]])],
        }
    )
    response = make(server).search(QUERY, ["wikipedia"], 50)

    pairs = [(f.engine, f.reason) for f in response.failed_engines]
    assert pairs == [("mwmbl_slow", "timeout"), ("bing", "captcha")]


def test_url_が空の要素は捨て_前後の空白を落とした_url_でまとめる():
    body = {
        "results": [
            {"url": "", "title": "空", "engine": "a"},
            {"url": " https://example.org/x ", "title": "X", "engine": "a"},
            {"url": "https://example.org/x", "title": "X2", "engine": "b"},
        ]
    }
    server = Server({JA_Q: [httpx.Response(200, json=body)], EN_Q: [empty_ok()]})
    response = make(server).search(QUERY, ["a"], 50)

    assert len(response.results) == 1
    assert response.results[0].url == "https://example.org/x"
    assert response.results[0].engines == ("a", "b")


def test_ja_と_en_で同じ_url_は1件にまとまる():
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    urls = [r.url for r in response.results]
    # ja・en を交互に取る（ja1, en1, ja2=en1 は取り済み, en2, ja3）
    assert urls == [
        "https://example.org/ja/quantum-intro",
        "https://example.org/shared/quantum-paper",
        "https://example.org/en/quantum-docs",
        "https://example.org/ja/quantum-blog",
    ]
    shared = response.results[1]
    assert shared.matched == (Lang.JA, Lang.EN)
    assert shared.engine == "arxiv"
    assert shared.engines == ("arxiv", "crossref")
    assert response.results[0].matched == (Lang.JA,)
    assert response.results[2].matched == (Lang.EN,)


def test_en_が無ければ_ja_だけを1回問い合わせる():
    query = NormalizedQuery(original="x", ja=JA_Q, en=None, dictionary_version="d1")
    server = Server({JA_Q: [ja_ok()]})
    response = make(server).search(query, ["wikipedia"], 50)

    assert len(server.requests) == 1
    assert response.verdict == SearchVerdict.HAS_RESULTS


def test_max_results_件で切る():
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 2)

    assert [r.url for r in response.results] == [
        "https://example.org/ja/quantum-intro",
        "https://example.org/shared/quantum-paper",
    ]


def _urls_ok(urls: list[str]) -> httpx.Response:
    body = {"results": [{"url": u, "title": u, "content": u, "engine": "github"} for u in urls]}
    return httpx.Response(200, json=body)


def test_日本語版が上限まで埋まっても英語版だけの結果は交互に残る():
    """遷移: 結果が1件以上届いた
    never: 一部のエンジンや Jev が失敗したことを理由に、取れた結果を捨てない
    """
    ja = [f"https://{c}.test/" for c in "abc"]
    en = [f"https://{c}.test/" for c in "de"]
    server = Server({JA_Q: [_urls_ok(ja)], EN_Q: [_urls_ok(en)]})
    response = make(server).search(QUERY, ["github"], 3)

    assert [r.url for r in response.results] == [
        "https://a.test/",
        "https://d.test/",
        "https://b.test/",
    ]
    assert [r.matched for r in response.results] == [(Lang.JA,), (Lang.EN,), (Lang.JA,)]


def test_両方で当たった_url_は交互に取っても1件で_両方の言語が_matched_に入る():
    ja = ["https://a.test/", "https://b.test/", "https://c.test/"]
    en = ["https://b.test/", "https://d.test/"]
    server = Server({JA_Q: [_urls_ok(ja)], EN_Q: [_urls_ok(en)]})
    response = make(server).search(QUERY, ["github"], 3)

    # ja1=a, en1=b, ja2=b（取り済み）, en2=d
    assert [r.url for r in response.results] == [
        "https://a.test/",
        "https://b.test/",
        "https://d.test/",
    ]
    assert response.results[1].matched == (Lang.JA, Lang.EN)


def test_ja_だけなら順番も切り方も変わらない():
    query = NormalizedQuery(original="x", ja=JA_Q, en=None, dictionary_version="d1")
    ja = [f"https://{c}.test/" for c in "abcde"]
    server = Server({JA_Q: [_urls_ok(ja)]})
    response = make(server).search(query, ["github"], 3)

    assert [r.url for r in response.results] == [f"https://{c}.test/" for c in "abc"]


def test_DecodingError_も接続エラーと同じく1回やり直し_だめなら_unreachable():
    """遷移: やり直しても検索エンジンにつながらず、結果が1件も届かなかった"""
    query = NormalizedQuery(original="x", ja=JA_Q, en=None, dictionary_version="d1")
    server = Server({JA_Q: [httpx.DecodingError("bad gzip"), httpx.DecodingError("bad gzip")]})
    response = make(server).search(query, ["wikipedia"], 50)

    assert server.calls(JA_Q) == 2
    assert response.verdict == SearchVerdict.FAILED
    assert response.retried_queries == (Lang.JA,)
    assert response.unreachable_queries == (Lang.JA,)


def test_DecodingError_の後に成功すれば結果が残る():
    """遷移: 検索エンジンにつながらず、1回だけやり直す"""
    server = Server({JA_Q: [httpx.DecodingError("bad gzip"), ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert server.calls(JA_Q) == 2
    assert response.retried_queries == (Lang.JA,)
    assert response.unreachable_queries == ()
    assert response.verdict == SearchVerdict.HAS_RESULTS


def test_つながらない検索語だけを1回やり直す():
    """遷移: 検索エンジンにつながらず、1回だけやり直す"""
    server = Server({JA_Q: [connect_error(), ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert server.calls(JA_Q) == 2
    assert server.calls(EN_Q) == 1
    assert response.retried_queries == (Lang.JA,)
    assert response.unreachable_queries == ()
    assert response.verdict == SearchVerdict.HAS_RESULTS


@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(404, text="nope"),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["not", "a", "dict"]),
        httpx.ReadTimeout("slow"),
    ],
    ids=["500", "404", "html", "json-list", "timeout"],
)
def test_HTTP_エラーや_JSON_でない応答や_タイムアウトも1回だけやり直す(first):
    """遷移: 検索エンジンにつながらず、1回だけやり直す"""
    server = Server({JA_Q: [first, ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert server.calls(JA_Q) == 2
    assert response.retried_queries == (Lang.JA,)
    assert len(response.results) == 4


def test_全検索語が2回とも接続エラーなら失敗で結果は空():
    """遷移: やり直しても検索エンジンにつながらず、結果が1件も届かなかった"""
    server = Server(
        {JA_Q: [connect_error(), connect_error()], EN_Q: [connect_error(), connect_error()]}
    )
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert response.verdict == SearchVerdict.FAILED
    assert response.results == ()
    assert response.retried_queries == (Lang.JA, Lang.EN)
    assert response.unreachable_queries == (Lang.JA, Lang.EN)
    assert len(server.requests) == 4


def test_全検索語が200で0件なら_unresponsive_engines_があっても_NO_RESULTS():
    """遷移: どのエンジンからも結果が返ってこなかった
    never: 0件を失敗として記録しない
    """
    server = Server(
        {JA_Q: [empty_ok([["bing", "timeout"]])], EN_Q: [empty_ok()]}
    )
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert response.verdict == SearchVerdict.NO_RESULTS
    assert response.results == ()
    assert response.unreachable_queries == ()
    assert response.retried_queries == ()
    assert [(f.engine, f.reason) for f in response.failed_engines] == [("bing", "timeout")]


def test_結果が1件以上届けば_HAS_RESULTS():
    """遷移: 結果が1件以上届いた"""
    server = Server({JA_Q: [ja_ok()], EN_Q: [en_ok()]})
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert response.verdict == SearchVerdict.HAS_RESULTS
    assert response.unreachable_queries == ()
    assert response.retried_queries == ()


def test_en_だけがやり直しても届かなくても_ja_の結果は全部残る():
    """遷移: 一部の検索語は結果が届き、ほかはやり直してもつながらなかった
    never: 一部のエンジンや Jev が失敗したことを理由に、取れた結果を捨てない
    """
    server = Server(
        {
            JA_Q: [ja_ok()],
            EN_Q: [httpx.Response(500, text="x"), httpx.Response(500, text="x")],
        }
    )
    response = make(server).search(QUERY, ["wikipedia"], 50)

    assert response.verdict == SearchVerdict.HAS_RESULTS
    assert len(response.results) == 3
    assert response.unreachable_queries == (Lang.EN,)
    assert response.retried_queries == (Lang.EN,)
    assert server.calls(EN_Q) == 2


def test_API_キーはどのリクエストの_URL_やヘッダーにも出ない(monkeypatch):
    """never: API キーを、TypeSafe への認証ヘッダー以外に送らない"""
    sentinel = "sk-test-SENTINEL-0000"
    monkeypatch.setenv("TYPESAFE_API_KEY", sentinel)
    server = Server({JA_Q: [connect_error(), ja_ok()], EN_Q: [en_ok()]})
    make(server).search(QUERY, ["wikipedia"], 50)

    assert server.requests
    for request in server.requests:
        assert sentinel not in str(request.url)
        assert all(sentinel not in k and sentinel not in v for k, v in request.headers.items())
        assert "authorization" not in request.headers
