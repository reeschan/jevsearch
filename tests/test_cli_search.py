"""`jevsearch search` の通しのテスト（seam ⑥）。

CLI を `main(argv, deps=...)` で動かす。検索の接続口・Jev の ask・共有メモリは fake に差し替え、
保存先は `tmp_path` の設定ファイルで指す。最後の2つは make_* を既定のまま
`httpx.MockTransport` で通し、キーの行き先を確かめる。
"""

import dataclasses
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from jevsearch import credentials
from jevsearch.cli import Deps, _SystemClock, main
from jevsearch.config import load_config, scrape_engine_names
from jevsearch.jev_classify import JevClassifier
from jevsearch.jev_client import JevError, JevResult
from jevsearch.job import decide_outcome
from jevsearch.models import (
    FINAL_STATE,
    JEV_MODEL,
    EngineFailure,
    JevReport,
    JudgedBy,
    Judgment,
    Lang,
    Outcome,
    SearchResponse,
    SearchResult,
    SearchVerdict,
)
from jevsearch.store import ResultStore

KEY = "sk-test-SENTINEL-0000"
JST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 3, 21, 5, 12, tzinfo=JST)
QUERY = "rust メタサーチ"
QUERY_JA = "rust メタ検索"
QUERY_EN = "rust metasearch"


# --- 差し替え用の部品 ---


class TickClock:
    """呼ぶたびに 10 ミリ秒ずつ進む時計（tz 付き）。"""

    def __init__(self) -> None:
        self._now = T0

    def now(self) -> datetime:
        current = self._now
        self._now = current + timedelta(milliseconds=10)
        return current


def _result(n: int, url: str | None = None) -> SearchResult:
    return SearchResult(
        url=url or f"https://site{n}.example.org/page/{n}",
        title=f"title {n}",
        snippet=f"snippet {n}",
        engine="wikipedia",
        published_date=None,
        retrieved_at=T0,
        engines=("wikipedia",),
        matched=(Lang.JA,),
    )


# 1件目は github.com なので、規則で repo に決まる。
RESULTS = (_result(1, "https://github.com/example/repo"), _result(2), _result(3))


def _response(
    verdict=SearchVerdict.HAS_RESULTS,
    results=RESULTS,
    failed=(),
    unreachable=(),
    retried=(),
) -> SearchResponse:
    return SearchResponse(
        verdict=verdict,
        results=tuple(results),
        failed_engines=tuple(failed),
        unreachable_queries=tuple(unreachable),
        retried_queries=tuple(retried),
        rustxng_version="0.26.0",
    )


class FakeSearch:
    """決まった SearchResponse を返し、受けた呼び出しを記録する。"""

    def __init__(self, response: SearchResponse) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def search(self, query, engines, max_results):
        self.calls.append((query, list(engines), max_results))
        return self.response


def _answers_for(questions: dict) -> dict:
    answers: dict = {}
    for key in questions:
        if key.startswith("type_"):
            answers[key] = {"choice": "blog", "confidence": 0.9}
        elif key.startswith("genre_") or key == "query_genre":
            answers[key] = {"choice": "tech", "confidence": 0.9}
        elif key.startswith("relevant_"):
            answers[key] = {"probabilities": {"1": 0.9, "0": 0.1}}
    return answers


class FakeJev:
    """JevClient.ask の代わり。error があれば常にそれを投げる。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, state: str, questions: dict) -> JevResult:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return JevResult(
            answers=_answers_for(questions),
            usage={"input_tokens": 100, "output_tokens": 20},
            duration_ms=5,
        )


class FakeMemory:
    """共有メモリの代わり。write の呼び出しを記録する。"""

    def __init__(self, write_error: Exception | None = None) -> None:
        self.write_error = write_error
        self.writes: list[dict] = []
        self.searches: list[str] = []

    def search(self, query, *, scope, tags, limit):
        self.searches.append(query)
        return []

    def write(self, **kwargs):
        if self.write_error is not None:
            raise self.write_error
        self.writes.append(kwargs)
        return f"M{len(self.writes)}"


def _fail_factory(name):
    def make(*args):
        raise AssertionError(f"{name} が呼ばれた")

    return make


@pytest.fixture(autouse=True)
def _no_real_credentials(tmp_path_factory, monkeypatch):
    """手元の認証ファイルを読まない（キーは deps.env だけから来る）。"""
    missing = tmp_path_factory.mktemp("nocred") / "credentials.json"
    monkeypatch.setattr(credentials, "default_credentials_path", lambda: missing)


def _write_config(tmp_path: Path, extra: str = "", storage: Path | None = None) -> Path:
    data = storage if storage is not None else tmp_path / "data"
    path = tmp_path / "config.toml"
    path.write_text(f"[storage]\ndir = '{data.as_posix()}'\n{extra}", encoding="utf-8")
    return path


def _deps(search=None, jev=None, memory=None, env=None) -> Deps:
    clock = TickClock()
    search = search if search is not None else FakeSearch(_response())
    jev = jev if jev is not None else FakeJev()
    classifier = JevClassifier(
        jev,
        model=JEV_MODEL,
        batch_size=25,
        max_request_tokens=10**9,
        threshold=0.55,
        clock=clock,
    )
    return Deps(
        env=env if env is not None else {},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        clock=clock,
        make_search_backend=lambda config, allow_scrape: search,
        make_classifier=lambda config: classifier,
        make_memory_client=(lambda config: memory)
        if memory is not None
        else _fail_factory("make_memory_client"),
    )


def _run(tmp_path, deps, *extra, query=QUERY, config=None):
    cfg = config if config is not None else _write_config(tmp_path)
    code = main(["search", query, "--config", str(cfg), *extra], deps=deps)
    return code, deps.stdout.getvalue(), deps.stderr.getvalue()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _jobs(tmp_path) -> list[dict]:
    return _jsonl(tmp_path / "data" / "jobs.jsonl")


def _rows(tmp_path) -> list[dict]:
    return _jsonl(tmp_path / "data" / "results.jsonl")


def _scenario(kind: str):
    """結末ごとの (SearchResponse, Jev の fake)。"""
    engine_failure = (EngineFailure("arxiv", "timeout"),)
    if kind == "saved":
        return _response(), FakeJev()
    if kind == "partial_engine":
        return _response(failed=engine_failure), FakeJev()
    if kind == "partial_jev":
        return _response(failed=engine_failure), FakeJev(JevError("http", "busy", 529))
    if kind == "no_results":
        return _response(SearchVerdict.NO_RESULTS, (), failed=engine_failure), FakeJev()
    if kind == "failed":
        response = _response(
            SearchVerdict.FAILED,
            (),
            unreachable=(Lang.JA, Lang.EN),
            retried=(Lang.JA, Lang.EN),
        )
        return response, FakeJev()
    raise AssertionError(kind)


# --- 結末ごとの記録 ---


def test_accepted_job_record_has_id_query_and_topic(tmp_path):
    """遷移: 検索語を受け取った

    ジョブの記録に空でない job_id、入力のままの query、topic が入る。
    """
    deps = _deps(memory=FakeMemory())
    code, _, err = _run(tmp_path, deps, "--topic", "t1")
    assert code == 0, err
    (job,) = _jobs(tmp_path)
    assert job["job_id"]
    assert job["query"] == QUERY
    assert job["topic"] == "t1"
    assert job["job_id"].count("-") == 1


def test_saved_job_record_states_exit_code_and_row_count(tmp_path):
    """遷移: 検索を始めた
    遷移: 用語辞書で検索語をそろえ終えた
    遷移: 結果が1件以上届いた
    遷移: ドメインで分かる種類を付け終えた
    遷移: すべての検索語とエンジンが答え、Jev もすべて答えた
    """
    jev = FakeJev()
    deps = _deps(jev=jev)
    code, out, err = _run(tmp_path, deps)
    assert code == 0, err
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "saved"
    assert job["states"] == [
        "accepted",
        "normalizing",
        "searching",
        "rule_tagging",
        "jev_classify",
        "saved",
    ]
    assert job["query_norm"] == {"ja": QUERY_JA, "en": QUERY_EN}
    assert job["n_results"] == len(RESULTS) == len(_rows(tmp_path))
    assert job["n_unclassified"] == 0
    assert job["versions"]["jev_model"] == JEV_MODEL
    # Jev へは正規化した日本語の検索語を渡す
    assert json.loads(jev.calls[0][0])["query"] == QUERY_JA
    assert err == ""
    assert out.count("\n") == 1 and "保存済み" in out and "3件" in out


def test_rule_decided_type_is_kept_and_jev_fills_the_rest(tmp_path):
    deps = _deps()
    _run(tmp_path, deps)
    rows = _rows(tmp_path)
    assert rows[0]["type"] == {"value": "repo", "by": "rule", "confidence": 1.0}
    assert rows[1]["type"]["by"] == "jev" and rows[1]["type"]["value"] == "blog"
    assert rows[1]["relevant"]["value"] is True


def test_retry_adds_a_second_searching_state(tmp_path):
    """遷移: 検索エンジンにつながらず、1回だけやり直す"""
    search = FakeSearch(_response(retried=(Lang.JA,)))
    code, _, _ = _run(tmp_path, _deps(search=search))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["states"][:5] == [
        "accepted",
        "normalizing",
        "searching",
        "searching",
        "rule_tagging",
    ]
    assert job["outcome"] == "saved"


def test_failed_exits_1_without_results_or_classification(tmp_path):
    """遷移: やり直しても検索エンジンにつながらず、結果が1件も届かなかった"""
    response, jev = _scenario("failed")
    deps = _deps(search=FakeSearch(response), jev=jev)
    code, out, err = _run(tmp_path, deps)
    assert code == 1
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "failed"
    assert job["states"] == ["accepted", "normalizing", "searching", "searching", "failed"]
    assert job["reasons"]["unreachable_queries"] == ["ja", "en"]
    assert _rows(tmp_path) == []
    assert jev.calls == []
    assert "失敗" in out
    assert err


def test_no_results_is_not_a_failure(tmp_path):
    """遷移: どのエンジンからも結果が返ってこなかった
    never: 0件を失敗として記録しない
    """
    search = FakeSearch(_response(SearchVerdict.NO_RESULTS, ()))
    jev = FakeJev()
    code, out, err = _run(tmp_path, _deps(search=search, jev=jev))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "no_results"
    assert job["outcome"] != "failed"
    assert job["states"] == ["accepted", "normalizing", "searching", "no_results"]
    assert job["n_results"] == 0
    assert _rows(tmp_path) == []
    assert jev.calls == []
    assert "0件" in out
    assert err == ""


def test_no_results_with_failed_engines_warns_on_stderr(tmp_path):
    response, jev = _scenario("no_results")
    code, out, err = _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "no_results"
    assert job["reasons"]["failed_engines"] == [{"engine": "arxiv", "reason": "timeout"}]
    assert "arxiv" in err and "arxiv" in out


def test_partial_when_an_engine_did_not_answer(tmp_path):
    """遷移: 一部のエンジンか検索語が答えなかった"""
    response, jev = _scenario("partial_engine")
    code, out, err = _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "partial"
    assert job["reasons"]["failed_engines"] == [{"engine": "arxiv", "reason": "timeout"}]
    assert len(_rows(tmp_path)) == len(RESULTS)
    assert "警告" in err and "arxiv" in err
    assert "一部だけ保存" in out


def test_partial_when_only_some_queries_reached_the_engines(tmp_path):
    """遷移: 一部の検索語は結果が届き、ほかはやり直してもつながらなかった"""
    search = FakeSearch(_response(unreachable=(Lang.EN,), retried=(Lang.EN,)))
    code, _, err = _run(tmp_path, _deps(search=search))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "partial"
    assert job["reasons"]["unreachable_queries"] == ["en"]
    assert job["states"][2:4] == ["searching", "searching"]
    assert len(_rows(tmp_path)) == len(RESULTS)
    assert "警告" in err


def test_partial_when_jev_fails_keeps_every_result_unclassified(tmp_path):
    """遷移: Jev が使えない、または上限を超えたので未分類として残した
    never: 一部のエンジンや Jev が失敗したことを理由に、取れた結果を捨てない
    """
    response, jev = _scenario("partial_jev")
    code, _, err = _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "partial"
    assert job["states"][-1] == "partially_saved"
    assert job["reasons"]["jev_failure"] == "http 529"
    assert job["reasons"]["failed_engines"]
    rows = _rows(tmp_path)
    assert len(rows) == len(RESULTS) == job["n_results"]
    assert all(row["genre"]["value"] is None for row in rows)
    assert all(row["relevant"]["value"] is None for row in rows)
    # 規則で決まった種類は Jev が失敗しても残る
    assert rows[0]["type"] == {"value": "repo", "by": "rule", "confidence": 1.0}
    assert rows[1]["type"]["value"] is None
    assert job["n_unclassified"] == len(RESULTS)
    assert "警告" in err and "http 529" in err


def test_jev_failure_alone_makes_the_job_partial(tmp_path):
    jev = FakeJev(JevError("daily_limit", "limit"))
    code, _, err = _run(tmp_path, _deps(jev=jev))
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "partial"
    assert job["reasons"]["jev_failure"] == "daily_limit"
    assert job["reasons"]["failed_engines"] == []
    assert len(_rows(tmp_path)) == len(RESULTS)
    assert "daily_limit" in err


def test_every_result_row_has_a_query_and_an_offset_timestamp(tmp_path):
    """never: 検索語と取得日時のない結果を保存しない"""
    _run(tmp_path, _deps())
    rows = _rows(tmp_path)
    assert rows
    for row in rows:
        assert row["query"] == QUERY
        stamp = datetime.fromisoformat(row["retrieved_at"])
        assert stamp.utcoffset() is not None
        assert row["job_id"] == _jobs(tmp_path)[0]["job_id"]


@pytest.mark.parametrize(
    "kind", ["saved", "partial_engine", "partial_jev", "no_results", "failed"]
)
def test_last_state_matches_the_outcome(tmp_path, kind):
    response, jev = _scenario(kind)
    _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    (job,) = _jobs(tmp_path)
    assert job["states"][-1] == FINAL_STATE[Outcome(job["outcome"])].value
    assert job["states"][0] == "accepted"


def test_eval_record_is_written_for_a_classified_job(tmp_path):
    _run(tmp_path, _deps())
    evals = _jsonl(tmp_path / "data" / "eval.jsonl")
    assert len(evals) == 1
    assert evals[0]["job_id"] == _jobs(tmp_path)[0]["job_id"]
    assert evals[0]["n_results"] == len(RESULTS)


@pytest.mark.parametrize("kind", ["no_results", "failed"])
def test_eval_record_is_written_for_jobs_that_never_reached_jev(tmp_path, kind):
    response, jev = _scenario(kind)
    _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    (job,) = _jobs(tmp_path)
    (row,) = _jsonl(tmp_path / "data" / "eval.jsonl")
    assert row["job_id"] == job["job_id"]
    assert row["n_results"] == 0
    assert row["jev_tokens"] == {"input": None, "output": None, "estimated_input": 0}
    assert row["failed_engines"] == job["reasons"]["failed_engines"]
    assert jev.calls == []


def test_eval_record_of_a_no_result_job_lists_the_failed_engine(tmp_path):
    response, jev = _scenario("no_results")
    _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    (row,) = _jsonl(tmp_path / "data" / "eval.jsonl")
    assert row["failed_engines"] == [{"engine": "arxiv", "reason": "timeout"}]


def test_job_record_holds_the_query_genre_judged_by_jev(tmp_path):
    _run(tmp_path, _deps())
    (job,) = _jobs(tmp_path)
    assert job["query_genre"] == {"value": "tech", "by": "jev", "confidence": 0.9}


def test_job_record_query_genre_is_null_without_a_jev_report(tmp_path):
    for kind in ("no_results", "failed"):
        sub = tmp_path / kind
        sub.mkdir()
        response, jev = _scenario(kind)
        _run(sub, _deps(search=FakeSearch(response), jev=jev))
        (job,) = _jobs(sub)
        assert job["query_genre"] is None


def test_job_record_query_genre_is_unclassified_when_jev_fails(tmp_path):
    response, jev = _scenario("partial_jev")
    _run(tmp_path, _deps(search=FakeSearch(response), jev=jev))
    (job,) = _jobs(tmp_path)
    assert job["query_genre"]["value"] is None
    assert job["query_genre"]["by"] == "jev"


# --- decide_outcome ---


def _report(failure):
    return JevReport(query_genre=Judgment(None, JudgedBy.JEV, None), usage=(), failure=failure)


@pytest.mark.parametrize(
    ("response", "report", "expected"),
    [
        (_response(SearchVerdict.FAILED, ()), None, Outcome.FAILED),
        (_response(SearchVerdict.NO_RESULTS, (), failed=(EngineFailure("a", ""),)), None,
         Outcome.NO_RESULTS),
        (_response(), _report(None), Outcome.SAVED),
        (_response(failed=(EngineFailure("a", ""),)), _report(None), Outcome.PARTIAL),
        (_response(unreachable=(Lang.EN,)), _report(None), Outcome.PARTIAL),
        (_response(), _report("http 529"), Outcome.PARTIAL),
    ],
)
def test_decide_outcome(response, report, expected):
    assert decide_outcome(response, report) is expected


# --- キーを出さない ---


def _all_outputs(tmp_path, *streams):
    texts = [s for s in streams]
    for path in tmp_path.rglob("*"):
        if path.is_file():
            texts.append(path.read_bytes().decode("utf-8", errors="replace"))
    return texts


def test_api_key_never_appears_in_files_or_screen(tmp_path):
    """never: API キーを結果ログ・記録・画面の表示に出さない"""
    env = {"TYPESAFE_API_KEY": KEY}
    streams: list[str] = []
    jev_with_key_in_error = FakeJev(RuntimeError(f"boom {KEY}"))
    for kind, jev in (
        ("saved", None),
        ("partial_engine", None),
        ("partial_jev", jev_with_key_in_error),
        ("no_results", None),
        ("failed", None),
    ):
        response, default_jev = _scenario(kind)
        deps = _deps(
            search=FakeSearch(response), jev=jev or default_jev, memory=FakeMemory(), env=env
        )
        _run(tmp_path, deps, "--topic", "t1")
        streams += [deps.stdout.getvalue(), deps.stderr.getvalue()]

    dry = Deps(
        env=env,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        make_search_backend=_fail_factory("make_search_backend"),
        make_classifier=_fail_factory("make_classifier"),
        make_memory_client=_fail_factory("make_memory_client"),
    )
    code = main(["search", QUERY, "--dry-run", "--config", str(_write_config(tmp_path))], deps=dry)
    assert code == 0
    streams += [dry.stdout.getvalue(), dry.stderr.getvalue()]

    assert len(_jobs(tmp_path)) == 5
    for text in _all_outputs(tmp_path, *streams):
        assert KEY not in text


# --- 本物の経路（make_* は既定のまま、通信だけを MockTransport にする） ---

_RUSTXNG_BODY = {
    "results": [
        {
            "url": "https://github.com/example/repo",
            "title": "repo",
            "content": "a repository",
            "engine": "github",
        },
        {"url": "https://a.example.org/1", "title": "one", "content": "c1", "engine": "wikipedia"},
        {"url": "https://b.example.org/2", "title": "two", "content": "c2", "engine": "mwmbl"},
    ],
    "unresponsive_engines": [],
}


def _mcp_reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    method = body["method"]
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18"}
    elif method == "notifications/initialized":
        return httpx.Response(202)
    elif body["params"]["name"] == "search":
        result = {"content": [{"type": "text", "text": "[]"}], "isError": False}
    else:
        result = {"content": [{"type": "text", "text": json.dumps({"id": "MEM1"})}], "isError": False}
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def _transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        host, port = request.url.host, request.url.port
        if host == "127.0.0.1" and port == 29483:
            return httpx.Response(200, json=_RUSTXNG_BODY)
        if host == "api.typesafe.ai":
            questions = json.loads(request.content)["questions"]
            return httpx.Response(
                200,
                json={
                    "answers": _answers_for(questions),
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            )
        if host == "127.0.0.1" and port == 7820:
            return _mcp_reply(request)
        raise AssertionError(f"想定外の宛先: {request.url}")

    return httpx.MockTransport(handler)


def _real_deps(seen, env) -> Deps:
    return Deps(
        env=env,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        clock=TickClock(),
        http_transport=_transport(seen),
    )


def _carries_key(request: httpx.Request, *, skip_authorization: bool) -> bool:
    if KEY in str(request.url) or KEY.encode() in request.content:
        return True
    for name, value in request.headers.items():
        if skip_authorization and name.lower() == "authorization":
            continue
        if KEY in value:
            return True
    return False


def test_real_path_sends_the_key_only_in_the_typesafe_auth_header(tmp_path):
    """never: API キーを、TypeSafe への認証ヘッダー以外に送らない
    never: API キーを結果ログ・記録・画面の表示に出さない
    """
    seen: list[httpx.Request] = []
    deps = _real_deps(seen, {"TYPESAFE_API_KEY": KEY})
    code, out, err = _run(tmp_path, deps, "--topic", "t1")
    assert code == 0, err

    jev_requests = [r for r in seen if r.url.host == "api.typesafe.ai"]
    rustxng_requests = [r for r in seen if r.url.port == 29483]
    memory_requests = [r for r in seen if r.url.port == 7820]
    assert jev_requests and memory_requests
    assert len(rustxng_requests) == 2  # 日本語と英語の検索語
    for request in jev_requests:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert not _carries_key(request, skip_authorization=True)
    for request in rustxng_requests + memory_requests:
        assert not _carries_key(request, skip_authorization=False)

    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "saved"
    assert job["reasons"]["digest_failure"] is None
    rows = _rows(tmp_path)
    assert len(rows) == 3
    assert rows[0]["type"]["by"] == "rule"
    assert rows[1]["type"]["by"] == "jev"
    writes = [
        json.loads(r.content)
        for r in memory_requests
        if json.loads(r.content).get("method") == "tools/call"
        and json.loads(r.content)["params"]["name"] == "write"
    ]
    assert len(writes) == 1
    for text in _all_outputs(tmp_path, out, err):
        assert KEY not in text


def test_real_path_without_a_key_leaves_results_unclassified(tmp_path):
    seen: list[httpx.Request] = []
    deps = _real_deps(seen, {})
    code, _, err = _run(tmp_path, deps)
    assert code == 0, err
    assert not [r for r in seen if r.url.host == "api.typesafe.ai"]
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "partial"
    assert job["reasons"]["jev_failure"] == "no_api_key"
    rows = _rows(tmp_path)
    assert len(rows) == 3
    assert all(row["genre"]["value"] is None for row in rows)
    assert rows[0]["type"]["value"] == "repo"


# --- 想定外の例外 ---


def test_store_failure_is_an_internal_error_with_one_line(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")  # 保存先がファイルなので書けない
    config = _write_config(tmp_path, storage=blocker)
    deps = _deps(env={"TYPESAFE_API_KEY": KEY})
    code, out, err = _run(tmp_path, deps, config=config)
    assert code == 1
    assert err == "内部エラー: StoreError\n"
    assert out == ""


def test_internal_error_message_is_not_printed(tmp_path, monkeypatch):
    def boom(self, row):
        raise RuntimeError(f"secret {KEY}")

    monkeypatch.setattr(ResultStore, "append_job", boom)
    deps = _deps(env={"TYPESAFE_API_KEY": KEY})
    code, _, err = _run(tmp_path, deps)
    assert code == 1
    assert err == "内部エラー: RuntimeError\n"


# --- ダイジェスト ---


def test_digest_is_written_once_then_superseded_on_the_same_day(tmp_path):
    memory = FakeMemory()
    for _ in range(2):
        code, _, err = _run(tmp_path, _deps(memory=memory), "--topic", "t1")
        assert code == 0, err
        if len(memory.writes) == 1:
            assert memory.writes[0]["supersedes"] is None
    assert len(memory.writes) == 2
    assert memory.writes[1]["supersedes"] == "M1"
    assert memory.writes[1]["scope"] == "global"
    assert [job["reasons"]["digest_failure"] for job in _jobs(tmp_path)] == [None, None]


def _results_retrieved_at(*moments: datetime) -> tuple[SearchResult, ...]:
    return tuple(
        dataclasses.replace(_result(n), retrieved_at=moment)
        for n, moment in enumerate(moments, start=1)
    )


def test_digest_day_follows_the_retrieved_at_of_this_jobs_rows(tmp_path):
    """取得が日付をまたいだ結果でも、その行の日付のダイジェストが書かれる。"""
    next_day = T0 + timedelta(days=1)
    memory = FakeMemory()
    search = FakeSearch(_response(results=_results_retrieved_at(next_day, next_day)))
    code, _, err = _run(tmp_path, _deps(search=search, memory=memory), "--topic", "t1")
    assert code == 0, err
    (write,) = memory.writes
    assert next_day.date().isoformat() in write["title"]
    assert T0.date().isoformat() not in write["title"]
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "saved"
    assert job["reasons"]["digest_failure"] is None
    assert not (tmp_path / "data" / "digest-pending.jsonl").exists()


def test_digest_is_written_once_per_retrieved_day(tmp_path):
    """結果が2つの日付にまたがるときは、日ごとに1回ずつ書く。"""
    next_day = T0 + timedelta(days=1)
    memory = FakeMemory()
    search = FakeSearch(_response(results=_results_retrieved_at(T0, next_day, T0)))
    code, _, err = _run(tmp_path, _deps(search=search, memory=memory), "--topic", "t1")
    assert code == 0, err
    titles = [write["title"] for write in memory.writes]
    assert len(titles) == 2
    assert T0.date().isoformat() in titles[0]
    assert next_day.date().isoformat() in titles[1]


def test_digest_failure_on_a_later_day_is_recorded(tmp_path):
    """どれかの日の失敗が reasons.digest_failure に残り、結末は saved のまま。"""
    memory = FakeMemory(write_error=RuntimeError("memory down"))
    next_day = T0 + timedelta(days=1)
    search = FakeSearch(_response(results=_results_retrieved_at(next_day)))
    code, _, _ = _run(tmp_path, _deps(search=search, memory=memory), "--topic", "t1")
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "saved"
    assert job["reasons"]["digest_failure"]


def test_default_clock_returns_local_time_with_its_offset():
    """既定の時計はローカル時刻（オフセット付き）を返す。UTC 固定ではない。"""
    before = datetime.now().astimezone()
    now = _SystemClock().now()
    after = datetime.now().astimezone()
    assert now.tzinfo is not None
    assert now.utcoffset() == before.utcoffset() == after.utcoffset()
    assert before <= now <= after


def test_digest_failure_keeps_the_outcome_saved(tmp_path):
    memory = FakeMemory(write_error=RuntimeError("memory down"))
    code, _, err = _run(tmp_path, _deps(memory=memory), "--topic", "t1")
    assert code == 0
    (job,) = _jobs(tmp_path)
    assert job["outcome"] == "saved"
    assert job["states"][-1] == "saved"
    assert job["reasons"]["digest_failure"]
    assert "ダイジェスト" in err
    assert len(_rows(tmp_path)) == len(RESULTS)


def test_no_digest_without_a_topic_or_when_memory_is_disabled(tmp_path):
    memory = FakeMemory()
    # topic なし: make_memory_client も呼ばれない
    code, _, _ = _run(tmp_path, _deps())
    assert code == 0
    # topic ありでも memory.enabled = false なら書かない
    config = _write_config(tmp_path, "[memory]\nenabled = false\n")
    code, _, _ = _run(tmp_path, _deps(), "--topic", "t1", config=config)
    assert code == 0
    assert memory.writes == []
    assert len(_jobs(tmp_path)) == 2


def test_no_digest_for_failed_or_no_results(tmp_path):
    memory = FakeMemory()
    for kind in ("failed", "no_results"):
        response, jev = _scenario(kind)
        _run(tmp_path, _deps(search=FakeSearch(response), jev=jev, memory=memory), "--topic", "t")
    assert memory.writes == [] and memory.searches == []


# --- 画面読み取りの二重の許可 ---


def test_scrape_needs_both_the_flag_and_the_config(tmp_path):
    """never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない"""
    scrape = scrape_engine_names(load_config(_write_config(tmp_path)))
    cases = [
        ("[search]\nallow_scrape = true\n", ["--allow-scrape"], True),
        ("[search]\nallow_scrape = true\n", [], False),
        ("", ["--allow-scrape"], False),
    ]
    for extra, flags, expected in cases:
        search = FakeSearch(_response())
        deps = _deps(search=search)
        asked: list[bool] = []

        def make(config, allow_scrape, asked=asked, search=search):
            asked.append(allow_scrape)
            return search

        deps.make_search_backend = make
        _run(tmp_path, deps, "--genre", "general", *flags, config=_write_config(tmp_path, extra))
        assert asked == [expected]
        engines = set(search.calls[0][1])
        assert bool(scrape & engines) is expected


# --- --dry-run ---


def test_dry_run_calls_no_factory_and_shows_normalization_and_jev_preview(tmp_path):
    deps = Deps(
        env={},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        make_search_backend=_fail_factory("make_search_backend"),
        make_classifier=_fail_factory("make_classifier"),
        make_memory_client=_fail_factory("make_memory_client"),
    )
    code, out, err = _run(tmp_path, deps, "--dry-run")
    assert code == 0, err
    data = json.loads(out)
    assert data["query_norm"] == {"ja": QUERY_JA, "en": QUERY_EN}
    requests = data["rustxng"]["requests"]
    assert [r["lang"] for r in requests] == ["ja", "en"]
    assert requests[0]["params"]["q"] == QUERY_JA
    assert requests[1]["params"]["q"] == QUERY_EN
    assert requests[1]["params"]["language"] == "en"
    preview = data["jev"]["preview"]
    assert preview["questions"]
    assert preview["state"]["query"] == QUERY_JA
    assert preview["state"]["results"][0]["id"] == "r01"
    # 記録は書かない
    assert not (tmp_path / "data").exists()


def test_dry_run_has_no_en_request_when_the_dictionary_has_no_english(tmp_path):
    deps = Deps(
        env={},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        make_search_backend=_fail_factory("make_search_backend"),
        make_classifier=_fail_factory("make_classifier"),
        make_memory_client=_fail_factory("make_memory_client"),
    )
    code, out, _ = _run(tmp_path, deps, "--dry-run", query="plain words")
    assert code == 0
    data = json.loads(out)
    assert data["query_norm"]["en"] is None
    assert [r["lang"] for r in data["rustxng"]["requests"]] == ["ja"]
