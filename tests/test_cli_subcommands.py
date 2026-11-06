"""`eval`・`stats`・`digest --retry` の通しのテスト。

CLI を `main(argv, deps=...)` で動かす。Jev と共有メモリは fake に差し替え、保存先は tmp_path の
設定ファイルで指す。batch-vs-single の一部は、Jev の通信を httpx.MockTransport にして既定の
make_classifier を通し、batch_size が本物の分類器に届くことと、Jev が答えないときの終了コードを確かめる。
"""

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from jevsearch import credentials
from jevsearch.cli import Deps, main
from jevsearch.evaluation import CSV_COLUMNS, eval_row
from jevsearch.models import (
    JEV_MODEL,
    ClassifiedResult,
    Genre,
    JevReport,
    JudgedBy,
    Judgment,
    Lang,
    NormalizedQuery,
    ResultType,
    SearchResult,
    Versions,
)
from jevsearch.store import ResultStore, result_rows

KEY = "sk-test-SENTINEL-0000"
JST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 2, 21, 5, 12, tzinfo=JST)
VERSIONS = Versions(
    config="cfg-test",
    dictionary="dict-test",
    rules="rules-test",
    jev_model=JEV_MODEL,
    rustxng="0.26.0",
)


@pytest.fixture(autouse=True)
def _no_real_credentials(tmp_path_factory, monkeypatch):
    """手元の認証ファイルを読まない（キーは deps.env だけから来る）。"""
    missing = tmp_path_factory.mktemp("nocred") / "credentials.json"
    monkeypatch.setattr(credentials, "default_credentials_path", lambda: missing)


# --- 部品 ---


def _result(n: int, title: str | None = None) -> SearchResult:
    return SearchResult(
        url=f"https://site{n}.example.org/page/{n}",
        title=title or f"title {n}",
        snippet=f"snippet {n}",
        engine="wikipedia",
        published_date=None,
        retrieved_at=T0,
        engines=("wikipedia",),
        matched=(Lang.JA,),
    )


def _judged(position: int, result: SearchResult) -> ClassifiedResult:
    """Jev で決めた判定（種類は blog、ジャンルは tech、関連ありは真）。"""
    return ClassifiedResult(
        id=f"r{position + 1:02d}",
        result=result,
        type=Judgment(ResultType.BLOG, JudgedBy.JEV, 0.9),
        genre=Judgment(Genre.TECH, JudgedBy.JEV, 0.9),
        relevant=Judgment(True, JudgedBy.JEV, 0.9),
    )


def _rows(
    job_id: str, query: str, results: list[SearchResult], topic: str | None = None
) -> list[dict]:
    """結果ログの行（保存と同じ形）。"""
    return result_rows(
        job_id=job_id,
        query=query,
        query_norm=NormalizedQuery(
            original=query, ja=query, en=None, dictionary_version="dict-test"
        ),
        topic=topic,
        items=[_judged(position, r) for position, r in enumerate(results)],
        versions=VERSIONS,
    )


def _eval_row(job_id: str, started_at: datetime, n: int) -> dict:
    """評価記録の1行（evaluation.eval_row で作る）。結果は n 件で、Jev の使用量は無い。"""
    return eval_row(
        job_id=job_id,
        started_at=started_at,
        items=[_judged(i, _result(i)) for i in range(n)],
        report=JevReport(query_genre=Judgment(None, JudgedBy.JEV, None), usage=(), failure=None),
        failed_engines=(),
        price_per_mtok_input=None,
        price_per_mtok_output=None,
    )


def _store(tmp_path: Path) -> ResultStore:
    """tmp_path/data の保存先。書く側の番兵は既定のしきい値（0.55）で見る。"""
    return ResultStore(tmp_path / "data", threshold=0.55, api_key=None)


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    """保存先を tmp_path/data にする設定ファイル。extra は追加の節。"""
    data = (tmp_path / "data").as_posix()
    path = tmp_path / "config.toml"
    path.write_text(f"[storage]\ndir = '{data}'\n{extra}", encoding="utf-8")
    return path


def _fail(name: str):
    """呼ばれてはいけない make_* の代わり。呼ばれたら AssertionError（内部エラーで終了コード 1）。"""

    def make(*args):
        raise AssertionError(f"{name} が呼ばれた")

    return make


def _deps(env: dict | None = None, **overrides) -> Deps:
    """標準出力と標準エラーを StringIO にした Deps。make_* は、差し替えない限り呼ばれたら失敗する。"""
    base = {
        "env": env if env is not None else {},
        "stdout": io.StringIO(),
        "stderr": io.StringIO(),
        "make_search_backend": _fail("make_search_backend"),
        "make_classifier": _fail("make_classifier"),
        "make_memory_client": _fail("make_memory_client"),
    }
    base.update(overrides)
    return Deps(**base)


def _run(argv: list[str], deps: Deps) -> tuple[int, str, str]:
    code = main(argv, deps=deps)
    return code, deps.stdout.getvalue(), deps.stderr.getvalue()


class FakeClassifier:
    """Jev の代わり。どの結果にも同じ判定を返し、受け取った batch_size は覚えるだけにする。"""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def classify(self, query, results, rule_types):
        items = [_judged(position, r) for position, r in enumerate(results)]
        report = JevReport(
            query_genre=Judgment(Genre.TECH, JudgedBy.JEV, 0.9), usage=(), failure=None
        )
        return items, report


class FakeMemory:
    """共有メモリの代わり。write の呼び出しを記録し、write_error があれば毎回投げる。"""

    def __init__(self, write_error: Exception | None = None) -> None:
        self.write_error = write_error
        self.writes: list[dict] = []

    def search(self, query, *, scope, tags, limit):
        return []

    def write(self, **kwargs):
        if self.write_error is not None:
            raise self.write_error
        self.writes.append(kwargs)
        return f"M{len(self.writes)}"


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


def _jev_transport(seen: list[httpx.Request], *, refuse: bytes | None = None) -> httpx.MockTransport:
    """Jev の宛先（api.typesafe.ai）だけに答える。受けた要求は seen に溜める。

    refuse を本文に含む要求だけは 401 で断る（一部の要求だけ失敗させるとき）。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host != "api.typesafe.ai":
            raise AssertionError(f"想定外の宛先: {request.url}")
        if refuse is not None and refuse in request.content:
            return httpx.Response(401)
        questions = json.loads(request.content)["questions"]
        usage = {"input_tokens": 100, "output_tokens": 20}
        return httpx.Response(200, json={"answers": _answers_for(questions), "usage": usage})

    return httpx.MockTransport(handler)


def _write_labels(path: Path, rows: list[dict]) -> None:
    """人が正解を付けた CSV（列は CSV_COLUMNS と同じ。書かない列は空）。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" for key in CSV_COLUMNS} | row)


def _write_log(tmp_path: Path) -> Path:
    """結果ログ。ジョブ j1 は日本語の結果が2件、ジョブ j2 は英語の結果が1件。"""
    rows = _rows("j1", "rust メタ検索", [_result(1, "日本語の記事"), _result(2, "別の記事")])
    rows += _rows("j2", "rust", [_result(3, "English article")])
    ResultStore(tmp_path / "logs", threshold=0.55, api_key=None).append_results(rows)
    return tmp_path / "logs" / "results.jsonl"


def _write_pending(tmp_path: Path, entries: list[dict]) -> None:
    path = tmp_path / "data" / "digest-pending.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    path.write_text(text, encoding="utf-8")


def _pending_entries(tmp_path: Path) -> list[dict]:
    path = tmp_path / "data" / "digest-pending.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _all_files_text(tmp_path: Path) -> list[str]:
    return [p.read_bytes().decode("utf-8", errors="replace") for p in tmp_path.rglob("*") if p.is_file()]


# --- stats ---


def test_stats_counts_the_eval_rows_as_jobs(tmp_path):
    cfg = _write_config(tmp_path)
    store = _store(tmp_path)
    for job, n in (("j1", 2), ("j2", 3), ("j3", 4)):
        store.append_eval(_eval_row(job, T0, n))
    code, out, err = _run(["stats", "--config", str(cfg)], _deps())
    assert code == 0, err
    summary = json.loads(out)
    assert summary["jobs"] == 3
    assert summary["results"] == 9


def test_stats_since_keeps_the_jobs_from_that_local_date_on(tmp_path):
    cfg = _write_config(tmp_path)
    now = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
    store = _store(tmp_path)
    for days_ago in (10, 1, 0):
        store.append_eval(_eval_row(f"j{days_ago}", now - timedelta(days=days_ago), 2))
    since = (now - timedelta(days=5)).date().isoformat()
    code, out, err = _run(["stats", "--since", since, "--config", str(cfg)], _deps())
    assert code == 0, err
    assert json.loads(out)["jobs"] == 2


def test_stats_without_an_eval_file_reports_zero_jobs(tmp_path):
    cfg = _write_config(tmp_path)
    code, out, err = _run(["stats", "--config", str(cfg)], _deps())
    assert code == 0, err
    assert json.loads(out)["jobs"] == 0


# --- eval sample ---


def test_eval_sample_prints_a_header_and_n_rows(tmp_path):
    cfg = _write_config(tmp_path)
    _store(tmp_path).append_results(_rows("j1", "rust", [_result(n) for n in range(5)]))
    code, out, err = _run(["eval", "sample", "--n", "3", "--config", str(cfg)], _deps())
    assert code == 0, err
    lines = list(csv.reader(io.StringIO(out)))
    assert lines[0] == list(CSV_COLUMNS)
    assert len(lines) == 1 + 3


def test_eval_sample_without_results_prints_only_the_header(tmp_path):
    cfg = _write_config(tmp_path)
    code, out, err = _run(["eval", "sample", "--config", str(cfg)], _deps())
    assert code == 0, err
    assert list(csv.reader(io.StringIO(out))) == [list(CSV_COLUMNS)]


def test_eval_sample_out_writes_the_same_text_as_stdout(tmp_path):
    cfg = _write_config(tmp_path)
    _store(tmp_path).append_results(_rows("j1", "rust", [_result(n) for n in range(5)]))
    target = tmp_path / "sample.csv"
    code, out, err = _run(
        [
            "eval",
            "sample",
            "--n",
            "3",
            "--seed",
            "7",
            "--out",
            str(target),
            "--config",
            str(cfg),
        ],
        _deps(),
    )
    assert code == 0, err
    assert out == ""
    written = target.read_bytes()
    assert b"\r" not in written  # 改行は LF のまま
    code, to_stdout, err = _run(
        ["eval", "sample", "--n", "3", "--seed", "7", "--config", str(cfg)], _deps()
    )
    assert code == 0, err
    assert written.decode("utf-8") == to_stdout


# --- eval accuracy ---


def test_eval_accuracy_prints_json_split_by_language(tmp_path):
    cfg = _write_config(tmp_path)
    labels = tmp_path / "labels.csv"
    _write_labels(
        labels,
        [
            {
                "lang": "ja",
                "type": "blog",
                "genre": "tech",
                "relevant": "true",
                "label_type": "blog",
                "label_genre": "news",
                "label_relevant": "1",
            },
            {
                "lang": "en",
                "type": "",
                "genre": "tech",
                "relevant": "false",
                "label_type": "blog",
                "label_genre": "tech",
                "label_relevant": "false",
            },
        ],
    )
    code, out, err = _run(["eval", "accuracy", str(labels), "--config", str(cfg)], _deps())
    assert code == 0, err
    acc = json.loads(out)
    assert set(acc) == {"type", "genre", "relevant"}
    assert acc["type"]["ja"] == {"correct": 1, "total": 1, "unclassified": 0, "accuracy": 1.0}
    assert acc["type"]["en"] == {"correct": 0, "total": 1, "unclassified": 1, "accuracy": 0.0}
    assert acc["genre"]["ja"]["accuracy"] == 0.0
    assert acc["genre"]["en"]["accuracy"] == 1.0
    assert acc["relevant"]["ja"]["accuracy"] == 1.0


# --- eval batch-vs-single ---


def test_eval_batch_vs_single_prints_ja_and_en_for_each_judgment(tmp_path):
    cfg = _write_config(tmp_path)
    log = _write_log(tmp_path)
    made: list[int] = []

    def make_classifier(config):
        made.append(config.jev.batch_size)
        return FakeClassifier(config.jev.batch_size)

    deps = _deps(env={"TYPESAFE_API_KEY": KEY}, make_classifier=make_classifier)
    code, out, err = _run(["eval", "batch-vs-single", str(log), "--config", str(cfg)], deps)
    assert code == 0, err
    summary = json.loads(out)
    assert set(summary) == {"type", "genre", "relevant"}
    for key in summary:
        assert set(summary[key]) == {"ja", "en"}
    assert summary["genre"]["ja"]["paired"] == 2
    assert summary["genre"]["en"]["paired"] == 1
    assert summary["genre"]["ja"]["agreement_rate"] == 1.0
    # まとめて（設定の batch_size）と1件ずつ（1）の2つの分類器を作った
    assert sorted(made) == [1, 25]
    assert KEY not in out


def test_eval_batch_vs_single_without_a_key_exits_1_and_builds_nothing(tmp_path):
    cfg = _write_config(tmp_path)
    log = _write_log(tmp_path)
    code, out, err = _run(
        ["eval", "batch-vs-single", str(log), "--config", str(cfg)], _deps(env={})
    )
    assert code == 1
    assert out == ""
    assert "API キー" in err


def test_eval_batch_vs_single_with_the_real_classifier_sends_batches_and_singles(tmp_path):
    """make_classifier は既定のまま、Jev の通信だけを MockTransport にする。

    群（ジョブと検索語）は2つ、結果は3件。まとめては群ごとに1回、1件ずつは結果ごとに1回なので
    2 + 3 = 5 回の要求になる。
    """
    cfg = _write_config(tmp_path)
    log = _write_log(tmp_path)
    seen: list[httpx.Request] = []
    deps = Deps(
        env={"TYPESAFE_API_KEY": KEY},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        http_transport=_jev_transport(seen),
    )
    code, out, err = _run(["eval", "batch-vs-single", str(log), "--config", str(cfg)], deps)
    assert code == 0, err
    assert len(seen) == 2 + 3
    for request in seen:
        assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in out
    assert all(KEY not in text for text in _all_files_text(tmp_path))


def test_eval_batch_vs_single_exits_1_when_jev_rejects_the_key(tmp_path):
    """Jev が 401 で断ったら、JSON は出すが終了コードは 1。理由は標準エラーに出し、キーは出さない。"""
    cfg = _write_config(tmp_path)
    log = _write_log(tmp_path)
    seen: list[httpx.Request] = []

    def reject(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401)

    deps = Deps(
        env={"TYPESAFE_API_KEY": KEY},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        http_transport=httpx.MockTransport(reject),
    )
    code, out, err = _run(["eval", "batch-vs-single", str(log), "--config", str(cfg)], deps)
    assert code == 1
    assert "http 401" in err
    assert KEY not in err
    assert KEY not in out
    summary = json.loads(out)
    assert summary["jev_failures"] == ["http 401"]
    assert summary["genre"]["ja"]["paired"] == 0
    # まとめて2回と1件ずつ3回の要求が、どれも401で断られる（401 は再試行しない）
    assert len(seen) == 2 + 3


def test_eval_batch_vs_single_exits_1_when_only_some_requests_fail(tmp_path):
    """一部の要求だけ Jev が断ったら、答えの揃った分は比べて出し、終了コードは 1 と jev_failures を返す。"""
    cfg = _write_config(tmp_path)
    log = _write_log(tmp_path)
    deps = Deps(
        env={"TYPESAFE_API_KEY": KEY},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        http_transport=_jev_transport([], refuse=b"English article"),
    )
    code, out, err = _run(["eval", "batch-vs-single", str(log), "--config", str(cfg)], deps)
    assert code == 1
    assert "http 401" in err
    assert KEY not in err
    assert KEY not in out
    summary = json.loads(out)
    assert summary["jev_failures"] == ["http 401"]
    # 英語の1件（title が English article）の要求だけ断られ、日本語の2件は両方で答えが揃う
    assert summary["genre"]["ja"]["paired"] == 2
    assert summary["genre"]["ja"]["agreement_rate"] == 1.0
    assert summary["genre"]["en"]["paired"] == 0


# --- digest --retry ---


def test_digest_retry_writes_the_pending_day_and_exits_0(tmp_path):
    cfg = _write_config(tmp_path)
    day = T0.date().isoformat()
    _store(tmp_path).append_results(_rows("j1", "rust", [_result(1), _result(2)], topic="t1"))
    _write_pending(tmp_path, [{"topic": "t1", "date": day, "error": "RuntimeError: down"}])
    memory = FakeMemory()

    def make_memory(config):
        return memory

    code, out, err = _run(
        ["digest", "--retry", "--config", str(cfg)], _deps(make_memory_client=make_memory)
    )
    assert code == 0, err
    assert out == "書けた 1 件・まだ 0 件\n"
    (write,) = memory.writes
    assert f"t1 {day}" in write["title"]
    assert "news:radar-bot:t1" in write["tags"]
    assert _pending_entries(tmp_path) == []
    index = json.loads((tmp_path / "data" / "digest-index.json").read_text(encoding="utf-8"))
    assert index == {f"t1|{day}": "M1"}


def test_digest_retry_keeps_a_digest_with_content_when_the_result_log_has_no_rows(tmp_path):
    """結果ログが無い（行が1件も無い）ときは、索引に前のダイジェストがあっても空で置き換えない。"""
    cfg = _write_config(tmp_path)
    day = T0.date().isoformat()
    _write_pending(tmp_path, [{"topic": "t1", "date": day, "error": "RuntimeError: down"}])
    index_path = tmp_path / "data" / "digest-index.json"
    index_path.write_text(json.dumps({f"t1|{day}": "PREV"}), encoding="utf-8")
    memory = FakeMemory()

    def make_memory(config):
        return memory

    code, out, err = _run(
        ["digest", "--retry", "--config", str(cfg)], _deps(make_memory_client=make_memory)
    )
    assert code == 0, err
    assert out == "書けた 0 件・まだ 0 件\n"
    assert memory.writes == []
    assert _pending_entries(tmp_path) == []
    assert json.loads(index_path.read_text(encoding="utf-8")) == {f"t1|{day}": "PREV"}


def test_digest_retry_exits_1_for_as_long_as_the_memory_keeps_failing(tmp_path):
    cfg = _write_config(tmp_path)
    day = T0.date().isoformat()
    _store(tmp_path).append_results(_rows("j1", "rust", [_result(1)], topic="t1"))
    _write_pending(tmp_path, [{"topic": "t1", "date": day, "error": "x"}])
    memory = FakeMemory(write_error=RuntimeError("memory down"))

    def make_memory(config):
        return memory

    for _ in range(2):
        code, out, err = _run(
            ["digest", "--retry", "--config", str(cfg)], _deps(make_memory_client=make_memory)
        )
        assert code == 1
        assert out == "書けた 0 件・まだ 1 件\n"
        assert len(_pending_entries(tmp_path)) == 1


def test_digest_retry_refuses_when_the_memory_is_disabled(tmp_path):
    cfg = _write_config(tmp_path, "[memory]\nenabled = false\n")
    _write_pending(tmp_path, [{"topic": "t1", "date": T0.date().isoformat(), "error": "x"}])
    memory = FakeMemory()

    def make_memory(config):
        return memory

    code, out, err = _run(
        ["digest", "--retry", "--config", str(cfg)], _deps(make_memory_client=make_memory)
    )
    assert code == 2
    assert out == ""
    assert err
    assert memory.writes == []
    assert len(_pending_entries(tmp_path)) == 1


def test_digest_retry_with_nothing_pending_exits_0(tmp_path):
    cfg = _write_config(tmp_path)
    code, out, err = _run(
        ["digest", "--retry", "--config", str(cfg)],
        _deps(make_memory_client=lambda config: FakeMemory()),
    )
    assert code == 0, err
    assert out == "書けた 0 件・まだ 0 件\n"


# --- 使い方・入力の誤り、--config ---


def test_bad_arguments_and_missing_inputs_exit_2(tmp_path):
    cfg = str(_write_config(tmp_path))
    missing = str(tmp_path / "missing")
    cases = [
        ["eval"],
        ["eval", "bogus"],
        ["eval", "sample", "--n", "0", "--config", cfg],
        ["eval", "sample", "--n", "x", "--config", cfg],
        ["stats", "--since", "2026/11/01", "--config", cfg],
        ["digest", "--config", cfg],
        ["eval", "accuracy", missing + ".csv", "--config", cfg],
        ["eval", "batch-vs-single", missing + ".jsonl", "--config", cfg],
        ["stats", "--config", missing + ".toml"],
        ["eval", "sample", "--out", str(tmp_path / "no-dir" / "out.csv"), "--config", cfg],
    ]
    for argv in cases:
        code, out, err = _run(argv, _deps(env={"TYPESAFE_API_KEY": KEY}))
        assert code == 2, argv
        assert out == "", argv
        assert err, argv


def test_every_subcommand_accepts_config(tmp_path):
    cfg = str(_write_config(tmp_path))
    store = _store(tmp_path)
    store.append_results(_rows("j1", "rust", [_result(1)], topic="t1"))
    store.append_eval(_eval_row("j1", T0, 1))
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [])
    log = _write_log(tmp_path)
    _write_pending(tmp_path, [{"topic": "t1", "date": T0.date().isoformat(), "error": "x"}])
    memory = FakeMemory()

    def make_memory(config):
        return memory

    def make_classifier(config):
        return FakeClassifier(config.jev.batch_size)

    runs = [
        (["stats", "--config", cfg], _deps()),
        (["eval", "sample", "--config", cfg], _deps()),
        (["eval", "accuracy", str(labels), "--config", cfg], _deps()),
        (
            ["eval", "batch-vs-single", str(log), "--config", cfg],
            _deps(env={"TYPESAFE_API_KEY": KEY}, make_classifier=make_classifier),
        ),
        (["digest", "--retry", "--config", cfg], _deps(make_memory_client=make_memory)),
    ]
    for argv, deps in runs:
        code, _, err = _run(argv, deps)
        assert code == 0, (argv, err)


def test_help_lists_the_subcommands(tmp_path):
    code, out, _ = _run(["--help"], _deps())
    assert code == 0
    for word in ("search", "eval", "stats", "digest"):
        assert word in out
