"""保存（store.py）のテスト。一時ディレクトリに書き、書く側の番兵を確かめる。"""

import datetime
import json
from dataclasses import replace
from pathlib import Path

import pytest

from jevsearch.models import (
    ClassifiedResult,
    EngineFailure,
    JevUsage,
    JobRecord,
    JobState,
    JudgedBy,
    Judgment,
    Lang,
    NormalizedQuery,
    Outcome,
    Reasons,
    SearchResult,
    Versions,
)
from jevsearch.store import ResultStore, StoreError, job_row, result_rows

SENTINEL = "sk-test-SENTINEL-0000"
THRESHOLD = 0.55
TZ = datetime.timezone(datetime.timedelta(hours=9))
NOW = datetime.datetime(2026, 11, 3, 12, 0, 0, tzinfo=TZ)

VERSIONS = Versions(config="c1", dictionary="d1", rules="r1", jev_model="jev-1.14.0", rustxng="x1")
NORM = NormalizedQuery(original="量子コンピュータ", ja="量子コンピュータ", en="quantum computer", dictionary_version="d1")


def _item(n: int, title: str = "量子コンピュータ入門", *, type_: Judgment | None = None) -> ClassifiedResult:
    result = SearchResult(
        url=f"https://example.com/{n}",
        title=title,
        snippet="説明",
        engine="duckduckgo",
        published_date=None,
        retrieved_at=NOW,
        engines=("duckduckgo", "brave"),
        matched=(Lang.JA, Lang.EN),
    )
    return ClassifiedResult(
        id=f"r{n:02d}",
        result=result,
        type=type_ or Judgment("blog", JudgedBy.RULE, 1.0),
        genre=Judgment("tech", JudgedBy.JEV, 0.9),
        relevant=Judgment(True, JudgedBy.JEV, 0.8),
    )


def _rows(n: int = 2) -> list[dict]:
    items = [_item(i + 1) for i in range(n)]
    return result_rows(
        job_id="job-1", query="量子コンピュータ", query_norm=NORM, topic="quantum", items=items, versions=VERSIONS
    )


def _store(tmp_path: Path, api_key: str | None = None) -> ResultStore:
    return ResultStore(tmp_path / "data", threshold=THRESHOLD, api_key=api_key)


def _record(outcome: Outcome, state: JobState) -> JobRecord:
    return JobRecord(
        job_id=f"job-{outcome.value}",
        query="量子コンピュータ",
        query_norm=NORM,
        topic=None,
        genre=None,
        started_at=NOW,
        duration_ms=120,
        outcome=outcome,
        reasons=Reasons(failed_engines=(EngineFailure("brave", "timeout"),), unreachable_queries=(Lang.EN,)),
        states=(JobState.ACCEPTED, JobState.NORMALIZING, JobState.SEARCHING, state),
        n_results=0,
        n_unclassified=0,
        versions=VERSIONS,
        jev_usage=(JevUsage("jev-1.14.0", 10, 5, 12, 300, 3, 1, None),),
    )


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_result_rows_have_required_keys():
    rows = _rows(3)
    assert len(rows) == 3
    for row in rows:
        assert row["query"]
        assert datetime.datetime.fromisoformat(row["retrieved_at"]).utcoffset() is not None
        assert row["job_id"] == "job-1"
        assert set(row["versions"]) == {"config", "dictionary", "rules", "jev_model", "rustxng"}
        for key in ("type", "genre", "relevant"):
            assert set(row[key]) == {"value", "by", "confidence"}
    assert rows[0]["id"] == "r01"
    assert rows[0]["query_norm"] == {"ja": "量子コンピュータ", "en": "quantum computer"}
    assert rows[0]["matched"] == ["ja", "en"]
    assert rows[0]["type"]["by"] == "rule"


@pytest.mark.parametrize(
    "patch",
    [
        {"retrieved_at": "2026-11-03T12:00:00"},
        {"retrieved_at": ""},
        {"retrieved_at": None},
        {"query": ""},
        {"query": "   "},
    ],
)
def test_rejects_rows_without_query_or_retrieved_at(tmp_path, patch):
    """never: 検索語と取得日時のない結果を保存しない"""
    store = _store(tmp_path)
    rows = _rows(2)
    rows[1] = {**rows[1], **patch}  # 2行目だけ違反でも、1行目も書かれない
    with pytest.raises(StoreError):
        store.append_results(rows)
    assert _lines(tmp_path / "data" / "results.jsonl") == []


def test_rejects_jev_value_below_threshold(tmp_path):
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない"""
    store = _store(tmp_path)
    rows = _rows(1)
    rows[0]["type"] = {"value": "tech", "by": "jev", "confidence": 0.5}
    with pytest.raises(StoreError):
        store.append_results(rows)
    assert _lines(tmp_path / "data" / "results.jsonl") == []
    # confidence が null の Jev 判定も同じ
    rows[0]["type"] = {"value": "tech", "by": "jev", "confidence": None}
    with pytest.raises(StoreError):
        store.append_results(rows)
    # 3つの判定のどれでも見る
    rows = _rows(1)
    rows[0]["relevant"] = {"value": False, "by": "jev", "confidence": 0.1}
    with pytest.raises(StoreError):
        store.append_results(rows)


def test_accepts_unclassified_low_confidence_and_threshold_edge(tmp_path):
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない"""
    store = _store(tmp_path)
    rows = _rows(2)
    rows[0]["type"] = {"value": None, "by": "jev", "confidence": 0.5}  # 未分類は書ける
    rows[1]["genre"] = {"value": "tech", "by": "jev", "confidence": THRESHOLD}  # ちょうどしきい値は書ける
    rows[1]["type"] = {"value": "docs", "by": "rule", "confidence": None}  # rule は確信度を見ない
    store.append_results(rows)
    assert len(_lines(tmp_path / "data" / "results.jsonl")) == 2


def test_rejects_api_key_in_any_file(tmp_path):
    """never: API キーを結果ログ・記録・画面の表示に出さない"""
    store = _store(tmp_path, api_key=SENTINEL)
    rows = _rows(2)
    rows[1]["snippet"] = f"leak {SENTINEL} here"
    with pytest.raises(StoreError) as e1:
        store.append_results(rows)
    job = job_row(_record(Outcome.FAILED, JobState.FAILED))
    job["reasons"]["jev_failure"] = f"401 key={SENTINEL}"
    with pytest.raises(StoreError) as e2:
        store.append_job(job)
    with pytest.raises(StoreError) as e3:
        store.append_eval({"job_id": "j", "note": SENTINEL})
    for e in (e1, e2, e3):
        assert SENTINEL not in str(e.value)
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").iterdir())


def test_api_key_none_skips_key_check_and_clean_rows_pass(tmp_path):
    """never: API キーを結果ログ・記録・画面の表示に出さない"""
    _store(tmp_path, api_key=None).append_eval({"job_id": "j", "note": SENTINEL})
    store = _store(tmp_path, api_key=SENTINEL)
    store.append_results(_rows(1))
    assert len(_lines(tmp_path / "data" / "results.jsonl")) == 1


def test_no_results_and_failed_jobs_are_each_recorded(tmp_path):
    """never: 0件を失敗として記録しない"""
    store = _store(tmp_path)
    store.append_job(job_row(_record(Outcome.NO_RESULTS, JobState.NO_RESULTS)))
    store.append_job(job_row(_record(Outcome.FAILED, JobState.FAILED)))
    jobs = list(store.read("jobs.jsonl"))
    assert [j["outcome"] for j in jobs] == ["no_results", "failed"]
    assert jobs[0]["states"][-1] == "no_results"
    assert jobs[1]["states"][-1] == "failed"
    assert jobs[0]["reasons"]["failed_engines"] == [{"engine": "brave", "reason": "timeout"}]
    assert jobs[0]["reasons"]["unreachable_queries"] == ["en"]
    assert jobs[0]["jev_usage"][0]["model"] == "jev-1.14.0"
    assert len(_lines(tmp_path / "data" / "jobs.jsonl")) == 2


def test_job_row_with_no_query_norm_and_genre():
    row = job_row(replace(_record(Outcome.FAILED, JobState.FAILED), query_norm=None))
    assert row["query_norm"] is None
    assert row["genre"] is None
    assert row["started_at"] == NOW.isoformat()


def test_japanese_is_written_unescaped_and_read_back(tmp_path):
    store = _store(tmp_path)
    rows = _rows(2)
    store.append_results(rows)
    text = (tmp_path / "data" / "results.jsonl").read_text(encoding="utf-8")
    assert "量子コンピュータ入門" in text
    assert "\\u" not in text
    assert [json.loads(line)["id"] for line in text.splitlines()] == ["r01", "r02"]
    assert list(store.read("results.jsonl")) == rows


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), True, "0.9", 1.5, -0.1, [0.9], {"x": 1}],
)
def test_rejects_non_numeric_or_out_of_range_confidence(tmp_path, bad):
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない"""
    store = _store(tmp_path)
    rows = _rows(1)
    rows[0]["genre"] = {"value": "tech", "by": "jev", "confidence": bad}
    with pytest.raises(StoreError):
        store.append_results(rows)
    assert _lines(tmp_path / "data" / "results.jsonl") == []


def test_accepts_int_confidence_within_range(tmp_path):
    store = _store(tmp_path)
    rows = _rows(1)
    rows[0]["genre"] = {"value": "tech", "by": "jev", "confidence": 1}
    store.append_results(rows)
    assert len(_lines(tmp_path / "data" / "results.jsonl")) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_never_writes_nan_or_infinity_anywhere(tmp_path, value):
    store = _store(tmp_path)
    with pytest.raises(StoreError):
        store.append_eval({"job_id": "j", "score": value})
    with pytest.raises(StoreError):
        store.append_job({"job_id": "j", "duration_ms": value})
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").iterdir())


def test_append_after_truncated_line_keeps_both_readable(tmp_path):
    store = _store(tmp_path)
    path = tmp_path / "data" / "jobs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"job_id": "a"', encoding="utf-8")  # 途中で切れた行（改行なし）
    store.append_job({"job_id": "b"})
    store.append_job({"job_id": "c"})
    assert _lines(path)[1:] == ['{"job_id": "b"}', '{"job_id": "c"}']
    # 壊れた行は飛ばし、あとの行は読める
    assert [r["job_id"] for r in store.read("jobs.jsonl")] == ["b", "c"]


def test_append_after_complete_line_adds_no_blank_line(tmp_path):
    store = _store(tmp_path)
    store.append_job({"job_id": "a"})
    store.append_job({"job_id": "b"})
    assert _lines(tmp_path / "data" / "jobs.jsonl") == ['{"job_id": "a"}', '{"job_id": "b"}']


def test_read_skips_invalid_lines_and_returns_the_rest(tmp_path):
    store = _store(tmp_path)
    path = tmp_path / "data" / "eval.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"n": 1}\n{"n": 2\n\n[1, 2]\n{"n": 3}\n\xff\xfe\n{"n": 4}\n')
    assert [r["n"] for r in store.read("eval.jsonl")] == [1, 3, 4]


def test_append_results_accepts_a_generator(tmp_path):
    store = _store(tmp_path)
    store.append_results(r for r in _rows(3))
    assert len(_lines(tmp_path / "data" / "results.jsonl")) == 3


def test_generator_with_violating_row_writes_nothing(tmp_path):
    store = _store(tmp_path)
    rows = _rows(2)
    rows[1]["query"] = ""
    with pytest.raises(StoreError):
        store.append_results(r for r in rows)
    assert _lines(tmp_path / "data" / "results.jsonl") == []


def test_unencodable_row_raises_store_error_and_writes_nothing(tmp_path):
    store = _store(tmp_path)
    rows = _rows(2)
    rows[1]["title"] = "bad \ud800 title"  # 単独のサロゲートは UTF-8 にできない
    with pytest.raises(StoreError) as e:
        store.append_results(rows)
    assert "bad" not in str(e.value)
    assert not (tmp_path / "data" / "results.jsonl").exists()
    with pytest.raises(StoreError):
        store.append_eval({"note": "\udcff"})
    assert not (tmp_path / "data" / "eval.jsonl").exists()


def test_os_error_on_write_becomes_store_error(tmp_path):
    # data_dir の場所に通常のファイルがあると、ディレクトリを作れず OSError になる
    blocker = tmp_path / "data"
    blocker.write_text("x", encoding="utf-8")
    store = _store(tmp_path)
    with pytest.raises(StoreError) as e:
        store.append_job({"job_id": "SECRET-CONTENT"})
    assert "SECRET-CONTENT" not in str(e.value)


def test_append_creates_dir_appends_and_read_missing_is_empty(tmp_path):
    store = _store(tmp_path)
    assert list(store.read("eval.jsonl")) == []
    store.append_eval({"job_id": "a"})
    store.append_eval({"job_id": "b", "title": "日本語"})
    assert [r["job_id"] for r in store.read("eval.jsonl")] == ["a", "b"]
    assert len(_lines(tmp_path / "data" / "eval.jsonl")) == 2
