"""評価記録の作成・集計、まとめて判定と1件ずつ判定の比較、抜き出しと正解率のテスト。"""

import csv
import io
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from jevsearch.evaluation import (
    CSV_COLUMNS,
    accuracy_from_labels,
    aggregate,
    batch_vs_single,
    eval_row,
    is_japanese,
    rows_to_results,
    sample_csv,
)
from jevsearch.models import (
    ClassifiedResult,
    EngineFailure,
    JevReport,
    JevUsage,
    JudgedBy,
    Judgment,
    Lang,
    ResultType,
    SearchResult,
)

JST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 2, 21, 5, 12, tzinfo=JST)


def _result(n: int, title: str = "title") -> SearchResult:
    return SearchResult(
        url=f"https://example.com/{n}",
        title=title,
        snippet="snippet",
        engine="github",
        published_date=None,
        retrieved_at=T0,
        engines=("github",),
        matched=(Lang.JA,),
    )


def _j(value, by=JudgedBy.JEV, confidence=0.8) -> Judgment:
    return Judgment(value=value, by=by, confidence=confidence)


def _item(n: int, type_: Judgment, genre: Judgment, relevant: Judgment) -> ClassifiedResult:
    return ClassifiedResult(
        id=f"r{n:02d}", result=_result(n), type=type_, genre=genre, relevant=relevant
    )


def _usage(input_tokens=1000, output_tokens=200) -> JevUsage:
    return JevUsage(
        model="jev-1.14.0",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_input_tokens=900,
        duration_ms=100,
        n_results=6,
        n_questions=18,
        error=None,
    )


def _report(*usage: JevUsage) -> JevReport:
    return JevReport(query_genre=_j("tech"), usage=tuple(usage), failure=None)


def _six_items() -> list[ClassifiedResult]:
    """type は規則2件・Jev3件・未分類1件。genre と relevant は全て Jev で分類済み。"""
    rule = JudgedBy.RULE
    return [
        _item(1, _j("repo", rule, 1.0), _j("tech", confidence=0.95), _j(True, confidence=0.9)),
        _item(2, _j("paper", rule, 1.0), _j("paper", confidence=0.85), _j(True, confidence=0.7)),
        _item(3, _j("blog"), _j("tech", confidence=0.75), _j(False, confidence=0.65)),
        _item(4, _j("news"), _j("news", confidence=0.6), _j(True, confidence=0.55)),
        _item(5, _j("docs"), _j("tech", confidence=0.05), _j(True, confidence=0.99)),
        _item(6, _j(None, confidence=0.4), _j("tech", confidence=0.58), _j(True, confidence=0.8)),
    ]


def _row(items=None, report=None, started_at=T0, failed=(), pin=None, pout=None) -> dict:
    return eval_row(
        job_id="J1",
        started_at=started_at,
        items=_six_items() if items is None else items,
        report=_report(_usage()) if report is None else report,
        failed_engines=failed,
        price_per_mtok_input=pin,
        price_per_mtok_output=pout,
    )


def test_eval_row_rule_jev_unclassified_rates():
    row = _row()
    t = row["type"]
    assert (t["rule"], t["jev"], t["unclassified"]) == (2, 3, 1)
    assert t["rule_rate"] == pytest.approx(2 / 6)
    assert t["jev_rate"] == pytest.approx(3 / 6)
    assert t["unclassified_rate"] == pytest.approx(1 / 6)
    assert row["genre"]["jev"] == 6
    assert row["job_id"] == "J1"
    assert row["n_results"] == 6
    assert row["started_at"] == T0.isoformat()


def test_eval_row_empty_items_rates_are_zero():
    row = _row(items=[], report=_report())
    assert row["n_results"] == 0
    for key in ("type", "genre", "relevant"):
        assert row[key]["rule_rate"] == 0.0
        assert row[key]["jev_rate"] == 0.0
        assert row[key]["unclassified_rate"] == 0.0
    assert row["confidence_distribution"] == [0] * 10


def test_confidence_distribution_has_ten_bins_and_sums_to_judgments():
    row = _row()
    dist = row["confidence_distribution"]
    assert len(dist) == 10
    # 6 件 x 3 判定 = 18 判定すべてに確信度がある（未分類の 0.4 も数える）
    assert sum(dist) == 18
    assert dist[0] == 1  # 0.05
    assert dist[4] == 1  # 0.4
    assert dist[9] == 5  # 1.0 x2 と 0.95 と 0.9 と 0.99


def test_confidence_distribution_skips_judgments_without_confidence():
    items = [_item(1, _j(None, confidence=None), _j("tech", confidence=0.5), _j(True, confidence=1.0))]
    dist = _row(items=items)["confidence_distribution"]
    assert sum(dist) == 2
    assert dist[5] == 1
    assert dist[9] == 1


def test_cost_is_null_without_prices_and_number_with_prices():
    assert _row()["cost_usd"] is None
    assert _row(pin=3.0)["cost_usd"] is None  # 片方だけでは出さない
    row = _row(pin=3.0, pout=15.0)
    assert row["cost_usd"] == pytest.approx((1000 * 3.0 + 200 * 15.0) / 1_000_000)
    assert row["jev_tokens"] == {"input": 1000, "output": 200, "estimated_input": 900}


def test_cost_is_null_when_tokens_unknown():
    report = _report(_usage(input_tokens=None, output_tokens=None))
    row = _row(report=report, pin=3.0, pout=15.0)
    assert row["cost_usd"] is None
    assert row["jev_tokens"]["input"] is None


def test_cost_is_null_when_some_usage_lacks_tokens():
    report = _report(_usage(), _usage(input_tokens=None, output_tokens=None))
    row = _row(report=report, pin=3.0, pout=15.0)
    assert row["cost_usd"] is None
    # 取れた分のトークン数は記録に残る
    assert row["jev_tokens"]["input"] == 1000
    only_output_missing = _report(_usage(), _usage(output_tokens=None))
    assert _row(report=only_output_missing, pin=3.0, pout=15.0)["cost_usd"] is None
    both_known = _report(_usage(), _usage())
    assert _row(report=both_known, pin=3.0, pout=15.0)["cost_usd"] == pytest.approx((2000 * 3.0 + 400 * 15.0) / 1_000_000)


def test_eval_row_records_failed_engines():
    row = _row(failed=[EngineFailure("arxiv", "timeout")])
    assert row["failed_engines"] == [{"engine": "arxiv", "reason": "timeout"}]


def test_eval_row_without_report_records_failures_and_counts():
    failures = [EngineFailure("arxiv", "timeout")]
    row = eval_row(
        job_id="J0",
        started_at=T0,
        items=[],
        report=None,
        failed_engines=failures,
        price_per_mtok_input=3.0,
        price_per_mtok_output=15.0,
    )
    assert row["n_results"] == 0
    assert row["jev_tokens"] == {"input": None, "output": None, "estimated_input": 0}
    assert row["cost_usd"] is None
    assert row["failed_engines"] == [{"engine": "arxiv", "reason": "timeout"}]
    assert row["type"]["rule_rate"] == 0.0
    assert row["confidence_distribution"] == [0] * 10
    # 行の形は report ありのときと同じ
    assert set(row) == set(_row(items=[], report=_report()))
    assert aggregate([row])["failed_engines"] == {"arxiv": 1}
    # 結果があるときは件数・割合を items から数える
    with_items = eval_row(
        job_id="J0",
        started_at=T0,
        items=_six_items(),
        report=None,
        failed_engines=[],
        price_per_mtok_input=3.0,
        price_per_mtok_output=15.0,
    )
    assert with_items["type"]["rule"] == 2
    assert with_items["jev_tokens"]["input"] is None
    assert with_items["cost_usd"] is None


def test_aggregate_uses_same_json_keys():
    agg = aggregate([_row()])
    for key in ("type", "genre", "relevant"):
        assert set(agg[key]) == {
            "rule",
            "jev",
            "unclassified",
            "rule_rate",
            "jev_rate",
            "unclassified_rate",
        }
    assert set(_row()["type"]) == set(agg["type"])


def test_aggregate_sums_rows_and_engine_failures():
    r1 = _row(failed=[EngineFailure("arxiv", "timeout")], pin=3.0, pout=15.0)
    r2 = _row(failed=[EngineFailure("arxiv", "http 500"), EngineFailure("github", "x")])
    agg = aggregate([r1, r2])
    assert agg["jobs"] == 2
    assert agg["results"] == 12
    assert agg["type"]["rule"] == 4
    assert agg["type"]["jev"] == 6
    assert agg["type"]["unclassified"] == 2
    assert agg["type"]["rule_rate"] == pytest.approx(4 / 12)
    assert agg["jev_tokens"] == {"input": 2000, "output": 400, "estimated_input": 1800}
    assert agg["cost_usd"] == pytest.approx(r1["cost_usd"])
    assert agg["failed_engines"] == {"arxiv": 2, "github": 1}


def test_aggregate_since_drops_older_rows():
    old = _row(started_at=datetime(2026, 10, 1, 12, 0, tzinfo=JST))
    new = _row(started_at=datetime(2026, 11, 2, 12, 0, tzinfo=JST))
    agg = aggregate([old, new], since=date(2026, 11, 1))
    assert agg["jobs"] == 1
    assert agg["results"] == 6
    assert aggregate([old, new])["jobs"] == 2


def test_aggregate_since_is_a_local_date(monkeypatch):
    stored = "2026-11-01T23:30:00+00:00"  # UTC で保存された時刻
    row = {"started_at": stored, "n_results": 1}
    # 機械の時間帯に依らず、同じ時刻の astimezone() の日付を基準にする
    local_date = datetime.fromisoformat(stored).astimezone().date()
    assert aggregate([row], since=local_date)["jobs"] == 1
    assert aggregate([row], since=local_date + timedelta(days=1))["jobs"] == 0

    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
        try:
            # JST では 11-02 08:30 なので、11-02 を since にしても含まれる
            assert aggregate([row], since=date(2026, 11, 2))["jobs"] == 1
            assert aggregate([row], since=date(2026, 11, 3))["jobs"] == 0
        finally:
            monkeypatch.undo()
            time.tzset()


def test_aggregate_empty():
    agg = aggregate([])
    assert agg["jobs"] == 0
    assert agg["type"]["rule_rate"] == 0.0
    assert agg["cost_usd"] is None
    assert agg["jev_tokens"]["input"] is None


def test_is_japanese():
    assert is_japanese("サーバー") is True
    assert is_japanese("server") is False
    assert is_japanese("検索エンジン") is True
    assert is_japanese("ひらがな mixed") is True
    assert is_japanese("") is False


def _log_row(job_id: str, n: int, title: str, query: str = "q", **extra) -> dict:
    row = {
        "schema": 1,
        "job_id": job_id,
        "query": query,
        "query_norm": {"ja": query, "en": None},
        "retrieved_at": T0.isoformat(),
        "topic": None,
        "id": f"r{n:02d}",
        "url": f"https://example.com/{job_id}/{n}",
        "title": title,
        "snippet": "s",
        "engine": "github",
        "engines": ["github", "arxiv"],
        "matched": ["ja", "en"],
        "published_date": None,
        "type": {"value": "repo", "by": "rule", "confidence": 1.0},
        "genre": {"value": "tech", "by": "jev", "confidence": 0.8},
        "relevant": {"value": True, "by": "jev", "confidence": 0.9},
    }
    row.update(extra)
    return row


def test_rows_to_results_groups_by_job_and_query():
    rows = [
        _log_row("J1", 1, "a"),
        _log_row("J1", 2, "b"),
        _log_row("J2", 1, "c"),
        _log_row("J1", 3, "d", query="other"),
    ]
    groups = rows_to_results(rows)
    assert set(groups) == {("J1", "q"), ("J2", "q"), ("J1", "other")}
    assert [r.title for r in groups[("J1", "q")]] == ["a", "b"]
    first = groups[("J1", "q")][0]
    assert isinstance(first, SearchResult)
    assert first.retrieved_at == T0
    assert first.engines == ("github", "arxiv")
    assert first.matched == (Lang.JA, Lang.EN)


class FakeClassifier:
    """batch_size によって答えを変える fake。

    batch_size が 1 のとき: 全て type=blog、genre=tech、relevant=True。
    それ以外: 偶数番号のタイトルは type=blog、奇数は type=news。genre は tech。
    relevant は batch のときは全て未分類。
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def classify(self, query, results, rule_types):
        items = []
        for i, res in enumerate(results):
            if self.batch_size == 1:
                type_value = "blog"
                relevant = _j(True)
            else:
                type_value = "blog" if i % 2 == 0 else "news"
                relevant = _j(None, confidence=0.3)
            items.append(
                ClassifiedResult(
                    id=f"r{i + 1:02d}",
                    result=res,
                    type=_j(type_value),
                    genre=_j("tech"),
                    relevant=relevant,
                )
            )
        return items, _report()


def test_batch_vs_single_splits_ja_and_en():
    rows = [
        _log_row("J1", 1, "サーバーの話"),  # index 0 ja: batch=blog single=blog 一致
        _log_row("J1", 2, "検索エンジン"),  # index 1 ja: batch=news single=blog 不一致
        _log_row("J1", 3, "english one"),  # index 2 en: 一致
        _log_row("J1", 4, "english two"),  # index 3 en: 不一致
        _log_row("J1", 5, "english three"),  # index 4 en: 一致
    ]
    sizes = []

    def make(n):
        sizes.append(n)
        return FakeClassifier(n)

    result = batch_vs_single(rows, make, batch_size=25, rule_types_for=lambda rs: {})
    assert sorted(sizes) == [1, 25]
    ja, en = result["type"]["ja"], result["type"]["en"]
    assert (ja["paired"], ja["agreed"]) == (2, 1)
    assert ja["agreement_rate"] == pytest.approx(0.5)
    assert (en["paired"], en["agreed"]) == (3, 2)
    assert en["agreement_rate"] == pytest.approx(2 / 3)
    assert result["genre"]["ja"]["agreement_rate"] == 1.0
    assert result["genre"]["en"]["agreement_rate"] == 1.0
    # relevant は batch が全て未分類、single は分類済み: 片方だけ未分類
    assert result["relevant"]["ja"]["paired"] == 0
    assert result["relevant"]["ja"]["only_one_unclassified"] == 2
    assert result["relevant"]["en"]["only_one_unclassified"] == 3
    assert result["relevant"]["en"]["agreement_rate"] == 0.0


class RuleAwareClassifier:
    """rule_types のある結果は by=RULE でその type を返す。残りは Jev で、batch_size で答えが変わる。"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def classify(self, query, results, rule_types):
        items = []
        for i, res in enumerate(results):
            if i in rule_types:
                type_ = Judgment(value=rule_types[i].value, by=JudgedBy.RULE, confidence=1.0)
            else:
                type_ = _j("news" if self.batch_size != 1 else "blog")
            items.append(
                ClassifiedResult(
                    id=f"r{i + 1:02d}",
                    result=res,
                    type=type_,
                    genre=_j("tech"),
                    relevant=_j(True),
                )
            )
        return items, _report()


def test_batch_vs_single_excludes_rule_decided_types():
    rows = [_log_row("J1", n, f"english {n}") for n in range(1, 6)]
    result = batch_vs_single(
        rows,
        RuleAwareClassifier,
        batch_size=25,
        rule_types_for=lambda rs: {i: ResultType.REPO for i in range(4)},
    )
    en = result["type"]["en"]
    assert en["paired"] == 1
    assert en["agreed"] == 0
    assert en["agreement_rate"] == 0.0
    assert en["rule_decided"] == 4
    assert result["type"]["ja"]["rule_decided"] == 0
    # genre は変わらず全件が対象
    assert result["genre"]["en"]["paired"] == 5
    assert "rule_decided" not in result["genre"]["en"]


def test_batch_vs_single_passes_normalized_japanese_query():
    seen = []

    class Spy:
        def classify(self, query, results, rule_types):
            seen.append(query)
            return [], _report()

    rows = [
        _log_row("J1", 1, "a", query="ユーザビリティ", query_norm={"ja": "UX", "en": None}),
        _log_row("J1", 2, "b", query="ユーザビリティ", query_norm={"ja": "UX", "en": None}),
        _log_row("J2", 1, "c", query="未正規化", query_norm=None),
        _log_row("J3", 1, "d", query="ja なし", query_norm={"ja": None, "en": "x"}),
        _log_row("J4", 1, "e", query="キーなし", query_norm={"en": "x"}),
    ]
    del rows[2]["query_norm"]
    batch_vs_single(rows, lambda n: Spy(), batch_size=25, rule_types_for=lambda rs: {})
    assert sorted(seen) == sorted(["UX", "UX", "未正規化", "未正規化", "ja なし", "ja なし", "キーなし", "キーなし"])
    assert set(rows_to_results(rows)) == {
        ("J1", "UX"),
        ("J2", "未正規化"),
        ("J3", "ja なし"),
        ("J4", "キーなし"),
    }


def test_batch_vs_single_passes_rule_types():
    seen = []

    class Spy:
        def __init__(self, n):
            self.n = n

        def classify(self, query, results, rule_types):
            seen.append((self.n, query, dict(rule_types), len(results)))
            return [], _report()

    rows = [_log_row("J1", 1, "a"), _log_row("J1", 2, "b")]
    batch_vs_single(
        rows,
        lambda n: Spy(n),
        batch_size=25,
        rule_types_for=lambda rs: {0: ResultType.REPO},
    )
    assert sorted(seen) == [
        (1, "q", {0: ResultType.REPO}, 2),
        (25, "q", {0: ResultType.REPO}, 2),
    ]


def _many_rows(count: int = 30) -> list[dict]:
    return [
        _log_row("J1", i, "日本語タイトル" if i % 2 else "english title") for i in range(1, count + 1)
    ]


def test_sample_csv_columns_count_and_reproducible():
    rows = _many_rows()
    text = sample_csv(rows, 10, seed=3)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert list(parsed[0].keys()) == list(CSV_COLUMNS)
    assert text.splitlines()[0] == (
        "job_id,id,query,url,title,snippet,lang,type,genre,relevant,"
        "label_type,label_genre,label_relevant"
    )
    assert len(parsed) == 10
    assert all(r["label_type"] == r["label_genre"] == r["label_relevant"] == "" for r in parsed)
    assert len({r["url"] for r in parsed}) == 10
    assert sample_csv(rows, 10, seed=3) == text
    assert sample_csv(rows, 10, seed=4) != text
    by_id = {r["id"]: r for r in parsed}
    assert {r["lang"] for r in parsed} <= {"ja", "en"}
    assert all(
        (r["lang"] == "ja") == (int(i[1:]) % 2 == 1) for i, r in by_id.items()
    )
    assert parsed[0]["type"] == "repo"
    assert parsed[0]["relevant"] == "true"


def test_sample_csv_n_larger_than_rows_and_unclassified_cell():
    rows = [_log_row("J1", 1, "a", relevant={"value": None, "by": "jev", "confidence": 0.3})]
    parsed = list(csv.DictReader(io.StringIO(sample_csv(rows, 50))))
    assert len(parsed) == 1
    assert parsed[0]["relevant"] == ""


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
def test_sample_csv_neutralizes_formula_cells(prefix):
    dangerous = f'{prefix}HYPERLINK("http://x","y")'
    rows = [
        _log_row(
            "J1",
            1,
            dangerous,
            query=dangerous,
            url=dangerous,
            snippet=dangerous,
        )
    ]
    parsed = list(csv.DictReader(io.StringIO(sample_csv(rows, 1)), strict=True))
    assert len(parsed) == 1
    for col in ("query", "url", "title", "snippet"):
        assert parsed[0][col] == "'" + dangerous


def test_sample_csv_formula_title_and_safe_text_untouched():
    rows = [_log_row("J1", 1, "=1+1", query="plain")]
    text = sample_csv(rows, 1)
    assert ",'=1+1," in text
    assert ",=1+1," not in text
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["query"] == "plain"
    assert parsed[0]["snippet"] == "s"
    # 途中にある記号は変えない
    mid = list(csv.DictReader(io.StringIO(sample_csv([_log_row("J1", 1, "a=b -c")], 1))))
    assert mid[0]["title"] == "a=b -c"


def test_accuracy_from_labels_ignores_escaped_text_columns():
    rows = [_log_row("J1", 1, "=1+1")]
    parsed = list(csv.DictReader(io.StringIO(sample_csv(rows, 1))))
    parsed[0]["label_type"] = "repo"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(parsed[0])
    acc = accuracy_from_labels(out.getvalue())
    assert acc["type"]["en"] == {"correct": 1, "total": 1, "accuracy": 1.0, "unclassified": 0}


def test_accuracy_from_labels_by_language():
    text = sample_csv(_many_rows(6), 6, seed=0)
    parsed = list(csv.DictReader(io.StringIO(text)))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    ja_seen = en_seen = 0
    for r in parsed:
        if r["lang"] == "ja":
            ja_seen += 1
            # 1件目だけ type を正解、2件目以降は誤り。genre は全部正解。relevant は未記入
            r["label_type"] = "repo" if ja_seen == 1 else "paper"
            r["label_genre"] = "tech"
        else:
            en_seen += 1
            r["label_type"] = "repo"
            r["label_relevant"] = "TRUE" if en_seen == 1 else "no"
        writer.writerow(r)
    acc = accuracy_from_labels(out.getvalue())
    assert ja_seen == 3 and en_seen == 3
    assert acc["type"]["ja"] == {
        "correct": 1,
        "total": 3,
        "accuracy": pytest.approx(1 / 3),
        "unclassified": 0,
    }
    assert acc["type"]["en"] == {"correct": 3, "total": 3, "accuracy": 1.0, "unclassified": 0}
    assert acc["genre"]["ja"]["accuracy"] == 1.0
    assert acc["genre"]["en"]["total"] == 0
    assert acc["genre"]["en"]["accuracy"] == 0.0
    assert acc["relevant"]["ja"]["total"] == 0
    # relevant: 機械は true。TRUE は正解、no は不正解
    assert acc["relevant"]["en"] == {
        "correct": 1,
        "total": 3,
        "accuracy": pytest.approx(1 / 3),
        "unclassified": 0,
    }


def test_accuracy_from_labels_unclassified_prediction_counts_as_wrong():
    rows = [_log_row("J1", 1, "a", genre={"value": None, "by": "jev", "confidence": 0.2})]
    parsed = list(csv.DictReader(io.StringIO(sample_csv(rows, 1))))
    parsed[0]["label_genre"] = "tech"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(parsed[0])
    acc = accuracy_from_labels(out.getvalue())
    # 未分類は不正解として total に数え、unclassified にも別に数える
    assert acc["genre"]["en"] == {"correct": 0, "total": 1, "accuracy": 0.0, "unclassified": 1}
    assert acc["type"]["en"]["unclassified"] == 0
    assert acc["genre"]["ja"]["unclassified"] == 0
