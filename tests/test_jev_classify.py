"""Jev によるまとめて分類（jev_classify.py）のテスト。

質問の組み立て（build_requests）と、答えの写し（map_answers）を分けて確かめる。
Jev 本体（JevClient.ask）は、受け取った state と questions を記録して固定データを返す fake に差し替える。
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jevsearch.jev_classify import (
    JevClassifier,
    build_requests,
    estimate_tokens,
    map_answers,
    preview_request,
    result_id,
)
from jevsearch.jev_client import JevError, JevResult
from jevsearch.models import (
    JEV_MODEL,
    Genre,
    JudgedBy,
    Judgment,
    Lang,
    ResultType,
    SearchResult,
)

FIXTURE = Path(__file__).parent / "fixtures" / "jev_answers.json"
FIXTURE_ANSWERS = json.loads(FIXTURE.read_text(encoding="utf-8"))["answers"]
JST = timezone(timedelta(hours=9))
THRESHOLD = 0.55
BIG = 10**9
UNCLASSIFIED = Judgment(None, JudgedBy.JEV, None)


def _result(n: int = 1, *, url: str | None = None, title: str | None = None) -> SearchResult:
    return SearchResult(
        url=url or f"https://site{n}.example.org/page/{n}",
        title=title or f"title {n}",
        snippet=f"snippet {n}",
        engine="github",
        published_date="2026-11-01",
        retrieved_at=datetime(2026, 11, 3, 21, 5, 12, tzinfo=JST),
        engines=("github", "arxiv"),
        matched=(Lang.JA, Lang.EN),
    )


def _results(count: int) -> list[SearchResult]:
    return [_result(n) for n in range(1, count + 1)]


class FakeClock:
    """呼ぶたびに 100 ミリ秒ずつ進む時計。"""

    def __init__(self) -> None:
        self._now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=JST)

    def now(self) -> datetime:
        current = self._now
        self._now += timedelta(milliseconds=100)
        return current


class FakeAsk:
    """受け取った state と questions を記録し、固定データの答えを返す。"""

    def __init__(self, answers=None, *, usage=None, error: Exception | None = None) -> None:
        self.answers = FIXTURE_ANSWERS if answers is None else answers
        self.usage = usage
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, state: str, questions: dict) -> JevResult:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return JevResult(answers=self.answers, usage=self.usage, duration_ms=42)


def _classifier(ask, *, batch_size: int = 25, max_request_tokens: int = BIG) -> JevClassifier:
    return JevClassifier(
        ask,
        model=JEV_MODEL,
        batch_size=batch_size,
        max_request_tokens=max_request_tokens,
        threshold=THRESHOLD,
        clock=FakeClock(),
    )


def _build(results, rule_types=None, *, batch_size: int = 25, max_request_tokens: int = BIG):
    return build_requests(
        "検索語",
        results,
        rule_types or {},
        batch_size=batch_size,
        max_request_tokens=max_request_tokens,
    )


def _map(answers, results=None, rule_types=None):
    results = results or _results(1)
    request = _build(results, rule_types)[0]
    return map_answers(request, answers, results, rule_types or {}, threshold=THRESHOLD)


def test_state_has_only_the_seven_items():
    """state は検索語と、結果ごとの6項目だけを持つ。

    never: 検索語・タイトル・URL・抜粋と、内部のID・ドメイン名・エンジン名以外は Jev に送らない
    """
    request = _build(_results(2))[0]
    state = json.loads(request.state)
    assert set(state) == {"query", "results"}
    assert state["query"] == "検索語"
    for item in state["results"]:
        assert set(item) == {"id", "title", "url", "domain", "snippet", "engine"}
    assert state["results"][0] == {
        "id": "r01",
        "title": "title 1",
        "url": "https://site1.example.org/page/1",
        "domain": "site1.example.org",
        "snippet": "snippet 1",
        "engine": "github",
    }
    assert "2026-11-01" not in request.state
    assert "arxiv" not in request.state


def test_api_key_is_never_in_state_or_questions(monkeypatch):
    """環境変数のキーは state にも質問にも入らない。

    never: API キーを、TypeSafe への認証ヘッダー以外に送らない
    """
    sentinel = "sk-test-SENTINEL-0000"
    monkeypatch.setenv("TYPESAFE_API_KEY", sentinel)
    ask = FakeAsk()
    _classifier(ask).classify("検索語", _results(3), {})
    requests = _build(_results(3)) + [preview_request("検索語", batch_size=25, max_request_tokens=BIG)]
    for request in requests:
        assert sentinel not in request.state
        assert sentinel not in json.dumps(request.questions, ensure_ascii=False)
    for state, questions in ask.calls:
        assert sentinel not in state
        assert sentinel not in json.dumps(questions, ensure_ascii=False)


def test_question_ids_match_results_positions():
    """質問 id はジョブ通しで、instructions は results[k] と id の両方を持つ。"""
    requests = _build(_results(26))
    assert len(requests) == 2
    second = requests[1]
    assert second.ids == ("r26",)
    assert second.positions == (25,)
    assert json.loads(second.state)["results"][0]["id"] == "r26"
    for key in ("type_r26", "genre_r26", "relevant_r26"):
        instructions = second.questions[key]["instructions"]
        assert "results[0]" in instructions
        assert "r26" in instructions
    first = requests[0]
    assert "results[24]" in first.questions["type_r25"]["instructions"]
    assert "r25" in first.questions["type_r25"]["instructions"]


def test_question_shapes():
    """質問の形は契約どおり（choice は英語の criteria、noul は 1 / 0）。"""
    questions = _build(_results(1))[0].questions
    assert set(questions) == {"type_r01", "genre_r01", "relevant_r01", "query_genre"}
    assert set(questions["type_r01"]["criteria"]) == {t.value for t in ResultType}
    assert set(questions["genre_r01"]["criteria"]) == {g.value for g in Genre}
    assert set(questions["query_genre"]["criteria"]) == {g.value for g in Genre}
    assert questions["query_genre"]["instructions"] == "What genre is the query itself?"
    assert questions["type_r01"]["type"] == "choice"
    assert questions["relevant_r01"] == {
        "type": "noul",
        "instructions": "Does results[0] (id r01) match the intent of the query?",
        "criteria": {
            "1": "It matches the intent of the query.",
            "0": "It does not match the intent of the query.",
        },
    }


def test_requests_split_by_batch_size_and_token_limit():
    """26件で2リクエスト（25+1）、25件で1リクエスト。トークン上限を小さくするとさらに分かれる。"""
    assert [len(r.ids) for r in _build(_results(26))] == [25, 1]
    assert [len(r.ids) for r in _build(_results(25))] == [25]
    assert _build([]) == []
    full = _build(_results(25))[0]
    limit = full.estimated_tokens // 2
    split = _build(_results(25), max_request_tokens=limit)
    assert len(split) > 1
    assert [rid for r in split for rid in r.ids] == [f"r{n:02d}" for n in range(1, 26)]
    assert [p for r in split for p in r.positions] == list(range(25))
    for request in split:
        assert len(request.ids) == 1 or request.estimated_tokens <= limit
    one_each = _build(_results(5), max_request_tokens=1)
    assert [len(r.ids) for r in one_each] == [1] * 5


def test_estimated_tokens_counts_state_and_questions():
    """見積もりは state と質問の JSON の UTF-8 バイト数 ÷ 3 の切り上げ。"""
    request = _build(_results(2))[0]
    text = request.state + json.dumps(request.questions, ensure_ascii=False)
    assert request.estimated_tokens == -(-len(text.encode("utf-8")) // 3)
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("あ") == 1
    assert estimate_tokens("ああ") == 2


def test_result_id_width():
    assert result_id(0, 5) == "r01"
    assert result_id(8, 99) == "r09"
    assert result_id(98, 99) == "r99"
    assert result_id(0, 100) == "r001"
    assert result_id(99, 100) == "r100"
    assert result_id(0, 1000) == "r0001"


def test_rule_decided_results_get_no_type_question():
    """規則で決まった結果には type の質問を出さない。

    遷移: ドメインで分かる種類を付け終えた
    """
    request = _build(_results(2), {0: ResultType.REPO})[0]
    assert "type_r01" not in request.questions
    assert "genre_r01" in request.questions
    assert "relevant_r01" in request.questions
    assert "type_r02" in request.questions


def test_rule_type_is_not_overwritten_by_answer():
    """答えに type_r01 が混ざっても、規則の種類のまま。

    never: ドメイン規則で決まった種類を Jev の答えで上書きしない
    """
    answers = {**FIXTURE_ANSWERS, "type_r01": {"choice": "blog", "confidence": 0.99}}
    classified, _ = _map(answers, _results(1), {0: ResultType.REPO})
    assert classified[0].type == Judgment(ResultType.REPO, JudgedBy.RULE, 1.0)

    ask = FakeAsk(answers)
    classified, _ = _classifier(ask).classify("検索語", _results(3), {0: ResultType.REPO})
    assert classified[0].type == Judgment("repo", JudgedBy.RULE, 1.0)
    assert classified[1].type.by == JudgedBy.JEV


def test_choice_threshold_boundary():
    """確信度 0.54 は未分類（確信度は残す）、0.55 は採用。

    never: 確信度がしきい値より低い判定を「分類済み」として保存しない
    """
    classified, query_genre = _map(FIXTURE_ANSWERS, _results(3))
    # type: r01 0.9 / r02 0.55 / r03 0.54
    assert classified[0].type == Judgment("blog", JudgedBy.JEV, 0.9)
    assert classified[1].type == Judgment("news", JudgedBy.JEV, 0.55)
    assert classified[2].type == Judgment(None, JudgedBy.JEV, 0.54)
    # genre: r01 0.9 / r02 0.54 / r03 0.55
    assert classified[0].genre == Judgment("tech", JudgedBy.JEV, 0.9)
    assert classified[1].genre == Judgment(None, JudgedBy.JEV, 0.54)
    assert classified[2].genre == Judgment("paper", JudgedBy.JEV, 0.55)
    assert query_genre == Judgment("tech", JudgedBy.JEV, 0.9)
    assert [c.id for c in classified] == ["r01", "r02", "r03"]


def test_noul_confidence_is_max_of_p_and_complement():
    """noul は確信度 max(p, 1-p)。p=0.52 は未分類、p=0.1 は False（確信度 0.9）。

    never: 確信度がしきい値より低い判定を「分類済み」として保存しない
    """
    classified, _ = _map(FIXTURE_ANSWERS, _results(3))
    assert classified[0].relevant == Judgment(True, JudgedBy.JEV, 0.9)
    assert classified[1].relevant.value is None
    assert classified[1].relevant.confidence == pytest.approx(0.52)
    assert classified[2].relevant.value is False
    assert classified[2].relevant.confidence == pytest.approx(0.9)
    # "1" が無ければ probability を p とみる。
    only_probability, _ = _map({"relevant_r01": {"probability": 0.8}})
    assert only_probability[0].relevant == Judgment(True, JudgedBy.JEV, 0.8)


def test_missing_answers_are_unclassified():
    """答えが欠けた質問は value None・confidence None。

    never: 確信度がしきい値より低い判定を「分類済み」として保存しない
    """
    classified, query_genre = _map({})
    assert classified[0].type == UNCLASSIFIED
    assert classified[0].genre == UNCLASSIFIED
    assert classified[0].relevant == UNCLASSIFIED
    assert query_genre == UNCLASSIFIED


def test_choice_outside_options_is_unclassified_but_keeps_confidence():
    answers = {"genre_r01": {"choice": "cooking", "confidence": 0.99}}
    classified, _ = _map(answers)
    assert classified[0].genre == Judgment(None, JudgedBy.JEV, 0.99)


def test_jev_error_keeps_all_results_unclassified():
    """Jev が失敗しても結果は減らず、全部未分類で返る。

    遷移: Jev が使えない、または上限を超えたので未分類として残した
    never: 一部のエンジンや Jev が失敗したことを理由に、取れた結果を捨てない
    """
    ask = FakeAsk(error=JevError(code="http", status=529))
    results = _results(4)
    classified, report = _classifier(ask).classify("検索語", results, {1: ResultType.DOCS})
    assert len(classified) == 4
    assert [c.result for c in classified] == results
    for c in classified:
        assert c.genre == UNCLASSIFIED
        assert c.relevant == UNCLASSIFIED
    assert classified[0].type == UNCLASSIFIED
    assert classified[1].type == Judgment(ResultType.DOCS, JudgedBy.RULE, 1.0)
    assert report.failure == "http 529"
    assert report.query_genre == UNCLASSIFIED
    assert report.usage[0].error == "http 529"
    assert report.usage[0].input_tokens is None
    assert report.usage[0].duration_ms == 100


def test_failed_request_does_not_stop_next_requests():
    """失敗したリクエストの結果だけ未分類にして、次のリクエストは進む。"""

    class FailFirst(FakeAsk):
        def __call__(self, state, questions):
            self.calls.append((state, questions))
            if len(self.calls) == 1:
                raise JevError(code="daily_limit")
            return JevResult(answers=FIXTURE_ANSWERS, usage=None, duration_ms=1)

    ask = FailFirst()
    classified, report = _classifier(ask, batch_size=2).classify("検索語", _results(4), {})
    assert len(classified) == 4
    assert len(ask.calls) == 2
    assert classified[0].genre == UNCLASSIFIED
    assert classified[1].genre == UNCLASSIFIED
    # 2つ目のリクエストは r03・r04 で、固定データに答えはないので未分類（ただし呼ばれている）。
    assert report.failure == "daily_limit"
    assert [u.error for u in report.usage] == ["daily_limit", None]


def test_failure_codes_are_deduplicated_and_joined():
    ask = FakeAsk(error=JevError(code="http", status=529))
    _, report = _classifier(ask, batch_size=1).classify("検索語", _results(3), {})
    assert report.failure == "http 529"
    assert len(report.usage) == 3


def test_usage_records_model_and_tokens():
    """JevUsage にモデル名と応答の usage のトークン数が入る。usage がなければ None。"""
    ask = FakeAsk(usage={"input_tokens": 1234, "output_tokens": 56})
    classified, report = _classifier(ask).classify("検索語", _results(3), {})
    assert len(classified) == 3
    usage = report.usage[0]
    assert usage.model == "jev-1.14.0"
    assert usage.input_tokens == 1234
    assert usage.output_tokens == 56
    assert usage.estimated_input_tokens == _build(_results(3))[0].estimated_tokens
    assert usage.duration_ms == 42
    assert usage.n_results == 3
    assert usage.n_questions == 3 * 3 + 1
    assert usage.error is None
    assert report.failure is None

    _, report = _classifier(FakeAsk(usage=None)).classify("検索語", _results(3), {})
    assert report.usage[0].input_tokens is None
    assert report.usage[0].output_tokens is None


def test_query_genre_is_the_most_confident_one():
    class PerRequest(FakeAsk):
        def __call__(self, state, questions):
            self.calls.append((state, questions))
            confidence = 0.6 if len(self.calls) == 1 else 0.8
            choice = "news" if len(self.calls) == 1 else "paper"
            return JevResult(
                answers={"query_genre": {"choice": choice, "confidence": confidence}},
                usage=None,
                duration_ms=1,
            )

    _, report = _classifier(PerRequest(), batch_size=1).classify("検索語", _results(2), {})
    assert report.query_genre == Judgment("paper", JudgedBy.JEV, 0.8)


def test_broken_answers_do_not_raise():
    """壊れた答えは例外にせず、その質問だけ未分類にする。"""
    answers = {
        **FIXTURE_ANSWERS,
        "type_r01": {"choice": "blog", "confidence": "high"},
        "genre_r01": "tech",
        "relevant_r01": {"probabilities": "yes"},
        "type_r02": {"choice": "news", "confidence": 0.9, "probabilities": [1]},
        "relevant_r02": {"probabilities": {"0": 0.3}},
        "genre_r03": {"choice": "paper", "confidence": True},
        "relevant_r03": None,
    }
    classified, _ = _map(answers, _results(3))
    assert classified[0].type == UNCLASSIFIED
    assert classified[0].genre == UNCLASSIFIED
    assert classified[0].relevant == UNCLASSIFIED
    assert classified[1].type == UNCLASSIFIED
    assert classified[1].relevant == UNCLASSIFIED
    assert classified[2].genre == UNCLASSIFIED
    assert classified[2].relevant == UNCLASSIFIED
    # 壊れていない答えは影響を受けない。
    assert classified[1].genre == Judgment(None, JudgedBy.JEV, 0.54)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.5, -0.2])
def test_out_of_range_choice_confidence_is_unclassified(bad):
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない。

    NaN・inf・範囲外の確信度は壊れた答えで、confidence None の未分類にする。
    """
    answers = {
        **FIXTURE_ANSWERS,
        "type_r01": {"choice": "blog", "confidence": bad},
        "genre_r01": {"choice": "tech", "confidence": bad},
        "query_genre": {"choice": "tech", "confidence": bad},
    }
    classified, query_genre = _map(answers, _results(3))
    assert classified[0].type == UNCLASSIFIED
    assert classified[0].genre == UNCLASSIFIED
    assert query_genre == UNCLASSIFIED


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5])
def test_out_of_range_noul_probability_is_unclassified(bad):
    answers = {**FIXTURE_ANSWERS, "relevant_r01": {"probabilities": {"1": bad}}}
    classified, _ = _map(answers, _results(3))
    assert classified[0].relevant == UNCLASSIFIED


def test_nan_query_genre_does_not_block_a_valid_one_in_a_later_request():
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない。"""

    class PerRequest(FakeAsk):
        def __call__(self, state, questions):
            self.calls.append((state, questions))
            if len(self.calls) == 1:
                answer = {"choice": "news", "confidence": float("nan")}
            else:
                answer = {"choice": "paper", "confidence": 0.7}
            return JevResult(answers={"query_genre": answer}, usage=None, duration_ms=1)

    _, report = _classifier(PerRequest(), batch_size=1).classify("検索語", _results(2), {})
    assert report.query_genre == Judgment("paper", JudgedBy.JEV, 0.7)


def test_classified_query_genre_wins_over_a_more_confident_unclassified_one():
    """never: 確信度がしきい値より低い判定を「分類済み」として保存しない。"""
    from jevsearch.jev_classify import _best_genre

    tech = Judgment(value="tech", by=JudgedBy.JEV, confidence=0.8)
    unclassified = Judgment(value=None, by=JudgedBy.JEV, confidence=0.95)
    assert _best_genre([tech, unclassified]) == tech
    assert _best_genre([unclassified, tech]) == tech
    # 分類済みがなければ、確信度がいちばん高い未分類（今までの規則）
    lower = Judgment(value=None, by=JudgedBy.JEV, confidence=0.3)
    assert _best_genre([lower, unclassified]) == unclassified
    assert _best_genre([]) == UNCLASSIFIED


def test_unexpected_exception_does_not_stop_the_job():
    """JevError 以外の例外でも全結果が返り、failure は unexpected:<クラス名>。"""
    ask = FakeAsk(error=ValueError("boom"))
    classified, report = _classifier(ask).classify("検索語", _results(3), {})
    assert len(classified) == 3
    assert all(c.genre == UNCLASSIFIED for c in classified)
    assert report.failure == "unexpected:ValueError"
    assert report.usage[0].error == "unexpected:ValueError"


def test_error_in_mapping_is_also_unexpected(monkeypatch):
    import jevsearch.jev_classify as module

    def boom(*args, **kwargs):
        raise KeyError("x")

    monkeypatch.setattr(module, "map_answers", boom)
    classified, report = _classifier(FakeAsk()).classify("検索語", _results(2), {})
    assert len(classified) == 2
    assert report.failure == "unexpected:KeyError"


def test_japanese_is_not_escaped_in_state():
    """日本語の title は \\u エスケープされずに state に入る。"""
    request = _build([_result(1, title="日本語のタイトル")])[0]
    assert "日本語のタイトル" in request.state
    assert "\\u" not in request.state
    assert "検索語" in request.state


def test_preview_request_uses_a_sample_result():
    request = preview_request("検索語", batch_size=25, max_request_tokens=BIG)
    state = json.loads(request.state)
    assert state["query"] == "検索語"
    assert state["results"] == [
        {
            "id": "r01",
            "title": "<title>",
            "url": "https://example.com/",
            "domain": "example.com",
            "snippet": "<snippet>",
            "engine": "<engine>",
        }
    ]
    assert request.ids == ("r01",)
    assert "type_r01" in request.questions
    assert "query_genre" in request.questions


def test_empty_results_make_no_calls():
    ask = FakeAsk()
    classified, report = _classifier(ask).classify("検索語", [], {})
    assert classified == []
    assert ask.calls == []
    assert report.usage == ()
    assert report.failure is None
    assert report.query_genre == UNCLASSIFIED
