"""Jev によるまとめて分類。

質問を組み立てる部分（build_requests）と、答えを判定に写す部分（map_answers）を分け、
Jev 本体（JevClient.ask）は差し替えられる形で受け取る。
Jev が失敗しても結果は捨てず、未分類として返す。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from jevsearch.jev_client import JevError, JevResult
from jevsearch.models import (
    ClassifiedResult,
    Genre,
    JevReport,
    JevUsage,
    JudgedBy,
    Judgment,
    ResultType,
    SearchResult,
)
from jevsearch.ports import Clock
from jevsearch.rules import domain_of

_QUERY_GENRE_ID = "query_genre"
_BYTES_PER_TOKEN = 3
_MIN_ID_WIDTH = 2

_TYPE_CRITERIA: dict[str, str] = {
    ResultType.PAPER: "A research paper or preprint.",
    ResultType.BLOG: "A personal or company blog post, or an article.",
    ResultType.NEWS: "A news report from a news outlet.",
    ResultType.SNS: "A post on a social network or a discussion forum.",
    ResultType.DOCS: "Official documentation, a reference, or a tutorial.",
    ResultType.REPO: "A source code repository or a package page.",
    ResultType.OTHER: "None of the above.",
}

_GENRE_CRITERIA: dict[str, str] = {
    Genre.PAPER: "It is about academic research.",
    Genre.TECH: "It is about software or technology.",
    Genre.NEWS: "It is about recent news.",
    Genre.SNS: "It is about posts and discussions on social networks.",
    Genre.BLOG: "It is about personal or company blog posts.",
    Genre.GENERAL: "It is about something general that fits none of the above.",
}

_RELEVANT_CRITERIA: dict[str, str] = {
    "1": "It matches the intent of the query.",
    "0": "It does not match the intent of the query.",
}


@dataclass(frozen=True)
class JevRequest:
    """Jev への1回のリクエスト。positions は results の位置、ids はジョブ通しの id。"""

    state: str
    questions: dict
    ids: tuple[str, ...]
    positions: tuple[int, ...]
    estimated_tokens: int


def result_id(position: int, total: int) -> str:
    """位置（0 始まり）から id を作る。`r` + 位置+1 を2桁以上でゼロ詰め。"""
    width = max(_MIN_ID_WIDTH, len(str(total)))
    return f"r{position + 1:0{width}d}"


def estimate_tokens(text: str) -> int:
    """トークン数の見積もり。UTF-8 のバイト数 ÷ 3 の切り上げ。"""
    return -(-len(text.encode("utf-8")) // _BYTES_PER_TOKEN)


def _choice_question(instructions: str, criteria: Mapping[str, str]) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {str(key): value for key, value in criteria.items()},
    }


def _build_one(
    query: str,
    results: Sequence[SearchResult],
    rule_types: Mapping[int, ResultType],
    positions: Sequence[int],
    ids: Sequence[str],
) -> JevRequest:
    """results の positions の結果から、1つのリクエストを作る。"""
    state_results = []
    questions: dict = {}
    for k, (position, rid) in enumerate(zip(positions, ids, strict=True)):
        result = results[position]
        state_results.append(
            {
                "id": rid,
                "title": result.title,
                "url": result.url,
                "domain": domain_of(result.url),
                "snippet": result.snippet,
                "engine": result.engine,
            }
        )
        if position not in rule_types:
            questions[f"type_{rid}"] = _choice_question(
                f"What kind of content is results[{k}] (id {rid})?", _TYPE_CRITERIA
            )
        questions[f"genre_{rid}"] = _choice_question(
            f"What genre is results[{k}] (id {rid}) about?", _GENRE_CRITERIA
        )
        questions[f"relevant_{rid}"] = {
            "type": "noul",
            "instructions": f"Does results[{k}] (id {rid}) match the intent of the query?",
            "criteria": dict(_RELEVANT_CRITERIA),
        }
    questions[_QUERY_GENRE_ID] = _choice_question(
        "What genre is the query itself?", _GENRE_CRITERIA
    )
    state = json.dumps({"query": query, "results": state_results}, ensure_ascii=False)
    estimated = estimate_tokens(state + json.dumps(questions, ensure_ascii=False))
    return JevRequest(
        state=state,
        questions=questions,
        ids=tuple(ids),
        positions=tuple(positions),
        estimated_tokens=estimated,
    )


def _split_to_fit(
    query: str,
    results: Sequence[SearchResult],
    rule_types: Mapping[int, ResultType],
    positions: Sequence[int],
    all_ids: Sequence[str],
    max_request_tokens: int,
) -> list[JevRequest]:
    request = _build_one(query, results, rule_types, positions, [all_ids[p] for p in positions])
    if request.estimated_tokens <= max_request_tokens or len(positions) <= 1:
        return [request]
    half = (len(positions) + 1) // 2
    return _split_to_fit(
        query, results, rule_types, positions[:half], all_ids, max_request_tokens
    ) + _split_to_fit(query, results, rule_types, positions[half:], all_ids, max_request_tokens)


def build_requests(
    query: str,
    results: Sequence[SearchResult],
    rule_types: Mapping[int, ResultType],
    *,
    batch_size: int,
    max_request_tokens: int,
) -> list[JevRequest]:
    """結果を先頭から batch_size 件ずつ分け、大きすぎる塊は1件になるまで半分に割る。"""
    total = len(results)
    all_ids = [result_id(position, total) for position in range(total)]
    requests: list[JevRequest] = []
    for start in range(0, total, batch_size):
        positions = list(range(start, min(start + batch_size, total)))
        requests.extend(
            _split_to_fit(query, results, rule_types, positions, all_ids, max_request_tokens)
        )
    return requests


def preview_request(query: str, *, batch_size: int, max_request_tokens: int) -> JevRequest:
    """`--dry-run` 用。見本の結果1件で作ったリクエスト。"""
    sample = SearchResult(
        url="https://example.com/",
        title="<title>",
        snippet="<snippet>",
        engine="<engine>",
        published_date=None,
        retrieved_at=datetime(1970, 1, 1, tzinfo=UTC),
        engines=("<engine>",),
        matched=(),
    )
    return build_requests(
        query, [sample], {}, batch_size=batch_size, max_request_tokens=max_request_tokens
    )[0]


def _unclassified(*, confidence: float | None = None) -> Judgment:
    return Judgment(value=None, by=JudgedBy.JEV, confidence=confidence)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_unit_number(value: object) -> bool:
    """有限の数で 0 以上 1 以下。NaN・inf・範囲外は壊れた値として扱う。"""
    return _is_number(value) and math.isfinite(value) and 0 <= value <= 1


def _map_choice(answer: object, criteria: Mapping[str, str], threshold: float) -> Judgment:
    """choice の答えを判定に写す。壊れた答え・欠けた答えは未分類。"""
    if not isinstance(answer, dict):
        return _unclassified()
    confidence = answer.get("confidence")
    if not _is_unit_number(confidence):
        return _unclassified()
    if "probabilities" in answer and not isinstance(answer["probabilities"], dict):
        return _unclassified()
    confidence = float(confidence)
    choice = answer.get("choice")
    if confidence < threshold or not isinstance(choice, str) or choice not in criteria:
        return _unclassified(confidence=confidence)
    return Judgment(value=choice, by=JudgedBy.JEV, confidence=confidence)


def _map_noul(answer: object, threshold: float) -> Judgment:
    """noul の答えを判定に写す。確信度は max(p, 1-p)。"""
    if not isinstance(answer, dict):
        return _unclassified()
    probabilities = answer.get("probabilities")
    if probabilities is not None and not isinstance(probabilities, dict):
        return _unclassified()
    if isinstance(probabilities, dict) and "1" in probabilities:
        p = probabilities["1"]
    else:
        p = answer.get("probability")
    if not _is_unit_number(p):
        return _unclassified()
    p = float(p)
    confidence = max(p, 1 - p)
    if confidence < threshold:
        return _unclassified(confidence=confidence)
    return Judgment(value=p > 0.5, by=JudgedBy.JEV, confidence=confidence)


def _unclassified_results(
    request: JevRequest,
    results: Sequence[SearchResult],
    rule_types: Mapping[int, ResultType],
) -> list[ClassifiedResult]:
    """Jev の答えが無いときの結果。規則で決まった種類だけは規則の判定のまま残す。"""
    classified = []
    for position, rid in zip(request.positions, request.ids, strict=True):
        classified.append(
            ClassifiedResult(
                id=rid,
                result=results[position],
                type=_type_by_rule(position, rule_types) or _unclassified(),
                genre=_unclassified(),
                relevant=_unclassified(),
            )
        )
    return classified


def _type_by_rule(position: int, rule_types: Mapping[int, ResultType]) -> Judgment | None:
    if position not in rule_types:
        return None
    return Judgment(value=rule_types[position], by=JudgedBy.RULE, confidence=1.0)


def map_answers(
    request: JevRequest,
    answers: Mapping,
    results: Sequence[SearchResult],
    rule_types: Mapping[int, ResultType],
    *,
    threshold: float,
) -> tuple[list[ClassifiedResult], Judgment]:
    """Jev の答えを、結果ごとの判定と、そのリクエストの query_genre の判定に写す。"""
    classified = []
    for position, rid in zip(request.positions, request.ids, strict=True):
        rule_judgment = _type_by_rule(position, rule_types)
        result_type = rule_judgment or _map_choice(
            answers.get(f"type_{rid}"), _TYPE_CRITERIA, threshold
        )
        classified.append(
            ClassifiedResult(
                id=rid,
                result=results[position],
                type=result_type,
                genre=_map_choice(answers.get(f"genre_{rid}"), _GENRE_CRITERIA, threshold),
                relevant=_map_noul(answers.get(f"relevant_{rid}"), threshold),
            )
        )
    query_genre = _map_choice(answers.get(_QUERY_GENRE_ID), _GENRE_CRITERIA, threshold)
    return classified, query_genre


def _token_count(usage: Mapping | None, key: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _failure_code(error: JevError) -> str:
    if error.code == "http":
        return f"http {error.status}"
    return error.code


class JevClassifier:
    """Jev でまとめて分類する（ports.Classifier を満たす）。"""

    def __init__(
        self,
        ask: Callable[[str, dict], JevResult],
        *,
        model: str,
        batch_size: int,
        max_request_tokens: int,
        threshold: float,
        clock: Clock,
    ) -> None:
        self._ask = ask
        self._model = model
        self._batch_size = batch_size
        self._max_request_tokens = max_request_tokens
        self._threshold = threshold
        self._clock = clock

    def classify(
        self,
        query: str,
        results: Sequence[SearchResult],
        rule_types: Mapping[int, ResultType],
    ) -> tuple[list[ClassifiedResult], JevReport]:
        requests = build_requests(
            query,
            results,
            rule_types,
            batch_size=self._batch_size,
            max_request_tokens=self._max_request_tokens,
        )
        classified: list[ClassifiedResult] = []
        usages: list[JevUsage] = []
        codes: list[str] = []
        genres: list[Judgment] = []
        for request in requests:
            started = self._clock.now()
            input_tokens: int | None = None
            output_tokens: int | None = None
            duration_ms: int | None = None
            error: str | None = None
            try:
                response = self._ask(request.state, request.questions)
                usage = response.usage
                input_tokens = _token_count(usage, "input_tokens")
                output_tokens = _token_count(usage, "output_tokens")
                duration_ms = response.duration_ms
                mapped, query_genre = map_answers(
                    request, response.answers, results, rule_types, threshold=self._threshold
                )
                classified.extend(mapped)
                genres.append(query_genre)
            except JevError as e:
                error = _failure_code(e)
            except Exception as e:  # noqa: BLE001 - ジョブを止めずに未分類で進める
                error = f"unexpected:{type(e).__name__}"
            if error is not None:
                classified.extend(_unclassified_results(request, results, rule_types))
                codes.append(error)
            if duration_ms is None:
                elapsed = self._clock.now() - started
                duration_ms = int(elapsed.total_seconds() * 1000)
            usages.append(
                JevUsage(
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_input_tokens=request.estimated_tokens,
                    duration_ms=duration_ms,
                    n_results=len(request.ids),
                    n_questions=len(request.questions),
                    error=error,
                )
            )
        return classified, JevReport(
            query_genre=_best_genre(genres),
            usage=tuple(usages),
            failure=", ".join(dict.fromkeys(codes)) or None,
        )


def _best_genre(genres: Sequence[Judgment]) -> Judgment:
    """確信度がいちばん高いもの。確信度のあるものがなければ未分類。"""
    best = _unclassified()
    for genre in genres:
        if genre.confidence is None or not math.isfinite(genre.confidence):
            continue
        if best.confidence is None or genre.confidence > best.confidence:
            best = genre
    return best
