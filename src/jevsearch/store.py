"""保存（seam ⑤）。結果ログ・ジョブの記録・評価記録を JSONL で追記する。

書く側に番兵を置く。追記の前に全行を検査し、1行でも違反があれば何も書かずに StoreError にする。
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from jevsearch.models import (
    ClassifiedResult,
    JobRecord,
    JudgedBy,
    Judgment,
    NormalizedQuery,
    Versions,
)

SCHEMA = 1

RESULTS_FILE = "results.jsonl"
JOBS_FILE = "jobs.jsonl"
EVAL_FILE = "eval.jsonl"

# 結果の行で確信度の検査をかける判定のキー。
_JUDGED_KEYS = ("type", "genre", "relevant")


class StoreError(Exception):
    """保存の番兵に引っかかった、または書けなかった。メッセージに API キーを含めない。"""


def _iso(value: datetime.datetime) -> str:
    return value.isoformat()


def _judgment(j: Judgment) -> dict[str, Any]:
    return {
        "value": j.value,
        "by": str(j.by.value) if hasattr(j.by, "value") else str(j.by),
        "confidence": j.confidence,
    }


def _versions(v: Versions) -> dict[str, str]:
    return {
        "config": v.config,
        "dictionary": v.dictionary,
        "rules": v.rules,
        "jev_model": v.jev_model,
        "rustxng": v.rustxng,
    }


def _query_norm(q: NormalizedQuery | None) -> dict[str, str | None] | None:
    if q is None:
        return None
    return {"ja": q.ja, "en": q.en}


def result_rows(
    *,
    job_id: str,
    query: str,
    query_norm: NormalizedQuery,
    topic: str | None,
    items: Sequence[ClassifiedResult],
    versions: Versions,
) -> list[dict]:
    """分類済みの結果を、結果ログの1結果1行の形にする（純粋な関数）。"""
    rows: list[dict] = []
    for item in items:
        r = item.result
        rows.append(
            {
                "schema": SCHEMA,
                "job_id": job_id,
                "query": query,
                "query_norm": _query_norm(query_norm),
                "retrieved_at": _iso(r.retrieved_at),
                "topic": topic,
                "id": item.id,
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet,
                "engine": r.engine,
                "engines": list(r.engines),
                "matched": [str(m.value) if hasattr(m, "value") else str(m) for m in r.matched],
                "published_date": r.published_date,
                "type": _judgment(item.type),
                "genre": _judgment(item.genre),
                "relevant": _judgment(item.relevant),
                "versions": _versions(versions),
            }
        )
    return rows


def job_row(record: JobRecord) -> dict:
    """ジョブの記録を、jobs.jsonl の1ジョブ1行の形にする。"""
    reasons = record.reasons
    return {
        "schema": SCHEMA,
        "job_id": record.job_id,
        "query": record.query,
        "query_norm": _query_norm(record.query_norm),
        "topic": record.topic,
        "genre": None if record.genre is None else str(record.genre.value),
        "started_at": _iso(record.started_at),
        "duration_ms": record.duration_ms,
        "outcome": str(record.outcome.value),
        "reasons": {
            "failed_engines": [
                {"engine": f.engine, "reason": f.reason} for f in reasons.failed_engines
            ],
            "unreachable_queries": [str(q.value) for q in reasons.unreachable_queries],
            "jev_failure": reasons.jev_failure,
            "digest_failure": reasons.digest_failure,
        },
        "states": [str(s.value) for s in record.states],
        "n_results": record.n_results,
        "n_unclassified": record.n_unclassified,
        "versions": _versions(record.versions),
        "jev_usage": [
            {
                "model": u.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "estimated_input_tokens": u.estimated_input_tokens,
                "duration_ms": u.duration_ms,
                "n_results": u.n_results,
                "n_questions": u.n_questions,
                "error": u.error,
            }
            for u in record.jev_usage
        ],
    }


def _has_offset(value: object) -> bool:
    """ISO 8601 でオフセット付きの文字列か。"""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class ResultStore:
    """data_dir の下の JSONL に追記する。

    threshold は Jev の判定を「分類済み」と認める確信度の下限。
    """

    def __init__(self, data_dir: Path, *, threshold: float, api_key: str | None) -> None:
        self._dir = Path(data_dir)
        self._threshold = threshold
        self._api_key = api_key or None

    def append_results(self, rows: Iterable[Mapping[str, Any]]) -> None:
        listed = list(rows)  # ジェネレーターでも、検査と書き込みで同じ行を見る
        for i, row in enumerate(listed):
            self._check_result_row(i, row)
        self._append(RESULTS_FILE, listed)

    def append_job(self, row: Mapping[str, Any]) -> None:
        self._append(JOBS_FILE, [row])

    def append_eval(self, row: Mapping[str, Any]) -> None:
        self._append(EVAL_FILE, [row])

    def read(self, name: str) -> Iterator[dict]:
        """name のファイルの行を順に返す。ファイルが無ければ空。"""
        path = self._dir / name
        if not path.exists():
            return
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except ValueError:
                    continue  # 途中で切れた行などは飛ばし、ほかの行は返す
                if isinstance(parsed, dict):
                    yield parsed

    def _check_result_row(self, index: int, row: Mapping[str, Any]) -> None:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise StoreError(f"結果の行 {index}: 検索語（query）が空のため保存しない")
        if not _has_offset(row.get("retrieved_at")):
            raise StoreError(
                f"結果の行 {index}: 取得日時（retrieved_at）が"
                " ISO 8601 のオフセット付きでないため保存しない"
            )
        for key in _JUDGED_KEYS:
            j = row.get(key)
            if not isinstance(j, Mapping):
                continue
            if j.get("by") != JudgedBy.JEV or j.get("value") is None:
                continue
            conf = j.get("confidence")
            if (
                isinstance(conf, bool)
                or not isinstance(conf, (int, float))
                or not math.isfinite(conf)
                or not 0 <= conf <= 1
                or conf < self._threshold
            ):
                raise StoreError(
                    f"結果の行 {index}: {key} は確信度がしきい値（{self._threshold}）に"
                    "届かないため分類済みとして保存しない"
                )

    def _dump(self, name: str, row: Mapping[str, Any]) -> str:
        try:
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StoreError(
                f"{name}: JSON にできない内容のため保存しない（{type(exc).__name__}）"
            ) from None
        key = self._api_key
        if key is not None and (key in line or json.dumps(key, ensure_ascii=False)[1:-1] in line):
            raise StoreError(f"{name}: API キーを含む内容のため保存しない")
        return line

    def _append(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        lines = [self._dump(name, row) for row in rows]
        if not lines:
            return
        # 開く前に全行を符号化する。符号化できない行があれば何も書かない。
        try:
            data = "".join(line + "\n" for line in lines).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise StoreError(
                f"{name}: UTF-8 にできない内容のため保存しない（{type(exc).__name__}）"
            ) from None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with (self._dir / name).open("a+b") as f:
                # 前の書き込みが途中で切れていたら、改行を足して新しい行を別の行にする。
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        data = b"\n" + data
                f.write(data)
        except OSError as exc:
            raise StoreError(f"{name}: 書き込めなかった（{type(exc).__name__}）") from None
