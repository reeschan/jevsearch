"""段どうしで受け渡す共有の型。ロジックは持たない（frozen dataclass と StrEnum だけ）。"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum


class Lang(StrEnum):
    """検索語の言語。"""

    JA = "ja"
    EN = "en"


class Outcome(StrEnum):
    """ジョブの結末。"""

    SAVED = "saved"
    PARTIAL = "partial"
    NO_RESULTS = "no_results"
    FAILED = "failed"


class JobState(StrEnum):
    """ジョブの状態（図 jevsearch-job.mmd の状態に対応）。"""

    ACCEPTED = "accepted"
    NORMALIZING = "normalizing"
    SEARCHING = "searching"
    FAILED = "failed"
    NO_RESULTS = "no_results"
    RULE_TAGGING = "rule_tagging"
    JEV_CLASSIFY = "jev_classify"
    SAVED = "saved"
    PARTIALLY_SAVED = "partially_saved"


class Genre(StrEnum):
    """検索語のジャンル。"""

    PAPER = "paper"
    TECH = "tech"
    NEWS = "news"
    SNS = "sns"
    BLOG = "blog"
    GENERAL = "general"


class ResultType(StrEnum):
    """結果の種類。"""

    PAPER = "paper"
    BLOG = "blog"
    NEWS = "news"
    SNS = "sns"
    DOCS = "docs"
    REPO = "repo"
    OTHER = "other"


class JudgedBy(StrEnum):
    """判定の決め方。"""

    RULE = "rule"
    JEV = "jev"


class JudgmentKey(StrEnum):
    """判定の種類。ClassifiedResult の属性名・結果ログの行のキー・評価記録・CSV の列名と同じ。"""

    TYPE = "type"
    GENRE = "genre"
    RELEVANT = "relevant"


JUDGMENT_KEYS: tuple[str, ...] = tuple(k.value for k in JudgmentKey)


class SearchVerdict(StrEnum):
    """検索の総合結果。"""

    FAILED = "failed"
    NO_RESULTS = "no_results"
    HAS_RESULTS = "has_results"


JEV_MODEL = "jev-1.14.0"

# (元の状態。開始は None, 先の状態, 図のラベル)。docs/specs/statechart/jevsearch-job.mmd と同じ 12 件。
TRANSITIONS: tuple[tuple[JobState | None, JobState, str], ...] = (
    (None, JobState.ACCEPTED, "検索語を受け取った"),
    (JobState.ACCEPTED, JobState.NORMALIZING, "検索を始めた"),
    (JobState.NORMALIZING, JobState.SEARCHING, "用語辞書で検索語をそろえ終えた"),
    (JobState.SEARCHING, JobState.SEARCHING, "検索エンジンにつながらず、1回だけやり直す"),
    (JobState.SEARCHING, JobState.FAILED, "やり直しても検索エンジンにつながらず、結果が1件も届かなかった"),
    (JobState.SEARCHING, JobState.NO_RESULTS, "どのエンジンからも結果が返ってこなかった"),
    (JobState.SEARCHING, JobState.RULE_TAGGING, "結果が1件以上届いた"),
    (JobState.SEARCHING, JobState.RULE_TAGGING, "一部の検索語は結果が届き、ほかはやり直してもつながらなかった"),
    (JobState.RULE_TAGGING, JobState.JEV_CLASSIFY, "ドメインで分かる種類を付け終えた"),
    (JobState.JEV_CLASSIFY, JobState.SAVED, "すべての検索語とエンジンが答え、Jev もすべて答えた"),
    (JobState.JEV_CLASSIFY, JobState.PARTIALLY_SAVED, "一部のエンジンか検索語が答えなかった"),
    (JobState.JEV_CLASSIFY, JobState.PARTIALLY_SAVED, "Jev が使えない、または上限を超えたので未分類として残した"),
)

# 結末と終わりの状態の対応。
FINAL_STATE: dict[Outcome, JobState] = {
    Outcome.SAVED: JobState.SAVED,
    Outcome.PARTIAL: JobState.PARTIALLY_SAVED,
    Outcome.NO_RESULTS: JobState.NO_RESULTS,
    Outcome.FAILED: JobState.FAILED,
}


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    query: str
    topic: str | None
    genre: Genre | None
    max_results: int
    allow_scrape: bool
    dry_run: bool


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    ja: str
    en: str | None
    dictionary_version: str


@dataclass(frozen=True)
class SearchResult:
    """URL でまとめた後の1件。

    engine は最初に見えたエンジン、engines は返したエンジン全部、matched はどの検索語で当たったか。
    retrieved_at は tz 付き。
    """

    url: str
    title: str
    snippet: str
    engine: str
    published_date: str | None
    retrieved_at: datetime.datetime
    engines: tuple[str, ...]
    matched: tuple[Lang, ...]


@dataclass(frozen=True)
class EngineFailure:
    engine: str
    reason: str


@dataclass(frozen=True)
class SearchResponse:
    verdict: SearchVerdict
    results: tuple[SearchResult, ...]
    failed_engines: tuple[EngineFailure, ...]
    unreachable_queries: tuple[Lang, ...]
    retried_queries: tuple[Lang, ...]
    rustxng_version: str


@dataclass(frozen=True)
class Judgment:
    """判定。value が None なら未分類。"""

    value: str | bool | None
    by: JudgedBy
    confidence: float | None


@dataclass(frozen=True)
class ClassifiedResult:
    """id は r01 形式（2桁ゼロ詰め、100件以上は桁が増える。ジョブ全体で通し番号）。"""

    id: str
    result: SearchResult
    type: Judgment
    genre: Judgment
    relevant: Judgment


@dataclass(frozen=True)
class JevUsage:
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_input_tokens: int
    duration_ms: int
    n_results: int
    n_questions: int
    error: str | None


@dataclass(frozen=True)
class JevReport:
    query_genre: Judgment
    usage: tuple[JevUsage, ...]
    failure: str | None


@dataclass(frozen=True)
class Versions:
    config: str
    dictionary: str
    rules: str
    jev_model: str
    rustxng: str


@dataclass(frozen=True)
class Reasons:
    failed_engines: tuple[EngineFailure, ...] = ()
    unreachable_queries: tuple[Lang, ...] = ()
    jev_failure: str | None = None
    digest_failure: str | None = None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    query: str
    query_norm: NormalizedQuery | None
    topic: str | None
    genre: Genre | None
    started_at: datetime.datetime
    duration_ms: int
    outcome: Outcome
    reasons: Reasons
    states: tuple[JobState, ...]
    n_results: int
    n_unclassified: int
    versions: Versions
    jev_usage: tuple[JevUsage, ...]
    query_genre: Judgment | None = None
