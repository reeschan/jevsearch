"""検索ジョブの進行（状態の図 jevsearch-job.mmd どおりに段を進める）。

受付 → 正規化 → 検索中 → （失敗 | 0件 | 規則でタグ付け → Jevで分類 → 保存済み | 一部だけ保存）。
部品（接続口・分類器・保存・ダイジェスト・時計）は引数で受け取り、ここでは差し替えられる形のまま使う。
段ごとの `states` を記録に残し、結末は `decide_outcome` で決める。
"""

from __future__ import annotations

import datetime

from jevsearch.config import Config, select_engines
from jevsearch.digest import NO_ROWS, DigestWriter
from jevsearch.evaluation import eval_row
from jevsearch.models import (
    FINAL_STATE,
    JEV_MODEL,
    ClassifiedResult,
    JevReport,
    JevUsage,
    JobRecord,
    JobRequest,
    JobState,
    Outcome,
    Reasons,
    SearchResponse,
    SearchVerdict,
    Versions,
)
from jevsearch.normalize import Dictionary, normalize
from jevsearch.ports import Classifier, Clock, SearchBackend
from jevsearch.rules import DomainRules, rule_types
from jevsearch.store import RESULTS_FILE, ResultStore, job_row, result_rows


def decide_outcome(response: SearchResponse, report: JevReport | None) -> Outcome:
    """検索の総合結果と Jev の報告から、ジョブの結末を決める。

    接続口が FAILED なら failed、NO_RESULTS なら no_results。
    結果があり、失敗したエンジンか届かなかった検索語がある、または Jev の失敗があれば partial。
    それ以外は saved。確信度が低いだけの未分類は結末を変えない。
    """
    if response.verdict == SearchVerdict.FAILED:
        return Outcome.FAILED
    if response.verdict == SearchVerdict.NO_RESULTS:
        return Outcome.NO_RESULTS
    jev_failed = report is not None and report.failure is not None
    if response.failed_engines or response.unreachable_queries or jev_failed:
        return Outcome.PARTIAL
    return Outcome.SAVED


def _is_unclassified(item: ClassifiedResult) -> bool:
    """type・genre・relevant のどれか1つでも未分類か。"""
    return any(j.value is None for j in (item.type, item.genre, item.relevant))


def _write_digest(
    request: JobRequest,
    config: Config,
    store: ResultStore,
    digest: DigestWriter | None,
) -> str | None:
    """ダイジェストを書く。書けなかった理由を返す（書けたとき・書く対象がないときは None）。

    日付は、このジョブの（relevant が false でない）行の retrieved_at の日付ごと（行に書かれた
    オフセットのまま）。日ごとに1回 `upsert` する（ふつうは1日）。
    """
    if request.topic is None or digest is None or not config.memory.enabled:
        return None
    try:
        rows = list(store.read(RESULTS_FILE))
        days = _job_days(request, rows)
    except Exception as exc:  # noqa: BLE001 - ダイジェストの失敗で結末を変えない
        return type(exc).__name__
    failures: list[str] = []
    for day in days:
        try:
            new_id, reason = digest.upsert(request.topic, day, rows)
        except Exception as exc:  # noqa: BLE001 - ダイジェストの失敗で結末を変えない
            failures.append(type(exc).__name__)
            continue
        if new_id is None and reason != NO_ROWS and reason is not None:
            failures.append(reason)
    return "; ".join(failures) if failures else None


def _job_days(request: JobRequest, rows: list[dict]) -> list[datetime.date]:
    """このジョブの行の retrieved_at の日付を、最初に見えた順に重複なく返す。"""
    days: list[datetime.date] = []
    for row in rows:
        if row.get("job_id") != request.job_id or row.get("topic") != request.topic:
            continue
        relevant = row.get("relevant")
        if isinstance(relevant, dict) and relevant.get("value") is False:
            continue
        try:
            day = datetime.datetime.fromisoformat(row["retrieved_at"]).date()
        except (KeyError, TypeError, ValueError):
            continue
        if day not in days:
            days.append(day)
    return days


def run_job(
    request: JobRequest,
    *,
    config: Config,
    dictionary: Dictionary,
    rules: DomainRules,
    search: SearchBackend,
    classifier: Classifier,
    store: ResultStore,
    digest: DigestWriter | None,
    clock: Clock,
) -> JobRecord:
    """検索ジョブを1件進め、ジョブの記録を返す（保存も済ませる）。

    書く順番は、結果ログ（結果があるとき）→ ダイジェスト → ジョブの記録 → 評価記録（分類したとき）。
    ダイジェストの失敗は `reasons.digest_failure` に残すだけで、結末は変えない。

    遷移: 検索語を受け取った
    遷移: 検索を始めた
    """
    started_at = clock.now()
    versions = Versions(
        config=config.version,
        dictionary=dictionary.version,
        rules=rules.version,
        jev_model=JEV_MODEL,
        rustxng=config.rustxng.expected_version,
    )
    states: list[JobState] = [JobState.ACCEPTED]

    states.append(JobState.NORMALIZING)
    normalized = normalize(request.query, dictionary)

    states.append(JobState.SEARCHING)
    engines = select_engines(config, request.genre, request.allow_scrape)
    response = search.search(normalized, engines, request.max_results)
    if response.retried_queries:
        states.append(JobState.SEARCHING)

    classified: list[ClassifiedResult] = []
    report: JevReport | None = None
    if response.verdict == SearchVerdict.HAS_RESULTS:
        states.append(JobState.RULE_TAGGING)
        decided = rule_types(response.results, rules)
        states.append(JobState.JEV_CLASSIFY)
        classified, report = classifier.classify(normalized.ja, response.results, decided)

    outcome = decide_outcome(response, report)
    states.append(FINAL_STATE[outcome])

    # ① 結果ログ
    if classified:
        store.append_results(
            result_rows(
                job_id=request.job_id,
                query=request.query,
                query_norm=normalized,
                topic=request.topic,
                items=classified,
                versions=versions,
            )
        )

    # ② ダイジェスト
    digest_failure: str | None = None
    if outcome in (Outcome.SAVED, Outcome.PARTIAL):
        digest_failure = _write_digest(request, config, store, digest)

    # ③ ジョブの記録
    usage: tuple[JevUsage, ...] = report.usage if report is not None else ()
    elapsed = clock.now() - started_at
    record = JobRecord(
        job_id=request.job_id,
        query=request.query,
        query_norm=normalized,
        topic=request.topic,
        genre=request.genre,
        started_at=started_at,
        duration_ms=int(elapsed.total_seconds() * 1000),
        outcome=outcome,
        reasons=Reasons(
            failed_engines=response.failed_engines,
            unreachable_queries=response.unreachable_queries,
            jev_failure=report.failure if report is not None else None,
            digest_failure=digest_failure,
        ),
        states=tuple(states),
        n_results=len(classified),
        n_unclassified=sum(1 for item in classified if _is_unclassified(item)),
        versions=versions,
        jev_usage=usage,
    )
    store.append_job(job_row(record))

    # ④ 評価記録
    if report is not None:
        store.append_eval(
            eval_row(
                job_id=request.job_id,
                started_at=started_at,
                items=classified,
                report=report,
                failed_engines=response.failed_engines,
                price_per_mtok_input=config.jev.price_per_mtok_input,
                price_per_mtok_output=config.jev.price_per_mtok_output,
            )
        )
    return record
