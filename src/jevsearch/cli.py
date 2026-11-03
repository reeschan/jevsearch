"""コマンドラインの入口。

`search`（検索ジョブの実行と `--dry-run`）、`eval`、`stats`、`digest --retry` をつなぐ。
外部につながる部品（検索の接続口・Jev の分類器・共有メモリ）は `Deps` の make_* で作り、
テストでは差し替える。既定の make_* は本物を作る。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TextIO

import httpx

from jevsearch.config import (
    CONFIG_ENV_VAR,
    Config,
    ConfigError,
    load_config,
    scrape_engine_names,
    select_engines,
)
from jevsearch.credentials import get_api_key
from jevsearch.digest import DigestWriter
from jevsearch.evaluation import accuracy_from_labels, aggregate, batch_vs_single, sample_csv
from jevsearch.jev_classify import JevClassifier, preview_request
from jevsearch.jev_client import JevClient
from jevsearch.job import run_job
from jevsearch.mcp_client import McpHttpClient
from jevsearch.models import (
    ClassifiedResult,
    Genre,
    JevReport,
    JobRecord,
    JobRequest,
    Lang,
    Outcome,
    ResultType,
    SearchResult,
)
from jevsearch.normalize import Dictionary, load_dictionary, normalize
from jevsearch.ports import Classifier, Clock, MemoryClient, SearchBackend
from jevsearch.rules import DomainRules, load_rules, rule_types
from jevsearch.rustxng import RustxngAdapter
from jevsearch.search_request import build_search_request
from jevsearch.store import EVAL_FILE, RESULTS_FILE, ResultStore

EXIT_OK = 0
EXIT_FAILED = 1
"""処理が失敗したときの終了コード（検索の failed、キーが無い、想定外の例外など）。"""
EXIT_USAGE = 2
"""使い方・設定のエラーの終了コード。"""

_MEMORY_TIMEOUT_SECONDS = 30.0
"""共有メモリへの1回の通信を待つ秒数。"""

_OUTCOME_NAMES: dict[Outcome, str] = {
    Outcome.SAVED: "保存済み",
    Outcome.PARTIAL: "一部だけ保存",
    Outcome.NO_RESULTS: "0件",
    Outcome.FAILED: "失敗",
}
"""結末の日本語名（状態の図の表示名と同じ）。"""


class _SystemClock:
    """ローカル時刻をオフセット付き（tz 付き）で返す既定の時計。"""

    def now(self) -> datetime.datetime:
        return datetime.datetime.now().astimezone()


@dataclass
class Deps:
    """CLI が外から受け取るもの。テストでは差し替える。

    make_* を渡さなければ本物を作る既定の作り方になる。
    http_transport が None のときは本物の通信、渡せば既定の make_* が作る
    httpx.Client のすべてがそれを使う（テストでの差し替え用）。
    """

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    clock: Clock = field(default_factory=_SystemClock)
    http_transport: httpx.BaseTransport | None = None
    make_search_backend: Callable[[Config, bool], SearchBackend] = field(default=None)  # type: ignore[assignment]
    make_classifier: Callable[[Config], Classifier] = field(default=None)  # type: ignore[assignment]
    make_memory_client: Callable[[Config], MemoryClient] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.make_search_backend is None:
            self.make_search_backend = self._default_search_backend
        if self.make_classifier is None:
            self.make_classifier = self._default_classifier
        if self.make_memory_client is None:
            self.make_memory_client = self._default_memory_client

    def _http_client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self.http_transport)

    def _default_search_backend(self, config: Config, allow_scrape: bool) -> SearchBackend:
        """RustXNG の接続口。allow_scrape は CLI の許可と設定の許可の両方があるときだけ真。"""
        return RustxngAdapter(
            config.rustxng.url,
            config.rustxng.expected_version,
            scrape_engine_names(config),
            allow_scrape,
            self._http_client(config.rustxng.timeout_seconds),
            self.clock,
        )

    def _default_classifier(self, config: Config) -> Classifier:
        """Jev でまとめて分類する分類器。キーは env（と認証ファイル）から読む。"""
        jev = config.jev
        client = JevClient(
            api_key=get_api_key(self.env),
            model=jev.model,
            endpoint=jev.endpoint,
            client=self._http_client(jev.timeout_seconds),
            usage_path=config.storage.dir / "jev-usage.json",
            daily_request_limit=jev.daily_request_limit,
            max_rate_retries=jev.max_rate_retries,
            clock=self.clock,
            timeout_seconds=jev.timeout_seconds,
        )
        return JevClassifier(
            client.ask,
            model=jev.model,
            batch_size=jev.batch_size,
            max_request_tokens=jev.max_request_tokens,
            threshold=jev.threshold,
            clock=self.clock,
        )

    def _default_memory_client(self, config: Config) -> MemoryClient:
        return McpHttpClient(config.memory.url, self._http_client(_MEMORY_TIMEOUT_SECONDS))


def _path_arg(text: str) -> Path:
    """パスの引数。`~` は展開する。"""
    return Path(text).expanduser()


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で書く: {text}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 以上にする: {text}")
    return value


def _date_arg(text: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"YYYY-MM-DD で書く: {text}") from None


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="設定ファイル")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevsearch", description="メタ検索と分類")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="検索する")
    search.add_argument("query", help="検索語")
    search.add_argument("--topic", default=None, help="話題（記録に残す）")
    search.add_argument("--genre", choices=[g.value for g in Genre], default=None)
    search.add_argument("--max", dest="max_results", type=int, default=None, help="結果の上限")
    search.add_argument("--allow-scrape", action="store_true", help="画面読み取りのエンジンを許す")
    search.add_argument("--dry-run", action="store_true", help="外部を呼ばず送る内容を出す")
    _add_config(search)

    evaluation = sub.add_parser("eval", help="評価（比較・抜き出し・一致率）")
    eval_sub = evaluation.add_subparsers(dest="eval_command", required=True)

    batch = eval_sub.add_parser("batch-vs-single", help="まとめて判定と1件ずつ判定の一致率")
    batch.add_argument("log", type=_path_arg, help="結果ログ（JSONL）")
    _add_config(batch)

    sample = eval_sub.add_parser("sample", help="人が正解を付けるための抜き出し（CSV）")
    sample.add_argument("--n", type=_positive_int, default=50, help="抜き出す件数")
    sample.add_argument("--seed", type=int, default=0, help="抜き出しの乱数の種")
    sample.add_argument(
        "--out", type=_path_arg, default=None, help="書き出す先（無ければ標準出力）"
    )
    _add_config(sample)

    accuracy = eval_sub.add_parser("accuracy", help="付けた正解との一致率を集計する")
    accuracy.add_argument("csv", type=_path_arg, help="正解を付けた CSV")
    _add_config(accuracy)

    stats = sub.add_parser("stats", help="評価記録を集計する")
    stats.add_argument("--since", type=_date_arg, default=None, help="この日（ローカル）以降")
    _add_config(stats)

    digest = sub.add_parser("digest", help="書けなかったダイジェストを書き直す")
    digest.add_argument(
        "--retry", action="store_true", required=True, help="書き残しを書き直す"
    )
    _add_config(digest)
    return parser


def _parse(parser: argparse.ArgumentParser, argv: Sequence[str] | None, deps: Deps):
    """引数を読む。使い方の誤りは標準エラー（deps.stderr）に出し、終了コードを返す。"""
    try:
        with contextlib.redirect_stderr(deps.stderr), contextlib.redirect_stdout(deps.stdout):
            return parser.parse_args(argv), None
    except SystemExit as exc:
        code = exc.code
        return None, code if isinstance(code, int) else EXIT_USAGE


def _dry_run_payload(
    config: Config, args: argparse.Namespace, max_results: int, dictionary: Dictionary
) -> dict:
    genre = Genre(args.genre) if args.genre is not None else None
    engines = select_engines(config, genre, args.allow_scrape)
    normalized = normalize(args.query, dictionary)
    queries = [(Lang.JA, normalized.ja)]
    if normalized.en:
        queries.append((Lang.EN, normalized.en))
    requests = []
    for lang, text in queries:
        url, params = build_search_request(config.rustxng.url, text, lang, engines, max_results)
        requests.append({"lang": lang.value, "url": url, "params": params})
    preview = preview_request(
        normalized.ja,
        batch_size=config.jev.batch_size,
        max_request_tokens=config.jev.max_request_tokens,
    )
    return {
        "query": args.query,
        "query_norm": {"ja": normalized.ja, "en": normalized.en},
        "genre": args.genre,
        "topic": args.topic,
        "engines": engines,
        "rustxng": {"requests": requests},
        "jev": {
            "model": config.jev.model,
            "batch_size": config.jev.batch_size,
            "threshold": config.jev.threshold,
            "preview": {"state": json.loads(preview.state), "questions": preview.questions},
        },
        "versions": {"config": config.version},
    }


def _job_id(now: datetime.datetime) -> str:
    """`<YYYYMMDDTHHMMSS>-<uuid4 の先頭8文字>`。"""
    return f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def _engine_names(record: JobRecord) -> str:
    return ", ".join(f.engine for f in record.reasons.failed_engines) or "なし"


def _summary_line(record: JobRecord) -> str:
    return (
        f"{_OUTCOME_NAMES[record.outcome]}: {record.n_results}件, "
        f"未分類 {record.n_unclassified}件, 失敗したエンジン: {_engine_names(record)}"
    )


def _partial_reasons(record: JobRecord) -> list[str]:
    reasons = record.reasons
    parts: list[str] = []
    for failure in reasons.failed_engines:
        detail = f"（{failure.reason}）" if failure.reason else ""
        parts.append(f"エンジン {failure.engine} が答えなかった{detail}")
    if reasons.unreachable_queries:
        langs = ", ".join(q.value for q in reasons.unreachable_queries)
        parts.append(f"検索語 {langs} はつながらなかった")
    if reasons.jev_failure:
        parts.append(f"Jev が使えなかった（{reasons.jev_failure}）。未分類として残した")
    return parts


def _report(record: JobRecord, deps: Deps) -> int:
    """要約を標準出力に、警告を標準エラーに出し、終了コードを返す。"""
    print(_summary_line(record), file=deps.stdout)
    reasons = record.reasons
    if record.outcome == Outcome.PARTIAL:
        for part in _partial_reasons(record):
            print(f"警告: 一部だけ保存: {part}", file=deps.stderr)
    elif record.outcome == Outcome.NO_RESULTS and reasons.failed_engines:
        print(f"警告: 0件。答えなかったエンジン: {_engine_names(record)}", file=deps.stderr)
    elif record.outcome == Outcome.FAILED:
        print("エラー: 検索エンジンにつながらず、結果が1件も届かなかった", file=deps.stderr)
    if reasons.digest_failure:
        print(f"警告: ダイジェストを書けなかった: {reasons.digest_failure}", file=deps.stderr)
    return EXIT_FAILED if record.outcome == Outcome.FAILED else EXIT_OK


def _load_parts(config: Config) -> tuple[Dictionary, DomainRules]:
    return load_dictionary(config.dictionary.path), load_rules(config.rules.path)


def _config_path(args: argparse.Namespace, deps: Deps) -> Path | None:
    """--config、無ければ環境変数 JEVSEARCH_CONFIG の場所（deps.env から読む）。"""
    if args.config is None and deps.env.get(CONFIG_ENV_VAR):
        return Path(deps.env[CONFIG_ENV_VAR]).expanduser()
    return args.config


def _read_config(args: argparse.Namespace, deps: Deps) -> Config | None:
    """設定を読む。読めなければ理由を標準エラーに出し、None を返す。"""
    try:
        return load_config(_config_path(args, deps))
    except (ConfigError, OSError, ValueError) as exc:
        print(f"設定のエラー: {exc}", file=deps.stderr)
        return None


def _read_rows(data_dir: Path, name: str) -> list[dict]:
    """JSONL の行を読む。無ければ空。壊れた行は飛ばす。"""
    # 読むだけなので、しきい値と API キーは使わない。
    return list(ResultStore(data_dir, threshold=0.0, api_key=None).read(name))


def _run_search(args: argparse.Namespace, deps: Deps) -> int:
    if not args.query.strip():
        print("使い方のエラー: 検索語が空", file=deps.stderr)
        return EXIT_USAGE
    if args.max_results is not None and args.max_results < 1:
        print("使い方のエラー: --max は 1 以上", file=deps.stderr)
        return EXIT_USAGE

    try:
        config = load_config(_config_path(args, deps))
        dictionary, rules = _load_parts(config)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"設定のエラー: {exc}", file=deps.stderr)
        return EXIT_USAGE

    max_results = args.max_results if args.max_results is not None else config.search.max_results

    if args.dry_run:
        payload = _dry_run_payload(config, args, max_results, dictionary)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=deps.stdout)
        return EXIT_OK

    store = ResultStore(
        config.storage.dir, threshold=config.jev.threshold, api_key=get_api_key(deps.env)
    )
    search = deps.make_search_backend(config, args.allow_scrape and config.search.allow_scrape)
    classifier = deps.make_classifier(config)
    digest: DigestWriter | None = None
    if args.topic is not None and config.memory.enabled:
        digest = DigestWriter(
            deps.make_memory_client(config),
            data_dir=config.storage.dir,
            scope=config.memory.scope,
            source_agent=config.memory.source_agent,
        )
    request = JobRequest(
        job_id=_job_id(deps.clock.now()),
        query=args.query,
        topic=args.topic,
        genre=Genre(args.genre) if args.genre is not None else None,
        max_results=max_results,
        allow_scrape=args.allow_scrape,
        dry_run=False,
    )
    record = run_job(
        request,
        config=config,
        dictionary=dictionary,
        rules=rules,
        search=search,
        classifier=classifier,
        store=store,
        digest=digest,
        clock=deps.clock,
    )
    return _report(record, deps)


class _FailureRecorder:
    """分類器を包み、答えなかった Jev の要求を集める。判定はそのまま返す。"""

    def __init__(self, inner: Classifier, failures: list[str]) -> None:
        self._inner = inner
        self._failures = failures

    def classify(
        self,
        query: str,
        results: Sequence[SearchResult],
        rule_types: Mapping[int, ResultType],
    ) -> tuple[list[ClassifiedResult], JevReport]:
        items, report = self._inner.classify(query, results, rule_types)
        if report.failure:
            self._failures.append(report.failure)
        return items, report


def _jev_failure_codes(failures: Sequence[str]) -> list[str]:
    """JevReport.failure（", " でつなぐ）を code に分け、出てきた順に重複をのける。"""
    return list(dict.fromkeys(code for failure in failures for code in failure.split(", ")))


def _run_batch_vs_single(args: argparse.Namespace, deps: Deps) -> int:
    config = _read_config(args, deps)
    if config is None:
        return EXIT_USAGE
    if not args.log.is_file():
        print(f"使い方のエラー: 結果ログが無い: {args.log}", file=deps.stderr)
        return EXIT_USAGE
    try:
        rules = load_rules(config.rules.path)
    except (OSError, ValueError) as exc:
        print(f"設定のエラー: {exc}", file=deps.stderr)
        return EXIT_USAGE
    if get_api_key(deps.env) is None:
        # キーが無いと Jev の判定が出せない。キーの中身は出さない。
        print(
            "エラー: API キーが無いので Jev を呼べない（環境変数か認証ファイルに置く）",
            file=deps.stderr,
        )
        return EXIT_FAILED

    failures: list[str] = []

    def classifier_for(batch_size: int) -> Classifier:
        """batch_size だけ変えた分類器を作り、答えなかった Jev の要求を集める。"""
        sized = replace(config, jev=replace(config.jev, batch_size=batch_size))
        return _FailureRecorder(deps.make_classifier(sized), failures)

    summary = batch_vs_single(
        _read_rows(args.log.parent, args.log.name),
        classifier_for,
        batch_size=config.jev.batch_size,
        rule_types_for=lambda results: rule_types(results, rules),
    )
    codes = _jev_failure_codes(failures)
    if codes:
        summary["jev_failures"] = codes
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=deps.stdout)
    if codes:
        # JSON は出すが、答えなかった分があるので終了コードは 1。
        print(
            f"エラー: Jev が答えなかった要求がある（{', '.join(codes)}）。"
            "答えなかった結果は一致率に入っていない",
            file=deps.stderr,
        )
        return EXIT_FAILED
    return EXIT_OK


def _run_sample(args: argparse.Namespace, deps: Deps) -> int:
    config = _read_config(args, deps)
    if config is None:
        return EXIT_USAGE
    text = sample_csv(_read_rows(config.storage.dir, RESULTS_FILE), args.n, args.seed)
    if args.out is None:
        deps.stdout.write(text)
        return EXIT_OK
    try:
        # csv の書き方と同じく、改行を変換しない（LF のまま書く）
        with args.out.open("w", encoding="utf-8", newline="") as f:
            f.write(text)
    except OSError as exc:
        print(
            f"使い方のエラー: --out に書けない（{type(exc).__name__}）: {args.out}",
            file=deps.stderr,
        )
        return EXIT_USAGE
    return EXIT_OK


def _run_accuracy(args: argparse.Namespace, deps: Deps) -> int:
    if _read_config(args, deps) is None:
        return EXIT_USAGE
    try:
        text = args.csv.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"使い方のエラー: CSV が無い: {args.csv}", file=deps.stderr)
        return EXIT_USAGE
    except UnicodeDecodeError:
        print("使い方のエラー: CSV は UTF-8 で保存する", file=deps.stderr)
        return EXIT_USAGE
    summary = accuracy_from_labels(text)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=deps.stdout)
    return EXIT_OK


def _run_stats(args: argparse.Namespace, deps: Deps) -> int:
    config = _read_config(args, deps)
    if config is None:
        return EXIT_USAGE
    summary = aggregate(_read_rows(config.storage.dir, EVAL_FILE), since=args.since)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=deps.stdout)
    return EXIT_OK


def _run_digest(args: argparse.Namespace, deps: Deps) -> int:
    """書き残しを書き直し、書けた数と書けない数を出す。書けない数が 1 以上なら終了コード 1。"""
    config = _read_config(args, deps)
    if config is None:
        return EXIT_USAGE
    if not config.memory.enabled:
        print(
            "使い方のエラー: 共有メモリは無効（memory.enabled = false）",
            file=deps.stderr,
        )
        return EXIT_USAGE
    writer = DigestWriter(
        deps.make_memory_client(config),
        data_dir=config.storage.dir,
        scope=config.memory.scope,
        source_agent=config.memory.source_agent,
    )
    written, remaining = writer.retry_pending(_read_rows(config.storage.dir, RESULTS_FILE))
    print(f"書けた {written} 件・まだ {remaining} 件", file=deps.stdout)
    if remaining:
        print("警告: まだ書けないダイジェストが残った", file=deps.stderr)
    return EXIT_FAILED if remaining else EXIT_OK


_EVAL_COMMANDS: dict[str, Callable[[argparse.Namespace, Deps], int]] = {
    "batch-vs-single": _run_batch_vs_single,
    "sample": _run_sample,
    "accuracy": _run_accuracy,
}


def _dispatch(args: argparse.Namespace, deps: Deps) -> int:
    if args.command == "search":
        return _run_search(args, deps)
    if args.command == "eval":
        return _EVAL_COMMANDS[args.eval_command](args, deps)
    if args.command == "stats":
        return _run_stats(args, deps)
    return _run_digest(args, deps)


def main(argv: Sequence[str] | None = None, *, deps: Deps | None = None) -> int:
    """CLI の入口。戻り値が終了コード。"""
    if deps is None:
        deps = Deps()
        # 既定の標準出力のときだけ、文字化けしないよう UTF-8 にする
        for stream in (deps.stdout, deps.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")

    args, code = _parse(_build_parser(), argv, deps)
    if args is None:
        return code if code is not None else EXIT_USAGE

    try:
        return _dispatch(args, deps)
    except Exception as exc:  # noqa: BLE001 - 例外のメッセージにキーが混ざらないよう、クラス名だけ出す
        print(f"内部エラー: {type(exc).__name__}", file=deps.stderr)
        return EXIT_FAILED
