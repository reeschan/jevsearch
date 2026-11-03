"""コマンドラインの入口。

今は `search --dry-run`（外部を呼ばずに送る内容を確かめる）だけが動く。
実際の検索ジョブ・eval・stats・digest は後の PBI でつなぐ。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from jevsearch.config import CONFIG_ENV_VAR, Config, ConfigError, load_config, select_engines
from jevsearch.models import Genre, Lang
from jevsearch.ports import Classifier, Clock, MemoryClient, SearchBackend
from jevsearch.search_request import build_search_request

EXIT_OK = 0
EXIT_USAGE = 2
"""使い方・設定のエラーの終了コード。"""

_UNIMPLEMENTED_COMMANDS = ("eval", "stats", "digest")


class _SystemClock:
    """tz 付きの現在時刻を返す既定の時計。"""

    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.UTC)


def _not_wired(name: str) -> Callable[[Config], object]:
    """まだつないでいない部品の仮の作り方。呼ばれたら NotImplementedError。"""

    def make(config: Config) -> object:
        raise NotImplementedError(f"{name} はまだつないでいない")

    return make


@dataclass
class Deps:
    """CLI が外から受け取るもの。テストでは差し替える。"""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    clock: Clock = field(default_factory=_SystemClock)
    make_search_backend: Callable[[Config], SearchBackend] = field(
        default_factory=lambda: _not_wired("検索の接続口")  # type: ignore[arg-type,return-value]
    )
    make_classifier: Callable[[Config], Classifier] = field(
        default_factory=lambda: _not_wired("分類器")  # type: ignore[arg-type,return-value]
    )
    make_memory_client: Callable[[Config], MemoryClient] = field(
        default_factory=lambda: _not_wired("共有メモリ")  # type: ignore[arg-type,return-value]
    )


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
    search.add_argument("--config", type=Path, default=None, help="設定ファイル")

    for name in _UNIMPLEMENTED_COMMANDS:
        other = sub.add_parser(name)
        other.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def _parse(parser: argparse.ArgumentParser, argv: Sequence[str] | None, deps: Deps):
    """引数を読む。使い方の誤りは標準エラー（deps.stderr）に出し、終了コードを返す。"""
    try:
        with contextlib.redirect_stderr(deps.stderr), contextlib.redirect_stdout(deps.stdout):
            return parser.parse_args(argv), None
    except SystemExit as exc:
        code = exc.code
        return None, code if isinstance(code, int) else EXIT_USAGE


def _dry_run_payload(config: Config, args: argparse.Namespace, max_results: int) -> dict:
    genre = Genre(args.genre) if args.genre is not None else None
    engines = select_engines(config, genre, args.allow_scrape)
    url, params = build_search_request(
        config.rustxng.url, args.query, Lang.JA, engines, max_results
    )
    return {
        "query": args.query,
        "query_norm": {"ja": args.query, "en": None},
        "genre": args.genre,
        "topic": args.topic,
        "engines": engines,
        "rustxng": {"requests": [{"lang": Lang.JA.value, "url": url, "params": params}]},
        "jev": {
            "model": config.jev.model,
            "batch_size": config.jev.batch_size,
            "threshold": config.jev.threshold,
        },
        "versions": {"config": config.version},
    }


def _run_search(args: argparse.Namespace, deps: Deps) -> int:
    if not args.query.strip():
        print("使い方のエラー: 検索語が空", file=deps.stderr)
        return EXIT_USAGE
    if args.max_results is not None and args.max_results < 1:
        print("使い方のエラー: --max は 1 以上", file=deps.stderr)
        return EXIT_USAGE

    config_path = args.config
    if config_path is None and deps.env.get(CONFIG_ENV_VAR):
        config_path = Path(deps.env[CONFIG_ENV_VAR]).expanduser()
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"設定のエラー: {exc}", file=deps.stderr)
        return EXIT_USAGE

    if not args.dry_run:
        print("未実装: search（--dry-run なし）は PBI-012 でつなぐ", file=deps.stderr)
        return EXIT_USAGE

    max_results = args.max_results if args.max_results is not None else config.search.max_results
    payload = _dry_run_payload(config, args, max_results)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=deps.stdout)
    return EXIT_OK


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

    if args.command == "search":
        return _run_search(args, deps)

    print(f"未実装: {args.command} は PBI-012・PBI-013 でつなぐ", file=deps.stderr)
    return EXIT_USAGE
