"""設定ファイル（TOML）の読み込みと、ジャンルからのエンジン一覧の選び方。

設定は秘密を持たない（API キーは環境変数か認証ファイルから別に読む）。
ファイルにないキーは既定値で埋める（部分的な設定ファイルを許す）。
"""

from __future__ import annotations

import copy
import hashlib
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jevsearch.models import JEV_MODEL, Genre

CONFIG_ENV_VAR = "JEVSEARCH_CONFIG"
"""設定ファイルの場所を指す環境変数の名前。"""

DEFAULT_VERSION = "default"
"""ファイルがなく、既定値を使ったときの版。"""

VERSION_LENGTH = 12
"""版に使う sha256 の先頭の桁数。"""

# キー名に含まれてはいけない語（小文字で比べる）
_SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password")

# 上の語を含むが秘密ではないキー名（トークン数の上限など）
_SECRET_KEY_ALLOWLIST = frozenset({"max_request_tokens"})

JEV_ALLOWED_HOSTS = ("api.typesafe.ai",)
"""jev.endpoint に使ってよいホスト名。API キーは TypeSafe への認証ヘッダー以外に送らない。"""

_DATA_DIR = Path(__file__).resolve().parent / "data"

_GENRE_VALUES = tuple(g.value for g in Genre)


class ConfigError(Exception):
    """設定が不正なときの例外。メッセージにはキーの場所だけを書き、値は書かない。"""


@dataclass(frozen=True)
class RustxngConfig:
    """検索エンジンの接続先（RustXNG）。"""

    url: str
    expected_version: str
    timeout_seconds: float


@dataclass(frozen=True)
class SearchConfig:
    """検索全体の設定。"""

    max_results: int
    allow_scrape: bool


@dataclass(frozen=True)
class EnginesConfig:
    """エンジン名の一覧。キーはジャンル。"""

    official: Mapping[Genre, tuple[str, ...]]
    scrape: Mapping[Genre, tuple[str, ...]]
    unspecified: tuple[str, ...]
    scrape_unspecified: tuple[str, ...]


@dataclass(frozen=True)
class JevConfig:
    """Jev（分類モデル）の呼び方。"""

    model: str
    endpoint: str
    batch_size: int
    threshold: float
    max_request_tokens: int
    daily_request_limit: int | None
    max_rate_retries: int
    timeout_seconds: float
    price_per_mtok_input: float | None
    price_per_mtok_output: float | None


@dataclass(frozen=True)
class StorageConfig:
    """保存先。`~` は展開済み。"""

    dir: Path


@dataclass(frozen=True)
class MemoryConfig:
    """共有メモリ（ダイジェストの書き先）。"""

    enabled: bool
    url: str
    scope: str
    source_agent: str


@dataclass(frozen=True)
class DictionaryConfig:
    """用語辞書の場所。`~` は展開済み。"""

    path: Path


@dataclass(frozen=True)
class RulesConfig:
    """ドメイン規則の場所。`~` は展開済み。"""

    path: Path


@dataclass(frozen=True)
class Config:
    """読み込んだ設定。version は読んだファイルの中身の版。"""

    version: str
    rustxng: RustxngConfig
    search: SearchConfig
    engines: EnginesConfig
    jev: JevConfig
    storage: StorageConfig
    memory: MemoryConfig
    dictionary: DictionaryConfig
    rules: RulesConfig


# config.example.toml と同じ値。None になりうるキーは、省略されたとき None のままにする。
_DEFAULTS: dict[str, Any] = {
    "rustxng": {
        "url": "http://127.0.0.1:29483",
        "expected_version": "0.26.0",
        "timeout_seconds": 20,
    },
    "search": {"max_results": 50, "allow_scrape": False},
    "engines": {
        "official": {
            "paper": ["arxiv", "crossref", "semantic scholar", "openalex", "pubmed", "europepmc"],
            "tech": [
                "github",
                "stack overflow",
                "hackernews",
                "mdn",
                "microsoft_learn",
                "gitlab",
                "crates.io",
                "npm",
            ],
            "news": ["google news", "hackernews", "wikinews"],
            "sns": ["mastodon", "lemmy", "tootfinder"],
            "blog": ["wiby"],
            "general": ["wikipedia", "mwmbl"],
        },
        "scrape": {
            "paper": ["google_scholar"],
            "tech": ["pypi", "lib_rs"],
            "news": ["bing news"],
            "sns": [],
            "blog": ["searchmysite", "neocities"],
            "general": ["google", "bing", "brave", "duckduckgo", "mojeek", "yahoo japan"],
        },
        "unspecified": [
            "wikipedia",
            "mwmbl",
            "github",
            "stack overflow",
            "hackernews",
            "arxiv",
            "semantic scholar",
            "openalex",
        ],
        "scrape_unspecified": ["google", "bing", "brave", "duckduckgo"],
    },
    "jev": {
        "model": JEV_MODEL,
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "batch_size": 25,
        "threshold": 0.55,
        "max_request_tokens": 64000,
        "daily_request_limit": 200,
        "max_rate_retries": 3,
        "timeout_seconds": 120,
        "price_per_mtok_input": None,
        "price_per_mtok_output": None,
    },
    "storage": {"dir": "~/.local/share/jevsearch"},
    "memory": {
        "enabled": True,
        "url": "http://127.0.0.1:7820/mcp",
        "scope": "global",
        "source_agent": "claude",
    },
    "dictionary": {"path": str(_DATA_DIR / "dictionary.toml")},
    "rules": {"path": str(_DATA_DIR / "domain_rules.toml")},
}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """base に override を深く重ねた新しい辞書を返す。表（辞書）どうしだけ再帰する。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _check_no_secret_keys(value: Any, where: str = "") -> None:
    """どの節のどのキー名にも秘密を思わせる語が含まれていないことを確かめる。

    表だけでなく、配列の中（表の配列・インライン表・入れ子の配列）も深さを問わず調べる。
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{where}.{key}" if where else str(key)
            lowered = str(key).lower()
            if lowered not in _SECRET_KEY_ALLOWLIST and any(m in lowered for m in _SECRET_MARKERS):
                raise ConfigError(f"設定に秘密を書いてはいけない: キー {path}")
            _check_no_secret_keys(child, path)
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _check_no_secret_keys(child, f"{where}[{index}]")


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] は表でなければならない")
    return value


def _str(table: Mapping[str, Any], section: str, key: str) -> str:
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{section}.{key} は文字列でなければならない")
    return value


def _bool(table: Mapping[str, Any], section: str, key: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} は true か false でなければならない")
    return value


def _int(table: Mapping[str, Any], section: str, key: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section}.{key} は整数でなければならない")
    return value


def _number(table: Mapping[str, Any], section: str, key: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{section}.{key} は数でなければならない")
    return float(value)


def _optional_number(table: Mapping[str, Any], section: str, key: str) -> float | None:
    if table.get(key) is None:
        return None
    return _number(table, section, key)


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} は文字列の配列でなければならない")
    return tuple(value)


def _genre_lists(table: Mapping[str, Any], where: str) -> dict[Genre, tuple[str, ...]]:
    """ジャンル名をキーにしたエンジン一覧の表を読む。知らないジャンル名は拒む。"""
    unknown = [k for k in table if k not in _GENRE_VALUES]
    if unknown:
        raise ConfigError(f"{where} に知らないジャンル名がある: {', '.join(map(str, unknown))}")
    return {Genre(k): _str_list(v, f"{where}.{k}") for k, v in table.items()}


def _check_jev_endpoint(endpoint: str) -> None:
    """jev.endpoint が https で、許可したホストであることを確かめる（値はメッセージに書かない）。"""
    hosts = ", ".join(JEV_ALLOWED_HOSTS)
    error = ConfigError(f"jev.endpoint は https で、ホストが {hosts} のものでなければならない")
    if "\\" in endpoint or any(c.isspace() for c in endpoint):
        raise error
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise error from None
    if parsed.scheme != "https" or parsed.hostname not in JEV_ALLOWED_HOSTS:
        raise error
    if parsed.username is not None or parsed.password is not None or port not in (None, 443):
        raise error


def _check_rustxng_url(url: str) -> None:
    """rustxng.url が http か https の URL であることを確かめる（値はメッセージに書かない）。"""
    error = ConfigError("rustxng.url は http か https の URL（ホスト名つき）でなければならない")
    if "\\" in url or any(c.isspace() for c in url):
        raise error
    try:
        parsed = urlsplit(url)
        parsed.port  # noqa: B018  範囲外のポートは ValueError になる
        hostname = parsed.hostname
    except ValueError:
        raise error from None
    if parsed.scheme not in ("http", "https") or not hostname:
        raise error


def _check_engine_names(lists: list[tuple[str, tuple[str, ...]]]) -> None:
    """エンジン名が空・空白だけ・前後に空白・カンマ入りでないことを確かめる。

    接続口は名前をカンマで分けて空白を落としてから比べて送る。読み込み時の比べ方と
    ずれないよう、そうなる名前は最初から拒む。
    """
    for where, names in lists:
        for name in names:
            if not name.strip() or name != name.strip() or "," in name:
                raise ConfigError(
                    f"{where} のエンジン名は空・空白だけ・前後に空白・カンマ入りにできない"
                    f": {name!r}"
                )


def _check_engine_lists(engines: EnginesConfig) -> None:
    """エンジン名の形を確かめる。公式・未指定の一覧は空でなく、画面読み取りの名前を含まない。"""
    scrape: set[str] = set(engines.scrape_unspecified)
    for names in engines.scrape.values():
        scrape.update(names)

    lists: list[tuple[str, tuple[str, ...]]] = [
        (f"engines.official.{genre.value}", names) for genre, names in engines.official.items()
    ]
    lists.append(("engines.unspecified", engines.unspecified))
    scrape_lists: list[tuple[str, tuple[str, ...]]] = [
        (f"engines.scrape.{genre.value}", names) for genre, names in engines.scrape.items()
    ]
    scrape_lists.append(("engines.scrape_unspecified", engines.scrape_unspecified))
    _check_engine_names(lists + scrape_lists)
    for where, names in lists:
        if not names:
            raise ConfigError(f"{where} は空にできない")
        for name in names:
            if name in scrape:
                raise ConfigError(
                    f"{where} に画面読み取りのエンジンを入れてはいけない: {name}"
                    "（engines.scrape / engines.scrape_unspecified に書く）"
                )


def _expand(path_text: str) -> Path:
    return Path(path_text).expanduser()


def _build(raw: Mapping[str, Any], version: str) -> Config:
    """重ね終えた辞書を検査して Config にする。"""
    for name in _DEFAULTS:
        _section(raw, name)

    rx = _section(raw, "rustxng")
    rustxng = RustxngConfig(
        url=_str(rx, "rustxng", "url"),
        expected_version=_str(rx, "rustxng", "expected_version"),
        timeout_seconds=_number(rx, "rustxng", "timeout_seconds"),
    )
    _check_rustxng_url(rustxng.url)

    se = _section(raw, "search")
    search = SearchConfig(
        max_results=_int(se, "search", "max_results"),
        allow_scrape=_bool(se, "search", "allow_scrape"),
    )
    if not 1 <= search.max_results <= 200:
        raise ConfigError("search.max_results は 1〜200 でなければならない")

    en = _section(raw, "engines")
    engines = EnginesConfig(
        official=_genre_lists(_section(en, "official"), "engines.official"),
        scrape=_genre_lists(_section(en, "scrape"), "engines.scrape"),
        unspecified=_str_list(en["unspecified"], "engines.unspecified"),
        scrape_unspecified=_str_list(en["scrape_unspecified"], "engines.scrape_unspecified"),
    )
    _check_engine_lists(engines)

    jv = _section(raw, "jev")
    limit = jv.get("daily_request_limit")
    daily_limit: int | None = None
    if limit is not None:
        daily_limit = _int(jv, "jev", "daily_request_limit")
        if daily_limit <= 0:
            daily_limit = None
    jev = JevConfig(
        model=_str(jv, "jev", "model"),
        endpoint=_str(jv, "jev", "endpoint"),
        batch_size=_int(jv, "jev", "batch_size"),
        threshold=_number(jv, "jev", "threshold"),
        max_request_tokens=_int(jv, "jev", "max_request_tokens"),
        daily_request_limit=daily_limit,
        max_rate_retries=_int(jv, "jev", "max_rate_retries"),
        timeout_seconds=_number(jv, "jev", "timeout_seconds"),
        price_per_mtok_input=_optional_number(jv, "jev", "price_per_mtok_input"),
        price_per_mtok_output=_optional_number(jv, "jev", "price_per_mtok_output"),
    )
    if jev.model != JEV_MODEL:
        raise ConfigError(f"jev.model は {JEV_MODEL} でなければならない")
    _check_jev_endpoint(jev.endpoint)
    if not 20 <= jev.batch_size <= 30:
        raise ConfigError("jev.batch_size は 20〜30 でなければならない")
    if not 0 <= jev.threshold <= 1:
        raise ConfigError("jev.threshold は 0〜1 でなければならない")

    mem = _section(raw, "memory")
    memory = MemoryConfig(
        enabled=_bool(mem, "memory", "enabled"),
        url=_str(mem, "memory", "url"),
        scope=_str(mem, "memory", "scope"),
        source_agent=_str(mem, "memory", "source_agent"),
    )

    return Config(
        version=version,
        rustxng=rustxng,
        search=search,
        engines=engines,
        jev=jev,
        storage=StorageConfig(dir=_expand(_str(_section(raw, "storage"), "storage", "dir"))),
        memory=memory,
        dictionary=DictionaryConfig(
            path=_expand(_str(_section(raw, "dictionary"), "dictionary", "path"))
        ),
        rules=RulesConfig(path=_expand(_str(_section(raw, "rules"), "rules", "path"))),
    )


def _find_config_path(path: Path | None) -> Path | None:
    """読むファイルを決める。引数 → 環境変数 → 利用者の既定の場所 → なし（既定値）。"""
    if path is not None:
        return path
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser()
    default = Path.home() / ".config" / "jevsearch" / "config.toml"
    return default if default.is_file() else None


def load_config(path: Path | None = None) -> Config:
    """設定を読む。

    path が None なら環境変数 JEVSEARCH_CONFIG、次に ~/.config/jevsearch/config.toml を探す。
    どれも無ければ既定値を使い、version は "default" になる。
    path か環境変数で示したファイルが読めないときは ConfigError。
    """
    found = _find_config_path(path)
    if found is None:
        return _build(_DEFAULTS, DEFAULT_VERSION)

    try:
        data = found.read_bytes()
    except OSError as exc:
        raise ConfigError(f"設定ファイルを読めない: {found}") from exc
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"設定ファイルが TOML として読めない: {found}") from exc

    _check_no_secret_keys(parsed)
    version = hashlib.sha256(data).hexdigest()[:VERSION_LENGTH]
    return _build(_merge(_DEFAULTS, parsed), version)


def select_engines(config: Config, genre: Genre | None, allow_scrape_cli: bool) -> list[str]:
    """検索に使うエンジン名の一覧を返す（重複は最初の1つを残す）。

    genre があればそのジャンルの公式エンジン、なければジャンル未指定の一覧。
    画面読み取りのエンジンは、CLI の許可と設定の許可の両方があるときだけ後ろに足す。
    """
    if genre is not None:
        names = list(config.engines.official.get(genre, ()))
    else:
        names = list(config.engines.unspecified)

    if allow_scrape_cli and config.search.allow_scrape:
        if genre is not None:
            names.extend(config.engines.scrape.get(genre, ()))
        else:
            names.extend(config.engines.scrape_unspecified)

    return list(dict.fromkeys(names))


def scrape_engine_names(config: Config) -> set[str]:
    """画面読み取りの一覧に出てくる全エンジン名（許可がないとき接続口が落とすのに使う）。"""
    names: set[str] = set(config.engines.scrape_unspecified)
    for engines in config.engines.scrape.values():
        names.update(engines)
    return names
