"""ドメイン規則による種類付け。URL のホストだけで ResultType を決める。

規則は同梱の TOML（jevsearch/data/domain_rules.toml）に版付きで置く。
ジャンルと関連性は規則では決めない。
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from jevsearch.models import JudgedBy, Judgment, ResultType, SearchResult

BUNDLED_RULES_PATH = Path(__file__).resolve().parent / "data" / "domain_rules.toml"

_WWW_PREFIX = "www."


@dataclass(frozen=True)
class DomainRule:
    """1つの規則。"""

    type: ResultType
    domains: tuple[str, ...]


@dataclass(frozen=True)
class DomainRules:
    """版付きの規則の一覧。先に書いた規則が勝つ。"""

    version: str
    rules: tuple[DomainRule, ...]


def domain_of(url: str) -> str:
    """ホスト名を小文字で返す。先頭の www. は落とす。読めない URL は空文字。"""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return ""
    if not host:
        return ""
    host = host.rstrip(".")
    if host.startswith(_WWW_PREFIX) and len(host) > len(_WWW_PREFIX):
        host = host[len(_WWW_PREFIX) :]
    return host


def type_by_domain(url: str, rules: DomainRules) -> ResultType | None:
    """URL のホストに当たる最初の規則の種類。当たる規則がなければ None。"""
    host = domain_of(url)
    if not host:
        return None
    for rule in rules.rules:
        if any(_matches(host, pattern) for pattern in rule.domains):
            return rule.type
    return None


def rule_judgment(url: str, rules: DomainRules) -> Judgment | None:
    """規則で決まれば確信度 1.0・by rule の判定。決まらなければ None。"""
    result_type = type_by_domain(url, rules)
    if result_type is None:
        return None
    return Judgment(value=result_type, by=JudgedBy.RULE, confidence=1.0)


def rule_types(results: Sequence[SearchResult], rules: DomainRules) -> dict[int, ResultType]:
    """規則で決まった結果の位置（添字）から種類へ。決まらない結果は入れない。"""
    decided: dict[int, ResultType] = {}
    for position, result in enumerate(results):
        result_type = type_by_domain(result.url, rules)
        if result_type is not None:
            decided[position] = result_type
    return decided


def load_rules(path: Path) -> DomainRules:
    """TOML の規則ファイルを読む。version と各規則を検査する。"""
    with path.open("rb") as f:
        data = tomllib.load(f)
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"version が空か、文字列ではない: {path}")
    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"rules は [[rules]] の並びにする: {path}")
    return DomainRules(
        version=version,
        rules=tuple(_parse_rule(raw, path) for raw in raw_rules),
    )


def load_bundled_rules() -> DomainRules:
    """同梱の規則（jevsearch/data/domain_rules.toml）を読む。"""
    return load_rules(BUNDLED_RULES_PATH)


def _parse_rule(raw: object, path: Path) -> DomainRule:
    if not isinstance(raw, dict):
        raise ValueError(f"規則の形が不正: {path}")
    result_type = ResultType(raw.get("type"))
    domains = raw.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(f"domains は1つ以上の文字列の一覧にする: {path}")
    patterns: list[str] = []
    for domain in domains:
        if not isinstance(domain, str):
            raise ValueError(f"domains の要素は文字列にする: {path}")
        pattern = domain.strip().lower()
        _check_pattern(pattern)
        patterns.append(pattern)
    return DomainRule(type=result_type, domains=tuple(patterns))


def _check_pattern(pattern: str) -> None:
    """書き方は3種類だけ。完全一致、先頭の `*.`、先頭ラベルの `.*`。"""
    if pattern.startswith("*."):
        body = pattern[2:]
    elif pattern.endswith(".*"):
        body = pattern[:-2]
    else:
        body = pattern
    if not body or "*" in body or body.startswith(".") or body.endswith("."):
        raise ValueError(f"ドメインの書き方が不正: {pattern!r}")


def _matches(host: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        # 先頭の `*.` は、その下のホストだけに当てる（例えば `*.hatenablog.com`）。
        return host.endswith(pattern[1:])
    if pattern.endswith(".*"):
        # 先頭ラベルで当てる（例えば `docs.*` は docs.python.org に当たる）。
        return host.startswith(pattern[:-1])
    # 完全一致。サブドメインにも当たる（例えば export.arxiv.org は arxiv.org に当たる）。
    return host == pattern or host.endswith("." + pattern)
