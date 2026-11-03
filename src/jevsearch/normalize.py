"""正規化（用語辞書、seam ②）。

用語辞書だけで検索語の表記ゆれをそろえ、辞書に英語表記がある語は英語の検索語も作る。
辞書にない語はそのまま通す。LLM による言い換えや英訳はしない。
辞書を入力にした純粋な変換（ファイルを読むのは辞書の読み込み関数だけ）。
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from jevsearch.models import NormalizedQuery

_BUNDLED_PACKAGE = "jevsearch"
_BUNDLED_PATH = ("data", "dictionary.toml")

# 英数字の連なりの一部には当てないための判定
# （ASCII の英数字を含む語は、語まるごと一致のときだけ置き換える）。
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Term:
    """辞書の1項目。"""

    canonical: str
    aliases: tuple[str, ...] = ()
    en: str | None = None


@dataclass(frozen=True)
class Dictionary:
    """用語辞書。version は記録に残る。"""

    version: str
    entries: tuple[Term, ...]


def _parse(data: dict, source: str) -> Dictionary:
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{source}: version が空か、ありません")
    entries: list[Term] = []
    for i, raw in enumerate(data.get("terms", [])):
        canonical = raw.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"{source}: terms[{i}] に canonical がありません")
        entries.append(
            Term(
                canonical=canonical,
                aliases=tuple(raw.get("aliases", [])),
                en=raw.get("en") or None,
            )
        )
    return Dictionary(version=version, entries=tuple(entries))


def load_dictionary(path: Path) -> Dictionary:
    """TOML の用語辞書を読む。"""
    with Path(path).open("rb") as f:
        return _parse(tomllib.load(f), str(path))


def load_bundled_dictionary() -> Dictionary:
    """パッケージに同梱した用語辞書（jevsearch/data/dictionary.toml）を読む。"""
    resource = resources.files(_BUNDLED_PACKAGE)
    for part in _BUNDLED_PATH:
        resource = resource / part
    return _parse(tomllib.loads(resource.read_text(encoding="utf-8")), "/".join(_BUNDLED_PATH))


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@lru_cache(maxsize=8)
def _index(dictionary: Dictionary) -> tuple[dict[str, Term], re.Pattern[str] | None]:
    """見出し語・別名から項目を引く表と、連なりの中で当てる正規表現を作る。

    先に載っている項目を優先する。正規表現は長い語を先に当てる
    （見出し語自身を含めるので、「サーバー」の「サーバ」だけを置き換えて壊すことがない）。
    """
    lookup: dict[str, Term] = {}
    for term in dictionary.entries:
        for surface in (term.canonical, *term.aliases):
            lookup.setdefault(_fold(surface), term)
    inner = sorted(
        (key for key in lookup if key and not _ASCII_ALNUM.search(key)),
        key=len,
        reverse=True,
    )
    pattern = re.compile("|".join(re.escape(k) for k in inner)) if inner else None
    return lookup, pattern


def _clean(query: str) -> str:
    text = unicodedata.normalize("NFKC", query)
    return _SPACES.sub(" ", text).strip()


def normalize(query: str, dictionary: Dictionary) -> NormalizedQuery:
    """検索語を用語辞書でそろえる。

    遷移: 用語辞書で検索語をそろえ終えた

    文字の正規化（NFKC・前後の空白・連続する空白）の後、空白で区切った各語を
    見出し語・別名と大文字小文字を区別せず照らして、そろえた言い方に置き換える。
    空白で区切れない日本語の連なりの中の別名も置き換える（長い語を先に当てる）。
    ASCII の英数字を含む語は、語まるごと一致のときだけ置き換える
    （例えば「DBMS」の中の「DB」には当てない）。
    英語の検索語は、置き換えた語のうち1つ以上に英語表記があるときだけ作る。
    """
    lookup, pattern = _index(dictionary)
    ja_tokens: list[str] = []
    en_tokens: list[str] = []
    has_en = False

    for token in _clean(query).split(" ") if _clean(query) else []:
        whole = lookup.get(_fold(token))
        if whole is not None:
            ja_tokens.append(whole.canonical)
            en_tokens.append(whole.en or whole.canonical)
            has_en = has_en or whole.en is not None
            continue
        if pattern is None:
            ja_tokens.append(token)
            en_tokens.append(token)
            continue

        ja_parts: list[str] = []
        en_parts: list[str] = []
        pos = 0
        folded = token.casefold()
        # casefold で長さが変わる文字があると位置がずれるので、その語は連なりの置き換えをしない。
        if len(folded) != len(token):
            ja_tokens.append(token)
            en_tokens.append(token)
            continue
        for m in pattern.finditer(folded):
            term = lookup[m.group(0)]
            ja_parts += [token[pos : m.start()], term.canonical]
            en_parts += [token[pos : m.start()], term.en or term.canonical]
            has_en = has_en or term.en is not None
            pos = m.end()
        ja_parts.append(token[pos:])
        en_parts.append(token[pos:])
        ja_tokens.append("".join(ja_parts))
        en_tokens.append("".join(en_parts))

    return NormalizedQuery(
        original=query,
        ja=" ".join(ja_tokens),
        en=" ".join(en_tokens) if has_en else None,
        dictionary_version=dictionary.version,
    )
