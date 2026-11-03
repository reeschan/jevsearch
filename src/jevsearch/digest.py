"""検索結果を、トピックと日付ごとに1件のダイジェストとして共有メモリに書く。

同じ日・同じトピックで何度書いても、`supersedes` で最新の1件だけが生きる。
書けなかったものは書き残しファイルに残し、あとで `retry_pending` で書き直す。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from jevsearch.ports import MemoryClient

INDEX_FILE = "digest-index.json"
PENDING_FILE = "digest-pending.jsonl"
NO_ROWS = "no_rows"

_KIND = "episodic"
_PROVENANCE = "agent_inferred"
_UNKNOWN = "?"
_SEARCH_LIMIT = 20
_ERROR_LIMIT = 500


def _topic_tag(topic: str) -> str:
    return f"news:radar-bot:{topic}"


def _row_date(row: dict) -> date | None:
    """行に書かれたオフセットのままの日付（変換しない）。読めなければ None。"""
    try:
        return datetime.fromisoformat(row["retrieved_at"]).date()
    except (KeyError, TypeError, ValueError):
        return None


def _judged(row: dict, key: str):
    judgment = row.get(key)
    return judgment.get("value") if isinstance(judgment, dict) else None


def _one_line(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _label(value: object) -> str:
    return _UNKNOWN if value is None else str(value)


def _select_rows(topic: str, day: date, rows: Iterable[dict]) -> list[dict]:
    """topic と日付が合う行を、URL で重複を除いて最初に見えた順に返す。relevant が false は除く。"""
    selected: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("topic") != topic or _row_date(row) != day:
            continue
        if _judged(row, "relevant") is False:
            continue
        url = row.get("url")
        if url in seen:
            continue
        seen.add(url)
        selected.append(row)
    return selected


def build_digest(topic: str, day: date, rows: Iterable[dict]) -> tuple[str, str, list[str]]:
    """(title, 本文の Markdown, tags) を作る。"""
    selected = _select_rows(topic, day, rows)
    day_text = day.isoformat()
    title = f"jevsearch {topic} {day_text}: 検索結果のダイジェスト"

    queries: list[str] = []
    for row in selected:
        query = row.get("query")
        if query and query not in queries:
            queries.append(query)

    lines = [
        f"# {title}",
        "",
        f"- トピック: {topic}",
        f"- 日付: {day_text}",
        f"- 件数: {len(selected)}",
        "",
        "## 検索語",
        "",
    ]
    lines += [f"- {_one_line(q)}" for q in queries] or ["- （なし）"]
    lines += ["", "## 結果", ""]
    for row in selected:
        kind = _label(_judged(row, "type"))
        genre = _label(_judged(row, "genre"))
        lines.append(
            f"- [{kind}/{genre}] {_one_line(row.get('title', ''))} — {row.get('url', '')}"
            f"（{row.get('engine', '')}）"
        )
    tags = [_topic_tag(topic), "news", "radar-bot", "jevsearch", day_text]
    return title, "\n".join(lines) + "\n", tags


class DigestWriter:
    def __init__(
        self, memory: MemoryClient, *, data_dir: Path, scope: str, source_agent: str
    ) -> None:
        self._memory = memory
        self._index_path = Path(data_dir) / INDEX_FILE
        self._pending_path = Path(data_dir) / PENDING_FILE
        self._scope = scope
        self._source_agent = source_agent

    # --- index ---

    def _read_index(self) -> dict:
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict) -> None:
        self._atomic_write(self._index_path, json.dumps(index, ensure_ascii=False, indent=1))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # --- pending ---

    def _read_pending_lines(self) -> list[str]:
        try:
            text = self._pending_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return text.splitlines()

    @staticmethod
    def _parse_pending(lines: list[str]) -> list[dict]:
        entries = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and "topic" in entry and "date" in entry:
                entries.append(entry)
        return entries

    def _append_pending(self, topic: str, day: date, error: str) -> None:
        entry = {"topic": topic, "date": day.isoformat(), "error": error}
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- 書く ---

    def _try_upsert(
        self, topic: str, day: date, rows: list[dict]
    ) -> tuple[str | None, str | None]:
        """書き残しには触らない。失敗は例外のまま出る。"""
        if not _select_rows(topic, day, rows):
            return None, NO_ROWS
        title, content, tags = build_digest(topic, day, rows)
        key = f"{topic}|{day.isoformat()}"
        index = self._read_index()
        supersedes = index.get(key)
        if supersedes is None:
            found = self._memory.search(
                title, scope=self._scope, tags=[_topic_tag(topic)], limit=_SEARCH_LIMIT
            )
            for memory in found:
                if memory.get("title") == title and memory.get("id"):
                    supersedes = memory["id"]
                    break
        new_id = self._memory.write(
            title=title,
            content=content,
            scope=self._scope,
            kind=_KIND,
            provenance=_PROVENANCE,
            source_agent=self._source_agent,
            tags=tags,
            supersedes=supersedes,
        )
        index[key] = new_id
        self._write_index(index)
        return new_id, None

    def upsert(self, topic: str, day: date, rows: Iterable[dict]) -> tuple[str | None, str | None]:
        """(新しい id, 失敗の理由)。対象の行が0件なら何も書かず (None, "no_rows")。"""
        try:
            return self._try_upsert(topic, day, list(rows))
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"[:_ERROR_LIMIT]
            try:
                self._append_pending(topic, day, reason)
            except Exception as pending_error:
                reason = f"{reason}（書き残しにも失敗: {type(pending_error).__name__}）"
            return None, reason

    def retry_pending(self, rows: Iterable[dict]) -> tuple[int, int]:
        """(書けた数, まだ書けない数)。例外で書けなかったものだけを書き残しに残す。

        対象の行が無い（no_rows）ものと、日付が読めないものは、あとで行が増えることが
        無いので書き残しから外し、書けた数にも まだ書けない数にも数えない。
        最初に読んだあとで他から足された行は、そのまま残す。
        """
        rows = list(rows)
        first_lines = self._read_pending_lines()
        keys = dict.fromkeys(
            (str(entry["topic"]), str(entry["date"])) for entry in self._parse_pending(first_lines)
        )
        written = 0
        remaining: list[dict] = []
        for topic, day_text in keys:
            try:
                day = date.fromisoformat(day_text)
            except ValueError:
                continue  # 読めない日付は直しようが無いので外す
            try:
                new_id, reason = self._try_upsert(topic, day, rows)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"[:_ERROR_LIMIT]
                remaining.append({"topic": topic, "date": day_text, "error": error})
                continue
            if new_id is not None:
                written += 1
            # new_id が None で例外が無いのは no_rows だけ。外す。
        # 書き換える直前に読み直し、最初に読んだ分より後ろに足された行を残す。
        added = self._read_pending_lines()[len(first_lines) :]
        text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in remaining)
        text += "".join(line + "\n" for line in added)
        self._atomic_write(self._pending_path, text)
        return written, len(remaining)
