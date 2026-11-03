"""ダイジェスト書き込みのテスト。共有メモリは fake に差し替える。"""

import json
from datetime import date

from jevsearch.digest import DigestWriter, build_digest

SCOPE = "global"
DAY = date(2026, 11, 3)
OTHER_DAY = date(2026, 11, 4)


def _row(url, *, topic="ai", at="2026-11-03T09:00:00+09:00", relevant=True, **kw):
    row = {
        "query": kw.pop("query", "生成AI"),
        "retrieved_at": at,
        "topic": topic,
        "url": url,
        "title": kw.pop("title", f"title of {url}"),
        "engine": kw.pop("engine", "duckduckgo"),
        "type": {"value": kw.pop("type", "news"), "by": "rule", "confidence": None},
        "genre": {"value": kw.pop("genre", "tech"), "by": "jev", "confidence": 0.9},
        "relevant": {"value": relevant, "by": "jev", "confidence": 0.9},
    }
    row.update(kw)
    return row


class FakeMemory:
    def __init__(self, existing=None, fail=None):
        self.existing = existing or []
        self.fail = fail
        self.writes: list[dict] = []
        self.searches: list[dict] = []
        self._n = 0

    def search(self, query, *, scope, tags, limit):
        self.searches.append({"query": query, "scope": scope, "tags": list(tags), "limit": limit})
        if self.fail:
            raise self.fail
        return self.existing

    def write(self, **kwargs):
        if self.fail:
            raise self.fail
        self._n += 1
        self.writes.append(kwargs)
        return f"ID{self._n}"


def _writer(tmp_path, memory):
    return DigestWriter(memory, data_dir=tmp_path, scope=SCOPE, source_agent="claude")


def _pending(tmp_path):
    path = tmp_path / "digest-pending.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


ROWS = [_row("https://a.example/1"), _row("https://b.example/2")]


def test_1回目は_supersedes_なしで書く(tmp_path):
    memory = FakeMemory()
    new_id, reason = _writer(tmp_path, memory).upsert("ai", DAY, ROWS)
    assert (new_id, reason) == ("ID1", None)
    write = memory.writes[0]
    assert write["supersedes"] is None
    assert write["kind"] == "episodic"
    assert write["provenance"] == "agent_inferred"
    assert write["scope"] == SCOPE
    assert write["source_agent"] == "claude"
    index = json.loads((tmp_path / "digest-index.json").read_text(encoding="utf-8"))
    assert index == {"ai|2026-11-03": "ID1"}


def test_同じ日と同じトピックの2回目は_index_の_id_を_supersedes_にする(tmp_path):
    memory = FakeMemory()
    writer = _writer(tmp_path, memory)
    first, _ = writer.upsert("ai", DAY, ROWS)
    second, _ = writer.upsert("ai", DAY, ROWS)
    assert memory.writes[1]["supersedes"] == first
    assert second == "ID2"
    assert len(memory.searches) == 1  # 探したのは1回目だけ（2回目は index で足りる）
    index = json.loads((tmp_path / "digest-index.json").read_text(encoding="utf-8"))
    assert index["ai|2026-11-03"] == "ID2"


def test_index_が無くても_search_で同じ_title_を見つけて_supersedes_にする(tmp_path):
    title, _, _ = build_digest("ai", DAY, ROWS)
    memory = FakeMemory(existing=[{"id": "OTHER", "title": "別"}, {"id": "OLD", "title": title}])
    _writer(tmp_path, memory).upsert("ai", DAY, ROWS)
    assert memory.writes[0]["supersedes"] == "OLD"
    assert memory.searches == [
        {"query": title, "scope": SCOPE, "tags": ["news:radar-bot:ai"], "limit": 20}
    ]


def test_search_に同じ_title_が無ければ_supersedes_なし(tmp_path):
    memory = FakeMemory(existing=[{"id": "OTHER", "title": "別の日"}])
    _writer(tmp_path, memory).upsert("ai", DAY, ROWS)
    assert memory.writes[0]["supersedes"] is None


def test_別の日は_supersedes_なし(tmp_path):
    memory = FakeMemory()
    writer = _writer(tmp_path, memory)
    writer.upsert("ai", DAY, ROWS)
    rows = ROWS + [_row("https://c.example/3", at="2026-11-04T09:00:00+09:00")]
    writer.upsert("ai", OTHER_DAY, rows)
    assert memory.writes[1]["supersedes"] is None
    assert "2026-11-04" in memory.writes[1]["title"]
    assert "https://c.example/3" in memory.writes[1]["content"]
    assert "https://a.example/1" not in memory.writes[1]["content"]


def test_relevant_が_false_の結果は本文に無い_null_は残す():
    rows = [
        _row("https://keep.example/1"),
        _row("https://drop.example/2", relevant=False),
        {**_row("https://null.example/3"), "relevant": {"value": None, "by": "jev"}},
    ]
    _, content, _ = build_digest("ai", DAY, rows)
    assert "https://keep.example/1" in content
    assert "https://drop.example/2" not in content
    assert "https://null.example/3" in content


def test_tags_にトピックと日付が入る():
    _, _, tags = build_digest("ai", DAY, ROWS)
    assert tags == ["news:radar-bot:ai", "news", "radar-bot", "jevsearch", "2026-11-03"]


def test_本文の形と_URL_の重複除去と未分類():
    rows = [
        _row("https://a.example/1", title="一\n番", engine="brave", query="生成AI"),
        _row("https://b.example/2", type=None, genre=None, query="LLM"),
        _row("https://other.example/9", topic="other"),
    ]
    title, content, _ = build_digest("ai", DAY, rows)
    assert title == "jevsearch ai 2026-11-03: 検索結果のダイジェスト"
    assert "件数: 2" in content
    assert "- [news/tech] 一 番 — https://a.example/1（brave）" in content
    assert "- [?/?] title of https://b.example/2 — https://b.example/2（duckduckgo）" in content
    assert "https://other.example/9" not in content
    assert "- 生成AI" in content and "- LLM" in content


def test_同じ_URL_は最後の行で決まり_最初に見えた順を保つ():
    rows = [
        _row("https://a.example/1", title="古い", genre="tech", query="語1"),
        _row("https://b.example/2"),
        _row("https://a.example/1", title="新しい", genre="news", query="語2"),
    ]
    _, content, _ = build_digest("ai", DAY, rows)
    assert "件数: 2" in content
    assert "古い" not in content and "語1" not in content
    lines = [line for line in content.splitlines() if line.startswith("- [")]
    assert lines == [
        "- [news/news] 新しい — https://a.example/1（duckduckgo）",
        "- [news/tech] title of https://b.example/2 — https://b.example/2（duckduckgo）",
    ]


def test_同じ_URL_の最後の行が_relevant_false_ならその_URL_を除く():
    rows = [
        _row("https://a.example/1", relevant=True),
        _row("https://a.example/1", relevant=False, genre="news"),
        {**_row("https://b.example/2"), "relevant": {"value": None, "by": "jev"}},
    ]
    _, content, _ = build_digest("ai", DAY, rows)
    assert "https://a.example/1" not in content
    assert "https://b.example/2" in content
    assert "件数: 1" in content


def test_同じ_URL_の最後の行が_true_なら_前の_false_を打ち消す():
    rows = [
        _row("https://a.example/1", relevant=False),
        _row("https://a.example/1", relevant=True),
    ]
    _, content, _ = build_digest("ai", DAY, rows)
    assert "https://a.example/1" in content


def test_日付は行のオフセットのまま取る(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    rows = [_row("https://a.example/1", at="2026-11-03T00:30:00+09:00")]
    _, content, _ = build_digest("ai", DAY, rows)
    assert "https://a.example/1" in content
    _, content, _ = build_digest("ai", date(2026, 11, 2), rows)
    assert "https://a.example/1" not in content


def test_memory_が例外を投げても_upsert_は例外を出さず書き残しに足す(tmp_path):
    memory = FakeMemory(fail=ConnectionError("down"))
    new_id, reason = _writer(tmp_path, memory).upsert("ai", DAY, ROWS)
    assert new_id is None
    assert "down" in reason
    pending = _pending(tmp_path)
    assert len(pending) == 1
    assert pending[0]["topic"] == "ai"
    assert pending[0]["date"] == "2026-11-03"
    assert "down" in pending[0]["error"]
    assert not (tmp_path / "digest-index.json").exists()


def test_retry_pending_で書けたものが書き残しから消える(tmp_path):
    failing = FakeMemory(fail=ConnectionError("down"))
    _writer(tmp_path, failing).upsert("ai", DAY, ROWS)
    _writer(tmp_path, failing).upsert("ai", DAY, ROWS)  # 同じものが2行になる
    assert len(_pending(tmp_path)) == 2

    memory = FakeMemory()
    written, left = _writer(tmp_path, memory).retry_pending(ROWS)
    assert (written, left) == (1, 0)  # 重複は1回だけ書く
    assert len(memory.writes) == 1
    assert _pending(tmp_path) == []


def test_retry_pending_でまだ書けないものだけが残る(tmp_path):
    failing = FakeMemory(fail=ConnectionError("down"))
    writer = _writer(tmp_path, failing)
    writer.upsert("ai", DAY, ROWS)
    writer.upsert("ai", OTHER_DAY, [_row("https://c.example/3", at="2026-11-04T09:00:00+09:00")])

    class OnlyFirstDay(FakeMemory):
        def write(self, **kwargs):
            if "2026-11-04" in kwargs["title"]:
                raise ConnectionError("still down")
            return super().write(**kwargs)

    rows = ROWS + [_row("https://c.example/3", at="2026-11-04T09:00:00+09:00")]
    written, left = _writer(tmp_path, OnlyFirstDay()).retry_pending(rows)
    assert (written, left) == (1, 1)
    pending = _pending(tmp_path)
    assert [(p["topic"], p["date"]) for p in pending] == [("ai", "2026-11-04")]
    assert "still down" in pending[0]["error"]


def test_書き残しが無ければ_retry_pending_は何もしない(tmp_path):
    memory = FakeMemory()
    assert _writer(tmp_path, memory).retry_pending(ROWS) == (0, 0)
    assert memory.writes == [] and memory.searches == []


def test_rows_の読み出しで例外が出ても_upsert_は例外を出さず書き残しに足す(tmp_path):
    def gen():
        yield _row("https://a.example/1")
        raise OSError("log unreadable")

    memory = FakeMemory()
    new_id, reason = _writer(tmp_path, memory).upsert("ai", DAY, gen())
    assert new_id is None
    assert "log unreadable" in reason
    assert memory.writes == []
    pending = _pending(tmp_path)
    assert len(pending) == 1
    assert (pending[0]["topic"], pending[0]["date"]) == ("ai", "2026-11-03")


def test_retry_pending_は_no_rows_のものを書き残しから外す(tmp_path):
    _writer(tmp_path, FakeMemory(fail=ConnectionError("down"))).upsert("ai", DAY, ROWS)
    assert len(_pending(tmp_path)) == 1

    memory = FakeMemory()
    writer = _writer(tmp_path, memory)
    assert writer.retry_pending([]) == (0, 0)
    assert _pending(tmp_path) == []
    assert writer.retry_pending([]) == (0, 0)
    assert memory.writes == []


def test_retry_pending_は読めない日付のものを書き残しから外す(tmp_path):
    path = tmp_path / "digest-pending.jsonl"
    path.write_text(
        json.dumps({"topic": "ai", "date": "not-a-date", "error": "x"}) + "\n", encoding="utf-8"
    )
    assert _writer(tmp_path, FakeMemory()).retry_pending(ROWS) == (0, 0)
    assert _pending(tmp_path) == []


def test_retry_pending_は実行中に足された行を失わない(tmp_path):
    _writer(tmp_path, FakeMemory(fail=ConnectionError("down"))).upsert("ai", DAY, ROWS)

    class AppendsDuringWrite(FakeMemory):
        def write(self, **kwargs):
            # 別の upsert が失敗して、書き残しに別の日の行を足した、という状況
            with (tmp_path / "digest-pending.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"topic": "ai", "date": "2026-11-04", "error": "e"}) + "\n")
            return super().write(**kwargs)

    written, left = _writer(tmp_path, AppendsDuringWrite()).retry_pending(ROWS)
    assert (written, left) == (1, 0)
    assert [(p["topic"], p["date"]) for p in _pending(tmp_path)] == [("ai", "2026-11-04")]


def test_対象の行が0件なら_write_せず_no_rows_を返す(tmp_path):
    memory = FakeMemory()
    writer = _writer(tmp_path, memory)
    writer.upsert("ai", DAY, ROWS)  # 前の1件
    rows = [_row("https://x.example/1", relevant=False), _row("https://y.example/2", topic="x")]
    assert writer.upsert("ai", DAY, rows) == (None, "no_rows")
    assert len(memory.writes) == 1  # 増えていない（前の1件を supersedes で外さない）
    assert _pending(tmp_path) == []
    assert writer.upsert("ai", DAY, []) == (None, "no_rows")
