"""共有の型（models.py）と差し替え口（ports.py）の名前・形を固定するテスト。"""

import dataclasses
import datetime
import re
import socket
from enum import StrEnum
from pathlib import Path

import pytest

from jevsearch import models, ports
from jevsearch.models import JobState, Outcome
from jevsearch.store import result_rows

MMD = Path(__file__).resolve().parent.parent / "docs/specs/statechart/jevsearch-job.mmd"

STATE_RE = re.compile(r'^\s*state\s+"([^"]+)"\s+as\s+(\w+)\s*$')
EDGE_RE = re.compile(r"^\s*(\[\*\]|\w+)\s*-->\s*(\w+)\s*:\s*(.+?)\s*$")


def _snake(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


def _parse_mmd() -> set[tuple[JobState | None, JobState, str]]:
    edges = set()
    for line in MMD.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("%%"):
            continue
        m = EDGE_RE.match(line)
        if not m:
            continue
        src, dst, label = m.groups()
        source = None if src == "[*]" else JobState(_snake(src))
        edges.add((source, JobState(_snake(dst)), label))
    return edges


def test_transitions_match_diagram():
    """図そのものとの照合のテスト。図の12遷移（元, 先, ラベル）が TRANSITIONS と等しい。"""
    from_diagram = _parse_mmd()
    assert len(from_diagram) == 12
    assert len(models.TRANSITIONS) == 12
    assert set(models.TRANSITIONS) == from_diagram


def test_diagram_states_are_job_states():
    ids = set()
    for line in MMD.read_text(encoding="utf-8").splitlines():
        m = STATE_RE.match(line)
        if m:
            ids.add(JobState(_snake(m.group(2))))
    assert ids == set(JobState)


def test_outcome_values():
    assert {o.value for o in Outcome} == {"saved", "partial", "no_results", "failed"}


def test_final_state_covers_every_outcome():
    assert models.FINAL_STATE == {
        Outcome.SAVED: JobState.SAVED,
        Outcome.PARTIAL: JobState.PARTIALLY_SAVED,
        Outcome.NO_RESULTS: JobState.NO_RESULTS,
        Outcome.FAILED: JobState.FAILED,
    }


def test_jev_model():
    assert models.JEV_MODEL == "jev-1.14.0"


def test_network_sentinel_blocks_connections():
    with pytest.raises(RuntimeError):
        socket.create_connection(("127.0.0.1", 9))


ENUMS = {
    "Lang": {"JA": "ja", "EN": "en"},
    "Outcome": {
        "SAVED": "saved",
        "PARTIAL": "partial",
        "NO_RESULTS": "no_results",
        "FAILED": "failed",
    },
    "JobState": {
        "ACCEPTED": "accepted",
        "NORMALIZING": "normalizing",
        "SEARCHING": "searching",
        "FAILED": "failed",
        "NO_RESULTS": "no_results",
        "RULE_TAGGING": "rule_tagging",
        "JEV_CLASSIFY": "jev_classify",
        "SAVED": "saved",
        "PARTIALLY_SAVED": "partially_saved",
    },
    "Genre": {
        "PAPER": "paper",
        "TECH": "tech",
        "NEWS": "news",
        "SNS": "sns",
        "BLOG": "blog",
        "GENERAL": "general",
    },
    "ResultType": {
        "PAPER": "paper",
        "BLOG": "blog",
        "NEWS": "news",
        "SNS": "sns",
        "DOCS": "docs",
        "REPO": "repo",
        "OTHER": "other",
    },
    "JudgedBy": {"RULE": "rule", "JEV": "jev"},
    "JudgmentKey": {"TYPE": "type", "GENRE": "genre", "RELEVANT": "relevant"},
    "SearchVerdict": {
        "FAILED": "failed",
        "NO_RESULTS": "no_results",
        "HAS_RESULTS": "has_results",
    },
}


@pytest.mark.parametrize("name", sorted(ENUMS))
def test_enum_members(name):
    enum_cls = getattr(models, name)
    assert issubclass(enum_cls, StrEnum)
    assert {m.name: m.value for m in enum_cls} == ENUMS[name]


DATACLASSES = {
    "JobRequest": ["job_id", "query", "topic", "genre", "max_results", "allow_scrape", "dry_run"],
    "NormalizedQuery": ["original", "ja", "en", "dictionary_version"],
    "SearchResult": [
        "url",
        "title",
        "snippet",
        "engine",
        "published_date",
        "retrieved_at",
        "engines",
        "matched",
    ],
    "EngineFailure": ["engine", "reason"],
    "SearchResponse": [
        "verdict",
        "results",
        "failed_engines",
        "unreachable_queries",
        "retried_queries",
        "rustxng_version",
    ],
    "Judgment": ["value", "by", "confidence"],
    "ClassifiedResult": ["id", "result", "type", "genre", "relevant"],
    "JevUsage": [
        "model",
        "input_tokens",
        "output_tokens",
        "estimated_input_tokens",
        "duration_ms",
        "n_results",
        "n_questions",
        "error",
    ],
    "JevReport": ["query_genre", "usage", "failure"],
    "Versions": ["config", "dictionary", "rules", "jev_model", "rustxng"],
    "Reasons": ["failed_engines", "unreachable_queries", "jev_failure", "digest_failure"],
    "JobRecord": [
        "job_id",
        "query",
        "query_norm",
        "topic",
        "genre",
        "started_at",
        "duration_ms",
        "outcome",
        "reasons",
        "states",
        "n_results",
        "n_unclassified",
        "versions",
        "jev_usage",
        "query_genre",
    ],
}


@pytest.mark.parametrize("name", sorted(DATACLASSES))
def test_dataclass_fields(name):
    cls = getattr(models, name)
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert [f.name for f in dataclasses.fields(cls)] == DATACLASSES[name]


def test_judgment_keys_match_classified_result_and_store_row():
    """JudgmentKey の値は、ClassifiedResult の判定の属性名と、結果ログの行の判定のキーに一致する。"""
    judgment_fields = {
        f.name for f in dataclasses.fields(models.ClassifiedResult) if f.type in ("Judgment", models.Judgment)
    }
    keys = {k.value for k in models.JudgmentKey}
    assert keys == judgment_fields
    assert models.JUDGMENT_KEYS == tuple(k.value for k in models.JudgmentKey)

    now = datetime.datetime(2026, 11, 3, 12, 0, tzinfo=datetime.UTC)
    result = models.SearchResult(
        url="https://example.com/1",
        title="t",
        snippet="s",
        engine="e",
        published_date=None,
        retrieved_at=now,
        engines=("e",),
        matched=(models.Lang.EN,),
    )
    judgment = models.Judgment("x", models.JudgedBy.RULE, 1.0)
    item = models.ClassifiedResult("r01", result, judgment, judgment, judgment)
    versions = models.Versions("c", "d", "r", "j", "x")
    norm = models.NormalizedQuery("q", "q", None, "d")
    (row,) = result_rows(
        job_id="j", query="q", query_norm=norm, topic=None, items=[item], versions=versions
    )
    row_judgment_keys = {
        key for key, value in row.items() if isinstance(value, dict) and set(value) == {"value", "by", "confidence"}
    }
    assert row_judgment_keys == keys


def test_reasons_defaults():
    r = models.Reasons()
    assert r.failed_engines == ()
    assert r.unreachable_queries == ()
    assert r.jev_failure is None
    assert r.digest_failure is None


PORTS = {
    "SearchBackend": ["search"],
    "Classifier": ["classify"],
    "MemoryClient": ["search", "write"],
    "Clock": ["now"],
}


@pytest.mark.parametrize("name", sorted(PORTS))
def test_ports_protocols(name):
    proto = getattr(ports, name)
    assert getattr(proto, "_is_protocol", False)
    for method in PORTS[name]:
        assert callable(getattr(proto, method))
