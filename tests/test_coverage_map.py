"""状態の図の全遷移・全 never が、テストのどこかに書かれているかの検査。

遷移のラベルは docs/specs/statechart/jevsearch-job.mmd の `<元> --> <先>: <ラベル>` の行から、
never の文は docs/specs/statechart/agreed-verbatim.mmd の `%% never ` の行から読む。
tests/ の下の全 .py の本文に、遷移は「遷移: <ラベル>」、never は「never: <文>」が
1回以上あることを確かめる。足りないものは全部メッセージに並べる。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATECHART = ROOT / "docs" / "specs" / "statechart"
TESTS = ROOT / "tests"

EXPECTED_TRANSITIONS = 12
EXPECTED_NEVERS = 9

_TRANSITION_LINE = re.compile(r"^\s*(\S+)\s+-->\s+(\S+):\s*(.+?)\s*$")
_NEVER_PREFIX = "%% never "


def _transition_labels() -> list[str]:
    text = (STATECHART / "jevsearch-job.mmd").read_text(encoding="utf-8")
    labels = []
    for line in text.splitlines():
        match = _TRANSITION_LINE.match(line)
        if match:
            labels.append(match.group(3))
    return labels


def _never_sentences() -> list[str]:
    text = (STATECHART / "agreed-verbatim.mmd").read_text(encoding="utf-8")
    sentences = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_NEVER_PREFIX):
            sentences.append(stripped[len(_NEVER_PREFIX) :].strip())
    return sentences


def _tests_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TESTS.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def test_diagram_has_the_agreed_number_of_transitions_and_nevers():
    assert len(_transition_labels()) == EXPECTED_TRANSITIONS
    assert len(_never_sentences()) == EXPECTED_NEVERS


def test_every_transition_and_never_is_written_in_the_tests():
    labels = _transition_labels()
    nevers = _never_sentences()
    assert len(labels) == EXPECTED_TRANSITIONS
    assert len(nevers) == EXPECTED_NEVERS

    text = _tests_text()
    missing = [f"遷移: {label}" for label in labels if f"遷移: {label}" not in text]
    missing += [f"never: {sentence}" for sentence in nevers if f"never: {sentence}" not in text]
    assert not missing, "tests に書かれていない遷移・never:\n" + "\n".join(missing)
