"""API キーの読み込みのテスト。本物の認証ファイルは読まない。"""

import json

from jevsearch.credentials import get_api_key

SENTINEL_ENV = "sk-test-SENTINEL-ENV-0000"
SENTINEL_FILE = "sk-test-SENTINEL-FILE-0000"


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_環境変数が優先される(tmp_path):
    path = _write(tmp_path / "c.json", {"api_key": SENTINEL_FILE})
    assert get_api_key({"TYPESAFE_API_KEY": SENTINEL_ENV}, path) == SENTINEL_ENV


def test_環境変数が無ければファイルのapi_key(tmp_path):
    path = _write(tmp_path / "c.json", {"api_key": SENTINEL_FILE})
    assert get_api_key({}, path) == SENTINEL_FILE


def test_空の環境変数はファイルにフォールバックする(tmp_path):
    path = _write(tmp_path / "c.json", {"api_key": SENTINEL_FILE})
    assert get_api_key({"TYPESAFE_API_KEY": ""}, path) == SENTINEL_FILE


def test_どちらも無ければNone(tmp_path):
    assert get_api_key({}, tmp_path / "missing.json") is None


def test_壊れたファイルでNone(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    assert get_api_key({}, path) is None


def test_形が違うファイルでNone(tmp_path):
    assert get_api_key({}, _write(tmp_path / "a.json", ["api_key"])) is None
    assert get_api_key({}, _write(tmp_path / "b.json", {"api_key": 123})) is None
    assert get_api_key({}, _write(tmp_path / "c.json", {"api_key": ""})) is None
    assert get_api_key({}, _write(tmp_path / "d.json", {"other": "x"})) is None
