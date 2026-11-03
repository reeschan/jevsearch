"""Jev クライアントのテスト。HTTP は httpx.MockTransport で差し替える。"""

import json
from datetime import datetime

import httpx
import pytest

from jevsearch.jev_client import JevClient, JevError, JevResult
from jevsearch.models import JEV_MODEL

KEY = "sk-test-SENTINEL-0000"
ENDPOINT = "https://jev.example.test/v1/systemone"
QUESTIONS = {"q1": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}
OK_BODY = {"answers": {"q1": {"choice": "a", "confidence": 0.9}}}


class FakeClock:
    def __init__(self, now=None):
        self._now = now or datetime(2026, 11, 3, 12, 0).astimezone()

    def now(self):
        return self._now


class Recorder:
    """MockTransport に渡すハンドラー。応答を順に返し、送られた要求を記録する。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _json(status=200, body=None, headers=None):
    return httpx.Response(status, json=OK_BODY if body is None else body, headers=headers)


def _make(tmp_path, recorder, *, api_key=KEY, limit=None, clock=None, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    client = JevClient(
        api_key=api_key,
        model=JEV_MODEL,
        endpoint=ENDPOINT,
        client=httpx.Client(transport=httpx.MockTransport(recorder)),
        usage_path=tmp_path / "jev-usage.json",
        daily_request_limit=limit,
        sleep=sleeps.append,
        clock=clock or FakeClock(),
    )
    return client, sleeps


def test_キーは認証ヘッダーにだけある(tmp_path):
    """never: API キーを、TypeSafe への認証ヘッダー以外に送らない"""
    rec = Recorder(_json())
    client, _ = _make(tmp_path, rec)
    client.ask("state text", QUESTIONS)
    (req,) = rec.requests
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in req.content.decode("utf-8")
    assert KEY not in str(req.url)
    for name, value in req.headers.items():
        if name.lower() != "authorization":
            assert KEY not in value
    assert req.headers["content-type"] == "application/json"
    assert req.method == "POST"


def test_本文はstate_model_questionsの3つだけ(tmp_path):
    """never: 検索語・タイトル・URL・抜粋と、内部のID・ドメイン名・エンジン名以外は Jev に送らない"""
    rec = Recorder(_json())
    client, _ = _make(tmp_path, rec)
    client.ask("state text", QUESTIONS)
    body = json.loads(rec.requests[0].content)
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == "state text"
    assert body["questions"] == QUESTIONS


def test_429が2回のあと成功しsleepが2回(tmp_path):
    rec = Recorder(_json(429, {}), _json(429, {}), _json())
    client, sleeps = _make(tmp_path, rec)
    result = client.ask("s", QUESTIONS)
    assert isinstance(result, JevResult)
    assert result.answers == OK_BODY["answers"]
    assert sleeps == [1, 2]
    assert len(rec.requests) == 3


def test_529が4回続くとhttp_529で失敗する(tmp_path):
    rec = Recorder(*[_json(529, {}) for _ in range(4)])
    client, sleeps = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "http"
    assert ei.value.status == 529
    assert len(rec.requests) == 4
    assert len(sleeps) == 3


def test_retry_afterの秒数だけ待つ(tmp_path):
    rec = Recorder(_json(429, {}, {"retry-after": "3"}), _json())
    client, sleeps = _make(tmp_path, rec)
    client.ask("s", QUESTIONS)
    assert sleeps == [3]


def test_接続エラー1回のあと成功する(tmp_path):
    rec = Recorder(httpx.ConnectError("boom"), _json())
    client, sleeps = _make(tmp_path, rec)
    assert client.ask("s", QUESTIONS).answers == OK_BODY["answers"]
    assert len(rec.requests) == 2
    assert sleeps == []


def test_接続エラーが続くとnetworkで失敗する(tmp_path):
    rec = Recorder(httpx.ConnectError("boom"), httpx.ConnectError("boom"))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "network"
    assert len(rec.requests) == 2


def test_タイムアウトが続くとtimeoutで失敗する(tmp_path):
    rec = Recorder(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "timeout"


def test_500は1回だけやり直す(tmp_path):
    rec = Recorder(_json(500, {}), _json(500, {}))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert (ei.value.code, ei.value.status) == ("http", 500)
    assert len(rec.requests) == 2


def test_キーなしはHTTPを呼ばずno_api_key(tmp_path):
    rec = Recorder()
    client, _ = _make(tmp_path, rec, api_key=None)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "no_api_key"
    assert rec.requests == []


def test_今日の数が上限に達するとHTTPを呼ばずdaily_limit(tmp_path):
    """遷移: Jev が使えない、または上限を超えたので未分類として残した"""
    clock = FakeClock()
    today = clock.now().astimezone().date().isoformat()
    (tmp_path / "jev-usage.json").write_text(json.dumps({today: 5}), encoding="utf-8")
    rec = Recorder()
    client, _ = _make(tmp_path, rec, limit=5, clock=clock)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "daily_limit"
    assert rec.requests == []


def test_昨日の数は上限に数えない(tmp_path):
    (tmp_path / "jev-usage.json").write_text(json.dumps({"2000-01-01": 99}), encoding="utf-8")
    rec = Recorder(_json())
    client, _ = _make(tmp_path, rec, limit=1)
    client.ask("s", QUESTIONS)
    assert len(rec.requests) == 1


def test_HTTPを1回送るたびに今日の数が1増える(tmp_path):
    clock = FakeClock()
    today = clock.now().astimezone().date().isoformat()
    rec = Recorder(_json(429, {}), _json())
    client, _ = _make(tmp_path, rec, clock=clock)
    client.ask("s", QUESTIONS)
    saved = json.loads((tmp_path / "jev-usage.json").read_text(encoding="utf-8"))
    assert saved == {today: 2}


def test_400の応答本文にキーがあっても例外にキーが出ない(tmp_path):
    """never: API キーを結果ログ・記録・画面の表示に出さない"""
    rec = Recorder(httpx.Response(400, text=f"bad request, key={KEY} rejected"))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "http"
    assert ei.value.status == 400
    assert KEY not in str(ei.value)
    assert KEY not in repr(ei.value)
    assert "***" in str(ei.value)
    assert KEY not in repr(client)


def test_応答にusageがあればJevResultに入る(tmp_path):
    body = {**OK_BODY, "usage": {"input_tokens": 10, "output_tokens": 2}}
    rec = Recorder(_json(200, body))
    client, _ = _make(tmp_path, rec)
    result = client.ask("s", QUESTIONS)
    assert result.usage == {"input_tokens": 10, "output_tokens": 2}
    assert result.duration_ms >= 0


def test_応答にusageが無ければNone(tmp_path):
    client, _ = _make(tmp_path, Recorder(_json()))
    assert client.ask("s", QUESTIONS).usage is None


def test_JSONでない200はbad_response(tmp_path):
    rec = Recorder(httpx.Response(200, text="not json"))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "bad_response"


def test_answersが辞書でない200はbad_response(tmp_path):
    rec = Recorder(_json(200, {"answers": ["x"]}))
    client, _ = _make(tmp_path, rec)
    with pytest.raises(JevError) as ei:
        client.ask("s", QUESTIONS)
    assert ei.value.code == "bad_response"
    assert ei.value.status == 200
