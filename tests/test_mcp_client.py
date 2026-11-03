"""MCP クライアントのテスト。HTTP は httpx.MockTransport で差し替える。"""

import json

import httpx
import pytest

from jevsearch import __version__
from jevsearch.mcp_client import McpError, McpHttpClient

URL = "http://127.0.0.1:7820/mcp"
KEY = "sk-test-SENTINEL-0000"


def _tool_result(payload, *, is_error=False):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": is_error}


class FakeServer:
    """MCP サーバーの代わり。受けた要求を記録し、メソッドごとに応答を返す。"""

    def __init__(self, *, session_id="sess-1", sse=False, tool_result=None, rpc_error=None):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.session_id = session_id
        self.sse = sse
        self.tool_result = tool_result
        self.rpc_error = rpc_error

    def _reply(self, request_id, result=None, headers=None):
        body = {"jsonrpc": "2.0", "id": request_id}
        if self.rpc_error is not None:
            body["error"] = self.rpc_error
        else:
            body["result"] = result
        if self.sse:
            # 別の id の通知を先に流し、id が同じものだけを使うことを確かめる。
            other = {"jsonrpc": "2.0", "id": 999, "result": {"content": []}}
            text = (
                f"event: message\ndata: {json.dumps(other)}\n\n"
                f"event: message\ndata: {json.dumps(body)}\n\n"
            )
            return httpx.Response(
                200, text=text, headers={"content-type": "text/event-stream", **(headers or {})}
            )
        return httpx.Response(200, json=body, headers=headers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        method = body["method"]
        if method == "initialize":
            headers = {"Mcp-Session-Id": self.session_id} if self.session_id else None
            return self._reply(body["id"], {"protocolVersion": "2025-06-18"}, headers)
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/call":
            result = self.tool_result
            if result is None:
                name = body["params"]["name"]
                result = _tool_result([] if name == "search" else {"id": "NEW"})
            return self._reply(body["id"], result)
        raise AssertionError(method)


def _make(server):
    return McpHttpClient(URL, httpx.Client(transport=httpx.MockTransport(server)))


def test_initialize_initialized_tools_call_の順で送る():
    server = FakeServer()
    client = _make(server)
    new_id = client.write(
        title="t",
        content="c",
        scope="s",
        kind="episodic",
        provenance="agent_inferred",
        source_agent="claude",
        tags=["a"],
        supersedes=None,
    )
    assert new_id == "NEW"
    assert [b["method"] for b in server.bodies] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    init = server.bodies[0]["params"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["capabilities"] == {}
    assert init["clientInfo"] == {"name": "jevsearch", "version": __version__}
    assert "id" not in server.bodies[1]
    call = server.bodies[2]["params"]
    assert call["name"] == "write"
    assert "supersedes" not in call["arguments"]
    for request in server.requests:
        assert request.headers["content-type"] == "application/json"
        assert request.headers["accept"] == "application/json, text/event-stream"


def test_supersedes_があれば引数に入れる():
    server = FakeServer()
    _make(server).write(
        title="t",
        content="c",
        scope="s",
        kind="episodic",
        provenance="agent_inferred",
        source_agent="claude",
        tags=[],
        supersedes="OLD",
    )
    assert server.bodies[-1]["params"]["arguments"]["supersedes"] == "OLD"


def test_2回目の呼び出しでは_initialize_しない():
    server = FakeServer(tool_result=_tool_result([{"id": "x", "title": "T"}]))
    client = _make(server)
    first = client.search("q", scope="s", tags=["t"], limit=20)
    second = client.search("q", scope="s", tags=["t"], limit=20)
    assert first == second == [{"id": "x", "title": "T"}]
    methods = [b["method"] for b in server.bodies]
    assert methods.count("initialize") == 1
    assert methods == ["initialize", "notifications/initialized", "tools/call", "tools/call"]
    args = server.bodies[2]["params"]
    assert args["name"] == "search"
    assert args["arguments"] == {"query": "q", "scope": "s", "tags": ["t"], "limit": 20}


def test_セッション_id_が以降のリクエストに付く():
    server = FakeServer(session_id="sess-xyz")
    _make(server).search("q", scope="s", tags=[], limit=1)
    assert "mcp-session-id" not in server.requests[0].headers
    assert server.requests[1].headers["mcp-session-id"] == "sess-xyz"
    assert server.requests[2].headers["mcp-session-id"] == "sess-xyz"


def test_セッション_id_が無ければヘッダーを付けない():
    server = FakeServer(session_id=None)
    _make(server).search("q", scope="s", tags=[], limit=1)
    assert all("mcp-session-id" not in r.headers for r in server.requests)


def test_SSE_の応答を読める():
    server = FakeServer(sse=True, tool_result=_tool_result([{"id": "a"}]))
    assert _make(server).search("q", scope="s", tags=[], limit=1) == [{"id": "a"}]


def test_isError_は_McpError():
    server = FakeServer(tool_result=_tool_result("拒否された", is_error=True))
    with pytest.raises(McpError):
        _make(server).search("q", scope="s", tags=[], limit=1)


def test_JSON_RPC_の_error_は_McpError():
    server = FakeServer(rpc_error={"code": -32601, "message": "no such method"})
    with pytest.raises(McpError, match="no such method"):
        _make(server).search("q", scope="s", tags=[], limit=1)


def test_HTTP_エラーは_McpError():
    client = McpHttpClient(
        URL, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    )
    with pytest.raises(McpError, match="500"):
        client.search("q", scope="s", tags=[], limit=1)


def test_接続エラーは_McpError():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = McpHttpClient(URL, httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(McpError):
        client.search("q", scope="s", tags=[], limit=1)


def test_初期化に失敗したら次の呼び出しでもう一度初期化する():
    server = FakeServer()
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return server(request)

    client = McpHttpClient(URL, httpx.Client(transport=httpx.MockTransport(flaky)))
    with pytest.raises(McpError):
        client.search("q", scope="s", tags=[], limit=1)
    client.search("q", scope="s", tags=[], limit=1)
    assert [b["method"] for b in server.bodies][0] == "initialize"


def test_API_キーを環境変数に入れても_どのリクエストにも番兵が無い(monkeypatch):
    """never: API キーを、TypeSafe への認証ヘッダー以外に送らない"""
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    server = FakeServer()
    client = _make(server)
    client.search("q", scope="s", tags=["t"], limit=1)
    client.write(
        title="t",
        content="c",
        scope="s",
        kind="episodic",
        provenance="agent_inferred",
        source_agent="claude",
        tags=["t"],
        supersedes=None,
    )
    assert server.requests
    for request in server.requests:
        assert KEY not in request.content.decode()
        assert KEY not in str(request.url)
        assert all(KEY not in v for v in request.headers.values())
