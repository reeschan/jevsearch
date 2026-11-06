"""共有メモリ（MCP、Streamable HTTP）を呼ぶクライアント。

`ports.MemoryClient` を満たす。API キーなどの秘密は扱わず、送る内容は引数と固定のヘッダーだけ。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

from jevsearch import __version__

PROTOCOL_VERSION = "2025-06-18"
_SESSION_HEADER = "Mcp-Session-Id"
_DETAIL_LIMIT = 300


class McpError(Exception):
    """共有メモリを呼べなかった、または呼び出しが失敗を返した。"""


def _parse_sse(text: str, request_id: int) -> dict:
    """SSE の本文から、リクエストの id と同じ id の JSON-RPC 応答を取り出す。"""
    events: list[str] = []
    data_lines: list[str] = []
    for line in text.splitlines():
        if line == "":
            if data_lines:
                events.append("\n".join(data_lines))
                data_lines = []
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        events.append("\n".join(data_lines))
    for data in events:
        try:
            message = json.loads(data)
        except ValueError:
            continue
        if isinstance(message, dict) and message.get("id") == request_id:
            return message
    raise McpError(f"応答に id {request_id} の結果が無い（SSE）")


class McpHttpClient:
    def __init__(self, url: str, client: httpx.Client) -> None:
        self._url = url
        self._client = client
        self._initialized = False
        self._session_id: str | None = None
        self._next_id = 1

    # --- 低レベル ---

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id is not None:
            headers[_SESSION_HEADER] = self._session_id
        return headers

    def _post(self, payload: dict) -> httpx.Response:
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers())
        except httpx.HTTPError as e:
            raise McpError(f"共有メモリに接続できない: {type(e).__name__}") from e
        if not response.is_success:
            detail = response.text[:_DETAIL_LIMIT]
            raise McpError(f"HTTP {response.status_code}: {detail}")
        return response

    def _request(self, method: str, params: dict) -> tuple[dict, httpx.Response]:
        """id 付きのリクエストを送り、JSON-RPC の result を返す。error は McpError。"""
        request_id = self._next_id
        self._next_id += 1
        response = self._post(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        content_type = response.headers.get("content-type", "")
        try:
            if content_type.lower().startswith("text/event-stream"):
                message = _parse_sse(response.text, request_id)
            else:
                message = response.json()
        except ValueError as e:
            raise McpError("応答が JSON ではない") from e
        if not isinstance(message, dict):
            raise McpError("応答が JSON-RPC の形ではない")
        if message.get("error") is not None:
            error = message["error"]
            text = error.get("message") if isinstance(error, dict) else error
            raise McpError(f"JSON-RPC エラー: {text}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpError("応答に result が無い")
        return result, response

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._session_id = None
        _, response = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "jevsearch", "version": __version__},
            },
        )
        self._session_id = response.headers.get(_SESSION_HEADER)
        # 通知は応答の本文が空（202）なので、2xx なら本文を読まずに成功とする。
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def _call_tool(self, name: str, arguments: dict) -> object:
        self._ensure_initialized()
        result, _ = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content")
        text = None
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text")
        if result.get("isError"):
            raise McpError(f"ツール {name} が失敗: {str(text)[:_DETAIL_LIMIT]}")
        if not isinstance(text, str):
            raise McpError(f"ツール {name} の結果に content[0].text が無い")
        try:
            return json.loads(text)
        except ValueError as e:
            raise McpError(f"ツール {name} の結果が JSON ではない") from e

    # --- MemoryClient ---

    def search(
        self, query: str, *, scope: str, tags: Sequence[str], limit: int
    ) -> list[dict]:
        data = self._call_tool(
            "search", {"query": query, "scope": scope, "tags": list(tags), "limit": limit}
        )
        if not isinstance(data, list):
            raise McpError("search の結果が配列ではない")
        return data

    def write(
        self,
        *,
        title: str,
        content: str,
        scope: str,
        kind: str,
        provenance: str,
        source_agent: str,
        tags: Sequence[str],
        supersedes: str | None,
    ) -> str:
        arguments: dict = {
            "title": title,
            "content": content,
            "scope": scope,
            "kind": kind,
            "provenance": provenance,
            "source_agent": source_agent,
            "tags": list(tags),
        }
        if supersedes is not None:
            arguments["supersedes"] = supersedes
        data = self._call_tool("write", arguments)
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            raise McpError("write の結果に id が無い")
        return data["id"]
