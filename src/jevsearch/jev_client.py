"""TypeSafe System One（Jev）を HTTP で呼ぶクライアント。

キーは認証ヘッダーにだけ入れる。例外のメッセージにはキーを出さない。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from jevsearch.ports import Clock

_RATE_STATUSES = (429, 529)
_MAX_WAIT_SECONDS = 30.0
_DETAIL_LIMIT = 500


class JevError(Exception):
    """Jev を呼べなかった。

    code: no_api_key / daily_limit / http / network / timeout / bad_response。
    status は HTTP ステータス（無ければ 0）。
    """

    def __init__(self, code: str, message: str = "", status: int = 0) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class JevResult:
    answers: dict
    usage: dict | None
    duration_ms: int


def _retry_wait(retry_after: str | None, n: int) -> float:
    """待ち時間。retry-after があればその秒数、なければ 2^n 秒。上限 30 秒。"""
    seconds: float | None = None
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = None
    if seconds is None or seconds < 0:
        seconds = float(2**n)
    return min(seconds, _MAX_WAIT_SECONDS)


class JevClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        endpoint: str,
        client: httpx.Client,
        usage_path: Path,
        daily_request_limit: int | None,
        max_rate_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Clock,
        timeout_seconds: float = 120,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._client = client
        self._usage_path = usage_path
        self._daily_limit = daily_request_limit
        self._max_rate_retries = max_rate_retries
        self._sleep = sleep
        self._clock = clock
        self._timeout = timeout_seconds

    def __repr__(self) -> str:
        return f"JevClient(model={self._model!r}, endpoint={self._endpoint!r})"

    def _redact(self, text: str) -> str:
        """文字列の中のキーを `***` に置き換える。"""
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text

    def _today(self) -> str:
        return self._clock.now().astimezone().date().isoformat()

    def _load_usage(self) -> dict:
        try:
            data = json.loads(self._usage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _today_count(self) -> int:
        count = self._load_usage().get(self._today())
        return count if isinstance(count, int) and not isinstance(count, bool) else 0

    def _count_request(self) -> None:
        """HTTP を1回送るたびに、今日の数を1増やして保存する。"""
        data = self._load_usage()
        today = self._today()
        count = data.get(today)
        base = count if isinstance(count, int) and not isinstance(count, bool) else 0
        data[today] = base + 1
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._usage_path.with_name(self._usage_path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._usage_path)

    def _detail(self, text: str) -> str:
        return self._redact(text)[:_DETAIL_LIMIT]

    def ask(self, state: str, questions: dict) -> JevResult:
        if not self._api_key:
            raise JevError("no_api_key", "Jev の API キーが見つからない")
        if self._daily_limit is not None and self._today_count() >= self._daily_limit:
            raise JevError("daily_limit", f"1日の上限 {self._daily_limit} 件に達した")

        body = json.dumps(
            {"state": state, "model": self._model, "questions": questions},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        started = time.monotonic()
        rate_retries = 0
        other_retried = False
        while True:
            self._count_request()
            try:
                response = self._client.post(
                    self._endpoint, content=body, headers=headers, timeout=self._timeout
                )
            except httpx.TimeoutException as e:
                err = JevError("timeout", self._detail(f"{type(e).__name__}: {e}"))
                if not other_retried:
                    other_retried = True
                    continue
                raise err from None
            except httpx.HTTPError as e:
                err = JevError("network", self._detail(f"{type(e).__name__}: {e}"))
                if not other_retried:
                    other_retried = True
                    continue
                raise err from None

            status = response.status_code
            if not 200 <= status < 300:
                err = JevError("http", self._detail(f"HTTP {status}: {response.text}"), status)
                if status in _RATE_STATUSES:
                    if rate_retries < self._max_rate_retries:
                        self._sleep(_retry_wait(response.headers.get("retry-after"), rate_retries))
                        rate_retries += 1
                        continue
                    raise err
                if 500 <= status <= 599 and not other_retried:
                    other_retried = True
                    continue
                raise err

            text = response.text
            try:
                data = json.loads(text)
            except ValueError:
                raise JevError(
                    "bad_response", self._detail(f"応答が JSON でない: {text}"), status
                ) from None
            if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                raise JevError(
                    "bad_response", self._detail(f"応答に answers が無い: {text}"), status
                )
            usage = data.get("usage")
            duration_ms = int((time.monotonic() - started) * 1000)
            return JevResult(
                answers=data["answers"],
                usage=usage if isinstance(usage, dict) else None,
                duration_ms=duration_ms,
            )
