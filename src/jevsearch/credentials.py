"""API キーの読み込み。環境変数か手元の認証ファイルからだけ読む。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "TYPESAFE_API_KEY"


def default_credentials_path() -> Path:
    """認証ファイルの既定の場所。"""
    return Path.home() / ".config" / "typesafe" / "credentials.json"


def get_api_key(env: Mapping[str, str], credentials_path: Path | None = None) -> str | None:
    """API キーを探す。環境変数、認証ファイルの `api_key` の順。無ければ None。

    読めない・形が違うファイルは None を返し、例外は出さない。
    """
    key = env.get(ENV_VAR)
    if key:
        return key
    path = credentials_path if credentials_path is not None else default_credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    key = data.get("api_key")
    if isinstance(key, str) and key:
        return key
    return None
