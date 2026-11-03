"""API キーが入ったファイルをコミットしにくくする検査。

使い方:
  python scripts/check_secrets.py [ファイル ...]

引数がなければ、ステージ済みのファイル
（`git diff --cached --name-only --diff-filter=ACMR`。名前の変わったファイルは新しい名前）を
調べる。このときはステージ済みの内容（`git show :<パス>`）を読み、作業ツリーのファイルは読まない。
引数でファイルを渡したときは、そのファイルを作業ツリーから読む。
見つけたら `<パス>:<行番号>: <種類>` を標準エラーに出して終了コード 1 にする。
値そのものは出さない。標準ライブラリだけで動く。

調べる種類:
  - ファイル名: 認証ファイル・環境変数ファイル
    （例の `.env.example` は除く。大文字小文字は区別しない）
  - 環境変数名への代入: 名前に空でない値を代入している行（`os.environ[...] = "値"` も含む）
  - JSON 風の `api_key`: 空でない文字列の値がある行
  - Bearer トークン: `Bearer ` のあとに16文字以上の英数字・`-_.` が続く行
値が `<...>`・`${...}`・`${{ ... }}`・`{{ ... }}` で包まれているものは、
鍵ではなく書き込み用の穴や参照とみなす。
`# check-secrets: ignore` を含む行は調べない。
"""

from __future__ import annotations

import codecs
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

EXIT_OK = 0
EXIT_FOUND = 1

IGNORE_MARK = "# check-secrets: ignore"
"""この文字列を含む行は調べない。"""

_KEY_NAME = "TYPESAFE_API_KEY"
"""キーを入れる環境変数の名前。"""

KIND_FILENAME = "鍵を入れるファイル名"
KIND_ENV_ASSIGN = "環境変数への代入"
KIND_JSON_KEY = "api_key の値"
KIND_BEARER = "Bearer トークン"

_CREDENTIAL_FILENAMES = ("credentials.json", ".env")
_ENV_EXAMPLE = ".env.example"

_QUOTED = r"""(?:"[^"\n]+"|'[^'\n]+')"""
_ASSIGN_RE = re.compile(
    rf"""(?P<q>["']?){_KEY_NAME}(?P=q)\]?\s*[=:](?!=)\s*(?P<value>{_QUOTED}|[^\s"'#,;)}}\]]+)"""
)
_BARE_NON_VALUE_START = ("$", "<", "%", "{")
"""引用符なしの値がこれで始まるなら、値ではなく参照や書き込み用の穴とみなす。"""
_JSON_KEY_RE = re.compile(r""""api_key"\s*:\s*"(?P<value>[^"\n]+)\"""")
_BEARER_RE = re.compile(r"bearer[ \t]+[a-z0-9\-_.]{16,}", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"<[^<>]*>|\$\{.*\}|\{\{.*\}\}", re.DOTALL)
"""値の全体が `<...>`・`${...}`・`${{ ... }}`・`{{ ... }}` なら、鍵ではない。"""


def _is_credential_filename(path: str) -> bool:
    name = Path(path).name.lower()
    if name == _ENV_EXAMPLE:
        return False
    return name in _CREDENTIAL_FILENAMES or name.startswith(".env.")


def _is_placeholder(value: str) -> bool:
    return _PLACEHOLDER_RE.fullmatch(value.strip()) is not None


def _assign_found(line: str) -> bool:
    for match in _ASSIGN_RE.finditer(line):
        value = match.group("value")
        quoted_value = value[0] in "\"'"
        if quoted_value:
            value = value[1:-1]
        elif match.group("q"):
            # 名前が引用符付き（辞書や JSON のキー）で値が裸なら、変数の参照とみなす
            continue
        elif value.startswith(_BARE_NON_VALUE_START):
            continue
        if _is_placeholder(value):
            continue
        return True
    return False


def _line_kinds(line: str) -> list[str]:
    """1行から見つかった種類の一覧。"""
    if IGNORE_MARK in line:
        return []
    kinds: list[str] = []
    if _assign_found(line):
        kinds.append(KIND_ENV_ASSIGN)
    if any(not _is_placeholder(m.group("value")) for m in _JSON_KEY_RE.finditer(line)):
        kinds.append(KIND_JSON_KEY)
    if _BEARER_RE.search(line):
        kinds.append(KIND_BEARER)
    return kinds


def decode_text(data: bytes) -> str:
    """BOM があれば UTF-8・UTF-16 LE/BE として、なければ UTF-8（壊れた所は置き換え）で読む。"""
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")  # BOM で並びを決め、BOM は取り除く
    return data.decode("utf-8", errors="replace")


def scan_bytes(path: str, data: bytes | None) -> Iterator[tuple[int, str]]:
    """(行番号, 種類) を順に返す。`data` が None なら、名前だけを調べる。"""
    if _is_credential_filename(path):
        yield 1, KIND_FILENAME
    if data is None:
        return
    for number, line in enumerate(decode_text(data).splitlines(), start=1):
        for kind in _line_kinds(line):
            yield number, kind


def scan_file(path: str) -> Iterator[tuple[int, str]]:
    """作業ツリーのファイルを調べる。読めないファイルは名前だけを調べる。"""
    try:
        data: bytes | None = Path(path).read_bytes()
    except OSError:
        data = None
    yield from scan_bytes(path, data)


def staged_files() -> list[str]:
    """ステージ済み（追加・コピー・変更・名前変更）のファイル。git が使えなければ空。"""
    try:
        done = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in done.stdout.decode("utf-8", errors="replace").split("\0") if p]


def staged_content(path: str) -> bytes | None:
    """ステージ済みの内容（`git show :<パス>`）。取れなければ None。"""
    try:
        done = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def _report(path: str, found: Iterator[tuple[int, str]]) -> bool:
    hit = False
    for number, kind in found:
        print(f"{path}:{number}: {kind}", file=sys.stderr)
        hit = True
    return hit


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")  # 文字化けしないよう UTF-8 にする
    args = list(sys.argv[1:] if argv is None else argv)
    found = False
    if args:
        for path in args:
            if Path(path).is_file():
                found = _report(path, scan_file(path)) or found
    else:
        for path in staged_files():
            found = _report(path, scan_bytes(path, staged_content(path))) or found
    if found:
        print(
            "鍵が入っているかもしれないファイルがあります。コミットを止めました。"
            f"意図したものなら、その行に `{IGNORE_MARK}` を付けてください。",
            file=sys.stderr,
        )
        return EXIT_FOUND
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
