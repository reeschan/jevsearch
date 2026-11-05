"""scripts/check_secrets.py（API キーが入ったファイルの検査）のテスト。

このファイルに鍵の形の行を直に書くと、コミット前の検査に自分が引っかかる。
そのため、検査が探す形は文字列をつなげて作る。
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_secrets.py"
SENTINEL = "sk-test-SENTINEL-0000"
KEY_NAME = "TYPESAFE_API_" + "KEY"
IGNORE = "# check-secrets: " + "ignore"


def _run(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, paths)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_env_assignment_is_found_and_value_is_not_printed(tmp_path):
    """never: API キーを結果ログ・記録・画面の表示に出さない"""
    target = _write(tmp_path / "run.sh", f"echo hi\n{KEY_NAME}={SENTINEL}\n")
    done = _run(target)
    assert done.returncode == 1
    assert f"{target}:2:" in done.stderr
    assert SENTINEL not in done.stdout + done.stderr


@pytest.mark.parametrize(
    "line",
    [
        f"export {KEY_NAME}='{SENTINEL}'",
        f'{KEY_NAME}: "{SENTINEL}"',
        f'env = {{"{KEY_NAME}": "{SENTINEL}"}}',
    ],
)
def test_env_assignment_forms_are_found(tmp_path, line):
    done = _run(_write(tmp_path / "a.txt", line + "\n"))
    assert done.returncode == 1
    assert SENTINEL not in done.stderr


@pytest.mark.parametrize("name", ["credentials.json", ".env", ".env.local"])
def test_credential_file_names_are_found(tmp_path, name):
    done = _run(_write(tmp_path / name, "{}\n"))
    assert done.returncode == 1
    assert f"{name}:1:" in done.stderr


def test_env_example_file_name_is_allowed(tmp_path):
    assert _run(_write(tmp_path / ".env.example", "A=1\n")).returncode == 0


def test_json_api_key_value_is_found(tmp_path):
    json_key = '"api_' + 'key"'
    target = _write(tmp_path / "c.txt", f'{{\n  {json_key}: "{SENTINEL}"\n}}\n')
    done = _run(target)
    assert done.returncode == 1
    assert f"{target}:2:" in done.stderr
    assert SENTINEL not in done.stderr


def test_json_api_key_empty_or_non_string_is_allowed(tmp_path):
    lines = ['{"api_key": ""}', '{"api_key": 123}', '{"api_key": KEY}']
    target = _write(tmp_path / "c.txt", "\n".join(lines) + "\n")
    assert _run(target).returncode == 0


def test_bearer_long_token_is_found(tmp_path):
    target = _write(tmp_path / "h.txt", f"Authorization: Bearer {SENTINEL}-abcdef\n")
    done = _run(target)
    assert done.returncode == 1
    assert SENTINEL not in done.stderr


def test_bearer_short_or_placeholder_is_allowed(tmp_path):
    target = _write(tmp_path / "h.txt", "Bearer abc\nBearer <キー>\nf'Bearer {key}'\n")
    assert _run(target).returncode == 0


def test_environment_reference_is_allowed(tmp_path):
    target = _write(
        tmp_path / "ok.py",
        f'import os\nkey = os.environ["{KEY_NAME}"]\nother = os.getenv("{KEY_NAME}")\n'
        f"export {KEY_NAME}=$OTHER\nexport {KEY_NAME}=\n"
        f'env = {{"{KEY_NAME}": KEY}}\n',
    )
    assert _run(target).returncode == 0


def test_ignore_mark_skips_the_line(tmp_path):
    target = _write(tmp_path / "doc.md", f"{KEY_NAME}={SENTINEL}  {IGNORE}\n")
    assert _run(target).returncode == 0


def test_ignore_mark_does_not_skip_other_lines(tmp_path):
    target = _write(tmp_path / "doc.md", f"{KEY_NAME}={SENTINEL}  {IGNORE}\n{KEY_NAME}={SENTINEL}\n")
    done = _run(target)
    assert done.returncode == 1
    assert f"{target}:2:" in done.stderr
    assert f"{target}:1:" not in done.stderr


def test_plain_file_passes(tmp_path):
    assert _run(_write(tmp_path / "plain.txt", "こんにちは\n")).returncode == 0


def test_script_itself_passes():
    assert _run(SCRIPT).returncode == 0


def test_environment_assignment_with_literal_is_found(tmp_path):
    lines = [
        f'os.environ["{KEY_NAME}"] = "{SENTINEL}"',
        f"os.environ['{KEY_NAME}']='{SENTINEL}'",
        f'os.environ["{KEY_NAME}"] = "{SENTINEL}"  {IGNORE}',
    ]
    target = _write(tmp_path / "setup.py", "\n".join(lines) + "\n")
    done = _run(target)
    assert done.returncode == 1
    assert f"{target}:1:" in done.stderr
    assert f"{target}:2:" in done.stderr
    assert f"{target}:3:" not in done.stderr
    assert SENTINEL not in done.stderr


def test_environment_reads_are_allowed(tmp_path):
    lines = [
        f'a = os.environ.get("{KEY_NAME}")',
        f'b = os.getenv("{KEY_NAME}")',
        f'c = os.environ["{KEY_NAME}"]',
        f'if os.environ["{KEY_NAME}"] == "x": pass',
    ]
    assert _run(_write(tmp_path / "ok.py", "\n".join(lines) + "\n")).returncode == 0


def test_utf16_files_are_decoded(tmp_path):
    for name, encoding in [("le.txt", "utf-16-le"), ("be.txt", "utf-16-be")]:
        bom = b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff"
        target = tmp_path / name
        target.write_bytes(bom + f"x\n{KEY_NAME}={SENTINEL}\n".encode(encoding))
        done = _run(target)
        assert done.returncode == 1, name
        assert f"{target}:2:" in done.stderr
        assert SENTINEL not in done.stderr


def test_utf8_bom_file_is_decoded(tmp_path):
    target = tmp_path / "bom.txt"
    target.write_bytes(b"\xef\xbb\xbf" + f"{KEY_NAME}={SENTINEL}\n".encode())
    assert f"{target}:1:" in _run(target).stderr


@pytest.mark.parametrize("prefix", ["bearer ", "BEARER ", "Bearer   ", "Bearer\t"])
def test_bearer_is_case_and_space_insensitive(tmp_path, prefix):
    target = _write(tmp_path / "h.txt", f"Authorization: {prefix}{SENTINEL}-abcdef\n")
    done = _run(target)
    assert done.returncode == 1
    assert SENTINEL not in done.stderr


@pytest.mark.parametrize("name", ["Credentials.json", "CREDENTIALS.JSON", ".ENV", ".Env.Local"])
def test_credential_file_names_ignore_case(tmp_path, name):
    done = _run(_write(tmp_path / name, "{}\n"))
    assert done.returncode == 1
    assert f"{name}:1:" in done.stderr


def test_env_example_file_name_ignores_case(tmp_path):
    assert _run(_write(tmp_path / ".ENV.Example", "A=1\n")).returncode == 0


def test_placeholders_are_not_keys(tmp_path):
    json_key = '"api_' + 'key"'
    lines = [
        f'{KEY_NAME}: "${{{{ secrets.{KEY_NAME} }}}}"',
        f'{KEY_NAME}="${{{KEY_NAME}}}"',
        f'{KEY_NAME}: "<your key>"',
        f"{KEY_NAME}='{{{{ key }}}}'",
        f'{{{json_key}: "<your key>"}}',
        f'{json_key}: "${{{{ secrets.X }}}}"',
        f'{json_key}: "{{{{ key }}}}"',
        f'{json_key}: "${{KEY}}"',
    ]
    target = _write(tmp_path / "ci.yml", "\n".join(lines) + "\n")
    done = _run(target)
    assert done.returncode == 0, done.stderr


def test_partial_placeholder_is_still_a_key(tmp_path):
    json_key = '"api_' + 'key"'
    target = _write(tmp_path / "c.txt", f'{json_key}: "<{SENTINEL}"\n')
    assert _run(target).returncode == 1


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _run_staged(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git が無い")
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    return path


def test_staged_mode_reads_the_staged_content(repo):
    target = _write(repo / "app.py", f'x = 1\n{KEY_NAME}="{SENTINEL}"\n')
    _git(repo, "add", "app.py")
    _write(target, "x = 1\n")  # 作業ツリーでは消したが、ステージ済みには残っている
    done = _run_staged(repo)
    assert done.returncode == 1
    assert "app.py:2:" in done.stderr
    assert SENTINEL not in done.stderr


def test_staged_mode_ignores_unstaged_edits(repo):
    target = _write(repo / "app.py", "x = 1\n")
    _git(repo, "add", "app.py")
    _write(target, f'x = 1\n{KEY_NAME}="{SENTINEL}"\n')  # ステージしていない
    assert _run_staged(repo).returncode == 0


def test_staged_mode_checks_renamed_files(repo):
    _write(repo / "notes.txt", "メモ\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-q", "-m", "first")
    _git(repo, "mv", "notes.txt", "credentials.json")
    done = _run_staged(repo)
    assert done.returncode == 1
    assert "credentials.json:1:" in done.stderr


def test_staged_mode_with_nothing_staged_passes(repo):
    assert _run_staged(repo).returncode == 0


API = "api_" + "key"
PUT = "put" + "env"


@pytest.mark.parametrize(
    "template",
    [
        'monkeypatch.setenv("{K}", "{S}")',
        "monkeypatch.setenv('{K}', '{S}')",
        'os.environ.setdefault("{K}", "{S}")',
        'os.{P}("{K}", "{S}")',
        'monkeypatch.setitem(os.environ, "{K}", "{S}")',
        'JevClient({A}="{S}")',
        'JevClient(base_url=u, {A}="{S}", timeout=3)',
        '{A} = "{S}"',
        "{A}='{S}'",
        '{A}: str = "{S}"',
        'cfg["{A}"] = "{S}"',
        '[jev]\n{A} = "{S}"',
        '{UA} = "{S}"',
    ],
)
def test_python_and_toml_ways_of_writing_a_key_are_found(tmp_path, template):
    line = template.format(K=KEY_NAME, S=SENTINEL, A=API, P=PUT, UA=API.upper())
    target = _write(tmp_path / "x.py", line + "\n")
    done = _run(target)
    assert done.returncode == 1, line
    assert SENTINEL not in done.stderr
    assert done.stderr.count(f"{target}:") >= 1


@pytest.mark.parametrize(
    "template",
    [
        'monkeypatch.setenv("{K}", "<your key>")',
        'monkeypatch.setenv("{K}", "${{X}}")',
        'monkeypatch.setenv("{K}", "")',
        'monkeypatch.setenv("{K}", key)',
        'monkeypatch.delenv("{K}", raising=False)',
        'os.environ.get("{K}")',
        'os.getenv("{K}")',
        'os.environ.get("{K}", "")',
        'os.getenv("{K}", "")',
        'os.getenv("{K}", "<your key>")',
        'os.environ.get("{K}", "${{X}}")',
        'os.getenv("{K}", "{{{{ secrets.X }}}}")',
        'os.getenv("{K}", default)',
        'os.getenv("{K}", default=None)',
        'os.getenv("{K}", default=fallback)',
        'os.getenv("{K}", default="")',
        'os.environ.get("{K}", default="")',
        'os.getenv("{K}", default="<your key>")',
        'os.getenv("{K}", default="${{X}}")',
        'os.environ.get("{K}", default="{{{{ secrets.X }}}}")',
        'os.getenv("{K}", default="{S}")  {I}',
        '{A}: "str | None" = None',
        'def __init__(self, {A}: "SecretStr") -> None:',
        '{A}: "Optional[str]" = None',
        "{A}: 'SecretStr'",
        '{A}: "pydantic.SecretStr | None"',
        "{A}: str",
        "{A}: str | None = None",
        "{A}: Optional[str] = None",
        "def f({A}: str) -> None:",
        "{A}: SomeLongTypeNameWithoutDigits",
        "{A}: short1",
        '{A}: ""',
        "{A}: ''",
        '{A}: "<your key>"',
        '{{"{A}": ""}}',
        "{A}: <your key>  # note",
        '{A}: "{S}"  {I}',
        "{A}: {S}  {I}",
        'JevClient({A}=key)',
        'JevClient({A}=os.environ["{K}"])',
        'JevClient({A}="")',
        'JevClient({A}=None)',
        'JevClient({A}="<your key>")',
        'JevClient({A}="${{X}}")',
        '{A} = "{{{{ secrets.X }}}}"',
        '{A} = os.environ["{K}"]',
        'if {A} == "{S}": pass',
        '{UA} = "{K}"',
        'NO_{UA} = "no_{A}"',
        'monkeypatch.setenv("{K}", "{S}")  {I}',
        'JevClient({A}="{S}")  {I}',
        '{A} = "{S}"  {I}',
    ],
)
def test_references_placeholders_and_empty_values_are_allowed(tmp_path, template):
    line = template.format(K=KEY_NAME, S=SENTINEL, A=API, I=IGNORE, UA=API.upper())
    done = _run(_write(tmp_path / "ok.py", line + "\n"))
    assert done.returncode == 0, (line, done.stderr)


@pytest.mark.parametrize(
    "template",
    [
        'os.getenv("{K}", "{S}")',
        "os.getenv('{K}', '{S}')",
        'os.environ.get("{K}", "{S}")',
        'x = os.environ.get( "{K}" , "{S}" )',
        'os.getenv("{K}", default="{S}")',
        "os.getenv('{K}', default='{S}')",
        'os.environ.get("{K}", default="{S}")',
        'os.environ.get("{K}", default = "{S}")',
    ],
)
def test_literal_default_of_get_and_getenv_is_found(tmp_path, template):
    line = template.format(K=KEY_NAME, S=SENTINEL)
    target = _write(tmp_path / "x.py", line + "\n")
    done = _run(target)
    assert done.returncode == 1, line
    assert SENTINEL not in done.stderr
    assert f"{target}:1: " in done.stderr


@pytest.mark.parametrize(
    "template",
    [
        "{{'{A}': '{S}'}}",
        "{{\"{A}\": '{S}'}}",
        "{{'{A}': \"{S}\"}}",
        '{{"{A}": "{S}"}}',
        '{A}: "{S}"',
        "{A}: '{S}'",
        "  {A}:   '{S}'",
        "{UA}: '{S}'",
        "Api_Key: \"{S}\"",
        '{A}: "tsk_live_abc12defghijklmnopq"',
        "{A}: 'tsk_live_abc12defghijklmnopq'",
        "{{'{A}': 'SecretStr'}}",
        '{{"{A}": "SecretStr"}}',
        "{A}: {S}",
        "{A}: {S}   ",
        "{A}: {S}  # my key",
        "  {A}: tsk_live_abcdefghijklmnop1",
    ],
)
def test_api_key_with_other_quoting_and_yaml_is_found(tmp_path, template):
    line = template.format(S=SENTINEL, A=API, UA=API.upper())
    target = _write(tmp_path / "x.yaml", line + "\n")
    done = _run(target)
    assert done.returncode == 1, line
    assert SENTINEL not in done.stderr
    assert f"{target}:1: " in done.stderr


def _load_module():
    spec = importlib.util.spec_from_file_location("check_secrets_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failing_run(fail_on_call: int, error: Exception, listing: bytes = b"a.py\0b.py\0"):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == fail_on_call:
            raise error
        return subprocess.CompletedProcess(cmd, 0, stdout=listing if len(calls) == 1 else b"x\n")

    return run, calls


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(128, "git"), FileNotFoundError("git")],
)
def test_staged_mode_stops_when_listing_the_staged_files_fails(monkeypatch, capsys, error):
    module = _load_module()
    run, _ = _failing_run(1, error)
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main([]) == module.EXIT_ERROR == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(128, "git"), FileNotFoundError("git")],
)
def test_staged_mode_stops_when_reading_a_staged_file_fails(monkeypatch, capsys, error):
    module = _load_module()
    run, calls = _failing_run(3, error)  # 1: 一覧、2: a.py、3: b.py
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main([]) == module.EXIT_ERROR
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "b.py" in err
    assert len(calls) == 3


def test_staged_mode_outside_a_git_repository_is_an_error(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git が無い")
    env = {**os.environ, "GIT_DIR": str(tmp_path / "no-such-git-dir")}
    done = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert done.returncode == 2
    assert len(done.stderr.strip().splitlines()) == 1


def test_file_arguments_do_not_need_git(tmp_path):
    env = {**os.environ, "GIT_DIR": str(tmp_path / "no-such-git-dir")}
    target = _write(tmp_path / "ok.txt", "x\n")
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert done.returncode == 0
