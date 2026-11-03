"""設定の読み込みとエンジンの選び方のテスト。"""

import dataclasses
import re
from pathlib import Path

import pytest

from jevsearch.config import (
    CONFIG_ENV_VAR,
    Config,
    ConfigError,
    load_config,
    scrape_engine_names,
    select_engines,
)
from jevsearch.models import Genre

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """利用者の本物の設定を読まないよう、ホームと環境変数を空にする。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    return home


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_file(isolated_home):
    config = load_config(None)
    assert config.version == "default"
    assert config.rustxng.url == "http://127.0.0.1:29483"
    assert config.search.max_results == 50
    assert config.search.allow_scrape is False
    assert config.jev.daily_request_limit == 200
    assert config.jev.price_per_mtok_input is None


def test_example_file_matches_code_defaults(isolated_home):
    from_file = load_config(EXAMPLE)
    assert re.fullmatch(r"[0-9a-f]{12}", from_file.version)
    assert dataclasses.replace(from_file, version="default") == load_config(None)


def test_toml_file_gives_hex_version_and_overrides(tmp_path):
    path = _write(tmp_path, '[search]\nmax_results = 10\n\n[rustxng]\nurl = "http://example.test:1"\n')
    config = load_config(path)
    assert re.fullmatch(r"[0-9a-f]{12}", config.version)
    assert config.search.max_results == 10
    assert config.rustxng.url == "http://example.test:1"
    # 書かなかったキーは既定値で埋まる
    assert config.rustxng.expected_version == "0.26.0"
    assert config.engines.official[Genre.BLOG] == ("wiby",)


def test_version_changes_with_content(tmp_path):
    a = load_config(_write(tmp_path, "[search]\nmax_results = 10\n"))
    b = load_config(_write(tmp_path, "[search]\nmax_results = 11\n"))
    assert a.version != b.version


def test_env_var_is_used_when_path_is_none(isolated_home, tmp_path, monkeypatch):
    path = _write(tmp_path, "[search]\nmax_results = 7\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert load_config(None).search.max_results == 7


def test_user_default_location_is_used(isolated_home):
    target = isolated_home / ".config" / "jevsearch" / "config.toml"
    target.parent.mkdir(parents=True)
    target.write_text("[search]\nmax_results = 9\n", encoding="utf-8")
    config = load_config(None)
    assert config.search.max_results == 9
    assert config.version != "default"


def test_missing_explicit_file_is_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nothing.toml")


@pytest.mark.parametrize(
    "text",
    [
        '[jev]\napi_key = "x"\n',
        '[jev]\nMy_Token_Value = "x"\n',
        '[memory]\nsecret = "x"\n',
        '[storage]\nPassword = "x"\n',
        '[rustxng]\nAPIKEY = "x"\n',
        '[extra.nested]\nxx_secret_yy = 1\n',
        # 配列の中（表の配列・配列内のインライン表・入れ子の配列）も調べる
        '[[extra]]\napi_key = "x"\n',
        'servers = [{token = "x"}]\n',
        'matrix = [[{password = "x"}]]\n',
    ],
)
def test_secret_looking_key_is_error(tmp_path, text):
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, text))
    assert '"x"' not in str(info.value)


def test_max_request_tokens_is_not_taken_for_a_secret(tmp_path):
    config = load_config(_write(tmp_path, "[jev]\nmax_request_tokens = 1000\n"))
    assert config.jev.max_request_tokens == 1000


def test_wrong_jev_model_is_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, '[jev]\nmodel = "jev-9.9.9"\n'))


@pytest.mark.parametrize(
    "text",
    [
        "[jev]\nbatch_size = 19\n",
        "[jev]\nbatch_size = 31\n",
        "[jev]\nthreshold = 1.5\n",
        "[jev]\nthreshold = -0.1\n",
        "[search]\nmax_results = 0\n",
        "[search]\nmax_results = 201\n",
        '[search]\nmax_results = "50"\n',
        '[engines.official]\nmovie = ["x"]\n',
        '[engines.scrape]\nmovie = ["x"]\n',
        "[search\n",
    ],
)
def test_invalid_values_are_errors(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text))


def test_boundary_values_are_accepted(tmp_path):
    text = "[jev]\nbatch_size = 20\nthreshold = 0\n[search]\nmax_results = 200\n"
    config = load_config(_write(tmp_path, text))
    assert (config.jev.batch_size, config.jev.threshold, config.search.max_results) == (20, 0.0, 200)


def test_daily_limit_zero_or_negative_means_no_limit(tmp_path):
    assert load_config(_write(tmp_path, "[jev]\ndaily_request_limit = 0\n")).jev.daily_request_limit is None
    assert load_config(_write(tmp_path, "[jev]\ndaily_request_limit = -1\n")).jev.daily_request_limit is None
    assert load_config(_write(tmp_path, "[jev]\ndaily_request_limit = 5\n")).jev.daily_request_limit == 5


def test_tilde_is_expanded(isolated_home, tmp_path):
    config = load_config(_write(tmp_path, '[dictionary]\npath = "~/my/dict.toml"\n'))
    assert config.dictionary.path == isolated_home / "my" / "dict.toml"
    default = load_config(None)
    assert "~" not in str(default.storage.dir)
    assert default.storage.dir == isolated_home / ".local" / "share" / "jevsearch"


def test_bundled_data_paths_are_defaults(isolated_home):
    config = load_config(None)
    assert config.dictionary.path.name == "dictionary.toml"
    assert config.dictionary.path.parent.name == "data"
    assert config.rules.path.name == "domain_rules.toml"


def test_select_engines_with_genre(isolated_home):
    config = load_config(None)
    assert select_engines(config, Genre.PAPER, False) == [
        "arxiv",
        "crossref",
        "semantic scholar",
        "openalex",
        "pubmed",
        "europepmc",
    ]
    assert select_engines(config, Genre.BLOG, False) == ["wiby"]


def test_select_engines_without_genre(isolated_home):
    config = load_config(None)
    assert select_engines(config, None, False) == list(config.engines.unspecified)
    assert "wikipedia" in select_engines(config, None, False)


def test_scrape_needs_both_cli_and_config(isolated_home, tmp_path):
    """never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない"""
    off = load_config(None)
    on = load_config(_write(tmp_path, "[search]\nallow_scrape = true\n"))
    scrape = scrape_engine_names(on)

    for genre in (*Genre, None):
        # CLI だけ（設定は false）
        assert not scrape & set(select_engines(off, genre, True))
        # 設定だけ（CLI のオプションなし）
        assert not scrape & set(select_engines(on, genre, False))
        # どちらもなし
        assert not scrape & set(select_engines(off, genre, False))

    # 両方あるとき
    both = select_engines(on, Genre.GENERAL, True)
    assert both[: len(on.engines.official[Genre.GENERAL])] == list(on.engines.official[Genre.GENERAL])
    assert "google" in both and "mojeek" in both
    assert select_engines(on, Genre.SNS, True) == list(on.engines.official[Genre.SNS])
    assert select_engines(on, None, True)[-4:] == ["google", "bing", "brave", "duckduckgo"]


def test_select_engines_removes_duplicates_keeping_first(tmp_path):
    text = (
        "[search]\nallow_scrape = true\n"
        '[engines.official]\ngeneral = ["a", "b", "a"]\n'
        '[engines.scrape]\ngeneral = ["c", "d", "c"]\n'
    )
    config = load_config(_write(tmp_path, text))
    assert select_engines(config, Genre.GENERAL, True) == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.typesafe.ai/v1/systemone",
        "https://evil.example/v1",
        "https://api.typesafe.ai.evil.example/v1",
        "https://api.typesafe.ai@evil.example/v1",
        "https://evil.example\\@api.typesafe.ai/v1",
        "https://api.typesafe.ai:8443/v1",
        "ftp://api.typesafe.ai/v1",
        "api.typesafe.ai/v1",
        "",
    ],
)
def test_jev_endpoint_must_be_https_on_allowed_host(tmp_path, endpoint):
    """never: API キーを、TypeSafe への認証ヘッダー以外に送らない"""
    with pytest.raises(ConfigError, match=r"jev\.endpoint") as info:
        load_config(_write(tmp_path, f'[jev]\nendpoint = "{endpoint.replace(chr(92), chr(92) * 2)}"\n'))
    assert "evil" not in str(info.value)


def test_jev_endpoint_default_and_allowed_host_load(tmp_path):
    assert load_config(_write(tmp_path, "")).jev.endpoint == "https://api.typesafe.ai/v1/systemone"
    config = load_config(_write(tmp_path, '[jev]\nendpoint = "https://API.typesafe.ai/v2/other"\n'))
    assert config.jev.endpoint == "https://API.typesafe.ai/v2/other"


@pytest.mark.parametrize("genre", [g.value for g in Genre])
def test_empty_official_list_is_error(tmp_path, genre):
    with pytest.raises(ConfigError, match=rf"engines\.official\.{genre}"):
        load_config(_write(tmp_path, f"[engines.official]\n{genre} = []\n"))


def test_empty_unspecified_list_is_error(tmp_path):
    with pytest.raises(ConfigError, match=r"engines\.unspecified"):
        load_config(_write(tmp_path, "[engines]\nunspecified = []\n"))


def test_scrape_engine_in_unspecified_list_is_error(tmp_path):
    with pytest.raises(ConfigError, match=r"engines\.unspecified.*google"):
        load_config(_write(tmp_path, '[engines]\nunspecified = ["wikipedia", "google"]\n'))


def test_scrape_engine_in_official_list_is_error(tmp_path):
    # 別のジャンルの画面読み取り一覧にある名前でも拒む
    with pytest.raises(ConfigError, match=r"engines\.official\.tech.*google_scholar"):
        load_config(_write(tmp_path, '[engines.official]\ntech = ["github", "google_scholar"]\n'))


def test_scrape_list_may_be_empty_and_defaults_load(isolated_home, tmp_path):
    assert load_config(_write(tmp_path, "[engines.scrape]\nsns = []\n")).engines.scrape[Genre.SNS] == ()
    assert load_config(None).version == "default"


def test_scrape_engine_names_covers_all_genres_and_unspecified(isolated_home):
    names = scrape_engine_names(load_config(None))
    assert {"google_scholar", "pypi", "lib_rs", "bing news", "searchmysite", "neocities"} <= names
    assert {"google", "bing", "brave", "duckduckgo", "mojeek", "yahoo japan"} <= names
    assert "wikipedia" not in names


def test_config_is_frozen(isolated_home):
    config = load_config(None)
    assert isinstance(config, Config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.version = "x"  # type: ignore[misc]
