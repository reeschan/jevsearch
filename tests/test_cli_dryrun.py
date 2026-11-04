"""CLI の骨組みと --dry-run、検索リクエストの組み立てのテスト。"""

import io
import json
from pathlib import Path

import httpx
import pytest

from jevsearch.cli import Deps, main
from jevsearch.config import load_config, scrape_engine_names
from jevsearch.jev_classify import JevClassifier
from jevsearch.mcp_client import McpHttpClient
from jevsearch.models import Lang
from jevsearch.rustxng import RustxngAdapter
from jevsearch.search_request import build_search_request


def _fail_factory(name):
    def make(config):
        raise AssertionError(f"{name} が呼ばれた")

    return make


def _deps(env=None):
    return Deps(
        env=env if env is not None else {},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        make_search_backend=_fail_factory("make_search_backend"),
        make_classifier=_fail_factory("make_classifier"),
        make_memory_client=_fail_factory("make_memory_client"),
    )


def _config(tmp_path: Path, text: str = "") -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _run(argv, deps):
    code = main(argv, deps=deps)
    return code, deps.stdout.getvalue(), deps.stderr.getvalue()


def _dry(tmp_path, *extra, text="", query="rust メタ検索"):
    deps = _deps()
    cfg = _config(tmp_path, text)
    code, out, err = _run(["search", query, "--dry-run", "--config", str(cfg), *extra], deps)
    assert code == 0, err
    return json.loads(out)


def test_dry_run_json_shape(tmp_path):
    data = _dry(tmp_path, "--genre", "paper", "--topic", "t1", "--max", "7")
    assert data["query"] == "rust メタ検索"
    assert data["query_norm"] == {"ja": "rust メタ検索", "en": "rust metasearch"}
    assert data["genre"] == "paper"
    assert data["topic"] == "t1"
    assert data["engines"][0] == "arxiv"
    request = data["rustxng"]["requests"][0]
    assert request["lang"] == "ja"
    assert request["url"] == "http://127.0.0.1:29483/search"
    assert request["params"]["max_results"] == "7"
    assert request["params"]["language"] == "ja"
    assert request["params"]["q"] == "rust メタ検索"
    jev = dict(data["jev"])
    assert jev.pop("preview")["questions"]
    assert jev == {"model": "jev-1.14.0", "batch_size": 25, "threshold": 0.55}
    assert data["versions"] == {"config": data["versions"]["config"]}
    assert data["versions"]["config"]


def test_dry_run_params_have_engines_format_json_and_no_categories(tmp_path):
    data = _dry(tmp_path)
    params = data["rustxng"]["requests"][0]["params"]
    assert params["engines"] == ",".join(data["engines"])
    assert "categories" not in params
    assert params["format"] == "json"
    assert data["genre"] is None and data["topic"] is None


def test_dry_run_uses_default_max_from_config(tmp_path):
    data = _dry(tmp_path, text="[search]\nmax_results = 12\n")
    assert data["rustxng"]["requests"][0]["params"]["max_results"] == "12"


def test_dry_run_prints_unicode_unescaped(tmp_path):
    deps = _deps()
    cfg = _config(tmp_path)
    code, out, _ = _run(["search", "日本語", "--dry-run", "--config", str(cfg)], deps)
    assert code == 0
    assert "日本語" in out
    assert out.startswith("{\n  ")


def test_dry_run_calls_no_external_factory(tmp_path):
    deps = _deps()
    cfg = _config(tmp_path)
    # どれかの make_* が呼ばれると AssertionError になり、このテストは失敗する
    code, _, _ = _run(["search", "abc", "--dry-run", "--config", str(cfg)], deps)
    assert code == 0


def test_allow_scrape_flag_alone_adds_no_scrape_engine(tmp_path):
    """never: 画面読み取りのエンジンを、オプションを付けていないのに呼ばない"""
    cfg = _config(tmp_path)  # 設定の allow_scrape は既定のオフ
    scrape = scrape_engine_names(load_config(cfg))
    for genre in (None, "paper", "tech", "news", "sns", "blog", "general"):
        extra = ["--allow-scrape", *(["--genre", genre] if genre else [])]
        data = _dry(tmp_path, *extra)
        assert not scrape & set(data["engines"])
        params = data["rustxng"]["requests"][0]["params"]
        assert not scrape & set(params["engines"].split(","))


def test_scrape_engines_need_both_flag_and_config(tmp_path):
    text = "[search]\nallow_scrape = true\n"
    with_flag = _dry(tmp_path, "--allow-scrape", "--genre", "general", text=text)
    assert "google" in with_flag["engines"]
    without_flag = _dry(tmp_path, "--genre", "general", text=text)
    assert "google" not in without_flag["engines"]


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_query_is_usage_error(tmp_path, query):
    deps = _deps()
    cfg = _config(tmp_path)
    code, out, err = _run(["search", query, "--dry-run", "--config", str(cfg)], deps)
    assert code == 2
    assert out == ""
    assert err


@pytest.mark.parametrize("value", ["-1", "0", "201", "1000"])
def test_max_outside_1_to_200_is_usage_error(tmp_path, value):
    deps = _deps()
    cfg = _config(tmp_path)
    code, out, err = _run(["search", "rust", "--max", value, "--dry-run", "--config", str(cfg)], deps)
    assert code == 2
    assert out == ""
    assert "--max" in err
    assert "1〜200" in err


@pytest.mark.parametrize("value", ["1", "200"])
def test_max_at_the_bounds_is_accepted(tmp_path, value):
    payload = _dry(tmp_path, "--max", value)
    assert payload["rustxng"]["requests"][0]["params"]["max_results"] == value


@pytest.mark.parametrize("topic", ["", "   ", "\t\n"])
def test_blank_topic_is_usage_error(tmp_path, topic):
    deps = _deps()
    # 設定ファイルが無くても、設定を読む前に使い方のエラーで止まる。
    code, out, err = _run(
        ["search", "rust", "--topic", topic, "--dry-run", "--config", str(tmp_path / "no.toml")],
        deps,
    )
    assert code == 2
    assert out == ""
    assert "--topic" in err


def test_config_with_api_key_is_error(tmp_path):
    deps = _deps()
    cfg = _config(tmp_path, '[jev]\napi_key = "sk-test-SENTINEL-0000"\n')  # check-secrets: ignore
    code, out, err = _run(["search", "abc", "--dry-run", "--config", str(cfg)], deps)
    assert code == 2
    assert out == ""
    assert "api_key" in err
    assert "sk-test-SENTINEL-0000" not in err


def test_missing_config_file_is_error(tmp_path):
    deps = _deps()
    code, _, err = _run(["search", "abc", "--dry-run", "--config", str(tmp_path / "no.toml")], deps)
    assert code == 2
    assert err


def test_env_config_path_is_read_from_deps_env(tmp_path):
    cfg = _config(tmp_path, "[search]\nmax_results = 3\n")
    deps = _deps(env={"JEVSEARCH_CONFIG": str(cfg)})
    code, out, _ = _run(["search", "abc", "--dry-run"], deps)
    assert code == 0
    assert json.loads(out)["rustxng"]["requests"][0]["params"]["max_results"] == "3"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["search"],
        ["search", "abc", "--genre", "movie"],
        ["search", "abc", "--max", "x"],
        ["search", "abc", "--max", "0"],
    ],
)
def test_usage_errors_return_2(tmp_path, argv):
    deps = _deps()
    code, out, err = _run(argv, deps)
    assert code == 2
    assert out == ""
    assert err


def test_default_deps_factories_build_real_parts(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    deps = Deps(env={}, http_transport=transport)
    config = load_config(_config(tmp_path))
    assert isinstance(deps.make_search_backend(config, False), RustxngAdapter)
    assert isinstance(deps.make_classifier(config), JevClassifier)
    assert isinstance(deps.make_memory_client(config), McpHttpClient)
    assert deps.clock.now().tzinfo is not None


def test_build_search_request_url_and_params():
    url, params = build_search_request(
        "http://127.0.0.1:29483/", "rust", Lang.EN, ["arxiv", "github"], 50
    )
    assert url == "http://127.0.0.1:29483/search"
    assert params == {
        "q": "rust",
        "format": "json",
        "engines": "arxiv,github",
        "language": "en",
        "max_results": "50",
    }
    assert "categories" not in params


def test_build_search_request_strips_all_trailing_slashes_and_time_range():
    url, params = build_search_request("http://h:1//", "q", Lang.JA, ["a"], 5, "month")
    assert url == "http://h:1/search"
    assert params["time_range"] == "month"
    assert params["language"] == "ja"
    assert params["max_results"] == "5"
