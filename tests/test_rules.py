"""ドメイン規則による種類付け（rules.py）のテスト。URL を入力にした純粋な判定だけを確かめる。"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jevsearch.models import JudgedBy, Judgment, Lang, ResultType, SearchResult
from jevsearch.rules import (
    DomainRule,
    DomainRules,
    domain_of,
    load_bundled_rules,
    load_rules,
    rule_judgment,
    rule_types,
    type_by_domain,
)

BUNDLED = load_bundled_rules()
JST = timezone(timedelta(hours=9))


def _result(url: str) -> SearchResult:
    return SearchResult(
        url=url,
        title="タイトル",
        snippet="抜粋",
        engine="github",
        published_date=None,
        retrieved_at=datetime(2026, 11, 3, 21, 5, 12, tzinfo=JST),
        engines=("github",),
        matched=(Lang.JA,),
    )


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://arxiv.org/abs/2401.00001", "arxiv.org"),
        ("https://WWW.Arxiv.ORG/abs/1", "arxiv.org"),
        ("https://export.arxiv.org:443/api/query?q=x", "export.arxiv.org"),
        ("https://user@github.com/reeschan/jevsearch", "github.com"),
        ("https://www.example.com", "example.com"),
        ("https://wwwexample.com/", "wwwexample.com"),
    ],
)
def test_domain_of_lowercases_host_and_drops_leading_www(url, host):
    assert domain_of(url) == host


@pytest.mark.parametrize(
    "url",
    ["", "not a url", "https:///no-host", "https://[::1/", "arxiv.org/abs/1"],
)
def test_domain_of_unreadable_url_is_empty(url):
    assert domain_of(url) == ""


SPEC_EXAMPLES = [
    # 論文
    ("https://arxiv.org/abs/2401.00001", ResultType.PAPER),
    ("https://doi.org/10.1000/xyz123", ResultType.PAPER),
    ("https://www.semanticscholar.org/paper/abc", ResultType.PAPER),
    ("https://openalex.org/W2741809807", ResultType.PAPER),
    # リポジトリ
    ("https://github.com/reeschan/jevsearch", ResultType.REPO),
    ("https://gitlab.com/group/project", ResultType.REPO),
    # ドキュメント
    ("https://developer.mozilla.org/ja/docs/Web/HTTP", ResultType.DOCS),
    ("https://learn.microsoft.com/ja-jp/azure/", ResultType.DOCS),
    ("https://docs.python.org/3/library/urllib.parse.html", ResultType.DOCS),
    # SNS
    ("https://mastodon.social/@someone/1", ResultType.SNS),
    ("https://lemmy.world/c/python", ResultType.SNS),
    ("https://x.com/someone/status/1", ResultType.SNS),
    # ブログ
    ("https://qiita.com/someone/items/abc", ResultType.BLOG),
    ("https://zenn.dev/someone/articles/abc", ResultType.BLOG),
    ("https://note.com/someone/n/n1234", ResultType.BLOG),
    ("https://medium.com/@someone/post-1", ResultType.BLOG),
    ("https://someone.hatenablog.com/entry/2026/11/03", ResultType.BLOG),
    ("https://dev.to/someone/post-1", ResultType.BLOG),
    # ニュース
    ("https://news.google.com/rss/articles/CBMiabc", ResultType.NEWS),
    ("https://wikinews.org/wiki/Example", ResultType.NEWS),
    ("https://ja.wikinews.org/wiki/Example", ResultType.NEWS),
]


@pytest.mark.parametrize(("url", "expected"), SPEC_EXAMPLES)
def test_spec_examples_get_their_type(url, expected):
    assert type_by_domain(url, BUNDLED) == expected


def test_plain_rule_also_matches_subdomains():
    assert type_by_domain("https://export.arxiv.org/api/query", BUNDLED) == ResultType.PAPER


def test_wildcard_rule_matches_only_subdomains():
    assert type_by_domain("https://someone.hatenablog.com/entry/1", BUNDLED) == ResultType.BLOG
    assert type_by_domain("https://hatenablog.com/", BUNDLED) is None


def test_leading_label_rule_matches_only_the_first_label():
    assert type_by_domain("https://docs.python.org/3/", BUNDLED) == ResultType.DOCS
    assert type_by_domain("https://mydocs.example.org/", BUNDLED) is None


@pytest.mark.parametrize(
    "url",
    ["https://docs.github.com/en/actions", "https://docs.gitlab.com/ee/ci/"],
)
def test_docs_host_under_a_repo_domain_is_docs(url):
    assert type_by_domain(url, BUNDLED) == ResultType.DOCS


def test_other_subdomains_of_a_repo_domain_stay_repo():
    assert type_by_domain("https://gist.github.com/reeschan/abc", BUNDLED) == ResultType.REPO


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "https://notgithub.com/reeschan/jevsearch",
        "https://github.com.example.net/reeschan/jevsearch",
        "https://arxiv.org.example.com/abs/1",
        "https://www.google.com/search?q=arxiv",
        "",
    ],
)
def test_unmatched_url_has_no_type(url):
    assert type_by_domain(url, BUNDLED) is None


def test_earlier_rule_wins_when_two_rules_match():
    rules = DomainRules(
        version="test-1",
        rules=(
            DomainRule(type=ResultType.BLOG, domains=("example.org",)),
            DomainRule(type=ResultType.NEWS, domains=("news.example.org",)),
        ),
    )
    assert type_by_domain("https://news.example.org/a", rules) == ResultType.BLOG


def test_rule_judgment_is_by_rule_with_confidence_one():
    judgment = rule_judgment("https://arxiv.org/abs/2401.00001", BUNDLED)
    assert judgment == Judgment(value=ResultType.PAPER, by=JudgedBy.RULE, confidence=1.0)
    assert judgment.by == JudgedBy.RULE
    assert judgment.confidence == 1.0


def test_rule_judgment_is_none_when_no_rule_matches():
    assert rule_judgment("https://example.com/", BUNDLED) is None


def test_rule_types_keeps_only_positions_decided_by_domain():
    """規則で決まった結果の位置だけを、種類と一緒に返す。

    遷移: ドメインで分かる種類を付け終えた
    never: ドメイン規則で決まった種類を Jev の答えで上書きしない
    """
    results = [
        _result("https://arxiv.org/abs/2401.00001"),
        _result("https://example.com/post"),
        _result("https://github.com/reeschan/jevsearch"),
        _result("https://notgithub.com/reeschan/jevsearch"),
    ]
    assert rule_types(results, BUNDLED) == {0: ResultType.PAPER, 2: ResultType.REPO}


def test_rule_types_is_empty_when_nothing_matches():
    assert rule_types([_result("https://example.com/")], BUNDLED) == {}
    assert rule_types([], BUNDLED) == {}


def test_bundled_rules_have_a_version():
    assert BUNDLED.version.strip() != ""


def test_bundled_rules_cover_the_six_spec_types():
    assert {rule.type for rule in BUNDLED.rules} == {
        ResultType.PAPER,
        ResultType.REPO,
        ResultType.DOCS,
        ResultType.SNS,
        ResultType.BLOG,
        ResultType.NEWS,
    }


def test_load_rules_reads_version_and_rules_in_order(tmp_path: Path):
    path = tmp_path / "rules.toml"
    path.write_text(
        'version = "test-1"\n'
        "\n"
        "[[rules]]\n"
        'type = "news"\n'
        'domains = ["Example.ORG", "*.Example.net", "x.*"]\n'
        "\n"
        "[[rules]]\n"
        'type = "blog"\n'
        'domains = ["dev.to"]\n',
        encoding="utf-8",
    )
    assert load_rules(path) == DomainRules(
        version="test-1",
        rules=(
            DomainRule(type=ResultType.NEWS, domains=("example.org", "*.example.net", "x.*")),
            DomainRule(type=ResultType.BLOG, domains=("dev.to",)),
        ),
    )


@pytest.mark.parametrize(
    "body",
    [
        'version = ""\n',
        'version = "test-1"\n[[rules]]\ntype = "video"\ndomains = ["example.org"]\n',
        'version = "test-1"\n[[rules]]\ntype = "news"\ndomains = []\n',
        'version = "test-1"\n[[rules]]\ntype = "news"\ndomains = ["a*.example.org"]\n',
        'version = "test-1"\n[[rules]]\ntype = "news"\ndomains = ["*"]\n',
    ],
)
def test_load_rules_rejects_invalid_files(tmp_path: Path, body: str):
    path = tmp_path / "rules.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(path)
