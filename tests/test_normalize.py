"""正規化（用語辞書）のテスト。辞書はテスト内で小さく作って渡す。"""

from jevsearch.normalize import (
    Dictionary,
    Term,
    load_bundled_dictionary,
    load_dictionary,
    normalize,
)

DICTIONARY = Dictionary(
    version="test.1",
    entries=(
        Term(canonical="サーバー", aliases=("サーバ",), en="server"),
        Term(canonical="データベース", aliases=("DB",), en="database"),
        Term(canonical="ネットワーク", aliases=("ネットワーク網",), en=None),
    ),
)


def test_表記ゆれが同じ_ja_にそろう():
    assert normalize("サーバ", DICTIONARY).ja == "サーバー"
    assert normalize("サーバー", DICTIONARY).ja == "サーバー"
    assert normalize("サーバ", DICTIONARY).ja == normalize("サーバー", DICTIONARY).ja


def test_全角英数と前後の空白と連続する空白をそろえる():
    result = normalize("　ＡＢＣ１２３　　監視 \t ツール　", DICTIONARY)
    assert result.ja == "ABC123 監視 ツール"
    assert result.en is None


def test_英語版は英語表記のある語だけ英語にして作る():
    result = normalize("サーバ 監視", DICTIONARY)
    assert result.ja == "サーバー 監視"
    assert result.en == "server 監視"


def test_辞書にない語だけなら_en_は_None_で_ja_は正規化しただけ():
    result = normalize("　Ｐｙｔｈｏｎ　入門 ", DICTIONARY)
    assert result.ja == "Python 入門"
    assert result.en is None


def test_空白で区切れない連なりの中の別名を置き換える():
    result = normalize("サーバ構築", DICTIONARY)
    assert result.ja == "サーバー構築"
    assert result.en == "server構築"


def test_見出し語自身の中の別名で壊さない():
    # 「サーバ」は「サーバー」の一部にも当たるが、長い方を先に当てるので「サーバーー」にならない。
    result = normalize("サーバー構築", DICTIONARY)
    assert result.ja == "サーバー構築"
    assert result.en == "server構築"


def test_長い別名を先に当てる():
    # 「ネットワーク網」は「ネットワーク」より長い別名。
    result = normalize("ネットワーク網設計", DICTIONARY)
    assert result.ja == "ネットワーク設計"


def test_英数字の連なりの一部には当てない():
    assert normalize("DBMS", DICTIONARY).ja == "DBMS"
    assert normalize("DBMS", DICTIONARY).en is None
    # 空白で区切った語がまるごと一致するときは、大文字小文字を区別せず置き換える。
    result = normalize("db 設計", DICTIONARY)
    assert result.ja == "データベース 設計"
    assert result.en == "database 設計"


def test_英語表記のない語だけを置き換えたときは_en_は作らない():
    result = normalize("ネットワーク網", DICTIONARY)
    assert result.ja == "ネットワーク"
    assert result.en is None


def test_original_と_dictionary_version_が入る():
    """遷移: 用語辞書で検索語をそろえ終えた"""
    result = normalize("  サーバ 監視 ", DICTIONARY)
    assert result.original == "  サーバ 監視 "
    assert result.dictionary_version == "test.1"


def test_空の検索語():
    result = normalize("   ", DICTIONARY)
    assert result.ja == ""
    assert result.en is None


def test_辞書ファイルを読める(tmp_path):
    path = tmp_path / "dict.toml"
    path.write_text(
        'version = "v9"\n\n[[terms]]\ncanonical = "サーバー"\naliases = ["サーバ"]\nen = "server"\n',
        encoding="utf-8",
    )
    dictionary = load_dictionary(path)
    assert dictionary.version == "v9"
    assert dictionary.entries == (Term("サーバー", ("サーバ",), "server"),)


def test_同梱辞書が読めて_version_が空でない():
    dictionary = load_bundled_dictionary()
    assert dictionary.version
    result = normalize("インタフェース 設計", dictionary)
    assert result.ja == "インターフェース 設計"
    assert result.en == "interface 設計"
