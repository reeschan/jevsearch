# jevsearch

検索語を RustXNG（自分の PC で動くメタ検索）に投げ、集まった結果を TypeSafe の Jev で「種類・ジャンル・関連性」に分類して、JSONL に残す Python の CLI です。トピックを付けた検索は、日ごとのダイジェストを共有メモリ（shared-memory MCP）にも書きます。

> **外部に送られるものがあります。** 検索語と結果の一部が米国のサービス（TypeSafe の Jev）に送られます。詳しくは「[外部に送られるもの](#外部に送られるもの)」を読んでから使ってください。

## 前提

- Python 3.11 以上
- [uv](https://docs.astral.sh/uv/)
- 127.0.0.1 で動いている RustXNG v0.26.0。RustXNG は開発が速く、既定のポートが版によって違います。このため **`rustxng.url` を必ず設定してください**（例の設定ファイルの値は `http://127.0.0.1:29483`）。作者の監査で未解決の問題が残っているので、127.0.0.1 だけで動かし、外には公開しないでください。
- TypeSafe の API キー（Jev を使うとき。無くても検索と保存はできますが、分類は全件が「未分類」になります）
- 共有メモリのダイジェストを書くときは、手元の shared-memory MCP（既定は `http://127.0.0.1:7820/mcp`）

## 導入

```sh
uv sync
git config core.hooksPath .githooks
```

2行目は、API キーが入ったファイルをコミットしにくくする検査（コミット前のフック）を有効にします。**クローンしたあとに、この `git config core.hooksPath .githooks` を必ず一度実行してください。** 実行しないと、フックは動かず、検査なしでコミットできます。Linux・macOS でフックが動かないときは、フックに実行権限があるかを確かめます（`git ls-files -s .githooks/pre-commit` の先頭が `100755` なら付いています。付いていなければ `chmod +x .githooks/pre-commit`）。

## 設定

`config.example.toml` を `~/.config/jevsearch/config.toml` に写して編集します。ファイルが無くても既定値で動きます。別の場所を使うときは、環境変数 `JEVSEARCH_CONFIG` か、各コマンドの `--config` で指します。

```sh
mkdir -p ~/.config/jevsearch
cp config.example.toml ~/.config/jevsearch/config.toml
```

主な項目:

- `rustxng.url`: RustXNG の接続先。必ず自分の環境に合わせます。
- `engines.unspecified`: ジャンルを指定しない検索で使うエンジンの一覧。ジャンルごとの公式エンジンは `engines.official`、画面読み取りのエンジンは `engines.scrape`。
- `jev.daily_request_limit`: Jev へのリクエスト数の 1 日の上限（既定 200。0 以下で上限なし）。上限に達すると Jev を呼ばず、残りは「未分類」になります。
- 画面読み取りのエンジンは **二重の許可** が要ります。設定の `search.allow_scrape = true` と、コマンドの `--allow-scrape` の両方があるときだけ使います。どちらかが欠けると呼びません。

設定ファイルに秘密は書けません（`api_key`・`token`・`secret`・`password` を含むキー名は読み込みで拒みます）。

## API キー

Jev を使うには TypeSafe の API キーが要ります。次のどちらかで渡します（上が先）。

1. 環境変数 `TYPESAFE_API_KEY`
2. `~/.config/typesafe/credentials.json` の `api_key`（`{"api_key": ...}` の形の JSON）

```sh
export TYPESAFE_API_KEY=<あなたのキー>
```

キーはこのリポジトリに入れません。コードにも、設定ファイルにも、テストにも書きません。キーは TypeSafe への認証ヘッダーにだけ入り、結果ログ・ジョブの記録・画面の表示・エラーメッセージには出ません。キーが無いときは、Jev が使えないときと同じ扱い（全件を未分類にして、結末は「一部だけ保存」）です。

## 使い方

```sh
# 検索する（トピックを付けると、共有メモリにダイジェストも書く）
uv run jevsearch search "量子コンピュータ 誤り訂正" --topic quantum --genre paper --max 30

# 外部を呼ばずに、送る内容だけを JSON で見る（API キーも読まない）
uv run jevsearch search "量子コンピュータ 誤り訂正" --dry-run

# 画面読み取りのエンジンも使う（設定の search.allow_scrape = true も必要）
uv run jevsearch search "rust async" --genre tech --allow-scrape

# まとめて判定と 1 件ずつ判定の一致率（日本語と英語を分けて出す）
uv run jevsearch eval batch-vs-single ~/.local/share/jevsearch/results.jsonl

# 人が正解を付けるための抜き出し（CSV）と、付けた正解との一致率
uv run jevsearch eval sample --n 50 --out sample.csv
uv run jevsearch eval accuracy sample.csv

# 評価記録の集計
uv run jevsearch stats --since 2026-11-01

# 共有メモリに書けなかったダイジェストを書き直す
uv run jevsearch digest --retry
```

`search` の引数: `--topic <名前>`、`--genre {paper,tech,news,sns,blog,general}`、`--max <件数>`（1 以上。1 ジョブの合計の上限）、`--allow-scrape`、`--dry-run`、`--config <ファイル>`。

トピックを付けない検索は、結果ログとジョブの記録だけに残り、共有メモリには何も書きません。

## 結末と終了コード

検索ジョブは 4 つの結末のどれかで終わります。要約は標準出力に、警告は標準エラーに出ます。

| 結末 | 意味 | 終了コード |
| --- | --- | --- |
| 保存済み | 結果が取れて、すべて保存できた | 0 |
| 一部だけ保存 | 取れた結果は保存した。ただし一部のエンジンや検索語が答えなかった、または Jev が使えなかった | 0（標準エラーに警告） |
| 0件 | すべての検索語が正常に答えたが、結果が 0 件だった（失敗ではない） | 0 |
| 失敗 | 検索エンジンにつながらず、結果が 1 件も届かなかった | 1 |
| （使い方・設定のエラー） | 引数の誤り、設定ファイルの誤り | 2 |

確信度がしきい値（既定 0.55）より低い判定は「未分類」として残し、結末は変えません。ダイジェストを書けなかったことも結末を変えません（結果ログは残り、`digest --retry` で書き直せます）。

## 保存されるファイル

保存先は設定の `storage.dir`（既定 `~/.local/share/jevsearch`）です。

| ファイル | 中身 |
| --- | --- |
| `results.jsonl` | 結果ログ。1 結果 1 行（検索語、URL、タイトル、抜粋、エンジン、判定と確信度、版など） |
| `jobs.jsonl` | ジョブの記録。1 ジョブ 1 行。0 件と失敗の分も必ず 1 行残る |
| `eval.jsonl` | 評価記録。1 ジョブ 1 行（判定の内訳、確信度の分布、トークン数、失敗したエンジン） |
| `digest-pending.jsonl` | 書けなかったダイジェストの書き残し |
| `digest-index.json` | トピックと日付ごとの、共有メモリの最新の id |
| `jev-usage.json` | 日ごとの Jev のリクエスト数（1 日の上限の判定に使う） |

日付はローカル時刻の日付です。書き込みの前に、API キーの文字列が含まれていないかを調べ、含まれていれば書かずに止めます。この検査をするのは `results.jsonl`・`jobs.jsonl`・`eval.jsonl` の 3 つだけです。`digest-pending.jsonl`・`digest-index.json`・`jev-usage.json` は、この検査の対象ではありません（API キーの文字列は入らない中身です）。

## 共有メモリのダイジェスト

トピックを付けた検索ジョブは、トピックと日付ごとに 1 件、タグ `news:radar-bot:<topic>` で共有メモリにダイジェストを書きます。中身は、その日のそのトピックの結果の URL・タイトル・種類・ジャンル・エンジンの一覧です（関連性が「合っていない」と判定されたものは除きます）。同じ日に同じトピックで再度検索すると、同じ 1 件を更新します。`memory.enabled = false` で書かなくできます。

## テスト

```sh
uv run pytest
uv run ruff check .
```

テストはネットワークなしで通ります。実際の RustXNG を呼ぶ結合テストは既定では動かさず、`-m integration` と環境変数 `JEVSEARCH_RUSTXNG_URL` が要ります。

```sh
JEVSEARCH_RUSTXNG_URL=http://127.0.0.1:29483 uv run pytest -m integration
```

## 外部に送られるもの

**検索語と結果の一部が、米国のサービス（TypeSafe の Jev）に送られます。** 秘密にしたい検索語や、公開できない内容を含む検索語では使わないでください。

- 送り先: TypeSafe の System One（`https://api.typesafe.ai/v1/systemone`、モデルは `jev-1.14.0` に固定）。サーバーは米国にあります。
- 送るもの: 検索語、結果のタイトル・URL・抜粋・ドメイン名・エンジン名、内部の ID（`r01` など）。分類に要る最小限だけを送り、これ以外は送りません。
- 送らないもの: API キーは認証ヘッダーにだけ入ります。Jev に送る内容（`state` と質問）には入りません。
- RustXNG と共有メモリは、自分の PC の 127.0.0.1 で動かす前提です。ただし RustXNG が各検索エンジンに検索語を送ることは、RustXNG 側の設定に従います。
- `--dry-run` を付けると、何も送らずに、送る内容を確かめられます。

## 公開リポジトリと鍵の検査

このリポジトリは公開です。API キーが入ったファイルをコミットしにくくするために、次を入れています。

- `.gitignore` が `.env`・`.env.*`（`.env.example` を除く）・`credentials.json`・`*.pem`・`*.key` を除外します。
- コミット前のフック `.githooks/pre-commit` が `scripts/check_secrets.py` を引数なしで呼びます。引数が無いときは、**ステージ済みの内容**（`git show :<パス>`）を調べます。作業ツリーのファイルは読みません。対象は、追加・コピー・変更・名前の変更（新しい名前）でステージされたファイルです。見つけると `<パス>:<行番号>: <種類>` を出してコミットを止めます（値そのものは出しません）。フックは `git config core.hooksPath .githooks` をしたときだけ動きます。
- 手動で調べるとき: `python scripts/check_secrets.py <ファイル> ...`。ファイルを渡したときは、そのファイルを**作業ツリーから**読みます（ステージ済みの内容ではありません）。
- 調べる種類は、ファイル名（`credentials.json`・`.env`・`.env.*`。`.env.example` は除く。大文字小文字は区別しません）、`TYPESAFE_API_KEY` への値の代入（`os.environ[...] = "値"` を含む。`os.environ.get(...)` などの参照は除く）、`"api_key"` の文字列の値、`Bearer ` に続く 16 文字以上のトークン（大文字小文字・空白の数は問いません）です。UTF-8 と、BOM 付きの UTF-16 を読みます。値が `<...>`・`${...}`・`${{ ... }}`・`{{ ... }}` で包まれているものは、書き込み用の穴や参照とみなして除きます。
- 意図した行は、行の中に `# check-secrets: ignore` を書くと除かれます。
