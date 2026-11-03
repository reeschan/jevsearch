# 状態の図と形式検証の再生成（pstack `/statechart`）

このフォルダ（`docs/specs/statechart/`）で実行する。pstack の statechart スクリプトは
Claude Code のプラグインとして手元に入っている前提（初回は TLC と Java のダウンロードがある）。

Windows（PowerShell）:

    cd docs\specs\statechart
    $sc = "$env:USERPROFILE\.claude\plugins\marketplaces\cursor-plugins\pstack\skills\statechart\scripts\statechart.mjs"
    node $sc render agreed-verbatim.mmd
    node $sc render jevsearch-job.mmd
    node $sc from-chart jevsearch-job.mmd --png

macOS / Linux:

    cd docs/specs/statechart
    sc=~/.claude/plugins/marketplaces/cursor-plugins/pstack/skills/statechart/scripts/statechart.mjs
    node "$sc" render agreed-verbatim.mmd
    node "$sc" render jevsearch-job.mmd
    node "$sc" from-chart jevsearch-job.mmd --png

## ファイル

- `agreed-verbatim.mmd` / `.png`: 合意した図（ユーザーの言葉のまま）。
- `jevsearch-job.mmd`: 検証用の図（`%% never` の決まりつき）。
- `JevsearchJob.tla` / `.cfg`: 図から作った TLA+ のモデル。`.tla` があれば作り直さずに使う。図から作り直すときは消してから実行する。
- `JevsearchJob.map.mjs`: TLC の ASCII の状態名を、図の日本語の名前に戻す対応表。

## 結果の見方

すべての決まりが守られていれば exit 0。どれかが破れると exit 1。
破れた決まりを `--expect-broken` に足して回避しないこと。図か仕様を直す。

出力フォルダ `JevsearchJob-statechart/`（TLC の出力、chart.png など）はコミットしない（`.gitignore` 済み）。
Windows では、このフォルダを置き換える rename が EPERM になることがある。
その場合は既存の `JevsearchJob-statechart` を別名に移してから実行する。

## 最後に検証した結果（2026-11-03）

10 の決まり（TypeOK と never 9 つ）がすべて守られた。9 状態・9 遷移、モデル上 3800 状態。
