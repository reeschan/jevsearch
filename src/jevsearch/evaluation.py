"""評価記録の作成・集計と、まとめて判定と1件ずつ判定の比較、人の正解との一致率の計算。

ファイルや Jev には触れない。行（dict）は呼び出し側が渡す。
"""

from __future__ import annotations

import csv
import io
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime

from jevsearch.models import (
    ClassifiedResult,
    EngineFailure,
    JevReport,
    JudgedBy,
    Judgment,
    Lang,
    ResultType,
    SearchResult,
)
from jevsearch.ports import Classifier

# 判定の種類（評価記録・比較・CSV の列名と同じ）。
JUDGMENT_KEYS: tuple[str, ...] = ("type", "genre", "relevant")

# 確信度の分布の区間数（0.1 刻み）。
N_BINS = 10

CSV_COLUMNS: tuple[str, ...] = (
    "job_id",
    "id",
    "query",
    "url",
    "title",
    "snippet",
    "lang",
    "type",
    "genre",
    "relevant",
    "label_type",
    "label_genre",
    "label_relevant",
)


def _rate(count: int, total: int) -> float:
    """割合。分母が 0 のときは 0.0。"""
    return count / total if total else 0.0


def _bin_index(confidence: float) -> int:
    """確信度を 0.1 刻みの区間番号（0〜9）にする。1.0 は最後の区間、範囲外は端に寄せる。"""
    return min(max(int(confidence * N_BINS), 0), N_BINS - 1)


def _judgment_counts(judgments: Sequence[Judgment], total: int) -> dict:
    rule = sum(1 for j in judgments if j.value is not None and j.by == JudgedBy.RULE)
    jev = sum(1 for j in judgments if j.value is not None and j.by == JudgedBy.JEV)
    unclassified = sum(1 for j in judgments if j.value is None)
    return {
        "rule": rule,
        "jev": jev,
        "unclassified": unclassified,
        "rule_rate": _rate(rule, total),
        "jev_rate": _rate(jev, total),
        "unclassified_rate": _rate(unclassified, total),
    }


def _sum_known(values: Iterable[int | None]) -> int | None:
    """None を除いて合計する。1つも値がなければ None。"""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def eval_row(
    *,
    job_id: str,
    started_at: datetime,
    items: Sequence[ClassifiedResult],
    report: JevReport,
    failed_engines: Sequence[EngineFailure],
    price_per_mtok_input: float | None,
    price_per_mtok_output: float | None,
) -> dict:
    """検索ジョブ1件分の評価記録の1行を作る。

    割合は結果の件数に対する割合で、件数 0 のときは 0.0。
    確信度の分布は、確信度のある判定（type・genre・relevant。未分類の判定も含む）を
    0.1 刻みの10区間に数えたもの。確信度のない判定は数えない。
    費用の目安（USD）は、単価が両方あり、全ての使用量の入力・出力トークン数が取れたときだけ
    数値、それ以外（1つでも欠けた場合を含む）は None。
    """
    total = len(items)
    row: dict = {
        "job_id": job_id,
        "started_at": started_at.isoformat(),
        "n_results": total,
    }
    bins = [0] * N_BINS
    for key in JUDGMENT_KEYS:
        judgments = [getattr(item, key) for item in items]
        row[key] = _judgment_counts(judgments, total)
        for j in judgments:
            if j.confidence is not None:
                bins[_bin_index(j.confidence)] += 1
    row["confidence_distribution"] = bins

    input_tokens = _sum_known(u.input_tokens for u in report.usage)
    output_tokens = _sum_known(u.output_tokens for u in report.usage)
    row["jev_tokens"] = {
        "input": input_tokens,
        "output": output_tokens,
        "estimated_input": sum(u.estimated_input_tokens for u in report.usage),
    }
    tokens_complete = bool(report.usage) and all(
        u.input_tokens is not None and u.output_tokens is not None for u in report.usage
    )
    cost: float | None = None
    if (
        price_per_mtok_input is not None
        and price_per_mtok_output is not None
        and tokens_complete
        and input_tokens is not None
        and output_tokens is not None
    ):
        cost = (
            input_tokens * price_per_mtok_input + output_tokens * price_per_mtok_output
        ) / 1_000_000
    row["cost_usd"] = cost
    row["failed_engines"] = [{"engine": f.engine, "reason": f.reason} for f in failed_engines]
    return row


def aggregate(rows: Iterable[dict], since: date | None = None) -> dict:
    """評価記録の行を集計する。since より前の started_at の行は除く。

    since はローカルの日付。started_at（UTC など保存時のオフセット）は
    ``astimezone()``（引数なし）でローカルの時刻にしてから日付を取って比べる。
    割合は結果の合計件数に対するもの（0 件なら 0.0）。
    トークンと費用は、値のある行だけ足し、1行も値がなければ None。
    """
    jobs = 0
    results = 0
    counts = {key: {"rule": 0, "jev": 0, "unclassified": 0} for key in JUDGMENT_KEYS}
    token_in: list[int | None] = []
    token_out: list[int | None] = []
    token_est: list[int | None] = []
    costs: list[float] = []
    failures: dict[str, int] = {}

    for row in rows:
        if since is not None:
            local_date = datetime.fromisoformat(row["started_at"]).astimezone().date()
            if local_date < since:
                continue
        jobs += 1
        results += row.get("n_results", 0)
        for key in JUDGMENT_KEYS:
            for kind in ("rule", "jev", "unclassified"):
                counts[key][kind] += row.get(key, {}).get(kind, 0)
        tokens = row.get("jev_tokens") or {}
        token_in.append(tokens.get("input"))
        token_out.append(tokens.get("output"))
        token_est.append(tokens.get("estimated_input"))
        if row.get("cost_usd") is not None:
            costs.append(row["cost_usd"])
        for failure in row.get("failed_engines", []):
            engine = failure["engine"]
            failures[engine] = failures.get(engine, 0) + 1

    out: dict = {"jobs": jobs, "results": results}
    for key in JUDGMENT_KEYS:
        c = counts[key]
        out[key] = {
            **c,
            "rule_rate": _rate(c["rule"], results),
            "jev_rate": _rate(c["jev"], results),
            "unclassified_rate": _rate(c["unclassified"], results),
        }
    out["jev_tokens"] = {
        "input": _sum_known(token_in),
        "output": _sum_known(token_out),
        "estimated_input": _sum_known(token_est),
    }
    out["cost_usd"] = sum(costs) if costs else None
    out["failed_engines"] = failures
    return out


# ひらがな・カタカナ・CJK 統合漢字（拡張 A を含む）。
_JAPANESE_RANGES: tuple[tuple[str, str], ...] = (
    ("぀", "ゟ"),
    ("゠", "ヿ"),
    ("㐀", "䶿"),
    ("一", "鿿"),
)


def is_japanese(text: str) -> bool:
    """ひらがな・カタカナ・CJK 統合漢字のどれかを含めば真。"""
    return any(lo <= ch <= hi for ch in text for lo, hi in _JAPANESE_RANGES)


def _lang_of(title: str) -> str:
    return Lang.JA.value if is_japanese(title) else Lang.EN.value


def rows_to_results(rows: Iterable[dict]) -> dict[tuple[str, str], list[SearchResult]]:
    """結果ログの行を (job_id, query) ごとの SearchResult の並びに戻す。行の順を保つ。"""
    groups: dict[tuple[str, str], list[SearchResult]] = {}
    for row in rows:
        result = SearchResult(
            url=row["url"],
            title=row["title"],
            snippet=row["snippet"],
            engine=row["engine"],
            published_date=row.get("published_date"),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            engines=tuple(row.get("engines") or (row["engine"],)),
            matched=tuple(Lang(m) for m in row.get("matched", ())),
        )
        groups.setdefault((row["job_id"], row["query"]), []).append(result)
    return groups


def batch_vs_single(
    rows: Iterable[dict],
    make_classifier: Callable[[int], Classifier],
    *,
    batch_size: int,
    rule_types_for: Callable[[Sequence[SearchResult]], Mapping[int, ResultType]],
) -> dict:
    """同じ結果を batch_size と 1 で分類し、判定ごとの一致率を日本語・英語に分けて返す。

    戻り値は {判定: {"ja"|"en": {"paired", "agreed", "agreement_rate", "only_one_unclassified"}}}。
    paired は両方とも分類済みの組の数、agreed はそのうち値が同じ数、
    only_one_unclassified は片方だけ未分類の数（両方未分類は数えない）。
    paired が 0 のとき一致率は 0.0。言語は結果の title の文字種で決める。
    type は、どちらかの判定が規則（JudgedBy.RULE）で決まった組は両方の実行で同じになるため
    paired に入れず、言語ごとに "rule_decided" として別に数える（type のみ）。
    2つの分類結果は URL で突き合わせる（URL でまとめた後の結果なので1グループ内で一意）。
    """
    stats = {
        key: {
            lang.value: {"paired": 0, "agreed": 0, "only_one_unclassified": 0} for lang in Lang
        }
        for key in JUDGMENT_KEYS
    }
    for s in stats["type"].values():
        s["rule_decided"] = 0
    batch_classifier = make_classifier(batch_size)
    single_classifier = make_classifier(1)

    for (_job_id, query), results in rows_to_results(rows).items():
        rule_types = rule_types_for(results)
        batch_items, _ = batch_classifier.classify(query, results, rule_types)
        single_items, _ = single_classifier.classify(query, results, rule_types)
        single_by_url = {item.result.url: item for item in single_items}
        for batch_item in batch_items:
            single_item = single_by_url.get(batch_item.result.url)
            if single_item is None:
                continue
            lang = _lang_of(batch_item.result.title)
            for key in JUDGMENT_KEYS:
                a = getattr(batch_item, key).value
                b = getattr(single_item, key).value
                s = stats[key][lang]
                if key == "type" and JudgedBy.RULE in (
                    batch_item.type.by,
                    single_item.type.by,
                ):
                    s["rule_decided"] += 1
                elif a is not None and b is not None:
                    s["paired"] += 1
                    if a == b:
                        s["agreed"] += 1
                elif a is not None or b is not None:
                    s["only_one_unclassified"] += 1

    for key in JUDGMENT_KEYS:
        for s in stats[key].values():
            s["agreement_rate"] = _rate(s["agreed"], s["paired"])
    return stats


def _cell(judgment: Mapping | None) -> str:
    """判定の値を CSV のセルの文字列にする。未分類は空、真偽は true / false。"""
    value = (judgment or {}).get("value")
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def sample_csv(rows: Iterable[dict], n: int, seed: int = 0) -> str:
    """人が正解を付けるための抜き出し（CSV）。label_* の列は空。同じ入力・seed なら同じ抜き出し。"""
    pool = list(rows)
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in picked:
        writer.writerow(
            [
                row["job_id"],
                row["id"],
                row["query"],
                row["url"],
                row["title"],
                row["snippet"],
                _lang_of(row["title"]),
                _cell(row.get("type")),
                _cell(row.get("genre")),
                _cell(row.get("relevant")),
                "",
                "",
                "",
            ]
        )
    return buf.getvalue()


_TRUE_WORDS = {"true", "1", "yes", "y"}
_FALSE_WORDS = {"false", "0", "no", "n"}


def _normalize_label(key: str, text: str) -> str:
    text = text.strip().lower()
    if key == "relevant":
        if text in _TRUE_WORDS:
            return "true"
        if text in _FALSE_WORDS:
            return "false"
    return text


def accuracy_from_labels(csv_text: str) -> dict:
    """label_* が空でない行について、判定ごとの正解率を ja / en に分けて返す。

    戻り値は {判定: {"ja"|"en": {"correct", "total", "accuracy", "unclassified"}}}。
    機械の判定が未分類（空）で正解が付いている行は、total に数えて不正解とする
    （accuracy では不正解扱い）。その行数は "unclassified" に別に数えるので、
    判定できた割合（coverage）と正解率を分けて見られる。
    total が 0 のとき accuracy は 0.0。
    """
    stats = {
        key: {lang.value: {"correct": 0, "total": 0, "unclassified": 0} for lang in Lang}
        for key in JUDGMENT_KEYS
    }
    for row in csv.DictReader(io.StringIO(csv_text)):
        lang = row.get("lang") or _lang_of(row.get("title", ""))
        if lang not in (Lang.JA.value, Lang.EN.value):
            lang = _lang_of(row.get("title", ""))
        for key in JUDGMENT_KEYS:
            label = (row.get(f"label_{key}") or "").strip()
            if not label:
                continue
            s = stats[key][lang]
            s["total"] += 1
            if not (row.get(key) or "").strip():
                s["unclassified"] += 1
            if _normalize_label(key, label) == _normalize_label(key, row.get(key) or ""):
                s["correct"] += 1
    for key in JUDGMENT_KEYS:
        for s in stats[key].values():
            s["accuracy"] = _rate(s["correct"], s["total"])
    return stats
