"""Curate ~50 human-review edge cases from audit outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    scored = pd.read_csv(OUT / "scored_corpus.csv", low_memory=False)
    fp = pd.read_csv(OUT / "false_positive_candidates.csv", low_memory=False)
    fn = pd.read_csv(OUT / "false_negative_candidates.csv", low_memory=False)
    star = pd.read_csv(OUT / "star_vs_text_conflicts.csv", low_memory=False)

    picks: list[pd.DataFrame] = []

    def take(df: pd.DataFrame, n: int, mode_col: str | None = None) -> pd.DataFrame:
        if df.empty:
            return df
        if mode_col and mode_col in df.columns:
            # stratified by failure_mode
            parts = []
            modes = df[mode_col].value_counts().index.tolist()
            per = max(1, n // max(len(modes), 1))
            for m in modes:
                parts.append(df[df[mode_col] == m].head(per))
            out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["record_id"])
            return out.head(n)
        return df.head(n)

    # Priority FP modes for human verification
    priority_modes = [
        "substring_false_positive_in_query",
        "advice_marked_positive",
        "query_marked_positive",
        "negation_or_sarcasm",
        "comparative_not_affective",
        "judge_disagreement_fp",
        "short_lexicon_positive",
    ]
    for mode in priority_modes:
        sub = fp[fp["failure_mode"] == mode]
        # Prefer Facebook / non-star for query FP; keep some store reviews
        picks.append(take(sub, 6 if mode.startswith("substring") or mode.startswith("advice") else 4, None))

    picks.append(take(fn, 8))

    # Star conflicts: high star + negative text model, or low star + positive text
    if not star.empty:
        hi = star[(star["star_rating"] >= 4) & (star["xlmr_label"] == "negative")].head(6)
        lo = star[(star["star_rating"] <= 2) & (star["xlmr_label"] == "positive")].head(4)
        picks.append(hi)
        picks.append(lo)

    # BN disagreements where production positive / bn not
    bn = scored[(scored["bn_label"] != "skip") & (scored["production"] != scored["bn_label"])]
    bn_fp = bn[(bn["production"] == "positive") & (bn["bn_label"] != "positive")].head(6)
    bn_fn = bn[(bn["production"] != "positive") & (bn["bn_label"] == "positive")].head(4)
    picks.append(bn_fp)
    picks.append(bn_fn)

    edge = pd.concat(picks, ignore_index=True)
    # unify columns
    for c in [
        "record_id",
        "source",
        "language",
        "brand",
        "star_rating",
        "text",
        "production",
        "lexicon_label",
        "xlmr_label",
        "vader_label",
        "bn_label",
        "intent_flags",
        "lex_pos_hits",
        "failure_mode",
    ]:
        if c not in edge.columns:
            edge[c] = ""

    # map production column name
    if "production_label" in edge.columns:
        edge["production"] = edge["production"].fillna(edge["production_label"])

    edge = edge.drop_duplicates(subset=["record_id"]).head(55).copy()
    edge["production_label"] = edge["production"].fillna(edge.get("production_label", ""))
    edge["vader_or_bn_label"] = edge.apply(
        lambda r: r["bn_label"] if str(r.get("bn_label", "skip")) != "skip" else r.get("vader_label", ""),
        axis=1,
    )
    if "failure_mode" not in edge.columns or edge["failure_mode"].isna().all():
        edge["failure_mode"] = edge.get("failure_mode", "review_needed")

    def why(r) -> str:
        flags = str(r.get("intent_flags") or "")
        hits = str(r.get("lex_pos_hits") or "")
        mode = str(r.get("failure_mode") or "")
        parts = []
        if mode and mode != "nan":
            parts.append(mode)
        if hits:
            parts.append(f"lex_hits=[{hits}]")
        if flags:
            parts.append(f"flags=[{flags}]")
        parts.append(
            f"prod={r.get('production_label')} xlmr={r.get('xlmr_label')} "
            f"ext={r.get('vader_or_bn_label')}"
        )
        return " | ".join(parts)

    edge["why_suspect"] = edge.apply(why, axis=1)
    edge["your_label"] = ""  # human fills: positive|neutral|negative

    cols = [
        "record_id",
        "source",
        "language",
        "brand",
        "star_rating",
        "text",
        "production_label",
        "lexicon_label",
        "xlmr_label",
        "vader_or_bn_label",
        "failure_mode",
        "why_suspect",
        "your_label",
    ]
    out = edge[cols]
    out.to_csv(OUT / "edge_cases_for_human.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} edge cases -> {OUT / 'edge_cases_for_human.csv'}")
    print(out["failure_mode"].value_counts().to_string())


if __name__ == "__main__":
    main()
