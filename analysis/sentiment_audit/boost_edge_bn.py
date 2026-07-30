"""Append Bangla disagreement rows to the edge pack."""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
edge = pd.read_csv(OUT / "edge_cases_for_human.csv")
scored = pd.read_csv(OUT / "scored_corpus.csv")
have = set(edge["record_id"].astype(str))
bn = scored[(scored["bn_label"] != "skip") & (scored["production"] != scored["bn_label"])].copy()
bn = bn[~bn["record_id"].astype(str).isin(have)]
a = bn[(bn["production"] == "positive") & (bn["bn_label"] != "positive")].head(4)
b = bn[(bn["production"] == "neutral") & (bn["bn_label"] == "negative")].head(4)
extra = pd.concat([a, b], ignore_index=True)
extra["production_label"] = extra["production"]
extra["vader_or_bn_label"] = extra["bn_label"]
extra["failure_mode"] = "bn_model_disagreement"
extra["why_suspect"] = [
    f"bn_disagree | prod={r.production} bn={r.bn_label} xlmr={r.xlmr_label}" for r in extra.itertuples()
]
extra["your_label"] = ""
for c in edge.columns:
    if c not in extra.columns:
        extra[c] = ""
extra = extra[edge.columns]
out = pd.concat([edge, extra], ignore_index=True).drop_duplicates("record_id").head(55)
out.to_csv(OUT / "edge_cases_for_human.csv", index=False, encoding="utf-8-sig")
print("final edge", len(out))
print(out["failure_mode"].value_counts().to_string())
