"""Analyze human verdicts vs judges; recover missing brands from clean CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "processed" / "unified_mentions_clean.csv"

df = pd.read_excel(OUT / "edge_cases_for_human.xlsx")
clean = pd.read_csv(CLEAN, usecols=["record_id", "brand", "source", "language", "star_rating", "text"], low_memory=False)

# recover brand / language from clean where missing
lookup = clean.set_index("record_id")
for i, row in df.iterrows():
    rid = row["record_id"]
    if rid not in lookup.index:
        continue
    if pd.isna(row.get("brand")) or str(row.get("brand")).strip() in ("", "nan"):
        df.at[i, "brand"] = lookup.at[rid, "brand"]
    if pd.isna(row.get("language")) or str(row.get("language")).strip() in ("", "nan"):
        df.at[i, "language"] = lookup.at[rid, "language"]

df["your_label"] = df["your_label"].astype(str).str.strip().str.lower()
df = df[df["your_label"].isin(["positive", "negative", "neutral"])].copy()

judges = {
    "production": "production_label",
    "lexicon": "lexicon_label",
    "xlmr": "xlmr_label",
    "vader_or_bn": "vader_or_bn_label",
}

lines = []
lines.append(f"n_labeled={len(df)}")
lines.append(f"brand_after_repair:\n{df['brand'].value_counts(dropna=False).to_string()}")
lines.append(f"\nyour_label:\n{df['your_label'].value_counts().to_string()}")

# agreement / confusion
for name, col in judges.items():
    y = df["your_label"].tolist()
    p = df[col].astype(str).str.lower().tolist()
    agree = sum(a == b for a, b in zip(y, p)) / len(y)
    lines.append(f"\n## vs {name} agreement={agree:.1%}")
    ct = pd.crosstab(df["your_label"], df[col].astype(str).str.lower(), margins=True)
    lines.append(ct.to_string())

# where human != production
diff = df[df["your_label"] != df["production_label"].astype(str).str.lower()]
lines.append(f"\n## human_disagrees_production n={len(diff)}")
lines.append(diff.groupby(["production_label", "your_label"]).size().to_string())

# majority of external judges (xlmr + vader_or_bn + lexicon) vs human
def maj(row):
    votes = [str(row["lexicon_label"]).lower(), str(row["xlmr_label"]).lower(), str(row["vader_or_bn_label"]).lower()]
    # skip empty
    votes = [v for v in votes if v in ("positive", "negative", "neutral")]
    if not votes:
        return "none"
    return max(set(votes), key=votes.count)

df["judge_majority"] = df.apply(maj, axis=1)
lines.append(
    f"\n## vs judge_majority agreement="
    f"{(df['your_label']==df['judge_majority']).mean():.1%}"
)
lines.append(pd.crosstab(df["your_label"], df["judge_majority"], margins=True).to_string())

# pattern slices
lines.append("\n## patterns: human neutral but production positive")
sub = df[(df["your_label"] == "neutral") & (df["production_label"].astype(str).str.lower() == "positive")]
lines.append(f"n={len(sub)}")
lines.append(sub[["failure_mode", "xlmr_label", "vader_or_bn_label", "lexicon_label"]].value_counts().head(15).to_string())

lines.append("\n## patterns: human negative but production positive")
sub2 = df[(df["your_label"] == "negative") & (df["production_label"].astype(str).str.lower() == "positive")]
lines.append(f"n={len(sub2)}")
for _, r in sub2.iterrows():
    lines.append(
        f"- [{r.source}/{r.brand}] xlmr={r.xlmr_label} ext={r.vader_or_bn_label} mode={r.failure_mode} "
        f"stars={r.star_rating} | {str(r.text)[:160].replace(chr(10), ' | ')}"
    )

lines.append("\n## patterns: human positive but production not")
sub3 = df[(df["your_label"] == "positive") & (df["production_label"].astype(str).str.lower() != "positive")]
lines.append(f"n={len(sub3)}")
for _, r in sub3.iterrows():
    lines.append(
        f"- [{r.source}/{r.brand}] prod={r.production_label} xlmr={r.xlmr_label} ext={r.vader_or_bn_label} "
        f"stars={r.star_rating} | {str(r.text)[:160].replace(chr(10), ' | ')}"
    )

# When XLM-R agrees with human
lines.append(
    f"\n## XLM-R==human rate={(df['your_label']==df['xlmr_label'].astype(str).str.lower()).mean():.1%}"
)
lines.append(
    f"## lexicon==human rate={(df['your_label']==df['lexicon_label'].astype(str).str.lower()).mean():.1%}"
)
lines.append(
    f"## production==human rate={(df['your_label']==df['production_label'].astype(str).str.lower()).mean():.1%}"
)

# Query/advice: human labels
qa = df[df["failure_mode"].astype(str).str.contains("query|advice|substring_false", case=False, na=False)]
lines.append(f"\n## query/advice failure modes human labels n={len(qa)}")
lines.append(qa["your_label"].value_counts().to_string())

# star rows
stars = df[df["star_rating"].notna()]
lines.append(f"\n## starred rows n={len(stars)}")
lines.append(
    stars.assign(
        star_bin=stars["star_rating"].apply(lambda x: "high4-5" if x >= 4 else ("low1-2" if x <= 2 else "mid3"))
    )
    .groupby(["star_bin", "your_label"])
    .size()
    .to_string()
)

out = "\n".join(lines)
(OUT / "human_vs_judges_analysis.txt").write_text(out, encoding="utf-8")
df.to_csv(OUT / "edge_cases_labeled_repaired.csv", index=False, encoding="utf-8-sig")
print(out[:4000])
print("\n... wrote", OUT / "human_vs_judges_analysis.txt")
