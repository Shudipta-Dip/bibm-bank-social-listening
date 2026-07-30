"""Write UTF-8 example snippets for the report."""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
fp = pd.read_csv(OUT / "false_positive_candidates.csv")
fn = pd.read_csv(OUT / "false_negative_candidates.csv")
star = pd.read_csv(OUT / "star_vs_text_conflicts.csv")
edge = pd.read_csv(OUT / "edge_cases_for_human.csv")

lines: list[str] = []
lines.append(f"edge_n={len(edge)} fp={len(fp)} fn={len(fn)} star={len(star)}")
for mode in [
    "substring_false_positive_in_query",
    "advice_marked_positive",
    "negation_or_sarcasm",
    "comparative_not_affective",
    "short_lexicon_positive",
]:
    lines.append(f"\n## {mode}")
    for _, r in fp[fp.failure_mode == mode].head(4).iterrows():
        lines.append(
            f"[{r.source}/{r.language}] prod={r.production_label} xlmr={r.xlmr_label} "
            f"bn={r.bn_label} vader={r.vader_label} hits={r.lex_pos_hits}"
        )
        lines.append(str(r.text)[:300].replace("\n", " | "))
        lines.append("")

lines.append("\n## FN sample")
for _, r in fn.head(5).iterrows():
    lines.append(
        f"[{r.source}/{r.language}] prod={r.production_label} xlmr={r.xlmr_label} vader={r.vader_label}"
    )
    lines.append(str(r.text)[:300].replace("\n", " | "))

lines.append("\n## star conflict: high star + xlmr negative")
hi = star[(star.star_rating >= 4) & (star.xlmr_label == "negative")].head(4)
for _, r in hi.iterrows():
    lines.append(
        f"[{r.source}] stars={r.star_rating} prod={r.production_label} xlmr={r.xlmr_label} lex={r.lexicon_label}"
    )
    lines.append(str(r.text)[:300].replace("\n", " | "))

(OUT / "example_snippets.txt").write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT / "example_snippets.txt")
