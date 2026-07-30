"""Reclassify clean corpus: XLM-R text + policy; stars stay parallel.

Uses audit cache when present; otherwise runs the HF model.

Usage (repo root):
  .\\.venv\\Scripts\\python.exe analysis/sentiment_audit/reclassify_corpus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from listening.nlp.policy import (  # noqa: E402
    apply_text_sentiment_policy,
    rating_sentiment_from_stars,
)

CLEAN = ROOT / "data" / "processed" / "unified_mentions_clean.csv"
CACHE = Path(__file__).resolve().parent / "cache" / "xlmr_full.json"
OUT_DIR = Path(__file__).resolve().parent / "out"
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_xlmr_labels(n: int) -> list[tuple[str, float]]:
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if len(data) == n:
            print(f"[info] using XLM-R cache ({n})")
            return [(d["label"], float(d["score"])) for d in data]
    print("[info] running XLM-R on full corpus…")
    from transformers import pipeline

    pipe = pipeline("sentiment-analysis", model=MODEL, truncation=True, max_length=512)
    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    df = pd.read_csv(CLEAN, usecols=["text"], low_memory=False)
    out = []
    for i, text in enumerate(df["text"].fillna("").tolist()):
        if not str(text).strip():
            out.append(("neutral", 0.0))
            continue
        raw = pipe(str(text)[:2000])[0]
        lab = mapping.get(str(raw["label"]).lower(), "neutral")
        out.append((lab, float(raw["score"])))
        if (i + 1) % 200 == 0:
            print(f"  … {i+1}/{n}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps([{"label": a, "score": b} for a, b in out]),
        encoding="utf-8",
    )
    return out


def main() -> None:
    df = pd.read_csv(CLEAN, encoding="utf-8", low_memory=False)
    n = len(df)
    raw = load_xlmr_labels(n)
    assert len(raw) == n

    old = df["sentiment_final"].astype(str).str.lower()
    texts = df["text"].fillna("").tolist()

    text_labels = []
    scores = []
    policies = []
    raw_labels = []
    for text, (lab, sc) in zip(texts, raw):
        new_lab, new_sc, tag = apply_text_sentiment_policy(text, lab, sc)
        text_labels.append(new_lab)
        scores.append(round(new_sc, 4))
        policies.append(tag)
        raw_labels.append(lab)

    df["sentiment_model_raw"] = raw_labels
    df["sentiment_policy"] = policies
    df["sentiment_text"] = text_labels
    df["sentiment_final"] = text_labels  # dashboard alias = text path
    df["sentiment_score"] = scores
    df["rating_sentiment"] = df["star_rating"].apply(
        lambda x: rating_sentiment_from_stars(x if pd.notna(x) else None)
    )

    # backup previous clean file once
    bak = CLEAN.with_suffix(".csv.bak_pre_reclassify")
    if not bak.exists():
        Path(CLEAN).replace(bak) if False else None
        # copy instead of replace
        import shutil

        shutil.copy2(CLEAN, bak)
        print(f"[info] backup -> {bak}")

    df.to_csv(CLEAN, index=False, encoding="utf-8", na_rep="")

    # also refresh pages data copy if present
    pages_csv = ROOT / "docs" / "data" / "unified_mentions_clean.csv"
    if pages_csv.parent.exists():
        df.to_csv(pages_csv, index=False, encoding="utf-8", na_rep="")

    new = df["sentiment_final"].astype(str).str.lower()
    changed = int((old != new).sum())
    stats = {
        "n": n,
        "changed_from_old_sentiment_final": changed,
        "old_dist": Counter(old).most_common(),
        "new_text_dist": Counter(new).most_common(),
        "rating_sentiment_dist": Counter(
            df["rating_sentiment"].dropna().astype(str)
        ).most_common(),
        "policy_tags": Counter(policies).most_common(),
        "agree_text_vs_rating_where_both": None,
    }
    both = df[df["rating_sentiment"].notna()]
    if len(both):
        stats["agree_text_vs_rating_where_both"] = round(
            100
            * (both["sentiment_text"] == both["rating_sentiment"]).mean(),
            2,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reclassify_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # agreement vs human labels if available
    human_path = OUT_DIR / "edge_cases_labeled_repaired.csv"
    if human_path.exists():
        hum = pd.read_csv(human_path)
        m = hum.merge(
            df[["record_id", "sentiment_text", "rating_sentiment", "sentiment_policy"]],
            on="record_id",
            how="inner",
        )
        if len(m):
            agr = (m["your_label"].str.lower() == m["sentiment_text"].str.lower()).mean()
            stats["human_agreement_after_reclassify"] = round(100 * agr, 2)
            (OUT_DIR / "reclassify_stats.json").write_text(
                json.dumps(stats, indent=2), encoding="utf-8"
            )
            print(f"[info] human agreement on edge pack: {100*agr:.1f}% ({len(m)} rows)")

    print(json.dumps(stats, indent=2))
    print(f"[done] wrote {CLEAN}")


if __name__ == "__main__":
    main()
