"""Metric computations for the brand listening dashboard."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "unified_mentions_clean.csv"

THEMES = [
    "theme_app_ux",
    "theme_cards",
    "theme_fees",
    "theme_service",
    "theme_transfers",
    "theme_security",
]
THEME_LABELS = {
    "theme_app_ux": "App / UX",
    "theme_cards": "Cards",
    "theme_fees": "Fees",
    "theme_service": "Service",
    "theme_transfers": "Transfers",
    "theme_security": "Security",
}
BRAND_LABELS = {"brac_bank": "BRAC Bank", "scb_bangladesh": "SCB Bangladesh"}
# Comparative pair: BRAC cerulean vs SCB green (max contrast in light & dark)
BRAND_COLORS = {"BRAC Bank": "#006CB5", "SCB Bangladesh": "#38D200"}
BRAC_SILVER = "#C2C2C2"
SCB_BLUE = "#0473EA"

COMPETITORS = {
    "BRAC Bank": [r"(?<!\w)brac(?!\w)"],
    "Standard Chartered": [r"(?<!\w)scb(?!\w)", r"standard\s+chartered"],
    "EBL": [r"(?<!\w)ebl(?!\w)", r"eastern\s+bank"],
    "MTB": [r"(?<!\w)mtb(?!\w)", r"mutual\s+trust"],
    "Dutch-Bangla (DBBL)": [r"(?<!\w)dbbl(?!\w)", r"dutch[\s\-]?bangla"],
    "City Bank": [r"city\s+bank"],
    "UCB": [r"(?<!\w)ucb(?!\w)", r"united\s+commercial"],
    "Prime Bank": [r"prime\s+bank"],
    "Pubali Bank": [r"pubali"],
    "Islami Bank": [r"islami\s+bank", r"(?<!\w)ibbl(?!\w)"],
    "Bank Asia": [r"bank\s+asia"],
    "HSBC": [r"(?<!\w)hsbc(?!\w)"],
    "bKash": [r"bkash"],
}

DEMO_PATTERNS = {
    "students": [
        r"student(s)?",
        r"university",
        r"campus",
        r"tuition",
        r"study\s+abroad",
        r"graduate\s+account",
        r"student\s+account",
        r"student\s+file",
    ],
    "freelancers": [
        r"freelancer(s)?",
        r"freelancing",
        r"upwork",
        r"fiverr",
        r"(?<!\w)erq(?!\w)",
        r"service\s+export",
    ],
    "sme": [
        r"(?<!\w)sme(?!\w)",
        r"small\s+business",
        r"entrepreneur(s)?",
        r"business\s+loan",
        r"trade\s+license",
        r"working\s+capital",
    ],
    "women": [
        r"women\s+banking",
        r"women\s+entrepreneur",
        r"female\s+student",
        r"for\s+women",
        r"women[\s\-]?owned",
    ],
}
DEMO_BN = {
    "students": ["ছাত্র", "ছাত্রী", "স্টুডেন্ট"],
    "freelancers": ["ফ্রিল্যান্সার", "ফ্রিল্যান্সিং"],
    "sme": ["ক্ষুদ্র ব্যবসা"],
    "women": ["নারী উদ্যোক্তা", "মহিলা উদ্যোক্তা"],
}

SCHEME_PATTERNS = {
    "BRAC Agami (student)": [
        r"agami\s+savers",
        r"agami\s+account",
        r"agami\s+card",
        r"brac\s+agami",
        r"tara\s+agami",
        r"(?<!\w)agami(?!\w)",
    ],
    "BRAC TARA (women)": [
        r"tara\s+agami",
        r"brac\s+tara",
        r"tara\s+account",
        r"tara\s+uddokta",
        r"tara\s+banking",
        r"tara\s+card",
    ],
    "BRAC Matrix (freelancer)": [
        r"matrix\s+account",
        r"freelancer\s+matrix",
        r"brac\s+matrix",
    ],
    "SCB Freelancer Account": [r"freelancer\s+account"],
    "SCB Student File": [r"student\s+file", r"right\s+start"],
    "SCB Saadiq Graduate": [r"saadiq\s+graduate", r"saadiq", r"graduate\s+account"],
    "SCB Orjon (women SME)": [r"(?<!\w)orjon(?!\w)", r"orjon[\s\-]?bil"],
}

SCHEME_TO_DEMO = {
    "BRAC Agami (student)": "students",
    "BRAC TARA (women)": "women",
    "BRAC Matrix (freelancer)": "freelancers",
    "SCB Freelancer Account": "freelancers",
    "SCB Student File": "students",
    "SCB Saadiq Graduate": "students",
    "SCB Orjon (women SME)": "women",
}


@lru_cache(maxsize=1)
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8", low_memory=False)
    for col in THEMES + ["low_quality", "is_review", "author_known"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["text"] = df["text"].fillna("")
    df["text_l"] = df["text"].str.lower()
    df["brand_label"] = df["brand"].map(BRAND_LABELS).fillna(df["brand"])
    return df


def apply_filters(
    df: pd.DataFrame,
    *,
    exclude_low_quality: bool = True,
    brands: list[str] | None = None,
    sources: list[str] | None = None,
    languages: list[str] | None = None,
    platform_types: list[str] | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if exclude_low_quality:
        out = out[~out["low_quality"]]
    if brands:
        out = out[out["brand"].isin(brands)]
    if sources:
        out = out[out["source"].isin(sources)]
    if languages:
        out = out[out["language"].isin(languages)]
    if platform_types:
        out = out[out["platform_type"].isin(platform_types)]
    if month_from and "month_year" in out.columns:
        out = out[out["month_year"].fillna("") >= month_from]
    if month_to and "month_year" in out.columns:
        out = out[out["month_year"].fillna("") <= month_to]
    return out


def _match_any(text: str, pats: list[str]) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


def find_banks(text: str, exclude: set[str] | None = None) -> set[str]:
    found: set[str] = set()
    for label, pats in COMPETITORS.items():
        if exclude and label in exclude:
            continue
        if _match_any(text, pats):
            found.add(label)
    if (not exclude or "BRAC Bank" not in exclude) and "ব্র্যাক" in text:
        found.add("BRAC Bank")
    return found


def match_demo(text: str, demo: str) -> bool:
    if _match_any(text, DEMO_PATTERNS[demo]):
        return True
    return any(bn in text for bn in DEMO_BN.get(demo, []))


def match_scheme(text: str, scheme: str) -> bool:
    return _match_any(text, SCHEME_PATTERNS[scheme])


def kpi_cards(df: pd.DataFrame) -> dict:
    n = len(df)
    by_brand = df["brand_label"].value_counts().to_dict()
    reviews = df[df["is_review"]]
    star = reviews.groupby("brand_label")["star_rating"].mean().to_dict()
    sent = (
        df.groupby("brand_label")["sentiment_final"]
        .value_counts(normalize=True)
        .mul(100)
        .unstack(fill_value=0)
    )
    return {
        "total": n,
        "by_brand": by_brand,
        "mean_star": star,
        "pos_pct": {b: float(sent.loc[b, "positive"]) if b in sent.index and "positive" in sent.columns else 0 for b in by_brand},
        "neg_pct": {b: float(sent.loc[b, "negative"]) if b in sent.index and "negative" in sent.columns else 0 for b in by_brand},
        "review_n": reviews.groupby("brand_label").size().to_dict(),
    }


def theme_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, g in df.groupby("brand_label"):
        n = len(g)
        if n == 0:
            continue
        for theme in THEMES:
            count = int(g[theme].sum())
            rows.append(
                {
                    "brand": brand,
                    "theme": THEME_LABELS[theme],
                    "theme_key": theme,
                    "count": count,
                    "pct": 100 * count / n,
                    "n_base": n,
                }
            )
    return pd.DataFrame(rows)


# schema bump: meaningfulness uses pct_positive (within-theme); see positive_differentiation_index
def meaningfulness(df: pd.DataFrame) -> pd.DataFrame:
    """% of theme-tagged mentions that are positive (within-theme praise rate)."""
    rows = []
    for brand, g in df.groupby("brand_label"):
        for theme in THEMES:
            themed = g[g[theme]]
            nt = len(themed)
            pos_n = int((themed["sentiment_final"] == "positive").sum()) if nt else 0
            rows.append(
                {
                    "brand": brand,
                    "theme": THEME_LABELS[theme],
                    "pos_count": pos_n,
                    "theme_count": nt,
                    "pct_positive": 100 * pos_n / nt if nt else 0.0,
                    "n_base": len(g),
                }
            )
    return pd.DataFrame(rows)


def differentiation_index(df: pd.DataFrame) -> pd.DataFrame:
    prev = theme_prevalence(df)
    if prev.empty:
        return prev
    pivot = prev.pivot(index="theme", columns="brand", values="pct").fillna(0)
    count_pivot = prev.pivot(index="theme", columns="brand", values="count").fillna(0)
    brands = list(pivot.columns)
    if len(brands) < 2:
        return pd.DataFrame()
    # Prefer BRAC - SCB ordering if both present
    if "BRAC Bank" in brands and "SCB Bangladesh" in brands:
        pass  # explicit columns below
    out = pd.DataFrame(
        {
            "theme": pivot.index,
            "brac_pct": pivot.get("BRAC Bank", pd.Series(0, index=pivot.index)),
            "scb_pct": pivot.get("SCB Bangladesh", pd.Series(0, index=pivot.index)),
            "brac_count": count_pivot.get("BRAC Bank", pd.Series(0, index=pivot.index)),
            "scb_count": count_pivot.get("SCB Bangladesh", pd.Series(0, index=pivot.index)),
        }
    )
    out["gap"] = out["brac_pct"] - out["scb_pct"]
    out["leader"] = out["gap"].apply(lambda x: "BRAC Bank" if x > 0 else ("SCB Bangladesh" if x < 0 else "Tie"))
    return out.sort_values("gap")


def positive_differentiation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Gap in share of brand rows that are positive ∩ theme (BRAC − SCB).

    Parallel to prevalence differentiation, but only praise-tagged theme talk —
    so a bank can lead volume yet trail on positive differentiation.
    """
    rows = []
    for brand, g in df.groupby("brand_label"):
        n = len(g)
        if n == 0:
            continue
        for theme in THEMES:
            themed = g[g[theme]]
            pos_n = int((themed["sentiment_final"] == "positive").sum()) if len(themed) else 0
            rows.append(
                {
                    "brand": brand,
                    "theme": THEME_LABELS[theme],
                    "pos_count": pos_n,
                    "theme_count": len(themed),
                    "pct": 100 * pos_n / n,
                    "n_base": n,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    pivot = long.pivot(index="theme", columns="brand", values="pct").fillna(0)
    count_pivot = long.pivot(index="theme", columns="brand", values="pos_count").fillna(0)
    out = pd.DataFrame(
        {
            "theme": pivot.index,
            "brac_pct": pivot.get("BRAC Bank", pd.Series(0, index=pivot.index)),
            "scb_pct": pivot.get("SCB Bangladesh", pd.Series(0, index=pivot.index)),
            "brac_count": count_pivot.get("BRAC Bank", pd.Series(0, index=pivot.index)),
            "scb_count": count_pivot.get("SCB Bangladesh", pd.Series(0, index=pivot.index)),
        }
    )
    out["gap"] = out["brac_pct"] - out["scb_pct"]
    out["leader"] = out["gap"].apply(lambda x: "BRAC Bank" if x > 0 else ("SCB Bangladesh" if x < 0 else "Tie"))
    return out.sort_values("gap")


def pop_pod_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, g in df.groupby("brand_label"):
        n = len(g)
        for theme in THEMES:
            themed = g[g[theme]]
            nt = len(themed)
            if nt == 0:
                pos_pct = neg_pct = net = 0.0
            else:
                pos_pct = 100 * (themed["sentiment_final"] == "positive").sum() / nt
                neg_pct = 100 * (themed["sentiment_final"] == "negative").sum() / nt
                net = pos_pct - neg_pct
            rows.append(
                {
                    "brand": brand,
                    "theme": THEME_LABELS[theme],
                    "theme_key": theme,
                    "prevalence": 100 * nt / n if n else 0,
                    "count": nt,
                    "pos_pct": pos_pct,
                    "neg_pct": neg_pct,
                    "net": net,
                    "n_base": n,
                }
            )
    return pd.DataFrame(rows)


def identify_pops_pods(pop_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """POPs = similar prevalence; PODs = large prevalence gap + sentiment edge."""
    if pop_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    themes = pop_df["theme"].unique()
    pop_rows, pod_rows = [], []
    for theme in themes:
        t = pop_df[pop_df["theme"] == theme]
        if set(t["brand"]) < {"BRAC Bank", "SCB Bangladesh"}:
            continue
        brac = t[t["brand"] == "BRAC Bank"].iloc[0]
        scb = t[t["brand"] == "SCB Bangladesh"].iloc[0]
        prev_gap = abs(brac["prevalence"] - scb["prevalence"])
        row = {
            "theme": theme,
            "brac_prevalence": brac["prevalence"],
            "scb_prevalence": scb["prevalence"],
            "brac_net": brac["net"],
            "scb_net": scb["net"],
            "brac_count": brac["count"],
            "scb_count": scb["count"],
            "prev_gap": prev_gap,
        }
        if prev_gap <= 2.0:  # similar volume = POP candidate
            row["quality_edge"] = (
                "BRAC Bank"
                if brac["net"] > scb["net"] + 5
                else ("SCB Bangladesh" if scb["net"] > brac["net"] + 5 else "Parity")
            )
            pop_rows.append(row)
        if prev_gap >= 3.0:
            leader = "BRAC Bank" if brac["prevalence"] > scb["prevalence"] else "SCB Bangladesh"
            leader_net = brac["net"] if leader == "BRAC Bank" else scb["net"]
            row["volume_leader"] = leader
            row["leader_net"] = leader_net
            row["pod_quality"] = "Positive POD" if leader_net > 5 else ("Complaint-driven" if leader_net < -5 else "Volume-only")
            pod_rows.append(row)
    return pd.DataFrame(pop_rows), pd.DataFrame(pod_rows)


def phygital_net(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, g in df.groupby("brand_label"):
        for label, theme in [("Digital (App/UX)", "theme_app_ux"), ("Physical (Service)", "theme_service")]:
            themed = g[g[theme]]
            n = len(themed)
            if n == 0:
                pos = neg = net = 0.0
            else:
                pos = 100 * (themed["sentiment_final"] == "positive").sum() / n
                neg = 100 * (themed["sentiment_final"] == "negative").sum() / n
                net = pos - neg
            rows.append(
                {
                    "brand": brand,
                    "touchpoint": label,
                    "pos_pct": pos,
                    "neg_pct": neg,
                    "net": net,
                    "count": n,
                }
            )
    return pd.DataFrame(rows)


def monthly_app_ux_positivity(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly App/UX positive and negative shares (of App/UX-tagged rows)."""
    themed = df[df["theme_app_ux"] & df["month_year"].notna()].copy()
    if themed.empty:
        return pd.DataFrame()
    rows = []
    for (brand, month), g in themed.groupby(["brand_label", "month_year"]):
        n = len(g)
        pos = 100 * (g["sentiment_final"] == "positive").sum() / n
        neg = 100 * (g["sentiment_final"] == "negative").sum() / n
        rows.append(
            {
                "brand": brand,
                "month": month,
                "pos_pct": pos,
                "neg_pct": neg,
                "count": n,
            }
        )
    return pd.DataFrame(rows).sort_values("month")


def salience_reference(df: pd.DataFrame) -> pd.DataFrame:
    """% of each brand's conversations that also mention each competitor."""
    rows = []
    mapping = [
        ("brac_bank", "BRAC Bank", {"BRAC Bank"}),
        ("scb_bangladesh", "SCB Bangladesh", {"Standard Chartered"}),
    ]
    for brand_key, brand_label, exclude in mapping:
        subset = df[df["brand"] == brand_key]
        n = len(subset)
        if n == 0:
            continue
        counts: dict[str, int] = {}
        any_other = 0
        for text in subset["text_l"]:
            found = find_banks(text, exclude=exclude)
            if found:
                any_other += 1
                for b in found:
                    counts[b] = counts.get(b, 0) + 1
        for bank, c in counts.items():
            rows.append(
                {
                    "focal_brand": brand_label,
                    "co_mentioned": bank,
                    "count": c,
                    "pct": 100 * c / n,
                    "n_base": n,
                }
            )
        rows.append(
            {
                "focal_brand": brand_label,
                "co_mentioned": "ANY other bank",
                "count": any_other,
                "pct": 100 * any_other / n,
                "n_base": n,
            }
        )
    return pd.DataFrame(rows)


def asymmetry(df: pd.DataFrame) -> dict:
    brac = df[df["brand"] == "brac_bank"]
    scb = df[df["brand"] == "scb_bangladesh"]
    brac_n, scb_n = len(brac), len(scb)
    brac_scb = sum(1 for t in brac["text_l"] if "Standard Chartered" in find_banks(t))
    scb_brac = sum(1 for t in scb["text_l"] if "BRAC Bank" in find_banks(t))
    return {
        "brac_mentions_scb_pct": 100 * brac_scb / brac_n if brac_n else 0,
        "brac_mentions_scb_n": brac_scb,
        "brac_n": brac_n,
        "scb_mentions_brac_pct": 100 * scb_brac / scb_n if scb_n else 0,
        "scb_mentions_brac_n": scb_brac,
        "scb_n": scb_n,
    }


def scheme_awareness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, g in df.groupby("brand_label"):
        n = len(g)
        for scheme in SCHEME_PATTERNS:
            count = int(g["text_l"].apply(lambda t: match_scheme(t, scheme)).sum())
            rows.append(
                {
                    "brand": brand,
                    "scheme": scheme,
                    "segment": SCHEME_TO_DEMO.get(scheme, ""),
                    "count": count,
                    "pct": 100 * count / n if n else 0,
                    "n_base": n,
                }
            )
    return pd.DataFrame(rows)


def demographic_gap(df: pd.DataFrame) -> pd.DataFrame:
    scheme_map = {
        "students": ["BRAC Agami (student)", "SCB Student File", "SCB Saadiq Graduate"],
        "freelancers": ["BRAC Matrix (freelancer)", "SCB Freelancer Account"],
        "women": ["BRAC TARA (women)", "SCB Orjon (women SME)"],
        "sme": ["BRAC TARA (women)", "SCB Orjon (women SME)"],
    }
    rows = []
    for brand, g in df.groupby("brand_label"):
        n = len(g)
        for demo, schemes in scheme_map.items():
            demo_mask = g["text_l"].apply(lambda t: match_demo(t, demo))
            demo_n = int(demo_mask.sum())
            if demo_n == 0:
                named = 0
                named_pct = 0.0
                demo_pct = 0.0
            else:
                demo_rows = g[demo_mask]
                named = int(
                    demo_rows["text_l"].apply(
                        lambda t: any(match_scheme(t, s) for s in schemes)
                    ).sum()
                )
                named_pct = 100 * named / demo_n
                demo_pct = 100 * demo_n / n
            rows.append(
                {
                    "brand": brand,
                    "segment": demo.title(),
                    "demo_count": demo_n,
                    "demo_pct": demo_pct,
                    "named_count": named,
                    "named_of_demo_pct": named_pct,
                    "unnamed_count": demo_n - named,
                    "n_base": n,
                }
            )
    return pd.DataFrame(rows)
