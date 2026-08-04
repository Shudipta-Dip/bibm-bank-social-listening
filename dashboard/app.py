"""BIBM Brand Listening Dashboard — BRAC vs SCB.

Run:  .\\.venv\\Scripts\\streamlit.exe run dashboard/app.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import analytics as A

# Streamlit can keep a stale analytics module in memory after partial reruns.
A = importlib.reload(A)

st.set_page_config(
    page_title="BIBM Brand Listening — BRAC vs SCB",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BRAC = "#006CB5"
SCB = "#38D200"
BRAC_SILVER = "#C2C2C2"
SCB_BLUE = "#0473EA"

OUTLINE_ICONS = {
    "mentions": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>""",
    "brac": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v10M8 10.5h8"/></svg>""",
    "scb": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg>""",
    "link": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>""",
    "star": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>""",
    "sentiment": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>""",
}

KPI_HELP = {
    "Mentions": "How many cleaned conversations are in the current filtered view. This is the evidence base behind every chart below.",
    "BRAC share": "How much of the filtered conversation volume is about BRAC. Higher share means BRAC dominates talk volume here — not necessarily that BRAC is liked more.",
    "SCB share": "How much of the filtered conversation volume is about SCB. Compare with BRAC share to see who owns more of the discussion under these filters.",
    "SCB → BRAC ref.": "Share of SCB conversations that also name BRAC. A high number means BRAC is the bank SCB customers compare against — usually when they are dissatisfied with SCB.",
    "BRAC → SCB ref.": "Share of BRAC conversations that also name SCB. Typically much lower than SCB → BRAC, which means BRAC customers do not need SCB as a reference point.",
    "BRAC mean star": "Average app-store star rating for BRAC. Higher than SCB here means BRAC's app experience is winning the rating battle.",
    "SCB mean star": "Average app-store star rating for SCB. Lower than BRAC is a digital-experience weakness relative to the local competitor.",
    "BRAC positive": "% of BRAC mentions with positive *text* sentiment (model + policy). App-store stars are a separate metric (mean star / rating_sentiment) — they are not merged into this KPI.",
    "SCB positive": "% of SCB mentions with positive *text* sentiment (model + policy). App-store stars are a separate metric (mean star / rating_sentiment) — they are not merged into this KPI.",
}


def theme_base() -> str:
    try:
        t = st.context.theme.type  # "light" | "dark"
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


def inject_css(mode: str) -> None:
    st.markdown(
        f"""
<style>
  :root {{
    --brac: {BRAC};
    --scb: {SCB};
    --brac-silver: {BRAC_SILVER};
    --scb-blue: {SCB_BLUE};
    --radius: 14px;
    --gap: 0.75rem;
  }}

  /* Quiet, neutral surfaces: brand colors are reserved for comparison marks. */
  :root, [data-theme="light"], .stApp {{
    --surface: #ffffff;
    --surface-2: #f7f9fb;
    --surface-3: #eef2f6;
    --ink: #17212e;
    --ink-muted: #64748b;
    --border: #e1e7ee;
    --shadow: 0 1px 2px rgba(26,35,50,0.035);
  }}

  .block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1440px;
  }}

  /* Hide default streamlit metric styling conflicts */
  div[data-testid="stMetric"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
  }}

  h1, h2, h3, h4 {{
    letter-spacing: -0.02em;
    color: var(--ink) !important;
  }}

  .dash-title {{
    font-size: clamp(1.35rem, 2.5vw, 1.85rem);
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 0.25rem 0;
  }}
  .dash-sub {{
    color: var(--ink-muted);
    font-size: 0.92rem;
    margin: 0 0 0.75rem 0;
  }}

  /* Bento grid */
  .bento {{
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: var(--gap);
    margin-bottom: var(--gap);
  }}
  .tile {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    padding: 0.9rem 1rem;
    min-width: 0;
  }}
  .tile-span-2 {{ grid-column: span 2; }}
  .tile-span-3 {{ grid-column: span 3; }}
  .tile-span-4 {{ grid-column: span 4; }}
  .tile-span-6 {{ grid-column: span 6; }}
  .tile-span-8 {{ grid-column: span 8; }}
  .tile-span-12 {{ grid-column: span 12; }}
  .bento > [class*="tile-span"] {{
    min-width: 0;
    display: flex;
  }}
  .bento > [class*="tile-span"] > .tile {{
    flex: 1;
    width: 100%;
  }}

  .kpi {{
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    height: 100%;
  }}
  .kpi-top {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--ink-muted);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .kpi-top svg {{ flex-shrink: 0; opacity: 0.9; }}
  .kpi-value {{
    font-size: clamp(1.35rem, 2.2vw, 1.75rem);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
  }}
  .kpi-hint {{
    font-size: 0.78rem;
    color: var(--ink-muted);
  }}
  .kpi.brac {{ border-top: 3px solid {BRAC}; }}
  .kpi.brac .kpi-top {{ color: {BRAC}; }}
  .kpi.scb {{ border-top: 3px solid {SCB}; }}
  .kpi.scb .kpi-top {{ color: {SCB_BLUE}; }}
  .kpi-inner {{
    border: 1px solid var(--border);
    border-radius: calc(var(--radius) - 4px);
    padding: 0.75rem 0.85rem;
  }}
  .kpi.scb .kpi-value, .kpi.brac .kpi-value, .kpi .kpi-value {{
    color: var(--ink) !important;
  }}
  .kpi .kpi-hint {{ color: var(--ink-muted) !important; }}

  .section-head {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0 0 0.35rem 0;
    color: var(--ink);
    font-weight: 650;
    font-size: 1rem;
  }}
  .section-head svg {{ color: var(--ink-muted); }}
  .help-tip {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border: 1px solid var(--ink-muted);
    border-radius: 50%;
    color: var(--ink-muted);
    cursor: help;
    font-size: 11px;
    font-weight: 700;
    line-height: 1;
    position: relative;
  }}
  .help-tip:hover::after {{
    content: attr(data-tip);
    position: absolute;
    z-index: 100;
    top: 22px;
    left: -8px;
    width: min(360px, 78vw);
    white-space: normal;
    padding: 0.65rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    color: var(--ink);
    font-size: 0.78rem;
    font-weight: 400;
    line-height: 1.4;
    letter-spacing: normal;
    text-transform: none;
  }}
  .section-note {{
    color: var(--ink-muted);
    font-size: 0.84rem;
    margin: 0 0 0.55rem 0;
    line-height: 1.35;
  }}
  .legend-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    color: var(--ink-muted);
    margin-right: 0.85rem;
  }}
  .swatch {{
    width: 12px; height: 12px; border-radius: 3px;
    border: 1px solid var(--border);
  }}
  .shared-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem 1rem;
    margin: 0.15rem 0 0.65rem 0;
    padding: 0.55rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-2);
  }}
  .shared-legend .item {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--ink-muted);
    white-space: nowrap;
  }}
  .shared-legend .dot {{
    width: 11px;
    height: 11px;
    border-radius: 3px;
    border: 1px solid rgba(23, 33, 46, 0.12);
    flex-shrink: 0;
  }}

  /* Tighten streamlit vertical rhythm inside tabs */
  div[data-testid="stVerticalBlock"] > div {{ gap: 0.55rem; }}
  div[data-testid="stTabs"] button {{ font-weight: 600; }}
  /* Hide only Streamlit cloud/deploy chrome; keep header + sidebar toggle. */
  [data-testid="stAppDeployButton"],
  footer {{
    display: none !important;
  }}

  @media (max-width: 900px) {{
    .tile-span-2, .tile-span-3, .tile-span-4, .tile-span-6, .tile-span-8 {{
      grid-column: span 12;
    }}
    .block-container {{
      padding-left: 0.6rem !important;
      padding-right: 0.6rem !important;
    }}
    .kpi-value {{ font-size: 1.35rem; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.5rem; }}
  }}
</style>
""",
        unsafe_allow_html=True,
    )


def kpi_tile(
    label: str,
    value: str,
    hint: str = "",
    icon: str = "mentions",
    variant: str = "",
    nested: bool = False,
) -> str:
    ic = OUTLINE_ICONS.get(icon, OUTLINE_ICONS["mentions"])
    cls = f"{'kpi-inner' if nested else 'tile'} kpi {variant}".strip()
    tip = KPI_HELP.get(label, "")
    help_icon = (
        f'<span class="help-tip" data-tip="{tip.replace(chr(34), "&quot;")}">?</span>'
        if tip
        else ""
    )
    return f"""
    <div class="{cls}">
      <div class="kpi-top">{ic}<span>{label}</span>{help_icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-hint">{hint}</div>
    </div>
    """


def section_title(title: str, explanation: str, icon: str = "mentions") -> str:
    """Render a compact title with an on-hover manager interpretation."""
    safe_tip = explanation.replace('"', "&quot;")
    return (
        '<div class="section-head">'
        f'{OUTLINE_ICONS.get(icon, OUTLINE_ICONS["mentions"])}'
        f"<span>{title}</span>"
        f'<span class="help-tip" data-tip="{safe_tip}">?</span>'
        "</div>"
    )


def brand_color_map():
    return {**A.BRAND_COLORS}


def fig_layout(fig, height: int, mode: str):
    is_dark = mode == "dark"
    ink = "#eef2f6" if is_dark else "#1a2332"
    muted = "#9aa8b6" if is_dark else "#5c6b7a"
    grid = "#334155" if is_dark else "#e6ebf1"
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, sans-serif", size=12, color=ink),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=ink),
        ),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted), title_font=dict(color=muted)),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted), title_font=dict(color=muted)),
        colorway=[BRAC, SCB, SCB_BLUE, BRAC_SILVER],
    )
    fig.update_xaxes(showline=True, linecolor=grid)
    fig.update_yaxes(showline=True, linecolor=grid)
    return fig


def chart_height(desktop: int = 340) -> int:
    # Slightly shorter on narrow viewports via session heuristic
    return desktop


# Named banks shown as individual slices in the co-mention donuts.
CO_MENTION_BANKS = [
    "EBL",
    "MTB",
    "City Bank",
    "Dutch-Bangla (DBBL)",
    "UCB",
]
# Translucent, distinct greys/teals for non-rival slices (readable on light UI).
CO_MENTION_COLORS = {
    "EBL": "rgba(100, 116, 139, 0.55)",
    "MTB": "rgba(148, 163, 184, 0.65)",
    "City Bank": "rgba(71, 85, 105, 0.50)",
    "Dutch-Bangla (DBBL)": "rgba(125, 141, 160, 0.55)",
    "UCB": "rgba(167, 174, 186, 0.70)",
    "Other banks": "rgba(203, 213, 225, 0.75)",
}


def reference_pie_data(
    data: pd.DataFrame,
    focal_key: str,
    focal_bank: str,
    target_bank: str,
    target_label: str,
) -> pd.DataFrame:
    """Co-mention composition among posts that name at least one other bank.

    Each named bank mentioned in a post increments that bank's count (a post can
    contribute to multiple slices). Rows with no other-bank mention are excluded.
    """
    subset = data[data["brand"] == focal_key]
    buckets = {target_label: 0, **{b: 0 for b in CO_MENTION_BANKS}, "Other banks": 0}
    co_posts = 0
    for text in subset["text_l"]:
        found = A.find_banks(text, exclude={focal_bank})
        if not found:
            continue
        co_posts += 1
        if target_bank in found:
            buckets[target_label] += 1
        for bank in CO_MENTION_BANKS:
            if bank in found:
                buckets[bank] += 1
        leftover = found - {target_bank} - set(CO_MENTION_BANKS)
        if leftover:
            buckets["Other banks"] += 1
    order = [target_label, *CO_MENTION_BANKS, "Other banks"]
    rows = [
        {
            "bucket": name,
            "count": buckets[name],
            "pct": 100 * buckets[name] / co_posts if co_posts else 0,
            "base": co_posts,
            "focal_n": len(subset),
        }
        for name in order
        if buckets[name] > 0
    ]
    return pd.DataFrame(rows)


def evidence_comparison_rows(data: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Prefer SCB talk that frames BRAC as better/alternative; skip praise-only positives."""
    import re

    prefer_re = re.compile(
        r"(learn\s+(?:something\s+)?from.{0,50}brac|"
        r"from\s+city\s+bank\s+or\s+brac|"
        r"copy\s+their|"
        r"take\s+suggestions\s+from|"
        r"দেখতে\s+পারেন|"
        r"go\s+for\s+brac|"
        r"move\s+to\s+brac|"
        r"quit\s+scb|"
        r"leave\s+scb|"
        r"switch.{0,40}brac|"
        r"সুইচ|"
        r"ছেড়ে)",
        re.I,
    )
    anti_re = re.compile(
        r"(brac\s+is\s+the\s+worst|"
        r"not\s+to\s+go\s+for\s+brac|"
        r"scb.{0,40}better\s+than\s+brac|"
        r"better\s+than\s+brac)",
        re.I,
    )

    scb = data[
        (data["brand"] == "scb_bangladesh")
        & data["text_l"].apply(lambda text: "BRAC Bank" in A.find_banks(text))
    ].copy()
    if scb.empty:
        return scb

    texts = scb["text"].fillna("")
    scb = scb[~texts.apply(lambda t: bool(anti_re.search(t)))].copy()
    texts = scb["text"].fillna("")
    scb["is_prefer"] = texts.apply(lambda t: bool(prefer_re.search(t)))
    # Keep comparative posts even if lexicon marks them positive; drop other positives.
    scb = scb[scb["is_prefer"] | (scb["sentiment_final"].astype(str).str.lower() != "positive")]
    scb["is_negative"] = scb["sentiment_final"].astype(str).str.lower() == "negative"
    scb = scb.sort_values(by=["is_prefer", "is_negative"], ascending=[False, False])

    picked = scb[scb["is_prefer"]].drop_duplicates(subset=["text"]).head(limit)
    if len(picked) < limit:
        rest = (
            scb[~scb.index.isin(picked.index) & scb["is_negative"]]
            .drop_duplicates(subset=["text"])
        )
        picked = pd.concat([picked, rest.head(limit - len(picked))])

    # Optional single reverse: BRAC talk that explicitly prefers SCB (rare contrast).
    brac = data[
        (data["brand"] == "brac_bank")
        & data["text_l"].apply(lambda text: "Standard Chartered" in A.find_banks(text))
        & data["text"].fillna("").str.contains(
            r"(?:scb|standard\s+chartered).{0,40}better|(?:better).{0,40}(?:scb|standard\s+chartered)",
            case=False,
            regex=True,
            na=False,
        )
    ]
    if not brac.empty and len(picked) >= 2:
        picked = pd.concat([picked.head(limit - 1), brac.head(1)], ignore_index=True)

    return picked.drop(columns=[c for c in ["is_prefer", "is_negative"] if c in picked.columns])


def make_co_mention_donut(
    pie_df: pd.DataFrame,
    title: str,
    target_label: str,
    target_color: str,
    mode: str,
):
    """Hover-only donut; no on-chart labels; legend suppressed (shared HTML legend)."""
    if pie_df.empty:
        return None
    color_map = {**CO_MENTION_COLORS, target_label: target_color}
    # Pre-bake hover text — pie customdata via px is unreliable across Plotly versions.
    hover = [
        (
            f"<b>{row['bucket']}</b><br>"
            f"{float(row['pct']):.1f}% of co-mention posts "
            f"({int(row['count'])} / {int(row['base'])})<br>"
            f"Focal corpus: {int(row['focal_n'])} posts"
        )
        for _, row in pie_df.iterrows()
    ]
    colors = [color_map.get(name, "rgba(203,213,225,0.75)") for name in pie_df["bucket"]]
    fig = go.Figure(
        go.Pie(
            labels=pie_df["bucket"].tolist(),
            values=pie_df["count"].tolist(),
            hole=0.58,
            sort=False,
            direction="clockwise",
            textinfo="none",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(colors=colors, line=dict(color="#ffffff", width=1.5)),
        )
    )
    fig.update_layout(
        showlegend=False,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        margin=dict(l=8, r=8, t=36, b=8),
    )
    laid = fig_layout(fig, chart_height(280), mode)
    laid.update_layout(showlegend=False, legend=None)
    return laid


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return A.load_clean()


mode = "light"
inject_css(mode)

# ── Sidebar ───────────────────────────────────────────────────────────────────
raw = get_data()
st.sidebar.markdown("### Filters")
st.sidebar.caption("Applied to KPIs, charts, and the table.")

exclude_lq = st.sidebar.toggle("Exclude low-quality text (<30 chars)", value=True)

all_brands = sorted(raw["brand"].dropna().unique().tolist())
brand_sel = st.sidebar.multiselect(
    "Brand",
    options=all_brands,
    default=all_brands,
    format_func=lambda x: A.BRAND_LABELS.get(x, x),
)
all_sources = sorted(raw["source"].dropna().unique().tolist())
source_sel = st.sidebar.multiselect("Source", options=all_sources, default=all_sources)
all_langs = sorted(raw["language"].dropna().unique().tolist())
lang_sel = st.sidebar.multiselect("Language", options=all_langs, default=all_langs)
all_pt = sorted(raw["platform_type"].dropna().unique().tolist())
pt_sel = st.sidebar.multiselect("Platform type", options=all_pt, default=all_pt)

months = sorted([m for m in raw["month_year"].dropna().unique().tolist() if m and m != "NaT"])
if months:
    m_from, m_to = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]))
else:
    m_from = m_to = None

df = A.apply_filters(
    raw,
    exclude_low_quality=exclude_lq,
    brands=brand_sel or all_brands,
    sources=source_sel or all_sources,
    languages=lang_sel or all_langs,
    platform_types=pt_sel or all_pt,
    month_from=m_from,
    month_to=m_to,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
<div class="legend-chip"><span class="swatch" style="background:{BRAC}"></span> BRAC · {BRAC}</div>
<div class="legend-chip"><span class="swatch" style="background:{SCB}"></span> SCB · {SCB}</div>
""",
    unsafe_allow_html=True,
)
st.sidebar.caption(f"{len(df):,} rows in view · {len(raw):,} cleaned total")

# ── Header + KPI bento ────────────────────────────────────────────────────────
st.markdown(
    '<p class="dash-title">BRAC Bank vs Standard Chartered Bangladesh</p>'
    '<p class="dash-sub">BIBM social listening — comparative brand metrics. Hover charts for counts. '
    "Use the filters to recompute the comparative evidence.</p>",
    unsafe_allow_html=True,
)

kpis = A.kpi_cards(df)
asym = A.asymmetry(df)
total = max(kpis["total"], 1)
brac_n = kpis["by_brand"].get("BRAC Bank", 0)
scb_n = kpis["by_brand"].get("SCB Bangladesh", 0)
brac_star = kpis["mean_star"].get("BRAC Bank")
scb_star = kpis["mean_star"].get("SCB Bangladesh")

st.markdown(
    f"""
<div class="bento">
  <div class="tile-span-2">{kpi_tile("Mentions", f"{kpis['total']:,}", "Filtered view", "mentions")}</div>
  <div class="tile-span-2">{kpi_tile("BRAC share", f"{100*brac_n/total:.0f}%", f"n={brac_n:,}", "brac", "brac")}</div>
  <div class="tile-span-2">{kpi_tile("SCB share", f"{100*scb_n/total:.0f}%", f"n={scb_n:,}", "scb", "scb")}</div>
  <div class="tile-span-3">{kpi_tile("SCB → BRAC ref.", f"{asym['scb_mentions_brac_pct']:.1f}%", f"{asym['scb_mentions_brac_n']} of {asym['scb_n']} SCB rows", "link", "brac")}</div>
  <div class="tile-span-3">{kpi_tile("BRAC → SCB ref.", f"{asym['brac_mentions_scb_pct']:.1f}%", f"{asym['brac_mentions_scb_n']} of {asym['brac_n']} BRAC rows", "link", "scb")}</div>
  <div class="tile-span-3">{kpi_tile("BRAC mean star", f"{brac_star:.2f}" if brac_star == brac_star and brac_star is not None else "—", f"Reviews n={kpis['review_n'].get('BRAC Bank', 0)}", "star", "brac")}</div>
  <div class="tile-span-3">{kpi_tile("SCB mean star", f"{scb_star:.2f}" if scb_star == scb_star and scb_star is not None else "—", f"Reviews n={kpis['review_n'].get('SCB Bangladesh', 0)}", "star", "scb")}</div>
  <div class="tile-span-3">{kpi_tile("BRAC positive", f"{kpis['pos_pct'].get('BRAC Bank', 0):.0f}%", "sentiment_final", "sentiment", "brac")}</div>
  <div class="tile-span-3">{kpi_tile("SCB positive", f"{kpis['pos_pct'].get('SCB Bangladesh', 0):.0f}%", "sentiment_final", "sentiment", "scb")}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_perf, tab_pos, tab_touch, tab_demo, tab_data = st.tabs(
    [
        "Performance & Themes",
        "Positioning",
        "Phygital & Trends",
        "Demographics",
        "Data",
    ]
)

# =============================================================================
with tab_perf:
    prev = A.theme_prevalence(df)
    mean_df = A.meaningfulness(df)
    diff = A.differentiation_index(df)

    st.markdown(
        f"""
<div class="tile tile-span-12" style="margin-bottom:var(--gap)">
  {section_title("Theme prevalence", "Shows what each bank is talked about for. Where BRAC's bar is higher, that theme is more central to BRAC talk than SCB talk — e.g. BRAC often leads on accessibility and digital banking volume. Where SCB leads, SCB owns that conversation topic. Check Meaningfulness next to see if the association is praise or complaint.", "mentions")}
  <p class="section-note">% of filtered rows tagged with each theme. Hover for absolute counts.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if not prev.empty:
        fig = px.bar(
            prev,
            x="pct",
            y="theme",
            color="brand",
            barmode="group",
            orientation="h",
            color_discrete_map=brand_color_map(),
            custom_data=["count", "n_base"],
            labels={"pct": "% of brand rows", "theme": "", "brand": ""},
            category_orders={"theme": list(A.THEME_LABELS.values())},
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% (%{customdata[0]} of %{customdata[1]})<extra>%{fullData.name}</extra>",
            marker_line_width=0,
        )
        st.plotly_chart(fig_layout(fig, chart_height(360), mode), use_container_width=True)

    st.markdown(
        f"""
<div class="tile tile-span-12" style="margin-bottom:var(--gap)">
  {section_title("Reference-point salience", "SCB users mention BRAC much more often than BRAC users mention SCB. This indicates BRAC is a comparison benchmark in SCB conversations; the evidence examples show that these comparisons often arise from SCB dissatisfaction.", "mentions")}
  <p class="section-note">Only co-mention conversations (posts that name another bank). Hover a slice for bank and count. Rival slices are solid; other banks are translucent.</p>
  <div class="shared-legend">
    <span class="item"><span class="dot" style="background:{SCB}"></span>Mentions SCB (in BRAC talk)</span>
    <span class="item"><span class="dot" style="background:{BRAC}"></span>Mentions BRAC (in SCB talk)</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['EBL']}"></span>EBL</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['MTB']}"></span>MTB</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['City Bank']}"></span>City Bank</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['Dutch-Bangla (DBBL)']}"></span>DBBL</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['UCB']}"></span>UCB</span>
    <span class="item"><span class="dot" style="background:{CO_MENTION_COLORS['Other banks']}"></span>Other banks</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns([1, 1, 1.35], gap="small")
    with s1:
        brac_pie = reference_pie_data(
            df, "brac_bank", "BRAC Bank", "Standard Chartered", "Mentions SCB"
        )
        fig = make_co_mention_donut(brac_pie, "BRAC co-mentions", "Mentions SCB", SCB, mode)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No BRAC co-mentions under current filters.")
    with s2:
        scb_pie = reference_pie_data(
            df, "scb_bangladesh", "Standard Chartered", "BRAC Bank", "Mentions BRAC"
        )
        fig = make_co_mention_donut(scb_pie, "SCB co-mentions", "Mentions BRAC", BRAC, mode)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SCB co-mentions under current filters.")
    with s3:
        st.markdown("**Evidence: SCB talk that cites BRAC as the better alternative**")
        examples = evidence_comparison_rows(df, limit=6)
        show = examples[["source", "text"]].copy() if not examples.empty else examples
        st.dataframe(
            show,
            column_config={
                "source": st.column_config.TextColumn("source", width="small"),
                "text": st.column_config.TextColumn("text", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            height=320,
        )

    c1 = st.container()
    with c1:
        st.markdown(
            f"""
<div class="tile">
  {section_title("Meaningfulness", "Of all mentions tagged with a theme, what share is positive? This is praise quality within the theme — not how common the theme is. Compare with Theme prevalence: a bank can lead App/UX volume yet trail on % positive App/UX, which means more talk is complaint, not preference.", "sentiment")}
  <p class="section-note">% positive among theme-tagged mentions (positive count ÷ theme count). Hover for both counts.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if not mean_df.empty:
            # Prefer within-theme % positive; fall back if a stale module is loaded.
            y_col = "pct_positive" if "pct_positive" in mean_df.columns else "per_100"
            custom = (
                ["pos_count", "theme_count"]
                if "theme_count" in mean_df.columns
                else ["pos_count", "n_base"]
            )
            hover = (
                "<b>%{x}</b><br>%{y:.1f}% positive (%{customdata[0]} of %{customdata[1]})"
                "<extra>%{fullData.name}</extra>"
                if y_col == "pct_positive"
                else "<b>%{x}</b><br>%{y:.1f}/100 · n=%{customdata[0]}<extra>%{fullData.name}</extra>"
            )
            fig = px.bar(
                mean_df,
                x="theme",
                y=y_col,
                color="brand",
                barmode="group",
                color_discrete_map=brand_color_map(),
                custom_data=custom,
                labels={y_col: "% positive within theme" if y_col == "pct_positive" else "Per 100 rows", "theme": ""},
            )
            fig.update_traces(hovertemplate=hover)
            st.plotly_chart(fig_layout(fig, chart_height(340), mode), use_container_width=True)

# =============================================================================
with tab_pos:
    diff = A.differentiation_index(df)
    pos_diff = A.positive_differentiation_index(df)

    st.markdown(
        f"""
<div class="tile tile-span-12" style="margin-bottom:var(--gap)">
  {section_title("Differentiation index (prevalence)", "How uneven total theme talk is between BRAC and SCB. A large gap means one bank is discussed more on that topic — volume ownership only. Pair with the positive differentiation chart below: leading volume does not mean leading preference.", "link")}
  <p class="section-note">Prevalence gap (BRAC % − SCB %) of all theme-tagged mentions. Brand-colored bars = who has more talk.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if not diff.empty:
        colors = [BRAC if gap > 0 else SCB for gap in diff["gap"]]
        fig = go.Figure(
            go.Bar(
                x=diff["gap"],
                y=diff["theme"],
                orientation="h",
                marker_color=colors,
                customdata=diff[["brac_pct", "scb_pct", "brac_count", "scb_count", "leader"]],
                hovertemplate=(
                    "<b>%{y}</b><br>Prevalence gap %{x:.1f}pp<br>"
                    "BRAC %{customdata[0]:.1f}% (n=%{customdata[2]})<br>"
                    "SCB %{customdata[1]:.1f}% (n=%{customdata[3]})<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=BRAC_SILVER)
        fig.update_layout(xaxis_title="pp gap (BRAC − SCB)", showlegend=False)
        st.plotly_chart(fig_layout(fig, chart_height(360), mode), use_container_width=True)

    st.markdown(
        f"""
<div class="tile tile-span-12" style="margin-bottom:var(--gap)">
  {section_title("Differentiation index (positive only)", "Same gap idea, but only positive theme mentions as a share of each bank’s corpus. If BRAC leads prevalence on App/UX but trails (or barely leads) here, the extra talk is not preference — it is often complaint. This is the chart that separates volume differentiation from perceived preference.", "star")}
  <p class="section-note">Gap in % of brand rows that are positive ∩ theme (BRAC − SCB). Hover for positive counts.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if not pos_diff.empty:
        colors = [BRAC if gap > 0 else SCB for gap in pos_diff["gap"]]
        fig = go.Figure(
            go.Bar(
                x=pos_diff["gap"],
                y=pos_diff["theme"],
                orientation="h",
                marker_color=colors,
                customdata=pos_diff[["brac_pct", "scb_pct", "brac_count", "scb_count", "leader"]],
                hovertemplate=(
                    "<b>%{y}</b><br>Positive-share gap %{x:.1f}pp<br>"
                    "BRAC %{customdata[0]:.1f}% (pos n=%{customdata[2]})<br>"
                    "SCB %{customdata[1]:.1f}% (pos n=%{customdata[3]})<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=BRAC_SILVER)
        fig.update_layout(xaxis_title="pp gap in positive theme share (BRAC − SCB)", showlegend=False)
        st.plotly_chart(fig_layout(fig, chart_height(360), mode), use_container_width=True)

# =============================================================================
with tab_touch:
    monthly = A.monthly_app_ux_positivity(df)
    st.markdown(
        f"""
<div class="tile">
  {section_title("Monthly App/UX sentiment %", "Solid lines = share of App/UX talk that is positive; dotted lines = share that is negative (same brand colors). When a bank’s dotted line sits above its solid line, complaint dominates digital talk that month. Compare BRAC vs SCB to see who is winning or losing digital goodwill over time.", "mentions")}
  <p class="section-note">Among App/UX-tagged mentions each month. Solid = positive % · dotted = negative %. Hover for monthly counts.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if not monthly.empty:
        fig = go.Figure()
        for brand, color in brand_color_map().items():
            sub = monthly[monthly["brand"] == brand].sort_values("month")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    name=f"{brand} · positive",
                    x=sub["month"],
                    y=sub["pos_pct"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="solid"),
                    marker=dict(size=7, color=color, line=dict(width=1, color="rgba(255,255,255,0.4)")),
                    customdata=sub[["count", "neg_pct"]],
                    hovertemplate=(
                        "<b>%{x}</b><br>%{y:.0f}% positive · %{customdata[1]:.0f}% negative"
                        " (n=%{customdata[0]})<extra>" + brand + "</extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    name=f"{brand} · negative",
                    x=sub["month"],
                    y=sub["neg_pct"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.25, dash="dot"),
                    marker=dict(size=6, color=color, symbol="diamond", line=dict(width=0)),
                    customdata=sub[["count", "pos_pct"]],
                    hovertemplate=(
                        "<b>%{x}</b><br>%{y:.0f}% negative · %{customdata[1]:.0f}% positive"
                        " (n=%{customdata[0]})<extra>" + brand + "</extra>"
                    ),
                )
            )
        fig.update_layout(yaxis_title="% of App/UX mentions", xaxis_title="")
        st.plotly_chart(fig_layout(fig, chart_height(420), mode), use_container_width=True)

# =============================================================================
with tab_demo:
    gap = A.demographic_gap(df)
    st.markdown(
        f"""
<div class="tile">
  {section_title("Segment talk vs branded scheme", "Compares how often people talk about a demographic need vs naming the bank's scheme (Agami, TARA, Priority, Probashi/Swadeshi, salary account, etc.). A large gap means the need is discussed but the branded offer is invisible. Salary/payroll only counts when the focal bank is the salary home — not income-for-CC profiles or salary at another bank. Priority excludes UCB Imperial and bug-report 'priority'. NRB excludes NRB Commercial Bank card lists.", "link")}
  <p class="section-note">Same scale (% of brand). Faded outer bar = all segment talk · solid inner bar = those who also name a scheme (subset). Hover for absolute counts.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    if not gap.empty:
        fig = go.Figure()
        # Nested overlay on one scale: draw faded FULL segment first (back),
        # then solid NAMED share on top (front). Solid-full-first hid the subset.
        for brand, color, faded in [
            ("BRAC Bank", BRAC, "rgba(0,108,181,0.28)"),
            ("SCB Bangladesh", SCB, "rgba(56,210,0,0.28)"),
        ]:
            sub = gap[gap["brand"] == brand]
            fig.add_trace(
                go.Bar(
                    name=f"{brand} · segment talk",
                    x=sub["segment"],
                    y=sub["demo_pct"],
                    offsetgroup=brand,
                    legendgroup=brand,
                    width=0.4,
                    marker=dict(
                        color=faded,
                        line=dict(color=color, width=1.5),
                    ),
                    customdata=sub[["demo_count", "n_base", "named_count"]],
                    hovertemplate=(
                        "<b>%{x}</b><br>Segment talk %{y:.1f}% of brand"
                        " (n=%{customdata[0]} / %{customdata[1]})<br>"
                        "Of which name a scheme: %{customdata[2]}"
                        "<extra>" + brand + "</extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Bar(
                    name=f"{brand} · names scheme",
                    x=sub["segment"],
                    y=sub["named_brand_pct"],
                    offsetgroup=brand,
                    legendgroup=brand,
                    width=0.22,
                    marker=dict(color=color, line=dict(color=color, width=0)),
                    customdata=sub[["named_count", "demo_count", "named_of_demo_pct"]],
                    hovertemplate=(
                        "<b>%{x}</b><br>Names scheme %{y:.2f}% of brand"
                        " (%{customdata[0]} of %{customdata[1]} segment"
                        " · %{customdata[2]:.0f}% of segment)"
                        "<extra>" + brand + "</extra>"
                    ),
                )
            )
        fig.update_layout(
            barmode="overlay",
            bargap=0.25,
            bargroupgap=0.15,
            yaxis_title="% of brand mentions",
        )
        st.plotly_chart(fig_layout(fig, chart_height(460), mode), use_container_width=True)

# =============================================================================
with tab_data:
    st.markdown(
        f"""
<div class="tile">
  {section_title("Cleaned mentions", "Raw filtered evidence behind the BRAC vs SCB charts. Search and open rows to verify why one bank looks stronger — e.g. switch advice, app complaints, or service praise — then export only what is in view.", "mentions")}
  <p class="section-note">Same filters as charts. Export downloads the current filtered rows.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    search = st.text_input("Search text contains", "")
    table = df.copy()
    if search.strip():
        table = table[table["text"].str.contains(search.strip(), case=False, na=False)]
    show_cols = [
        c
        for c in [
            "brand_label",
            "source",
            "platform_type",
            "month_year",
            "language",
            "sentiment_final",
            "rating_sentiment",
            "star_rating",
            "low_quality",
            "theme_app_ux",
            "theme_cards",
            "theme_fees",
            "theme_service",
            "theme_transfers",
            "theme_security",
            "text",
            "url",
            "record_id",
        ]
        if c in table.columns
    ]
    display = table[show_cols].rename(columns={"brand_label": "brand"})
    st.caption(f"Showing {len(display):,} rows")
    st.dataframe(display, use_container_width=True, height=420)
    st.download_button(
        "Export filtered CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="unified_mentions_filtered.csv",
        mime="text/csv",
    )
