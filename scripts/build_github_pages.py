"""Build a static GitHub Pages dashboard from the same analytics as Streamlit.

Does not import or modify dashboard/app.py — local Streamlit keeps working.

Usage (from repo root):
  python scripts/build_github_pages.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import analytics as A  # noqa: E402

DOCS = ROOT / "docs"
BRAC = "#006CB5"
SCB = "#38D200"
BRAC_SILVER = "#C2C2C2"
SCB_BLUE = "#0473EA"

CO_MENTION_BANKS = ["EBL", "MTB", "City Bank", "Dutch-Bangla (DBBL)", "UCB"]
CO_MENTION_COLORS = {
    "EBL": "rgba(100, 116, 139, 0.55)",
    "MTB": "rgba(148, 163, 184, 0.65)",
    "City Bank": "rgba(71, 85, 105, 0.50)",
    "Dutch-Bangla (DBBL)": "rgba(125, 141, 160, 0.55)",
    "UCB": "rgba(167, 174, 186, 0.70)",
    "Other banks": "rgba(203, 213, 225, 0.75)",
}


def brand_colors():
    return {**A.BRAND_COLORS}


def fig_layout(fig, height: int = 360):
    ink, muted, grid = "#1a2332", "#5c6b7a", "#e6ebf1"
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, sans-serif", size=12, color=ink),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted)),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=muted)),
        colorway=[BRAC, SCB, SCB_BLUE, BRAC_SILVER],
    )
    return fig


def fig_div(fig, div_id: str) -> str:
    if fig is None:
        return f'<p class="empty" id="{html.escape(div_id)}">No data for this view.</p>'
    payload = json.loads(pio.to_json(fig))
    spec = json.dumps({"data": payload["data"], "layout": payload["layout"]})
    var = "spec_" + re.sub(r"[^a-zA-Z0-9_]", "_", div_id)
    return (
        f'<div id="{html.escape(div_id)}" class="chart"></div>\n'
        f"<script>const {var} = {spec}; "
        f"Plotly.newPlot({json.dumps(div_id)}, {var}.data, {var}.layout, "
        "{responsive:true, displayModeBar:false});</script>"
    )


def reference_pie_data(data, focal_key, focal_bank, target_bank, target_label):
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
        if found - {target_bank} - set(CO_MENTION_BANKS):
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


def make_donut(pie_df, title, target_label, target_color):
    if pie_df.empty:
        return None
    color_map = {**CO_MENTION_COLORS, target_label: target_color}
    hover = [
        (
            f"<b>{row['bucket']}</b><br>{float(row['pct']):.1f}% of co-mention posts "
            f"({int(row['count'])} / {int(row['base'])})<br>Focal corpus: {int(row['focal_n'])} posts"
        )
        for _, row in pie_df.iterrows()
    ]
    colors = [color_map.get(n, "rgba(203,213,225,0.75)") for n in pie_df["bucket"]]
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
    fig.update_layout(showlegend=False, title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)))
    return fig_layout(fig, 280)


def evidence_rows(data: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    prefer_re = re.compile(
        r"(learn\s+(?:something\s+)?from.{0,50}brac|from\s+city\s+bank\s+or\s+brac|"
        r"copy\s+their|take\s+suggestions\s+from|দেখতে\s+পারেন|go\s+for\s+brac|"
        r"move\s+to\s+brac|quit\s+scb|leave\s+scb|switch.{0,40}brac|সুইচ|ছেড়ে)",
        re.I,
    )
    anti_re = re.compile(
        r"(brac\s+is\s+the\s+worst|not\s+to\s+go\s+for\s+brac|"
        r"scb.{0,40}better\s+than\s+brac|better\s+than\s+brac)",
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
    scb = scb[scb["is_prefer"] | (scb["sentiment_final"].astype(str).str.lower() != "positive")]
    scb["is_negative"] = scb["sentiment_final"].astype(str).str.lower() == "negative"
    scb = scb.sort_values(by=["is_prefer", "is_negative"], ascending=[False, False])
    picked = scb[scb["is_prefer"]].drop_duplicates(subset=["text"]).head(limit)
    if len(picked) < limit:
        rest = scb[~scb.index.isin(picked.index) & scb["is_negative"]].drop_duplicates(subset=["text"])
        picked = pd.concat([picked, rest.head(limit - len(picked))])
    return picked.drop(columns=[c for c in ["is_prefer", "is_negative"] if c in picked.columns])


def section(title: str, note: str, body: str) -> str:
    return f"""
<section class="tile">
  <h2>{html.escape(title)}</h2>
  <p class="note">{html.escape(note)}</p>
  {body}
</section>
"""


def kpi_html(df: pd.DataFrame) -> str:
    kpis = A.kpi_cards(df)
    asym = A.asymmetry(df)
    total = max(kpis["total"], 1)
    brac_n = kpis["by_brand"].get("BRAC Bank", 0)
    scb_n = kpis["by_brand"].get("SCB Bangladesh", 0)
    brac_star = kpis["mean_star"].get("BRAC Bank")
    scb_star = kpis["mean_star"].get("SCB Bangladesh")

    def card(label, value, sub, accent=""):
        return (
            f'<div class="kpi {accent}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>'
            f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
        )

    return f"""
<div class="kpis">
  {card("Mentions", f"{kpis['total']:,}", "Cleaned evidence base")}
  {card("BRAC share", f"{100*brac_n/total:.0f}%", f"n={brac_n:,}", "brac")}
  {card("SCB share", f"{100*scb_n/total:.0f}%", f"n={scb_n:,}", "scb")}
  {card("SCB → BRAC ref.", f"{asym['scb_mentions_brac_pct']:.1f}%", f"{asym['scb_mentions_brac_n']} of {asym['scb_n']} SCB", "brac")}
  {card("BRAC → SCB ref.", f"{asym['brac_mentions_scb_pct']:.1f}%", f"{asym['brac_mentions_scb_n']} of {asym['brac_n']} BRAC", "scb")}
  {card("BRAC mean star", f"{brac_star:.2f}" if brac_star == brac_star and brac_star is not None else "—", f"Reviews n={kpis['review_n'].get('BRAC Bank', 0)}", "brac")}
  {card("SCB mean star", f"{scb_star:.2f}" if scb_star == scb_star and scb_star is not None else "—", f"Reviews n={kpis['review_n'].get('SCB Bangladesh', 0)}", "scb")}
  {card("BRAC positive", f"{kpis['pos_pct'].get('BRAC Bank', 0):.0f}%", "sentiment", "brac")}
  {card("SCB positive", f"{kpis['pos_pct'].get('SCB Bangladesh', 0):.0f}%", "sentiment", "scb")}
</div>
"""


def build_figures(df: pd.DataFrame) -> dict[str, go.Figure | None]:
    figs: dict[str, go.Figure | None] = {}
    prev = A.theme_prevalence(df)
    if not prev.empty:
        fig = px.bar(
            prev, x="pct", y="theme", color="brand", barmode="group", orientation="h",
            color_discrete_map=brand_colors(), custom_data=["count", "n_base"],
            labels={"pct": "% of brand rows", "theme": "", "brand": ""},
            category_orders={"theme": list(A.THEME_LABELS.values())},
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% (%{customdata[0]} of %{customdata[1]})<extra>%{fullData.name}</extra>"
        )
        figs["theme_prev"] = fig_layout(fig, 360)
    else:
        figs["theme_prev"] = None

    figs["brac_donut"] = make_donut(
        reference_pie_data(df, "brac_bank", "BRAC Bank", "Standard Chartered", "Mentions SCB"),
        "BRAC co-mentions", "Mentions SCB", SCB,
    )
    figs["scb_donut"] = make_donut(
        reference_pie_data(df, "scb_bangladesh", "Standard Chartered", "BRAC Bank", "Mentions BRAC"),
        "SCB co-mentions", "Mentions BRAC", BRAC,
    )

    mean_df = A.meaningfulness(df)
    if not mean_df.empty:
        fig = px.bar(
            mean_df, x="theme", y="per_100", color="brand", barmode="group",
            color_discrete_map=brand_colors(), custom_data=["pos_count"],
            labels={"per_100": "Per 100 rows", "theme": ""},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.1f}/100 · n=%{customdata[0]}<extra>%{fullData.name}</extra>")
        figs["meaning"] = fig_layout(fig, 340)
    else:
        figs["meaning"] = None

    diff = A.differentiation_index(df)
    if not diff.empty:
        colors = [BRAC if g > 0 else SCB for g in diff["gap"]]
        fig = go.Figure(
            go.Bar(
                x=diff["gap"], y=diff["theme"], orientation="h", marker_color=colors,
                customdata=diff[["brac_pct", "scb_pct", "brac_count", "scb_count"]],
                hovertemplate=(
                    "<b>%{y}</b><br>Gap %{x:.1f}pp<br>BRAC %{customdata[0]:.1f}% (n=%{customdata[2]})<br>"
                    "SCB %{customdata[1]:.1f}% (n=%{customdata[3]})<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=BRAC_SILVER)
        fig.update_layout(xaxis_title="pp gap (BRAC − SCB)", showlegend=False)
        figs["diff"] = fig_layout(fig, 360)
    else:
        figs["diff"] = None

    pop_df = A.pop_pod_table(df)
    pops, pods = A.identify_pops_pods(pop_df)
    if not pods.empty:
        long = []
        for _, r in pods.iterrows():
            long.append({"theme": r["theme"], "brand": "BRAC Bank", "pct": r["brac_prevalence"], "count": r["brac_count"], "net": r["brac_net"], "quality": r["pod_quality"]})
            long.append({"theme": r["theme"], "brand": "SCB Bangladesh", "pct": r["scb_prevalence"], "count": r["scb_count"], "net": r["scb_net"], "quality": r["pod_quality"]})
        long_df = pd.DataFrame(long)
        fig = px.bar(
            long_df, x="theme", y="pct", color="brand", barmode="group",
            color_discrete_map=brand_colors(), custom_data=["count", "net", "quality"],
            labels={"pct": "Prevalence %", "theme": ""},
            text=long_df["net"].map(lambda x: f"{x:+.0f}"),
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% (n=%{customdata[0]}) · net %{customdata[1]:+.0f}pp<br>%{customdata[2]}<extra>%{fullData.name}</extra>",
        )
        figs["pods"] = fig_layout(fig, 340)
    else:
        figs["pods"] = None

    if not pops.empty:
        fig = go.Figure()
        for brand, color in brand_colors().items():
            sub = pop_df[(pop_df["brand"] == brand) & (pop_df["theme"].isin(pops["theme"]))]
            fig.add_trace(
                go.Bar(
                    name=brand, x=sub["theme"], y=sub["prevalence"], marker_color=color,
                    customdata=sub[["count", "net", "pos_pct", "neg_pct"]],
                    hovertemplate=(
                        "<b>%{x}</b><br>Prevalence %{y:.1f}% (n=%{customdata[0]})<br>"
                        "Net %{customdata[1]:+.0f}pp (pos %{customdata[2]:.0f}% / neg %{customdata[3]:.0f}%)"
                        f"<extra>{brand}</extra>"
                    ),
                )
            )
        fig.update_layout(barmode="group", yaxis_title="Prevalence %")
        figs["pops"] = fig_layout(fig, 340)
    else:
        figs["pops"] = None

    phy = A.phygital_net(df)
    if not phy.empty:
        fig = px.bar(
            phy, x="touchpoint", y="net", color="brand", barmode="group",
            color_discrete_map=brand_colors(), custom_data=["pos_pct", "neg_pct", "count"],
            labels={"net": "Net sentiment (pp)", "touchpoint": ""},
            text=phy["net"].map(lambda x: f"{x:+.0f}"),
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Net %{y:+.0f}pp · pos %{customdata[0]:.0f}% · neg %{customdata[1]:.0f}% · n=%{customdata[2]}<extra>%{fullData.name}</extra>",
        )
        fig.add_hline(y=0, line_width=1, line_color=BRAC_SILVER)
        figs["phy"] = fig_layout(fig, 380)
    else:
        figs["phy"] = None

    monthly = A.monthly_app_ux_positivity(df)
    if not monthly.empty:
        fig = px.line(
            monthly, x="month", y="pos_pct", color="brand", markers=True,
            color_discrete_map=brand_colors(), custom_data=["count"],
            labels={"pos_pct": "Positive %", "month": ""},
        )
        fig.update_traces(
            line=dict(width=2.5),
            hovertemplate="<b>%{x}</b><br>%{y:.0f}% positive (n=%{customdata[0]})<extra>%{fullData.name}</extra>",
        )
        figs["monthly"] = fig_layout(fig, 380)
    else:
        figs["monthly"] = None

    gap = A.demographic_gap(df)
    if not gap.empty:
        fig = go.Figure()
        for brand, color, faded in [
            ("BRAC Bank", BRAC, "rgba(0,108,181,0.28)"),
            ("SCB Bangladesh", SCB, "rgba(56,210,0,0.28)"),
        ]:
            sub = gap[gap["brand"] == brand]
            fig.add_trace(
                go.Bar(
                    name=f"{brand} · segment talk", x=sub["segment"], y=sub["demo_pct"],
                    marker=dict(color=faded, line=dict(color=color, width=1.5)),
                    customdata=sub[["demo_count"]],
                    hovertemplate="<b>%{x}</b><br>Segment %{y:.1f}% (n=%{customdata[0]})<extra>" + brand + "</extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name=f"{brand} · names scheme", x=sub["segment"], y=sub["named_of_demo_pct"],
                    marker=dict(color=color),
                    customdata=sub[["named_count", "demo_count"]],
                    hovertemplate="<b>%{x}</b><br>%{y:.0f}% name a scheme (%{customdata[0]}/%{customdata[1]})<extra>" + brand + "</extra>",
                )
            )
        fig.update_layout(barmode="group", yaxis_title="%")
        figs["demo"] = fig_layout(fig, 460)
    else:
        figs["demo"] = None

    return figs


def evidence_table_html(df: pd.DataFrame) -> str:
    rows = evidence_rows(df, 6)
    if rows.empty:
        return "<p class='empty'>No evidence rows.</p>"
    body = []
    for _, r in rows.iterrows():
        body.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('source', '')))}</td>"
            f"<td>{html.escape(str(r.get('text', ''))[:500])}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>source</th><th>text</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


CSS = """
:root { --brac:#006CB5; --scb:#38D200; --ink:#17212e; --muted:#64748b; --border:#e1e7ee; --bg:#f5f7fa; --card:#fff; }
* { box-sizing: border-box; }
body { margin:0; font-family: "Segoe UI", Inter, system-ui, sans-serif; color:var(--ink); background:var(--bg); }
.wrap { max-width:1280px; margin:0 auto; padding:1.25rem 1rem 3rem; }
h1 { font-size:clamp(1.35rem,2.5vw,1.85rem); margin:0 0 .25rem; letter-spacing:-.02em; }
.sub { color:var(--muted); margin:0 0 1rem; font-size:.95rem; }
.badge { display:inline-block; font-size:.75rem; padding:.2rem .5rem; border-radius:999px; background:#e8f1f8; color:var(--brac); margin-bottom:.75rem; }
.kpis { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:.75rem; margin-bottom:1rem; }
.kpi { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:.85rem .9rem; }
.kpi.brac { border-top:3px solid var(--brac); }
.kpi.scb { border-top:3px solid var(--scb); }
.kpi-label { font-size:.78rem; color:var(--muted); font-weight:600; }
.kpi-value { font-size:1.35rem; font-weight:700; margin:.15rem 0; }
.kpi-sub { font-size:.72rem; color:var(--muted); }
nav.tabs { display:flex; flex-wrap:wrap; gap:.4rem; margin:1rem 0; position:sticky; top:0; background:rgba(245,247,250,.92); backdrop-filter:blur(8px); padding:.5rem 0; z-index:5; }
nav.tabs button { border:1px solid var(--border); background:var(--card); color:var(--ink); padding:.45rem .8rem; border-radius:999px; cursor:pointer; font-weight:600; font-size:.85rem; }
nav.tabs button.active { background:var(--brac); color:#fff; border-color:var(--brac); }
.panel { display:none; }
.panel.active { display:block; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1rem; margin-bottom:.85rem; }
.tile h2 { margin:0 0 .35rem; font-size:1.05rem; }
.note { color:var(--muted); font-size:.84rem; margin:0 0 .6rem; line-height:1.4; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:.85rem; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1.35fr; gap:.85rem; }
.legend { display:flex; flex-wrap:wrap; gap:.65rem 1rem; margin:.4rem 0 .8rem; font-size:.8rem; color:var(--muted); }
.legend span { display:inline-flex; align-items:center; gap:.35rem; }
.dot { width:11px; height:11px; border-radius:3px; border:1px solid rgba(23,33,46,.12); }
.table-wrap { overflow:auto; max-height:340px; border:1px solid var(--border); border-radius:8px; }
table { width:100%; border-collapse:collapse; font-size:.82rem; }
th, td { padding:.55rem .65rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
th { position:sticky; top:0; background:#f8fafc; }
.empty { color:var(--muted); font-size:.9rem; }
footer { margin-top:1.5rem; color:var(--muted); font-size:.8rem; }
a { color:var(--brac); }
@media (max-width:900px) {
  .grid2, .grid3 { grid-template-columns:1fr; }
}
"""


JS_TABS = """
document.querySelectorAll('nav.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.panel).classList.add('active');
    window.dispatchEvent(new Event('resize'));
  });
});
"""


def render_html(df: pd.DataFrame, figs: dict) -> str:
    legend = f"""
<div class="legend">
  <span><i class="dot" style="background:{SCB}"></i>Mentions SCB (in BRAC talk)</span>
  <span><i class="dot" style="background:{BRAC}"></i>Mentions BRAC (in SCB talk)</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['EBL']}"></i>EBL</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['MTB']}"></i>MTB</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['City Bank']}"></i>City Bank</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['Dutch-Bangla (DBBL)']}"></i>DBBL</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['UCB']}"></i>UCB</span>
  <span><i class="dot" style="background:{CO_MENTION_COLORS['Other banks']}"></i>Other banks</span>
</div>
"""
    perf = (
        section(
            "Theme prevalence",
            "Where BRAC's bar is higher, that theme is more central to BRAC talk than SCB talk. Check Meaningfulness for praise vs complaint.",
            fig_div(figs.get("theme_prev"), "c-theme"),
        )
        + section(
            "Reference-point salience",
            "SCB users mention BRAC much more often than BRAC users mention SCB — BRAC is the comparison benchmark in SCB talk.",
            legend
            + '<div class="grid3">'
            + f'<div>{fig_div(figs.get("brac_donut"), "c-brac-donut")}</div>'
            + f'<div>{fig_div(figs.get("scb_donut"), "c-scb-donut")}</div>'
            + "<div><h3 style='font-size:.95rem;margin:0 0 .5rem'>Evidence: SCB talk that cites BRAC as the better alternative</h3>"
            + evidence_table_html(df)
            + "</div></div>",
        )
        + section(
            "Meaningfulness",
            "Positive themed mentions per 100 rows. A bank ahead here owns that benefit; high talk with a low score is a warning.",
            fig_div(figs.get("meaning"), "c-meaning"),
        )
    )
    pos = (
        section(
            "Differentiation index",
            "Prevalence gap (BRAC % − SCB %). Large gaps mean one bank owns that topic; confirm sentiment before calling it an advantage.",
            fig_div(figs.get("diff"), "c-diff"),
        )
        + '<div class="grid2">'
        + section("Identified PODs", "Themes with a clear volume leader; labels show net sentiment (pp).", fig_div(figs.get("pods"), "c-pods"))
        + section("Clarified POP table", "Table-stakes themes both banks must get right. Better net sentiment = delivering the shared expectation more credibly.", fig_div(figs.get("pops"), "c-pops"))
        + "</div>"
    )
    touch = (
        '<div class="grid2">'
        + section("Sentiment net by touchpoint", "Digital = App/UX · Physical = Service. Net = pos% − neg%.", fig_div(figs.get("phy"), "c-phy"))
        + section("Monthly App/UX positivity %", "When BRAC's line sits above SCB, BRAC is winning digital goodwill that month.", fig_div(figs.get("monthly"), "c-monthly"))
        + "</div>"
    )
    demo = section(
        "Segment talk vs branded scheme",
        "Faded = segment talk %. Solid = % of that talk naming a scheme. Large gaps mean the need is discussed but the branded offer is invisible.",
        fig_div(figs.get("demo"), "c-demo"),
    )
    data_panel = section(
        "Cleaned mentions",
        "Download the cleaned evidence base used for every chart on this page.",
        '<p><a href="data/unified_mentions_clean.csv">Download unified_mentions_clean.csv</a></p>'
        f"<p class='note'>{len(df):,} rows in the published snapshot.</p>",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>BRAC Bank vs Standard Chartered Bangladesh — BIBM</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <div class="badge">GitHub Pages · static snapshot</div>
    <h1>BRAC Bank vs Standard Chartered Bangladesh</h1>
    <p class="sub">BIBM social listening — comparative brand metrics. Interactive Plotly charts (hover for counts). For live sidebar filters, run the Streamlit app locally.</p>
    {kpi_html(df)}
    <nav class="tabs" aria-label="Sections">
      <button type="button" class="active" data-panel="perf">Performance &amp; Themes</button>
      <button type="button" data-panel="pos">Positioning</button>
      <button type="button" data-panel="touch">Phygital &amp; Trends</button>
      <button type="button" data-panel="demo">Demographics</button>
      <button type="button" data-panel="data">Data</button>
    </nav>
    <div id="perf" class="panel active">{perf}</div>
    <div id="pos" class="panel">{pos}</div>
    <div id="touch" class="panel">{touch}</div>
    <div id="demo" class="panel">{demo}</div>
    <div id="data" class="panel">{data_panel}</div>
    <footer>
      Source repo: <a href="https://github.com/Shudipta-Dip/bibm-bank-social-listening">Shudipta-Dip/bibm-bank-social-listening</a>
      · Rebuild with <code>python scripts/build_github_pages.py</code>
    </footer>
  </div>
  <script>{JS_TABS}</script>
</body>
</html>
"""


def main() -> None:
    raw = A.load_clean()
    df = A.apply_filters(raw, exclude_low_quality=True)
    figs = build_figures(df)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copy2(A.CLEAN_CSV, DOCS / "data" / "unified_mentions_clean.csv")
    (DOCS / "index.html").write_text(render_html(df, figs), encoding="utf-8")
    print(f"Wrote {DOCS / 'index.html'} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
