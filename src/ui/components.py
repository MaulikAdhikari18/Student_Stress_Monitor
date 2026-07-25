"""
Small HTML-snippet builder functions shared across pages.

num_card / summary_card / level_badge / stress_bg / stress_border are the
originals, now theme-aware (colors come from CSS variables set by
styles.py, so they repaint correctly on light/dark toggle).

aurora_hero_card / metric_tile / progress_ring_svg are new -- the Aurora
Glass look (frosted card, soft blurred color blobs, circular progress
ring) used on the Dashboard page's hero section.
"""

import streamlit as st
import pandas as pd
from src.config import COLORS as LEVEL_COLOR


def num_card(label: str, value: str, sub: str = "", color: str = "var(--accent-purple)",
             border_color: str | None = None) -> str:
    bc = border_color or color
    return (
        f'<div style="background:var(--bg-surface);border:0.5px solid var(--border);'
        f'border-top:3px solid {bc};border-radius:12px;padding:0.8rem 1rem;">'
        f'<div style="font-size:0.8rem;color:var(--text-muted);text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:1.7rem;font-weight:800;color:{color};line-height:1;">{value}</div>'
        f'<div style="font-size:0.85rem;color:var(--text-muted);margin-top:3px;">{sub}</div>'
        f'</div>'
    )


def summary_card(label: str, value: str, sub: str, color: str = "var(--accent-purple)") -> str:
    return (
        f'<div style="background:var(--bg-surface);border:0.5px solid var(--border);'
        f'border-radius:14px;padding:0.9rem 1.1rem;">'
        f'<div style="font-size:0.8rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;">{label}</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:{color};line-height:1.1;margin:4px 0 2px;">{value}</div>'
        f'<div style="font-size:0.86rem;color:var(--text-muted);">{sub}</div>'
        f'</div>'
    )


def level_badge(level: str) -> str:
    c = LEVEL_COLOR.get(level, "#888")
    return f'<span style="background:{c}33;color:{c};border-radius:20px;padding:3px 11px;font-size:0.82rem;font-weight:700;">{level}</span>'


def stress_bg(score) -> str:
    if score is None:
        return "rgba(128,128,128,0.08)"
    if score >= 90:
        return "rgba(163,45,45,0.45)"
    if score >= 75:
        return "rgba(153,60,29,0.40)"
    if score >= 30:
        return "rgba(186,117,23,0.35)"
    return "rgba(99,153,34,0.35)"


def stress_border(score) -> str:
    if score is None:
        return "var(--border)"
    if score >= 90:
        return "#E24B4A"
    if score >= 75:
        return "#D85A30"
    if score >= 30:
        return "#EF9F27"
    return "#639922"


def dataframe_html(df: pd.DataFrame, max_height: str = "420px") -> str:
    """
    Renders a pandas DataFrame as a plain HTML table using our own theme
    variables, instead of st.dataframe().

    st.dataframe() draws its contents on an HTML <canvas> (Streamlit's
    "Glide Data Grid" engine) rather than real DOM elements -- no CSS,
    however specific, can restyle canvas pixels. That component always
    follows Streamlit's own OS-driven theme, which is exactly why it
    showed up as a black box regardless of our light/dark toggle. This
    builds a real <table> instead, so it's just as themeable as every
    other card in the app.
    """
    header_cells = "".join(f"<th>{col}</th>" for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)

    return f"""
<div style="max-height:{max_height};overflow:auto;border:0.5px solid var(--border);border-radius:10px;">
<table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
<thead style="position:sticky;top:0;background:var(--bg-surface-raised);">
<tr>{header_cells}</tr>
</thead>
<tbody>{body_html}</tbody>
</table>
</div>
<style>
.stApp table th {{
    text-align:left;padding:8px 12px;color:var(--text-muted);
    text-transform:uppercase;font-size:0.78rem;letter-spacing:0.03em;
    border-bottom:1px solid var(--border-strong);
}}
.stApp table td {{
    padding:7px 12px;color:var(--text-primary);
    border-bottom:0.5px solid var(--border);
}}
.stApp table tr:hover td {{ background:var(--bg-surface); }}
</style>
"""


def render_html(html: str):
    st.markdown(html, unsafe_allow_html=True)


# --- Aurora Glass components ------------------------------------------------

def progress_ring_svg(percent: int, ring_color: str, size: int = 56, hole_pct: float = 0.78) -> str:
    """
    A circular progress ring built with conic-gradient (no external chart
    lib needed). `percent` is 0-100. Renders as a small inline div -- pair
    it with a value/label next to it, it doesn't include its own text.
    """
    percent = max(0, min(100, int(percent)))
    deg = int(percent / 100 * 360)
    hole = int(size * hole_pct)
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:conic-gradient({ring_color} 0deg {deg}deg, var(--border) {deg}deg 360deg);'
        f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
        f'<div class="progress-ring-hole" style="width:{hole}px;height:{hole}px;"></div>'
        f'</div>'
    )


def metric_tile(icon_class: str, value: str, label: str, icon_color: str = "var(--accent-purple)") -> str:
    """One small Aurora-style metric tile (icon + value + label), used in a 3-4 column grid."""
    return (
        f'<div class="metric-tile">'
        f'<i class="{icon_class}" style="font-size:21px;color:{icon_color};" aria-hidden="true"></i>'
        f'<div class="metric-tile-value">{value}</div>'
        f'<div class="metric-tile-label">{label}</div>'
        f'</div>'
    )


def aurora_hero_card(title: str, value_html: str, ring_html: str = "",
                      blob_colors=("aurora-blob-purple", "aurora-blob-teal")) -> str:
    """
    The frosted hero card with two soft blurred color blobs behind it.
    `value_html` is the main content (score, label, etc.) placed to the
    left; `ring_html` (typically from progress_ring_svg) sits to the right.
    """
    blob_a, blob_b = blob_colors
    ring_block = f'<div>{ring_html}</div>' if ring_html else ''
    return f"""
<div class="aurora-hero">
    <div class="aurora-blob {blob_a}" style="top:-40px;right:-30px;width:160px;height:160px;"></div>
    <div class="aurora-blob {blob_b}" style="bottom:-30px;left:-20px;width:140px;height:140px;"></div>
    <div class="aurora-content">
        <p style="font-size:0.95rem;color:var(--text-secondary) !important;margin:0 0 4px;">{title}</p>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>{value_html}</div>
            {ring_block}
        </div>
    </div>
</div>
"""