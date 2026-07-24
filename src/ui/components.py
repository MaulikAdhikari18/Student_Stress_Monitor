"""
Small HTML-snippet builder functions that were previously redefined inline
(sometimes multiple times) inside dashboard.py's page-rendering code:
num_card, summary_card, level_badge, and the calendar day cell colors.
"""

import streamlit as st
from src.config import COLORS as LEVEL_COLOR


def num_card(label: str, value: str, sub: str = "", color: str = "#AFA9EC",
             border_color: str | None = None) -> str:
    """Small stat card used on the History page (numbers row)."""
    bc = border_color or color
    return (
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid {bc}44;'
        f'border-top:3px solid {bc};border-radius:12px;padding:0.8rem 1rem;">'
        f'<div style="font-size:0.72rem;color:#666;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{color};line-height:1;">{value}</div>'
        f'<div style="font-size:0.75rem;color:#666;margin-top:3px;">{sub}</div>'
        f'</div>'
    )


def summary_card(label: str, value: str, sub: str, color: str = "#AFA9EC") -> str:
    """Slightly larger stat card used on the Summary page."""
    return (
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);'
        f'border-radius:14px;padding:0.9rem 1.1rem;">'
        f'<div style="font-size:0.72rem;color:#666;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>'
        f'<div style="font-size:1.7rem;font-weight:800;color:{color};line-height:1.1;margin:4px 0 2px;">{value}</div>'
        f'<div style="font-size:0.78rem;color:#666;">{sub}</div>'
        f'</div>'
    )


def level_badge(level: str) -> str:
    c = LEVEL_COLOR.get(level, "#888")
    return f'<span style="background:{c}33;color:{c};border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:700;">{level}</span>'


def stress_bg(score) -> str:
    """Background tint for a calendar day cell, based on that day's stress score."""
    if score is None:
        return "rgba(255,255,255,0.04)"
    if score >= 90:
        return "rgba(163,45,45,0.55)"
    if score >= 75:
        return "rgba(153,60,29,0.50)"
    if score >= 30:
        return "rgba(186,117,23,0.45)"
    return "rgba(99,153,34,0.45)"


def stress_border(score) -> str:
    """Border/accent color for a calendar day cell, based on that day's stress score."""
    if score is None:
        return "rgba(255,255,255,0.1)"
    if score >= 90:
        return "#E24B4A"
    if score >= 75:
        return "#D85A30"
    if score >= 30:
        return "#EF9F27"
    return "#639922"


def render_html(html: str):
    """Thin wrapper so pages don't repeat `unsafe_allow_html=True` everywhere."""
    st.markdown(html, unsafe_allow_html=True)
