"""
Theme tokens for light/dark mode.

Each mode is a dict of CSS custom-property values. styles.py turns these
into a `:root { --name: value; }` block; every other CSS rule and every
inline HTML snippet in components.py / pages/*.py should reference
`var(--name)` rather than hardcoding a color, so the whole app repaints
when the mode is switched.

Semantic stress-level colors (Low/Moderate/High/Critical, see config.COLORS)
are intentionally NOT part of this token set — they carry meaning (danger,
success, etc.) and stay the same saturated hex across both modes, same as
before.
"""

import streamlit as st

DARK = {
    "bg-page": "#0E0E14",
    "bg-surface": "rgba(255,255,255,0.04)",
    "bg-surface-raised": "rgba(255,255,255,0.06)",
    "border": "rgba(255,255,255,0.10)",
    "border-strong": "rgba(255,255,255,0.16)",
    "text-primary": "#F2F1FA",
    "text-secondary": "#B9B7C9",
    "text-muted": "#79778A",
    "accent-purple": "#AFA9EC",
    "accent-purple-strong": "#7F77DD",
    "accent-pink": "#D4537E",
    "accent-teal": "#5DCAA5",
    "blob-purple": "rgba(175,169,236,0.25)",
    "blob-teal": "rgba(93,202,165,0.20)",
    "card-shadow": "none",
}

LIGHT = {
    "bg-page": "#FAFAF8",
    "bg-surface": "#FFFFFF",
    "bg-surface-raised": "#FFFFFF",
    "border": "#E5E3DC",
    "border-strong": "#D3D1C7",
    "text-primary": "#1A1A2E",
    "text-secondary": "#5F5E5A",
    "text-muted": "#888780",
    "accent-purple": "#534AB7",
    "accent-purple-strong": "#3C3489",
    "accent-pink": "#993556",
    "accent-teal": "#0F6E56",
    "blob-purple": "rgba(175,169,236,0.35)",
    "blob-teal": "rgba(93,202,165,0.30)",
    "card-shadow": "0 1px 3px rgba(0,0,0,0.06)",
}


def is_dark_mode() -> bool:
    """Defaults to dark, matching the app's original look."""
    return st.session_state.get("dark_mode", True)


def current_theme() -> dict:
    return DARK if is_dark_mode() else LIGHT


def toggle_control():
    """
    Renders the dark/light toggle. Call once, in the navbar. Streamlit
    binds the widget directly to st.session_state['dark_mode'] via `key`,
    so no manual wiring is needed elsewhere — just read is_dark_mode()
    or current_theme() anywhere you need the active palette.
    """
    st.session_state.setdefault("dark_mode", True)
    st.toggle("🌙", key="dark_mode", help="Toggle dark / light mode")