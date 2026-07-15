"""
Theme tokens for light/dark mode.

Each mode is a dict of CSS custom-property values. styles.py turns these
into a `:root { --name: value; }` block; every other CSS rule and every
inline HTML snippet in components.py / pages/*.py should reference
`var(--name)` rather than hardcoding a color, so the whole app repaints
when the mode is switched.

Semantic stress-level colors (Low/Moderate/High/Critical, see config.COLORS)
are intentionally NOT part of this token set -- they carry meaning (danger,
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
    "button-bg": "rgba(255,255,255,0.06)",
    "button-bg-hover": "rgba(255,255,255,0.12)",
    "button-text": "#F2F1FA",
}

# Softer than a stark white: warm light-gray page, gently off-white cards,
# muted (not neon) accents so nothing reads as glaring.
LIGHT = {
    "bg-page": "#EDEBE3",
    "bg-surface": "#F8F7F2",
    "bg-surface-raised": "#FFFFFF",
    "border": "#D9D6C9",
    "border-strong": "#C6C2B2",
    "text-primary": "#2B2A26",
    "text-secondary": "#5C5A52",
    "text-muted": "#847F72",
    "accent-purple": "#5B4FBF",
    "accent-purple-strong": "#453C94",
    "accent-pink": "#A03A5D",
    "accent-teal": "#0F6E56",
    "blob-purple": "rgba(91,79,191,0.16)",
    "blob-teal": "rgba(15,110,86,0.14)",
    "card-shadow": "0 1px 4px rgba(43,42,38,0.08)",
    "button-bg": "#F8F7F2",
    "button-bg-hover": "#FFFFFF",
    "button-text": "#2B2A26",
}


def is_dark_mode() -> bool:
    """Defaults to dark, matching the app's original look."""
    return st.session_state.get("dark_mode", True)


def current_theme() -> dict:
    return DARK if is_dark_mode() else LIGHT


def toggle_control():
    """
    Renders the single, site-wide dark/light toggle. Bound to
    st.session_state['dark_mode'] via `key`, so it's one shared value
    across every page -- switching pages never resets it. Call once,
    in the navbar.
    """
    st.session_state.setdefault("dark_mode", True)
    label = "🌙 Dark" if is_dark_mode() else "☀️ Light"
    st.toggle(label, key="dark_mode", help="Site-wide theme -- applies to every page")