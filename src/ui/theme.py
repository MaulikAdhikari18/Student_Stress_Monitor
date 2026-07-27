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


def _sync_dark_mode_from_widget():
    """on_change callback: copies the widget's own value into our stable
    app-level 'dark_mode' key. Runs only when the user actually flips the
    toggle -- never on unrelated reruns (nav clicks, etc)."""
    st.session_state["dark_mode"] = st.session_state["_dark_mode_widget"]


def toggle_control():
    """
    Renders the single, site-wide dark/light toggle.

    Uses a DIFFERENT internal widget key ("_dark_mode_widget") than the
    stable app-level key ("dark_mode") that the rest of the app reads via
    is_dark_mode()/current_theme(). The widget is always given an explicit
    value= (read from the stable key) rather than relying on Streamlit's
    implicit key-based restoration, and an on_change callback copies the
    widget's value into the stable key only when the user actually
    interacts with it.

    This replaces an earlier version that bound the widget directly via
    key="dark_mode" and relied on implicit restoration. That was
    confirmed (via OCR'd screen recording, not just visual inspection --
    an earlier reading of the same footage was mistakenly read as "stays
    True" when it actually showed a flip to False) to actually reset the
    stored value to False on ordinary page navigation, not just a
    rendering glitch. The explicit value=/on_change pattern below is
    Streamlit's documented-safe approach for exactly this failure mode:
    the widget's displayed state is always explicitly seeded from OUR
    value every render, rather than trusting Streamlit to correctly
    restore it on its own.
    """
    st.session_state.setdefault("dark_mode", True)
    st.toggle(
        "Dark mode",
        value=st.session_state["dark_mode"],
        key="_dark_mode_widget",
        on_change=_sync_dark_mode_from_widget,
        help="Site-wide theme -- applies to every page",
    )
    st.caption("🌙 Dark" if is_dark_mode() else "☀️ Light")