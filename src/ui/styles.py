"""
All CSS for the app, theme-aware.

Every color in every rule below is a var(--token) reference resolved from
src/ui/theme.py's DARK or LIGHT dict -- nothing here is a hardcoded hex or
rgba() anymore. inject_global_css() builds the :root block from the
active theme and must be called on every page load (it already is, from
app.py) so a mode switch repaints instantly on the next rerun.

IMPORTANT: Streamlit ships its own built-in theme that auto-follows the
browser/OS dark-mode preference, completely independent of this file.
Without the _WIDGET_OVERRIDES block below, our custom CSS only recolors
the divs *we* build (cards, tiles) -- every native widget (buttons, tabs,
progress bars, metrics, captions, inputs) stays on Streamlit's own theme.
That mismatch is what caused white-on-white invisible text when a user's
OS was in dark mode but our toggle was set to light. _WIDGET_OVERRIDES
forces every native widget to also read from our CSS variables.

FONT / SIZE NOTE: _WIDGET_OVERRIDES also loads a webfont and bumps the
root <html> font-size from the browser default (16px) to 17px. Since most
of the small text throughout the app uses rem units (relative to root
font-size), that one change scales *all* of it up ~6% app-wide, on top of
the specific class-level bumps below for the smallest labels/badges.
"""

import streamlit as st
from src.ui.theme import current_theme


def _root_vars(theme: dict) -> str:
    """
    !important on custom-property declarations is valid CSS and follows
    normal !important cascade rules: if two competing `:root` blocks set
    the same --token, the one with !important wins regardless of DOM
    order, unless the other side also uses !important.

    This replaces an earlier, broken attempt at this same fix that used a
    <script> tag to set these variables via JS -- confirmed via Streamlit's
    own GitHub issues that st.markdown(unsafe_allow_html=True) does not
    execute <script> tags at all, so that "fix" was silently inert and
    never actually did anything. This is a real, testable CSS-only fix
    for the same underlying problem: our theme flipping to light on
    navigation even though st.session_state['dark_mode'] was confirmed
    (via screen recording) to never change -- meaning some other
    stylesheet was overriding ours after some reruns.
    """
    lines = "\n".join(f"  --{k}: {v} !important;" for k, v in theme.items())
    return f":root {{\n{lines}\n}}"





_TABLER_ICONS_LINK = (
    '<link rel="stylesheet" '
    'href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">'
)

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
)

_WIDGET_OVERRIDES = """
/* ── Font: Plus Jakarta Sans everywhere, plus a slightly larger base size ──
   NOTE: deliberately NOT using a blanket [data-testid] or [class*="css"]
   selector here. Streamlit renders its built-in icons (password show/hide,
   checkmarks, etc.) as text nodes containing the icon's *name* (e.g. the
   literal word "visibility"), displayed as a glyph only because a special
   Material Symbols icon font is applied to that element. A blanket
   font-family override with !important was clobbering that icon font,
   which is why the password field showed the literal word "visibility"
   instead of an eye icon. Scoping this to real text elements (and adding
   an explicit rescue rule for the icon elements below) fixes that. */
html { font-size: 17px; }
html, body, .stApp,
p, li, span, div, label, button, input, textarea, select,
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
span[translate="no"],
.material-icons,
.material-symbols-outlined,
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    line-height: 1.55 !important;
}

/* ── Force Streamlit's native (OS-theme-driven) widgets onto our palette ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg-page) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    color: var(--text-primary) !important;
}
[data-testid="stCaptionContainer"] p { color: var(--text-muted) !important; font-size: 0.9rem !important; }
[data-testid="stWidgetLabel"] p { color: var(--text-primary) !important; font-size: 0.95rem !important; }

[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
[data-testid="stMetricDelta"] { color: var(--text-secondary) !important; }

/* Buttons -- this is what fixes the black nav buttons */
.stButton > button {
    background: var(--button-bg) !important;
    color: var(--button-text) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.95rem !important;
}
.stButton > button:hover {
    background: var(--button-bg-hover) !important;
    border-color: var(--border-strong) !important;
    color: var(--accent-purple) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)) !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"] p { color: #FFFFFF !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: var(--text-secondary) !important; }
.stTabs [data-baseweb="tab"] p { color: var(--text-secondary) !important; font-size: 0.95rem !important; }
.stTabs [aria-selected="true"] { color: var(--accent-purple) !important; }
.stTabs [aria-selected="true"] p { color: var(--accent-purple) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent-purple) !important; }
.stTabs [data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* Progress bar -- fixes the invisible confidence-bar labels */
[data-testid="stProgress"] > div > div { background: var(--border) !important; }
[data-testid="stProgress"] > div > div > div { background: var(--accent-purple) !important; }

/* Text / number / date inputs, selects, textareas */
.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    font-size: 0.95rem !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}

/* Expander */
[data-testid="stExpander"] {
    border-color: var(--border) !important;
    background: var(--bg-surface) !important;
}

/* Our own theme toggle -- give it a clearly "global setting" look */
[data-testid="stWidgetLabel"]:has(+ div [data-baseweb="switch"]) p,
.stToggle p {
    font-weight: 600 !important;
}
"""


def inject_global_css():
    st.markdown(_TABLER_ICONS_LINK, unsafe_allow_html=True)
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    theme = current_theme()
    css = f"""
<style>
{_root_vars(theme)}

.stApp {{ background: var(--bg-page); }}

.main-title {{
    font-size:2rem; font-weight:700;
    background:linear-gradient(135deg,var(--accent-purple),var(--accent-pink));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.auth-title {{ font-size:1.5rem; font-weight:700; color:var(--accent-purple); margin-bottom:0.25rem; }}
.auth-sub   {{ font-size:0.95rem; color:var(--text-muted); margin-bottom:1.4rem; }}

.section-header {{ font-size:1.1rem; font-weight:600; color:var(--text-primary); margin-bottom:0.5rem; }}

/* ── Aurora Glass cards ── */
.aurora-hero {{
    position:relative; overflow:hidden; border-radius:16px; padding:1.25rem;
    background:var(--bg-surface); border:0.5px solid var(--border);
    box-shadow:var(--card-shadow);
}}
.aurora-blob {{ position:absolute; border-radius:50%; filter:blur(30px); pointer-events:none; z-index:0; }}
.aurora-blob-purple {{ background:var(--blob-purple); }}
.aurora-blob-teal {{ background:var(--blob-teal); }}
.aurora-content {{ position:relative; z-index:1; }}

.metric-tile {{
    background:var(--bg-surface); border:0.5px solid var(--border);
    border-radius:14px; padding:0.9rem; text-align:center;
    box-shadow:var(--card-shadow);
}}
.metric-tile-label {{ font-size:0.78rem; color:var(--text-muted); margin:4px 0 0; }}
.metric-tile-value {{ font-size:1.15rem; font-weight:600; margin:8px 0 0; color:var(--text-primary); }}

.progress-ring-hole {{ background:var(--bg-surface); border-radius:50%; }}

/* ── Generic boxes (still used on several pages) ── */
.stress-box {{ padding:1rem 1.4rem; border-radius:12px; border-left:6px solid; margin-bottom:1rem; background:var(--bg-surface); color:var(--text-primary); }}
.box-low      {{ border-color:#639922; }}
.box-moderate {{ border-color:#BA7517; }}
.box-high     {{ border-color:#993C1D; }}
.box-critical {{ border-color:#A32D2D; }}

.tip-box {{
    background:var(--bg-surface); border-left:4px solid var(--accent-purple-strong);
    border-radius:0 8px 8px 0; padding:0.8rem 1.1rem; margin-bottom:0.5rem;
    font-size:0.98rem; color:var(--text-primary);
}}
.tip-box strong {{ color:var(--accent-purple); }}

.goal-card {{
    background:var(--bg-surface); border:0.5px solid var(--border);
    border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem; color:var(--text-primary);
}}
.goal-title {{ font-size:0.95rem; font-weight:600; color:var(--text-primary); opacity:0.85; margin-bottom:6px; }}

.streak-badge {{
    display:inline-block; padding:3px 11px; border-radius:20px;
    font-size:0.85rem; font-weight:600;
    background:rgba(99,153,34,0.2); color:#97C459; margin-left:8px;
}}
.streak-zero {{ background:var(--bg-surface); color:var(--text-muted); }}

.user-chip {{
    display:inline-block; background:var(--bg-surface); color:var(--accent-purple);
    border:0.5px solid var(--border); border-radius:20px; padding:4px 13px;
    font-size:0.92rem; font-weight:600;
}}

.day-card {{
    border:0.5px solid var(--border); border-radius:12px;
    padding:0.8rem 1rem; margin-bottom:0.6rem; background:var(--bg-surface);
}}
.day-header {{ font-size:0.92rem; font-weight:600; color:var(--accent-purple); margin-bottom:0.5rem; letter-spacing:0.02em; }}

.task-row {{
    display:flex; align-items:center; gap:8px; padding:6px 0;
    border-bottom:0.5px solid var(--border); font-size:0.95rem; color:var(--text-primary);
}}
.task-row:last-child {{ border-bottom:none; }}
.pri-high   {{ background:rgba(163,45,45,0.2);  color:#F09595; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }}
.pri-medium {{ background:rgba(186,117,23,0.2); color:#FAC775; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }}
.pri-low    {{ background:rgba(99,153,34,0.2);  color:#C0DD97; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }}

.break-box {{
    background:var(--bg-surface); border-left:4px solid var(--accent-purple-strong);
    border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; margin-bottom:1rem; color:var(--text-primary);
}}
.break-stat {{ font-size:2rem; font-weight:700; color:var(--accent-purple); display:inline-block; margin-right:0.5rem; }}

.timer-container {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:20px; padding:2rem 1.5rem; text-align:center; margin-bottom:1.2rem;
}}
.timer-display {{
    font-size:4.5rem; font-weight:800; font-family:'Courier New', monospace;
    background:linear-gradient(135deg,var(--accent-purple),var(--accent-pink));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:0.05em; line-height:1;
}}
.timer-label {{
    font-size:0.92rem; font-weight:600; color:var(--accent-purple);
    text-transform:uppercase; letter-spacing:0.08em; margin-top:0.4rem; opacity:0.9;
}}
.timer-phase-study {{ border-top:3px solid #639922; }}
.timer-phase-break {{ border-top:3px solid var(--accent-purple-strong); }}
.session-log-row {{
    display:flex; align-items:center; gap:10px; padding:6px 10px; border-radius:8px;
    background:var(--bg-surface); margin-bottom:4px; font-size:0.9rem; color:var(--text-primary);
}}

.rec-card {{ border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.75rem; background:var(--bg-surface); border:0.5px solid var(--border); }}
.rec-critical {{ border-left:5px solid #E24B4A; }}
.rec-high     {{ border-left:5px solid #FAC775; }}
.rec-moderate {{ border-left:5px solid var(--accent-purple-strong); }}
.rec-positive {{ border-left:5px solid #97C459; }}
.rec-header   {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
.rec-badge {{ font-size:0.78rem; font-weight:700; padding:3px 9px; border-radius:20px; text-transform:uppercase; letter-spacing:0.04em; }}
.badge-critical {{ background:rgba(226,75,74,0.3); color:#F09595; }}
.badge-high     {{ background:rgba(250,199,117,0.3); color:#FAC775; }}
.badge-moderate {{ background:rgba(127,119,221,0.3); color:var(--accent-purple); }}
.badge-positive {{ background:rgba(151,196,89,0.3); color:#C0DD97; }}
.rec-title {{ font-size:1.02rem; font-weight:700; color:var(--text-primary); }}
.rec-body  {{ font-size:0.93rem; opacity:0.9; line-height:1.55; margin:0; color:var(--text-primary); }}
.rec-action {{ margin-top:8px; padding:6px 11px; border-radius:6px; font-size:0.86rem; font-weight:600; background:var(--bg-surface-raised); display:inline-block; color:var(--text-primary); }}
.rec-score-bar {{ height:4px; border-radius:2px; margin-top:10px; background:var(--border); overflow:hidden; }}

.insight-box {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem;
}}
.insight-title {{ font-size:1.1rem; font-weight:700; color:var(--accent-purple); margin-bottom:0.4rem; }}
.insight-body  {{ font-size:0.95rem; opacity:0.9; line-height:1.6; color:var(--text-primary); }}

{_WIDGET_OVERRIDES}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def inject_landing_css():
    st.markdown(_TABLER_ICONS_LINK, unsafe_allow_html=True)
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    theme = current_theme()
    css = f"""
<style>
{_root_vars(theme)}
.stApp {{ background: var(--bg-page); }}
#MainMenu, footer, header {{ visibility: hidden; }}

.landing-hero {{ text-align: center; padding: 3rem 1rem 2rem; }}
.landing-logo {{ font-size: 5rem; line-height: 1; margin-bottom: 0.5rem; }}
.landing-title {{
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin-bottom: 0.4rem;
}}
.landing-sub {{
    font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 2.5rem;
    max-width: 540px; margin-left: auto; margin-right: auto; line-height: 1.6;
}}
.feature-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    max-width: 880px; margin: 0 auto 2.5rem;
}}
.feature-card {{
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.3rem 1.1rem; text-align: left;
}}
.feature-icon {{ font-size: 1.9rem; margin-bottom: 0.5rem; }}
.feature-title {{ font-size: 1.02rem; font-weight: 700; color: var(--accent-purple); margin-bottom: 0.3rem; }}
.feature-desc  {{ font-size: 0.9rem; color: var(--text-muted); line-height: 1.5; }}
.stat-row {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 2.5rem; }}
.stat-item {{ text-align: center; }}
.stat-val  {{ font-size: 2.1rem; font-weight: 800; color: var(--accent-purple); line-height: 1; }}
.stat-lbl  {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
.tech-bar {{ display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 2rem; }}
.tech-pill {{
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 5px 15px; font-size: 0.88rem; color: var(--accent-purple); font-weight: 600;
}}
.cta-note {{ font-size: 0.88rem; color: var(--text-muted); margin-top: 0.5rem; }}

{_WIDGET_OVERRIDES}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def inject_entry_section_css():
    st.markdown(f"""
<style>
.entry-section {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.8rem;
}}
.entry-section-title {{
    font-size:0.88rem; font-weight:700; color:var(--accent-purple);
    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem;
}}
</style>
""", unsafe_allow_html=True)


def inject_calendar_css():
    st.markdown("""
<style>
.ssm-cal{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;margin-top:6px;}
.ssm-ch{text-align:center;font-size:0.78rem;font-weight:700;color:var(--text-muted);
         padding:3px 0;text-transform:uppercase;letter-spacing:0.03em;}
.ssm-cd{border-radius:10px;padding:5px 3px 4px;text-align:center;
         font-size:0.9rem;font-weight:600;border:2px solid transparent;
         min-height:48px;display:flex;flex-direction:column;
         align-items:center;justify-content:center;gap:2px;
         transition:transform .12s; color:var(--text-primary);}
.ssm-cd:hover{transform:scale(1.06);}
.ssm-num{font-size:0.95rem;line-height:1;}
.ssm-sc{font-size:0.72rem;opacity:0.85;line-height:1;}
.ssm-dot{width:5px;height:5px;border-radius:50%;}
</style>
""", unsafe_allow_html=True)