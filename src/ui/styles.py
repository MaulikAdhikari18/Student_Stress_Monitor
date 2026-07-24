"""
All CSS for the app, as a single function to call once per page load.
Previously this was a ~150-line inline st.markdown(...) block sitting at
the top of dashboard.py above all the actual logic. Now it's just:

    from src.ui.styles import inject_global_css
    inject_global_css()
"""

import streamlit as st

_CSS = """
<style>
    .main-title {
        font-size:2rem; font-weight:700;
        background:linear-gradient(135deg,#534AB7,#D4537E);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .auth-title { font-size:1.4rem; font-weight:700; color:#534AB7; margin-bottom:0.2rem; }
    .auth-sub   { font-size:0.88rem; color:#888; margin-bottom:1.4rem; }
    .stress-box { padding:1rem 1.4rem; border-radius:12px; border-left:6px solid; margin-bottom:1rem; }
    .box-low      { background:rgba(99,153,34,0.15);  border-color:#639922; color:inherit; }
    .box-moderate { background:rgba(186,117,23,0.15); border-color:#BA7517; color:inherit; }
    .box-high     { background:rgba(153,60,29,0.15);  border-color:#993C1D; color:inherit; }
    .box-critical { background:rgba(163,45,45,0.15);  border-color:#A32D2D; color:inherit; }
    .tip-box {
        background:rgba(83,74,183,0.12);
        border-left:4px solid #7F77DD;
        border-radius:0 8px 8px 0;
        padding:0.75rem 1rem;
        margin-bottom:0.5rem;
        font-size:0.92rem;
        color:inherit;
    }
    .tip-box strong { color:#AFA9EC; }
    .goal-card {
        background:rgba(255,255,255,0.05);
        border:0.5px solid rgba(255,255,255,0.12);
        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem;
        color:inherit;
    }
    .goal-title { font-size:0.85rem; font-weight:600; color:inherit; opacity:0.75; margin-bottom:6px; }
    .streak-badge {
        display:inline-block; padding:2px 10px; border-radius:20px;
        font-size:0.78rem; font-weight:600;
        background:rgba(99,153,34,0.2); color:#97C459; margin-left:8px;
    }
    .streak-zero { background:rgba(128,128,128,0.15); color:#888; }
    .user-chip {
        display:inline-block; background:rgba(83,74,183,0.2); color:#AFA9EC;
        border-radius:20px; padding:3px 12px; font-size:0.85rem; font-weight:600;
    }
    .day-card {
        border:0.5px solid rgba(255,255,255,0.12);
        border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.6rem;
        background:rgba(255,255,255,0.04);
    }
    .day-header {
        font-size:0.85rem; font-weight:600; color:#AFA9EC;
        margin-bottom:0.5rem; letter-spacing:0.03em;
    }
    .task-row {
        display:flex; align-items:center; gap:8px;
        padding:5px 0; border-bottom:0.5px solid rgba(255,255,255,0.06);
        font-size:0.88rem;
    }
    .task-row:last-child { border-bottom:none; }
    .pri-high   { background:rgba(163,45,45,0.2);   color:#F09595; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .pri-medium { background:rgba(186,117,23,0.2);  color:#FAC775; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .pri-low    { background:rgba(99,153,34,0.2);   color:#C0DD97; border-radius:4px; padding:1px 7px; font-size:0.75rem; font-weight:600; }
    .break-box {
        background:rgba(83,74,183,0.12); border-left:4px solid #7F77DD;
        border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; margin-bottom:1rem;
    }
    .break-stat { font-size:2rem; font-weight:700; color:#AFA9EC; display:inline-block; margin-right:0.5rem; }
    /* ── Timer styles ── */
    .timer-container {
        background: linear-gradient(135deg, rgba(83,74,183,0.18), rgba(212,83,126,0.12));
        border: 1px solid rgba(175,169,236,0.3);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .timer-display {
        font-size: 4.5rem;
        font-weight: 800;
        font-family: 'Courier New', monospace;
        background: linear-gradient(135deg, #AFA9EC, #D4537E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.05em;
        line-height: 1;
    }
    .timer-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #AFA9EC;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.4rem;
        opacity: 0.8;
    }
    .timer-phase-study { border-top: 3px solid #639922; }
    .timer-phase-break { border-top: 3px solid #534AB7; }
    .session-log-row {
        display: flex; align-items: center; gap: 10px;
        padding: 6px 10px; border-radius: 8px;
        background: rgba(255,255,255,0.04);
        margin-bottom: 4px; font-size: 0.84rem;
    }
    /* ── Advanced recommendation styles ── */
    .rec-card {
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        position: relative;
        overflow: hidden;
    }
    .rec-critical { background: rgba(163,45,45,0.18); border-left: 5px solid #E24B4A; }
    .rec-high     { background: rgba(186,117,23,0.18); border-left: 5px solid #FAC775; }
    .rec-moderate { background: rgba(83,74,183,0.15);  border-left: 5px solid #7F77DD; }
    .rec-positive { background: rgba(99,153,34,0.15);  border-left: 5px solid #97C459; }
    .rec-header   { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
    .rec-badge {
        font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge-critical { background: rgba(226,75,74,0.3); color: #F09595; }
    .badge-high     { background: rgba(250,199,117,0.3); color: #FAC775; }
    .badge-moderate { background: rgba(127,119,221,0.3); color: #AFA9EC; }
    .badge-positive { background: rgba(151,196,89,0.3); color: #C0DD97; }
    .rec-title  { font-size: 0.95rem; font-weight: 700; }
    .rec-body   { font-size: 0.87rem; opacity: 0.85; line-height: 1.55; margin: 0; }
    .rec-action {
        margin-top: 8px; padding: 5px 10px; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600;
        background: rgba(255,255,255,0.08);
        display: inline-block; opacity: 0.9;
    }
    .rec-score-bar {
        height: 4px; border-radius: 2px; margin-top: 10px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(83,74,183,0.2), rgba(212,83,126,0.15));
        border: 1px solid rgba(175,169,236,0.3);
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .insight-title { font-size: 1rem; font-weight: 700; color: #AFA9EC; margin-bottom: 0.4rem; }
    .insight-body  { font-size: 0.88rem; opacity: 0.85; line-height: 1.6; }
</style>
"""

_LANDING_CSS = """
<style>
/* Hide streamlit default chrome on landing */
#MainMenu, footer, header { visibility: hidden; }

.landing-hero { text-align: center; padding: 3rem 1rem 2rem; }
.landing-logo { font-size: 5rem; line-height: 1; margin-bottom: 0.5rem; }
.landing-title {
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, #AFA9EC, #D4537E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin-bottom: 0.4rem;
}
.landing-sub {
    font-size: 1.15rem; color: #aaa; margin-bottom: 2.5rem;
    max-width: 520px; margin-left: auto; margin-right: auto; line-height: 1.6;
}
.feature-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    max-width: 860px; margin: 0 auto 2.5rem;
}
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(175,169,236,0.18);
    border-radius: 16px; padding: 1.3rem 1.1rem; text-align: left;
}
.feature-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.feature-title { font-size: 0.95rem; font-weight: 700; color: #AFA9EC; margin-bottom: 0.3rem; }
.feature-desc  { font-size: 0.82rem; color: #888; line-height: 1.5; }
.stat-row { display: flex; justify-content: center; gap: 40px; margin-bottom: 2.5rem; }
.stat-item { text-align: center; }
.stat-val  { font-size: 2rem; font-weight: 800; color: #AFA9EC; line-height: 1; }
.stat-lbl  { font-size: 0.78rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.tech-bar { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 2rem; }
.tech-pill {
    background: rgba(83,74,183,0.15); border: 1px solid rgba(175,169,236,0.25);
    border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #AFA9EC; font-weight: 600;
}
.cta-note { font-size: 0.8rem; color: #555; margin-top: 0.5rem; }
</style>
"""

_ENTRY_SECTION_CSS = """
<style>
.entry-section {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.8rem;
}
.entry-section-title {
    font-size:0.8rem; font-weight:700; color:#AFA9EC;
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;
}
</style>
"""

_CALENDAR_CSS = """
<style>
.ssm-cal{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;margin-top:6px;}
.ssm-ch{text-align:center;font-size:0.68rem;font-weight:700;color:#555;
         padding:3px 0;text-transform:uppercase;letter-spacing:0.04em;}
.ssm-cd{border-radius:10px;padding:5px 3px 4px;text-align:center;
         font-size:0.8rem;font-weight:600;border:2px solid transparent;
         min-height:48px;display:flex;flex-direction:column;
         align-items:center;justify-content:center;gap:2px;
         transition:transform .12s;}
.ssm-cd:hover{transform:scale(1.06);}
.ssm-num{font-size:0.84rem;line-height:1;}
.ssm-sc{font-size:0.6rem;opacity:0.8;line-height:1;}
.ssm-dot{width:5px;height:5px;border-radius:50%;}
</style>
"""


def inject_global_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def inject_landing_css():
    st.markdown(_LANDING_CSS, unsafe_allow_html=True)


def inject_entry_section_css():
    st.markdown(_ENTRY_SECTION_CSS, unsafe_allow_html=True)


def inject_calendar_css():
    st.markdown(_CALENDAR_CSS, unsafe_allow_html=True)
