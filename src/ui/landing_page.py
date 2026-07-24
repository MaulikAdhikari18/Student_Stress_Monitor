"""The full-screen marketing/hero page shown before the user signs in."""

import streamlit as st

from src.ui.styles import inject_landing_css


def show_landing_page():
    inject_landing_css()

    st.markdown("""
    <div class="landing-hero">
        <div class="landing-logo">🧠</div>
        <div class="landing-title">Student Stress Monitor</div>
        <div class="landing-sub">
            An AI-powered daily wellness companion that tracks your stress,
            predicts burnout before it happens, and helps you study smarter.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-val">14</div>
            <div class="stat-lbl">Daily inputs tracked</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">81.5%</div>
            <div class="stat-lbl">Model accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">4</div>
            <div class="stat-lbl">Stress levels predicted</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">6</div>
            <div class="stat-lbl">Dashboard pages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📅</div>
            <div class="feature-title">Calendar Logging</div>
            <div class="feature-desc">Log any day directly from a colour-coded stress calendar. See your entire month at a glance.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">ML Prediction</div>
            <div class="feature-desc">GradBoost classifier predicts Low / Moderate / High / Critical with confidence probabilities.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Smart Recommendations</div>
            <div class="feature-desc">Severity-graded action cards sorted by urgency — Critical issues always appear first.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Rich Analytics</div>
            <div class="feature-desc">5 interactive Plotly charts — stress trends, sleep vs study, lifestyle heatmap, and more.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Weekly & Monthly Summary</div>
            <div class="feature-desc">7-day heatmap, sparklines, best/worst day callouts, and metric pass/fail grid.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⏱️</div>
            <div class="feature-title">Study Timer</div>
            <div class="feature-desc">Pomodoro timer with stress-adaptive block lengths and session logging built right in.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tech-bar">
        <span class="tech-pill">🐍 Python</span>
        <span class="tech-pill">⚡ Streamlit</span>
        <span class="tech-pill">🤖 scikit-learn</span>
        <span class="tech-pill">📈 Plotly</span>
        <span class="tech-pill">🗃️ SQLite</span>
        <span class="tech-pill">🐼 pandas</span>
    </div>
    """, unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([1.5, 1, 1, 1.5])
    with c1:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state["show_landing"] = False
            st.rerun()
    with c2:
        if st.button("🔑 Sign In", use_container_width=True):
            st.session_state["show_landing"] = False
            st.rerun()

    st.markdown(
        '<div class="cta-note" style="text-align:center;">'
        '🔒 Free · Private · All data stored locally on your device'
        '</div>', unsafe_allow_html=True)
