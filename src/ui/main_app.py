"""
Main app router — replaces the giant show_main_app() from dashboard.py.

Responsibilities:
  1. Render the top navbar and handle page switching (st.session_state["page"])
  2. Compute the "current" stress prediction shared across pages, from
     either the New Entry page's live inputs or the latest saved session
  3. Dispatch to the right page module in src/ui/pages/
"""

import numpy as np
import streamlit as st

from src.config import LABELS, COLORS, EMOJIS
from src.db.sessions import load_sessions
from src.db.goals import load_goals
from src.ml.model import load_model, predict_stress
from src.ml.scoring import daily_exercise_to_weekly
from src.config import classify_score
from src.ui.components import render_html
from src.ui.theme import toggle_control

from src.ui.pages import (
    entry_page,
    dashboard_page,
    history_page,
    summary_page,
    goals_page,
    planner_page,
)

NAV_PAGES = [
    ("dashboard", "📊 Dashboard"),
    ("entry", "✏️ New Entry"),
    ("summary", "📋 Summary"),
    ("history", "📈 History"),
    ("goals", "🎯 Goals"),
    ("planner", "📅 Planner"),
]


def _render_navbar(username: str):
    n1, n2, n3, n4, n5, n6, nt, nr = st.columns([1.2, 1.1, 1, 1, 1, 1, 0.5, 1.1])
    for col, (pg_key, pg_label) in zip([n1, n2, n3, n4, n5, n6], NAV_PAGES):
        with col:
            is_active = st.session_state["page"] == pg_key
            if st.button(pg_label, use_container_width=True,
                         type="primary" if is_active else "secondary",
                         key=f"nav_{pg_key}"):
                st.session_state["page"] = pg_key
                st.rerun()
    with nt:
        toggle_control()
    with nr:
        render_html(
            f'<div style="text-align:right;padding-top:2px;">'
            f'<span class="user-chip">👤 {username}</span></div>')
        if st.button("Sign out", use_container_width=True, key="so_btn"):
            del st.session_state["user"]
            st.rerun()

    st.markdown("<hr style='margin:0.5rem 0 1rem;border-color:rgba(255,255,255,0.07);'>",
                unsafe_allow_html=True)


def _latest_session_inputs(history_df):
    """
    Shared values for dashboard/goals/planner pages, taken from the most
    recent logged session. Mirrors dashboard.py's original defaults for
    the fields the sessions table doesn't store (assignments, exam, etc.)
    — those stay at fixed placeholder values, same as before.
    """
    if not history_df.empty:
        latest = history_df.iloc[-1]
        study = float(latest.get("study", 6))
        sleep = float(latest.get("sleep", 7))
        screen = float(latest.get("screen", 4))
        anxiety = int(latest.get("anxiety", 4))
        exercise = int(latest.get("exercise", 3))
        assignments, exam, performance = 3, 5, 7
        social, finance, family, peer, extra, rel = 5, 3, 7, 4, 0, 0
        import pandas as pd
        raw_score = pd.to_numeric(latest.get("stress_score", None), errors="coerce")
        stress_score = int(raw_score) if pd.notna(raw_score) else 30
        level_name = str(latest.get("stress_level", "Low"))
    else:
        study, sleep, screen, anxiety, exercise, assignments = 6, 7, 4, 4, 3, 3
        exam, performance, social, finance, family, peer, extra, rel = 5, 7, 5, 3, 7, 4, 0, 0
        stress_score, level_name = 0, "Low"

    return dict(
        study=study, sleep=sleep, screen=screen, anxiety=anxiety, exercise=exercise,
        assignments=assignments, exam=exam, performance=performance, social=social,
        finance=finance, family=family, peer=peer, extra=extra, rel=rel,
        stress_score=stress_score, level_name=level_name,
    )


def show_main_app(user: dict):
    model, scaler, meta = load_model()
    model_ready = model is not None
    user_id = user["id"]
    username = user["username"]

    history_df = load_sessions(user_id)
    saved_goals = load_goals(user_id)

    st.session_state.setdefault("page", "dashboard")

    _render_navbar(username)
    current_page = st.session_state["page"]

    # ── New Entry has its own self-contained state/logic ──────────
    if current_page == "entry":
        entry_page.render(user_id, history_df, model, scaler, model_ready)
        return

    # ── Shared "current" prediction for all other pages ───────────
    vals = _latest_session_inputs(history_df)

    pred_proba, pred_class = None, 0
    level_name = vals["level_name"]
    if model_ready and not history_df.empty:
        try:
            exercise_ml = daily_exercise_to_weekly(int(vals["exercise"]) >= 1)
            feature_row = [
                vals["study"], vals["assignments"], vals["exam"], vals["performance"],
                vals["sleep"], exercise_ml, vals["social"], vals["screen"],
                vals["anxiety"], vals["finance"], vals["family"], vals["peer"],
                vals["extra"], vals["rel"],
            ]
            level_name, pred_proba, pred_class = predict_stress(
                model, scaler, feature_row, vals["stress_score"])
        except Exception:
            pred_proba, pred_class = None, 0
            level_name = vals["level_name"]

    level_color = COLORS.get(level_name, "#639922")
    level_emoji = EMOJIS.get(level_name, "😊")

    if current_page == "dashboard":
        dashboard_page.render(
            user, history_df, model_ready, meta,
            vals["study"], vals["sleep"], vals["screen"], vals["anxiety"], vals["exercise"],
            vals["assignments"], vals["exam"], vals["performance"], vals["social"],
            vals["finance"], vals["family"], vals["peer"], vals["extra"], vals["rel"],
            vals["stress_score"], level_name, level_color, level_emoji,
            pred_proba, pred_class,
        )

    elif current_page == "history":
        history_page.render(username, history_df)

    elif current_page == "goals":
        goals_page.render(user_id, history_df, saved_goals,
                           vals["sleep"], vals["study"], vals["exercise"], vals["screen"])

    elif current_page == "planner":
        planner_page.render(user_id, vals["stress_score"], saved_goals)

    elif current_page == "summary":
        summary_page.render(history_df)

    st.divider()
    st.caption("🧠 Student Stress Monitor | Built with Streamlit & scikit-learn | For educational purposes only.")