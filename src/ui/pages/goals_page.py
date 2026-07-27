"""'Goals' page — set weekly wellness targets and track streaks/progress."""

import pandas as pd
import streamlit as st

from src.db.goals import save_goals_db
from src.services.streaks import compute_streaks, week_progress
from src.ui.components import render_html, aurora_hero_card, progress_ring_svg


def render(user_id: int, history_df: pd.DataFrame, saved_goals: dict,
           sleep, study, exercise, screen):
    st.markdown("#### 🎯 Set your weekly wellness goals")
    st.caption("Goals are saved per account and tracked against every session you log.")

    with st.form("goals_form"):
        st.markdown("##### Adjust your targets")
        gc1, gc2 = st.columns(2)
        with gc1:
            g_sleep = st.slider("😴 Sleep target (hrs/night, min)",
                                 4.0, 10.0, float(saved_goals.get("goal_sleep", 8.0)), 0.5)
            g_exercise = st.slider("🏃 Exercise target (days/week, min)",
                                    1, 7, int(saved_goals.get("goal_exercise", 4)))
        with gc2:
            g_study = st.slider("📚 Study limit (hrs/day, max)",
                                 2.0, 14.0, float(saved_goals.get("goal_study", 8.0)), 0.5)
            g_screen = st.slider("📱 Screen time limit (hrs/day, max)",
                                  1.0, 12.0, float(saved_goals.get("goal_screen", 4.0)), 0.5)
        save_goals_btn = st.form_submit_button("💾 Save Goals", use_container_width=True)

    if save_goals_btn:
        save_goals_db(user_id, g_sleep, g_study, g_exercise, g_screen)
        st.success("✅ Goals saved!")
        st.rerun()
    else:
        g_sleep = float(saved_goals.get("goal_sleep", 8.0))
        g_study = float(saved_goals.get("goal_study", 8.0))
        g_exercise = int(saved_goals.get("goal_exercise", 4))
        g_screen = float(saved_goals.get("goal_screen", 4.0))

    st.divider()
    st.markdown("##### How does today compare?")
    col_a, col_b = st.columns(2)
    for i, (label, actual, target, direction, u_a, u_t) in enumerate([
        ("😴 Sleep", sleep, g_sleep, "gte", "hrs tonight", "hrs target"),
        ("📚 Study", study, g_study, "lte", "hrs today", "hrs max"),
        ("🏃 Exercise", float(exercise), float(g_exercise), "gte", "days this week", "days target"),
        ("📱 Screen", screen, g_screen, "lte", "hrs today", "hrs limit"),
    ]):
        met = (actual >= target) if direction == "gte" else (actual <= target)
        icon = "✅" if met else "❌"
        status_txt = "Goal met!" if met else ("Need more" if direction == "gte" else "Too much")
        delta_str = (f"+{round(actual - target, 1)}" if actual >= target else str(round(actual - target, 1)))
        with (col_a if i % 2 == 0 else col_b):
            render_html(f"""
            <div class="goal-card">
                <div class="goal-title">{icon} {label}</div>
                <div style="font-size:1.4rem;font-weight:700;
                            color:{'#639922' if met else '#993C1D'};">
                    {actual}
                    <span style="font-size:0.85rem;font-weight:400;color:var(--text-muted);">{u_a}</span>
                </div>
                <div style="font-size:0.82rem;color:var(--text-muted);margin:2px 0 8px;">
                    Target: {target} {u_t} &nbsp;|&nbsp; {status_txt} ({delta_str})
                </div>
            </div>""")

    st.divider()
    st.markdown("##### Weekly progress & streaks")
    hdf_g = history_df.copy() if not history_df.empty else history_df
    streaks = compute_streaks(hdf_g, g_sleep, g_study, g_exercise, g_screen)
    progress = week_progress(hdf_g, g_sleep, g_study, g_exercise, g_screen)

    for label, key, rule in [
        ("😴 Sleep", "sleep", f"≥ {g_sleep}h/night"),
        ("📚 Study", "study", f"≤ {g_study}h/day"),
        ("🏃 Exercise", "exercise", f"≥ {g_exercise} days/week"),
        ("📱 Screen", "screen", f"≤ {g_screen}h/day"),
    ]:
        pct = progress[key]
        streak = streaks[key]
        bar_c = "#639922" if pct >= 70 else "#BA7517" if pct >= 40 else "#E24B4A"
        s_cls = "streak-badge" if streak > 0 else "streak-badge streak-zero"
        s_txt = f"🔥 {streak}-day streak" if streak > 0 else "No streak yet"
        render_html(f"""
        <div class="goal-card">
            <div style="display:flex;align-items:center;
                        justify-content:space-between;margin-bottom:6px;">
                <span class="goal-title" style="margin:0;">{label} &nbsp;
                    <span style="font-weight:400;color:var(--text-muted);font-size:0.8rem;">({rule})</span>
                </span>
                <span class="{s_cls}">{s_txt}</span>
            </div>
            <div style="background:var(--border);border-radius:8px;height:12px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{bar_c};
                            border-radius:8px;"></div>
            </div>
            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:4px;">
                {pct}% of last 7 sessions goal was met {"✅" if pct == 100 else ""}
            </div>
        </div>""")

    st.divider()
    st.markdown("##### Overall goal score")
    overall = int(sum(progress.values()) / len(progress))
    o_color = "#639922" if overall >= 70 else "#BA7517" if overall >= 40 else "#E24B4A"
    o_label = "Excellent 🌟" if overall >= 80 else "Good 👍" if overall >= 60 else "Needs work 💪"
    _, oc2, _ = st.columns([1, 2, 1])
    with oc2:
        value_html = (
            f'<h1 style="margin:0;font-size:32px;color:{o_color};font-weight:700;">{overall}%</h1>'
            f'<p style="margin:2px 0 0;font-size:0.95rem;color:var(--text-primary);">{o_label}</p>'
            f'<p style="margin:2px 0 0;font-size:0.8rem;color:var(--text-muted);">Based on your last 7 logged sessions</p>'
        )
        render_html(aurora_hero_card("Overall goal score", value_html, progress_ring_svg(overall, o_color)))

    if hdf_g.empty:
        st.info("💡 Start logging daily sessions to see your streaks and progress fill up!")