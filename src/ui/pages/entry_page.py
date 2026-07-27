"""
'New Entry' page — pick a date on a calendar, fill in today's inputs,
see a live stress prediction, and save it as a session.
"""

import calendar as cal_mod
import datetime

import numpy as np
import streamlit as st

from src.config import LABELS, COLORS, EMOJIS
from src.db.sessions import save_session
from src.ml.scoring import compute_stress_score, daily_exercise_to_weekly
from src.ml.model import predict_stress
from src.ui.styles import inject_entry_section_css, inject_calendar_css
from src.ui.components import stress_bg, stress_border, render_html, aurora_hero_card, progress_ring_svg, metric_tile


def render(user_id: int, history_df, model, scaler, model_ready: bool):
    st.markdown('<div class="main-title">✏️ Log a New Entry</div>', unsafe_allow_html=True)
    st.caption("Pick a date on the calendar, fill in your details, and save.")
    st.divider()

    # ── Calendar data ────────────────────────────────────────────────
    hdf_cal = history_df.copy() if not history_df.empty else history_df
    if not hdf_cal.empty:
        import pandas as pd
        hdf_cal["stress_score"] = pd.to_numeric(hdf_cal["stress_score"], errors="coerce")
        hdf_cal["cal_date"] = pd.to_datetime(hdf_cal["timestamp"]).dt.date
        date_stress = hdf_cal.groupby("cal_date")["stress_score"].mean().to_dict()
        date_level = hdf_cal.groupby("cal_date")["stress_level"].last().to_dict()
    else:
        date_stress, date_level = {}, {}

    today = datetime.date.today()
    st.session_state.setdefault("cal_year", today.year)
    st.session_state.setdefault("cal_month", today.month)
    st.session_state.setdefault("entry_date", today)

    cy = st.session_state["cal_year"]
    cm = st.session_state["cal_month"]

    cal_col, form_col = st.columns([1, 1.4])

    # ── Calendar ─────────────────────────────────────────────────────
    with cal_col:
        st.markdown('<div class="section-header">📅 Select Date</div>', unsafe_allow_html=True)

        mn1, mn2, mn3 = st.columns([1, 3, 1])
        with mn1:
            if st.button("◀", key="cal_prev"):
                if cm == 1:
                    st.session_state["cal_month"] = 12
                    st.session_state["cal_year"] -= 1
                else:
                    st.session_state["cal_month"] -= 1
                st.rerun()
        with mn2:
            st.markdown(
                f'<div style="text-align:center;font-weight:700;font-size:1rem;'
                f'color:var(--accent-purple);padding-top:4px;">{cal_mod.month_name[cm]} {cy}</div>',
                unsafe_allow_html=True)
        with mn3:
            if st.button("▶", key="cal_next"):
                if cm == 12:
                    st.session_state["cal_month"] = 1
                    st.session_state["cal_year"] += 1
                else:
                    st.session_state["cal_month"] += 1
                st.rerun()

        inject_calendar_css()
        cal_mod.setfirstweekday(6)
        month_weeks = cal_mod.monthcalendar(cy, cm)
        selected_date = st.session_state["entry_date"]

        cal_html = '<div class="ssm-cal">'
        for d in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
            cal_html += f'<div class="ssm-ch">{d}</div>'

        for week in month_weeks:
            for day in week:
                if day == 0:
                    cal_html += '<div></div>'
                    continue
                d_obj = datetime.date(cy, cm, day)
                score = date_stress.get(d_obj)
                bg = stress_bg(score)
                border = stress_border(score)
                is_sel = d_obj == selected_date
                is_tod = d_obj == today
                is_fut = d_obj > today

                ring = "box-shadow:0 0 0 3px #AFA9EC,0 0 0 5px rgba(175,169,236,0.2);" if is_sel else ""
                op = "opacity:0.3;pointer-events:none;" if is_fut else ""
                dash = "border-style:dashed;" if is_tod and not is_sel else ""
                sc_html = f'<div class="ssm-sc">{score:.0f}</div>' if score is not None else ""
                dot_col = border if score is not None else "transparent"
                dot = f'<div class="ssm-dot" style="background:{dot_col};"></div>'

                score_tip = f" • {score:.0f}" if score is not None else ""
                cal_html += (
                    '<div class="ssm-cd" '
                    f'style="background:{bg};border-color:{border};{ring}{op}{dash}" '
                    f'title="{d_obj:%b %d}{score_tip}">'
                    f'<div class="ssm-num">{day}</div>{sc_html}{dot}</div>'
                )
        cal_html += "</div>"
        render_html(cal_html)

        st.markdown("<div style='margin-top:0.7rem'></div>", unsafe_allow_html=True)
        picked = st.date_input("Pick date", value=selected_date,
                                max_value=today, key="dpick",
                                label_visibility="collapsed")
        if picked != st.session_state["entry_date"]:
            st.session_state["entry_date"] = picked
            st.session_state["cal_year"] = picked.year
            st.session_state["cal_month"] = picked.month
            st.rerun()

        render_html("""
<div style="display:flex;gap:8px;margin-top:0.6rem;flex-wrap:wrap;">
  <span style="font-size:0.7rem;color:var(--text-muted);display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(99,153,34,0.6);
                 border:1.5px solid #639922;display:inline-block;"></span>Low</span>
  <span style="font-size:0.7rem;color:var(--text-muted);display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(186,117,23,0.55);
                 border:1.5px solid #EF9F27;display:inline-block;"></span>Moderate</span>
  <span style="font-size:0.7rem;color:var(--text-muted);display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(153,60,29,0.55);
                 border:1.5px solid #D85A30;display:inline-block;"></span>High</span>
  <span style="font-size:0.7rem;color:var(--text-muted);display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;background:rgba(163,45,45,0.6);
                 border:1.5px solid #E24B4A;display:inline-block;"></span>Critical</span>
  <span style="font-size:0.7rem;color:var(--text-muted);display:flex;align-items:center;gap:4px;">
    <span style="width:9px;height:9px;border-radius:3px;border:1.5px dashed #AFA9EC;
                 display:inline-block;"></span>Today</span>
</div>""")

    # ── Entry form ────────────────────────────────────────────────────
    with form_col:
        sel_date = st.session_state["entry_date"]
        existing_level = date_level.get(sel_date)
        existing_score = date_stress.get(sel_date)

        if existing_level:
            ec = {"Low": "#639922", "Moderate": "#EF9F27", "High": "#D85A30", "Critical": "#E24B4A"}.get(existing_level, "#888")
            render_html(
                f'<div style="background:var(--bg-surface);border:1px solid {ec}44;'
                f'border-left:4px solid {ec};border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;">'
                f'<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">'
                f'Existing entry — {sel_date:%A, %d %b %Y}</div>'
                f'<div style="font-size:1.3rem;font-weight:800;color:{ec};">{existing_level} Stress</div>'
                f'<div style="font-size:0.82rem;color:var(--text-muted);">Score: {existing_score:.0f}/100</div></div>')
        else:
            render_html(
                f'<div style="background:var(--bg-surface);border:1px solid var(--border);'
                f'border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;">'
                f'<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">Logging entry for</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:var(--accent-purple);">{sel_date:%A, %d %b %Y}</div>'
                f'<div style="font-size:0.8rem;color:var(--text-muted);">No entry yet</div></div>')

        with st.form("entry_form"):
            inject_entry_section_css()

            st.markdown('<div class="entry-section-title">📚 Academic — Today</div>', unsafe_allow_html=True)
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                study = st.number_input("Hours studied today", 0.0, 24.0, 6.0, 0.5, key="e_study")
            with ac2:
                assignments = st.slider("Pending assignments", 0, 15, 3, key="e_asgn")
            with ac3:
                exam = st.slider("Exam pressure today (1–10)", 1, 10, 5, key="e_exam")
            with ac4:
                performance = st.slider("Academic performance (1–10)", 1, 10, 7, key="e_perf")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="entry-section-title">🌙 Lifestyle — Today</div>', unsafe_allow_html=True)
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                sleep = st.number_input("Hours slept last night", 0.0, 24.0, 7.0, 0.5, key="e_sleep")
            with lc2:
                exercised_today = st.radio("Exercised today?", ["Yes", "No"],
                                            index=1, key="e_exer_radio", horizontal=True)
                exercise = 1 if exercised_today == "Yes" else 0
            with lc3:
                screen = st.number_input("Screen time today (hrs)", 0.0, 24.0, 4.0, 0.5, key="e_screen")
            with lc4:
                social = st.slider("Meaningful interactions today", 0, 20, 5, key="e_soc")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="entry-section-title">🧠 Mental & Emotional — Today</div>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                anxiety = st.slider("Anxiety level today (1–10)", 1, 10, 4, key="e_anx")
            with mc2:
                finance = st.slider("Financial stress (1–10)", 1, 10, 3, key="e_fin")
            with mc3:
                family = st.slider("Family support felt today (1–10)", 1, 10, 7, key="e_fam")
            with mc4:
                peer = st.slider("Peer pressure today (1–10)", 1, 10, 4, key="e_peer")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="entry-section-title">✨ Other</div>', unsafe_allow_html=True)
            oc1, oc2 = st.columns(2)
            with oc1:
                extra = st.selectbox("Extracurricular activities (overall)", [0, 1, 2],
                                      format_func=lambda x: ["None", "1–2 activities", "3+ activities"][x],
                                      key="e_extra")
            with oc2:
                rel = st.selectbox("Relationship situation", [0, 1, 2],
                                    format_func=lambda x: ["Single / N/A", "Stable relationship", "Relationship issues"][x],
                                    key="e_rel")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            save_btn = st.form_submit_button("💾 Save Entry", use_container_width=True, type="primary")

        # ── Live prediction ────────────────────────────────────────
        exercise_weekly = daily_exercise_to_weekly(exercise == 1)
        stress_score = compute_stress_score(
            study_hours=study, assignments_pending=assignments, exam_pressure=exam,
            academic_performance=performance, sleep_hours=sleep,
            exercise_days_per_week=exercise_weekly, social_interactions_per_week=social,
            screen_time_hours=screen, anxiety_level=anxiety, financial_stress=finance,
            family_support=family, peer_pressure=peer,
            extracurricular_activities=extra, relationship_issues=rel,
        )

        if model_ready:
            feature_row = [study, assignments, exam, performance, sleep,
                            exercise_weekly, social, screen, anxiety, finance,
                            family, peer, extra, rel]
            level_name, pred_proba, _ = predict_stress(model, scaler, feature_row, stress_score)
        else:
            from src.config import classify_score
            level_name = classify_score(stress_score)
            pred_proba = None

        level_color = COLORS[level_name]
        level_emoji = EMOJIS[level_name]
        recovery = int(((exercise * 0.4) + (sleep / 10) * 0.4 + (social / 20) * 0.2) * 100)
        burnout = min(100, int(stress_score * 0.6 + max(0, study - 8) * 4 + max(0, 10 - sleep) * 3))

        rc1, rc2 = st.columns(2)
        with rc1:
            value_html = (
                f'<h1 style="margin:0;font-size:26px;color:var(--text-primary);font-weight:500;">'
                f'{level_emoji} {level_name} '
                f'<span style="font-size:15px;color:var(--text-secondary);font-weight:400;">'
                f'{stress_score}/100</span></h1>'
            )
            ring_html = progress_ring_svg(stress_score, level_color, size=52)
            render_html(aurora_hero_card("Live Result", value_html, ring_html))
        with rc2:
            exer_txt = "Yes" if exercise else "Rest day"
            tiles = [
                metric_tile("ti ti-battery", f"{recovery}%", "Recovery", "#5DCAA5"),
                metric_tile("ti ti-flame", f"{burnout}", "Burnout Risk", "#D4537E"),
                metric_tile("ti ti-moon", f"{sleep}h", "Sleep"),
                metric_tile("ti ti-run", exer_txt, "Exercise"),
            ]
            render_html(
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:0.5rem;">'
                + "".join(tiles) + '</div>'
            )

        if save_btn:
            save_session(user_id, stress_score, level_name,
                         sleep, study, screen, anxiety, exercise,
                         entry_date=st.session_state["entry_date"])
            st.success(f"✅ Entry saved for {sel_date:%A, %d %b %Y}!")
            st.rerun()