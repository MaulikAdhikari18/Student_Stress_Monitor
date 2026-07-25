"""
'Dashboard' page — the main landing page after login. Three tabs:
Stress Result, Factor Analysis, Management Tips (with severity-graded
recommendations and a wellness radar chart).
"""

import datetime as _dt2

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.config import LABELS, COLORS
from src.ml.scoring import factor_breakdown
from src.services.report import generate_stress_report
from src.ui.components import render_html, aurora_hero_card, progress_ring_svg, metric_tile


def render(user: dict, history_df: pd.DataFrame, model_ready: bool, meta: dict | None,
           study, sleep, screen, anxiety, exercise, assignments, exam, performance,
           social, finance, family, peer, extra, rel,
           stress_score, level_name, level_color, level_emoji,
           pred_proba, pred_class):

    if model_ready and meta is not None:
        acc = meta.get("accuracy", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("ML Model", meta.get("best_model", "Loaded"))
        c2.metric("Model Accuracy", f"{acc*100:.1f}%")
        c3.metric("Sessions logged", str(len(history_df)))
    else:
        st.warning("⚠️ ML model not found. Run `python src/train_model.py` to enable AI predictions.")

    if history_df.empty:
        st.info("👋 Welcome! Head to **✏️ New Entry** to log your first stress entry.")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Stress Result", "🔍 Factor Analysis", "💡 Management Tips"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Stress Result
    # ══════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([1.4, 1])
        with col_left:
            value_html = (
                f'<h1 style="margin:0;font-size:30px;color:var(--text-primary);font-weight:500;">'
                f'{level_emoji} {level_name} '
                f'<span style="font-size:16px;color:var(--text-secondary);font-weight:400;">'
                f'{stress_score}/100</span></h1>'
            )
            ring_html = progress_ring_svg(stress_score, level_color, size=56)
            render_html(aurora_hero_card("Current stress level", value_html, ring_html))

            if model_ready and pred_proba is not None:
                st.markdown("#### Confidence across levels")
                for lbl, p in zip(LABELS, pred_proba):
                    ca, cb = st.columns([3, 1])
                    ca.progress(float(p), text=lbl)
                    cb.write(f"**{p*100:.1f}%**")
            else:
                st.markdown("#### Score breakdown")
                st.write(f"- Sleep deficit: `{max(0, round(7 - sleep, 1))}h below target`")
                st.write(f"- Study overload: `{max(0, round(study - 8, 1))}h above 8h`")
                st.write(f"- Screen excess: `{max(0, round(screen - 4, 1))}h above 4h`")

        with col_right:
            if model_ready and pred_proba is not None:
                proba_vals = [float(p) for p in pred_proba]
                pull = [0.05 if i == pred_class else 0 for i in range(4)]
                fig_prob = go.Figure(go.Pie(
                    labels=LABELS, values=proba_vals, hole=0.45, pull=pull,
                    marker=dict(colors=['#639922', '#EF9F27', '#D85A30', '#E24B4A'],
                                line=dict(color='#1a1a2e', width=2)),
                    textinfo='percent', textposition='inside',
                    insidetextorientation='radial',
                    hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
                    sort=False,
                ))
                fig_prob.update_layout(
                    title=dict(text='Probability Distribution', font=dict(size=13, color='#ccc'),
                               x=0.5, xanchor='center'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='v', x=1.02, y=0.5, font=dict(color='#ccc', size=11),
                                bgcolor='rgba(0,0,0,0)'),
                    margin=dict(t=40, b=10, l=10, r=80), height=300,
                    annotations=[dict(
                        text=f"{proba_vals[pred_class]*100:.0f}%",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=18, color=COLORS.get(LABELS[pred_class], '#fff'), family='Arial'),
                    )]
                )
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(4, 1.5))
                ax.barh(['Stress'], [stress_score], color=level_color, height=0.5)
                ax.barh(['Stress'], [100 - stress_score], left=[stress_score], color='#e8e8e8', height=0.5)
                ax.set_xlim(0, 100)
                ax.axis('off')
                ax.set_title(f'Score: {stress_score}', fontsize=12)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.divider()
        st.markdown("#### Quick health snapshot")
        sleep_status = "Optimal" if sleep >= 8 else f"-{8 - sleep:.1f}h"
        study_load = ["Light", "Moderate", "Heavy", "Extreme"][min(3, int(study // 4))]
        recovery = int(((exercise / 7) * 0.4 + (sleep / 10) * 0.4 + (social / 20) * 0.2) * 100)
        burnout = min(100, int(stress_score * 0.6 + max(0, study - 8) * 4 + max(0, 10 - sleep) * 3))
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_tile("ti ti-moon", sleep_status, "Sleep status"), unsafe_allow_html=True)
        m2.markdown(metric_tile("ti ti-book", study_load, "Study load"), unsafe_allow_html=True)
        m3.markdown(metric_tile("ti ti-battery", f"{recovery}%", "Recovery score", "#5DCAA5"), unsafe_allow_html=True)
        m4.markdown(metric_tile("ti ti-flame", f"{burnout}/100", "Burnout risk", "#D4537E"), unsafe_allow_html=True)

        # ── PDF report export ─────────────────────────────────
        st.divider()
        render_html('<div class="section-header">📄 Export Report</div>')
        rpt_col1, rpt_col2 = st.columns([2, 1])
        with rpt_col1:
            render_html(
                '<div style="font-size:0.88rem;color:#888;padding-top:0.4rem;">'
                'Generate a one-page PDF summary of your current stress level, '
                'inputs, ML confidence, and top recommendations.</div>')
        with rpt_col2:
            if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Generating report…"):
                    try:
                        pdf_bytes = generate_stress_report(
                            user, history_df, stress_score, level_name,
                            level_color, sleep, study, screen, anxiety,
                            exercise, pred_proba, LABELS, COLORS
                        )
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.session_state["pdf_fn"] = (
                            f"stress_report_{user['username']}_{_dt2.date.today()}.pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state["pdf_bytes"],
                    file_name=st.session_state.get("pdf_fn", "stress_report.pdf"),
                    mime="application/pdf",
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Factor Analysis
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Which factors are driving your stress?")
        factor_scores = factor_breakdown(
            study_hours=study, assignments_pending=assignments, exam_pressure=exam,
            sleep_hours=sleep, anxiety_level=anxiety, financial_stress=finance,
            family_support=family, social_interactions_per_week=social,
            peer_pressure=peer, screen_time_hours=screen, exercise_days_per_week=exercise,
        )
        sorted_f = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [f[0] for f in sorted_f]
        vals = [f[1] for f in sorted_f]
        clrs = ['#A32D2D' if v >= 75 else '#993C1D' if v >= 55
                else '#BA7517' if v >= 30 else '#639922' for v in vals]
        bars = ax.barh(names, vals, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + 2, bar.get_y() + bar.get_height() / 2, str(v), va='center', fontsize=10)
        ax.axvline(55, ls='--', lw=1, color='#BA7517', alpha=0.6, label='High threshold')
        ax.axvline(75, ls='--', lw=1, color='#A32D2D', alpha=0.6, label='Critical threshold')
        ax.set_xlim(0, 115)
        ax.set_xlabel("Stress contribution score")
        ax.set_title("Stress Factor Breakdown", fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Your inputs at a glance")
        sc = st.columns(4)
        snap = [("Study hrs/day", f"{study}h"), ("Sleep hrs/night", f"{sleep}h"),
                ("Anxiety", f"{anxiety}/10"), ("Exercise days", str(exercise)),
                ("Assignments", str(assignments)), ("Financial stress", f"{finance}/10"),
                ("Family support", f"{family}/10"), ("Screen time", f"{screen}h")]
        for i, (k, v) in enumerate(snap):
            sc[i % 4].metric(k, v)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Management Tips (Advanced)
    # ══════════════════════════════════════════════════════════
    with tab3:
        _render_tips_tab(study, sleep, screen, anxiety, exercise, assignments,
                          exam, performance, social, finance, family, peer,
                          extra, stress_score, level_name)


def _render_tips_tab(study, sleep, screen, anxiety, exercise, assignments,
                      exam, performance, social, finance, family, peer,
                      extra, stress_score, level_name):
    # ── Smart insight summary ──────────────────────────────
    top_issue = ""
    issue_score = 0
    if sleep < 6 and (7 - sleep) * 15 > issue_score:
        issue_score = int((7 - sleep) * 15)
        top_issue = "sleep deprivation"
    if anxiety > 7 and anxiety * 9 > issue_score:
        issue_score = int(anxiety * 9)
        top_issue = "high anxiety"
    if study > 10 and (study - 8) * 8 > issue_score:
        issue_score = int((study - 8) * 8)
        top_issue = "study overload"
    if not top_issue:
        top_issue = "a generally balanced profile"

    recovery_days = ("2–3 days" if stress_score < 35
                      else "4–5 days" if stress_score < 60
                      else "7–10 days")
    insight_text = (
        f"Your stress score of **{stress_score}/100** places you in the **{level_name}** zone. "
        f"The primary driver appears to be **{top_issue}**. "
        f"With consistent corrective action, meaningful improvement typically takes **{recovery_days}**. "
        f"Focus on the Critical and High-priority recommendations below first."
    ) if stress_score > 20 else (
        f"Your stress indicators look well-balanced (score: **{stress_score}/100**). "
        f"Maintain your current routines and log daily to catch any early drift."
    )

    render_html(f"""
    <div class="insight-box">
        <div class="insight-title">🧠 AI Stress Insight</div>
        <div class="insight-body">{insight_text}</div>
    </div>""")

    CRITICAL, HIGH, MODERATE, POSITIVE = "critical", "high", "moderate", "positive"
    recs = []

    if sleep < 5:
        recs.append((CRITICAL, "😴", "Severe Sleep Deficit",
            f"Only {sleep}h sleep — this is a medical concern. Cognitive function drops 30%+ below 5h.",
            "Go to bed in the next 2 hours. No exceptions tonight.",
            int((7 - sleep) / 5 * 100)))
    elif sleep < 7:
        recs.append((HIGH, "😴", "Sleep Below Threshold",
            f"You're getting {sleep}h vs the recommended 7–9h. This raises cortisol and impairs memory consolidation.",
            "Set a hard lights-out alarm. Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s).",
            int((7 - sleep) / 3 * 70)))
    elif sleep >= 8:
        recs.append((POSITIVE, "😴", "Great Sleep",
            f"Excellent — {sleep}h of sleep supports memory, mood, and immune function.",
            "Keep your consistent sleep schedule.", 0))

    if study > 12:
        recs.append((CRITICAL, "📚", "Dangerous Study Load",
            f"{study}h/day is unsustainable and counterproductive. Retention collapses after 6–7h of quality study.",
            "Cut to max 6h today. Use Pomodoro 25/5. Schedule mandatory end-time.",
            min(100, int((study - 8) * 12))))
    elif study > 8:
        recs.append((HIGH, "📚", "Study Overload Risk",
            f"{study}h/day is above the effective threshold. Quality matters more than quantity.",
            "Cap at 8h. Use active recall and spaced repetition for higher retention.",
            min(100, int((study - 8) * 8))))

    if exercise < 2:
        recs.append((HIGH, "🏃", "Critical Exercise Deficit",
            "0–1 exercise days/week significantly raises stress hormones. Exercise is one of the strongest natural anxiolytics.",
            "Start with a 20-min walk today. You don't need a gym — just movement.", 80))
    elif exercise < 3:
        recs.append((MODERATE, "🏃", "Increase Physical Activity",
            f"{exercise} exercise days/week is below the recommended 3–5. Even light activity reduces cortisol by ~26%.",
            "Add 2 more days this week. Try a 15-min YouTube workout.", 50))
    elif exercise >= 5:
        recs.append((POSITIVE, "🏃", "Active & Resilient",
            f"{exercise} exercise days/week — excellent. Exercise is your best stress buffer.",
            "Maintain this habit. Consider adding yoga or stretching for recovery.", 0))

    if anxiety >= 8:
        recs.append((CRITICAL, "🧘", "High Anxiety — Immediate Action Needed",
            f"Anxiety at {anxiety}/10 is clinically significant. This is affecting your cognition and sleep.",
            "Try box breathing NOW: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 5×.",
            int(anxiety * 10)))
    elif anxiety >= 6:
        recs.append((HIGH, "🧘", "Elevated Anxiety",
            f"Anxiety at {anxiety}/10 is interfering with focus. Cognitive load increases sharply above 6/10.",
            "10-min daily mindfulness practice. Apps: Headspace, Insight Timer (free tier).",
            int(anxiety * 8)))

    if screen > 8:
        recs.append((HIGH, "📱", "Excessive Screen Time",
            f"{screen}h/day of screens raises cortisol and disrupts melatonin production, directly worsening sleep.",
            "Set app time limits. Use grayscale mode after 9 PM to reduce dopamine spikes.",
            min(100, int((screen - 4) * 10))))
    elif screen > 5:
        recs.append((MODERATE, "📱", "Moderate Screen Overuse",
            f"{screen}h/day is above the 4h guideline. Blue light affects sleep quality.",
            "Use blue-light glasses or Night Shift mode. No screens 30 min before bed.",
            min(100, int((screen - 4) * 7))))

    if finance >= 8:
        recs.append((HIGH, "💰", "High Financial Stress",
            "Financial stress is one of the top predictors of academic dropout and mental health issues.",
            "Contact your institution's student welfare office today. Emergency funds may be available.",
            int(finance * 9)))
    elif finance >= 6:
        recs.append((MODERATE, "💰", "Financial Pressure",
            "Moderate financial stress is draining background mental resources.",
            "Track expenses for 1 week. Identify one non-essential cost to cut or defer.",
            int(finance * 6)))

    if social < 3:
        recs.append((HIGH, "👥", "Social Isolation Risk",
            "Low social interaction is linked to depression and reduced stress resilience.",
            "Schedule one 20-min call or meet-up this week. Join a study group or club.", 75))
    elif social < 5:
        recs.append((MODERATE, "👥", "Limited Social Connection",
            "Moderate social contact — aim to increase meaningful interactions.",
            "Even brief positive exchanges count. Say hi to a classmate daily.", 45))

    if assignments >= 10:
        recs.append((CRITICAL, "📝", "Task Overload",
            f"{assignments} pending assignments creates decision paralysis and chronic low-grade panic.",
            "Eisenhower matrix: list tasks → sort by urgent+important → do top 1 NOW.",
            min(100, assignments * 7)))
    elif assignments >= 6:
        recs.append((MODERATE, "📝", "Heavy Task Queue",
            f"{assignments} pending items. Unfinished tasks occupy working memory (Zeigarnik effect).",
            "Write every task down — externalising it frees cognitive load immediately.",
            min(100, assignments * 5)))

    if peer >= 8:
        recs.append((HIGH, "🤝", "Severe Peer Pressure",
            f"Peer pressure at {peer}/10 is draining energy and distorting your decisions.",
            "Practice assertive phrases: 'I'm not able to commit to that right now.' Limit time with draining people.",
            int(peer * 9)))

    if family <= 3:
        recs.append((HIGH, "❤️", "Low Support Network",
            "Low family support increases psychological vulnerability significantly.",
            "Campus counselors and peer mentors provide structured support — reach out today.", 70))

    if not recs or all(r[0] == POSITIVE for r in recs):
        recs.append((POSITIVE, "🌟", "Strong Wellbeing Profile",
            "Your indicators are well-balanced. You're in the top tier for student wellbeing.",
            "Do weekly check-ins to detect drift early. Share what's working with peers.", 0))

    order = {CRITICAL: 0, HIGH: 1, MODERATE: 2, POSITIVE: 3}
    recs.sort(key=lambda r: order[r[0]])
    badge_labels = {CRITICAL: "Critical", HIGH: "High Priority", MODERATE: "Moderate", POSITIVE: "Positive"}

    st.markdown(f"#### 💡 {len([r for r in recs if r[0] != POSITIVE])} active recommendations")

    cols_r = st.columns(2)
    for idx, (severity, icon, title, body, action, score) in enumerate(recs):
        with cols_r[idx % 2]:
            bar_color = {"critical": "#E24B4A", "high": "#FAC775",
                         "moderate": "#7F77DD", "positive": "#97C459"}[severity]
            bar_width = score if severity != POSITIVE else 100
            render_html(f"""
            <div class="rec-card rec-{severity}">
                <div class="rec-header">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <div>
                        <span class="rec-badge badge-{severity}">{badge_labels[severity]}</span>
                        <div class="rec-title">{title}</div>
                    </div>
                </div>
                <p class="rec-body">{body}</p>
                <div class="rec-action">⚡ {action}</div>
                <div class="rec-score-bar">
                    <div style="width:{bar_width}%;height:100%;background:{bar_color};
                                border-radius:2px;"></div>
                </div>
            </div>""")

    st.divider()
    st.markdown("#### 📡 Wellness Radar")
    radar_cats = ['Sleep', 'Study Balance', 'Exercise', 'Social', 'Low Anxiety', 'Low Screen']
    norm_sleep = min(100, int(sleep / 9 * 100))
    norm_study = max(0, 100 - int(max(0, study - 6) / 10 * 100))
    norm_exercise = min(100, int(exercise / 7 * 100))
    norm_social = min(100, int(social / 15 * 100))
    norm_anxiety = max(0, 100 - int((anxiety - 1) / 9 * 100))
    norm_screen = max(0, 100 - int(max(0, screen - 3) / 13 * 100))
    radar_vals = [norm_sleep, norm_study, norm_exercise, norm_social, norm_anxiety, norm_screen]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]], theta=radar_cats + [radar_cats[0]],
        fill='toself', fillcolor='rgba(83,74,183,0.2)',
        line=dict(color='#AFA9EC', width=2), name='Your Profile'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[80] * len(radar_cats) + [80], theta=radar_cats + [radar_cats[0]],
        fill='toself', fillcolor='rgba(99,153,34,0.06)',
        line=dict(color='#639922', width=1.5, dash='dot'), name='Target Zone'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9),
                             gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, legend=dict(font=dict(color='#ccc')),
        margin=dict(t=30, b=30, l=30, r=30), height=380
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.markdown("#### 🗓️ 7-day recovery plan")
    plan_items = [
        ("Day 1", "Set a consistent sleep time and stick to it all week."),
        ("Day 2", "Write all pending tasks down. Cross off one small thing today."),
        ("Day 3", "Go for a 20-min walk — no phone, no earphones."),
        ("Day 4", "Call or message one friend or family member you trust."),
        ("Day 5", "Do one 25-min Pomodoro study block. Note your focus level."),
        ("Day 6", "Spend 10 min on box breathing or a simple meditation."),
        ("Day 7", "Review the week: what helped? Plan to repeat those habits."),
    ]
    p1, p2 = st.columns(2)
    for i, (day, action) in enumerate(plan_items):
        with (p1 if i % 2 == 0 else p2):
            st.checkbox(f"**{day}** — {action}", key=f"plan_{day}")