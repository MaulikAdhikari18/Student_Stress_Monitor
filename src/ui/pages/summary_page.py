"""'Summary' page — weekly and monthly rollups of stress, sleep, study, lifestyle."""

import calendar as cal_mod
import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS as LEVEL_COLOR
from src.ui.components import summary_card, render_html, dataframe_html, aurora_hero_card, progress_ring_svg


def render(history_df: pd.DataFrame):
    st.markdown('<div class="main-title">📋 Wellness Summary</div>', unsafe_allow_html=True)
    st.caption("Weekly and monthly breakdown of your stress, sleep, study, and lifestyle habits.")
    st.divider()

    if history_df.empty:
        st.info("No data yet — log entries on the ✏️ New Entry page to see your summary.")
        return

    hdf = _prep(history_df)

    today = datetime.date.today()
    week_start = today - datetime.timedelta(days=today.weekday())
    week_end = week_start + datetime.timedelta(days=6)
    month_start = today.replace(day=1)

    hdf_dedup = hdf.sort_values("entry_dt").groupby("date", as_index=False).last()
    week_df = hdf_dedup[hdf_dedup["date"].apply(lambda d: week_start <= d <= today)]
    month_df = hdf_dedup[hdf_dedup["date"].apply(lambda d: month_start <= d <= today)]

    _render_week_section(week_df, week_start, week_end, today)
    st.divider()
    _render_month_section(month_df, today)


def _prep(history_df: pd.DataFrame) -> pd.DataFrame:
    hdf = history_df.copy()
    for col in ["stress_score", "sleep", "study", "screen", "anxiety", "exercise"]:
        hdf[col] = pd.to_numeric(hdf[col], errors="coerce")
    hdf["entry_dt"] = pd.to_datetime(hdf["timestamp"], errors="coerce")
    hdf["date"] = hdf["entry_dt"].dt.date
    hdf["week"] = hdf["entry_dt"].dt.isocalendar().week.astype(int)
    hdf["month"] = hdf["entry_dt"].dt.month
    hdf["year"] = hdf["entry_dt"].dt.year
    hdf["year_month"] = hdf["entry_dt"].dt.to_period("M").astype(str)
    return hdf


def _render_week_section(week_df, week_start, week_end, today):
    render_html('<div class="section-header">📅 This Week</div>')
    st.caption(f"{week_start.strftime('%d %b')} – {min(week_end, today).strftime('%d %b %Y')} · {len(week_df)} entries")

    if week_df.empty:
        st.info("No entries this week yet.")
        return

    avg_stress_w = week_df["stress_score"].mean()
    avg_sleep_w = week_df["sleep"].mean()
    avg_study_w = week_df["study"].mean()
    wex = week_df.groupby("date")["exercise"].max()
    days_logged_w = len(wex)
    exercise_days_w = int((wex.clip(0, 1) > 0).sum())
    dominant_w = week_df["stress_level"].mode()[0] if "stress_level" in week_df.columns else "—"
    dc_w = LEVEL_COLOR.get(dominant_w, "#888")

    value_html = (
        f'<h1 style="margin:0;font-size:26px;color:var(--text-primary);font-weight:500;">'
        f'{avg_stress_w:.0f}<span style="font-size:15px;color:var(--text-secondary);font-weight:400;"> / 100</span></h1>'
        f'<p style="margin:2px 0 0;font-size:0.85rem;color:{dc_w};">Dominant: {dominant_w}</p>'
    )
    render_html(aurora_hero_card("Weekly average stress", value_html, progress_ring_svg(avg_stress_w, dc_w)))
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    wc1, wc2, wc3, wc4 = st.columns(4)
    cards_w = [
        ("Avg Stress Score", f"{avg_stress_w:.0f}", f"Dominant: {dominant_w}", dc_w, "ti ti-gauge"),
        ("Avg Sleep/Night", f"{avg_sleep_w:.1f}h", "Target: 7–9h", "var(--accent-purple)", "ti ti-moon"),
        ("Avg Study/Day", f"{avg_study_w:.1f}h", "Recommended: ≤8h", "var(--accent-purple)", "ti ti-book"),
        ("Exercise Days", f"{exercise_days_w}/{days_logged_w}", f"{exercise_days_w} of {days_logged_w} logged days",
         "#97C459" if days_logged_w > 0 and exercise_days_w / days_logged_w >= 0.5 else "#F09595", "ti ti-run"),
    ]
    for col, (label, val, sub, clr, icon) in zip([wc1, wc2, wc3, wc4], cards_w):
        col.markdown(summary_card(label, val, sub, clr, icon_class=icon), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    render_html('<div class="section-header">Day-by-Day Breakdown</div>')

    day_cols = ["date", "stress_score", "stress_level", "sleep", "study", "screen", "anxiety", "exercise"]
    wshow = week_df[[c for c in day_cols if c in week_df.columns]].copy()
    wshow = wshow.sort_values("date")
    wshow["date"] = wshow["date"].apply(lambda d: d.strftime("%a %d %b"))
    wshow.columns = [c.replace("_", " ").title() for c in wshow.columns]
    render_html(dataframe_html(wshow.reset_index(drop=True), max_height="320px"))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_html('<div class="section-header">Stress Heatmap — This Week</div>')

    heat_html = '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:6px;">'
    for i in range(7):
        d = week_start + datetime.timedelta(days=i)
        day_entry = week_df[week_df["date"] == d]
        if not day_entry.empty:
            sc = day_entry["stress_score"].values[0]
            lv = day_entry["stress_level"].values[0] if "stress_level" in day_entry.columns else "Low"
            bg = LEVEL_COLOR.get(lv, "#888") + "44"
            brd = LEVEL_COLOR.get(lv, "#888")
            sc_txt = f'<div style="font-size:1.1rem;font-weight:800;color:{brd};">{sc:.0f}</div>'
            lv_txt = f'<div style="font-size:0.65rem;color:{brd};font-weight:600;">{lv}</div>'
        else:
            bg = "rgba(255,255,255,0.03)"
            brd = "rgba(255,255,255,0.08)"
            sc_txt = '<div style="font-size:0.75rem;color:#444;">—</div>'
            lv_txt = '<div style="font-size:0.65rem;color:#444;">No entry</div>'

        ring = "box-shadow:0 0 0 2px #AFA9EC;" if d == today else ""
        heat_html += (
            f'<div style="background:{bg};border:1.5px solid {brd};border-radius:12px;'
            f'padding:0.7rem 0.4rem;text-align:center;{ring}">'
            f'<div style="font-size:0.7rem;color:#888;font-weight:600;">{d.strftime("%a")}</div>'
            f'<div style="font-size:0.75rem;color:#666;">{d.strftime("%d")}</div>'
            + sc_txt + lv_txt + '</div>'
        )
    heat_html += '</div>'
    render_html(heat_html)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    wk_sorted = week_df.sort_values("entry_dt")
    if len(wk_sorted) > 1:
        wfig = go.Figure()
        wfig.add_trace(go.Scatter(
            x=wk_sorted["date"].apply(lambda d: d.strftime("%a %d")),
            y=wk_sorted["stress_score"], mode="lines+markers+text",
            text=wk_sorted["stress_score"].apply(lambda v: f"{v:.0f}"), textposition="top center",
            line=dict(color="#AFA9EC", width=2.5, shape="spline"),
            marker=dict(size=9, color=[LEVEL_COLOR.get(l, "#888") for l in wk_sorted.get("stress_level", ["Low"] * len(wk_sorted))],
                        line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
            fill="tozeroy", fillcolor="rgba(83,74,183,0.09)"
        ))
        wfig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#777", size=10)),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#777", size=10)),
            margin=dict(t=20, b=10, l=10, r=10), height=200, showlegend=False
        )
        st.plotly_chart(wfig, use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    insights = []
    if avg_sleep_w < 7:
        insights.append(("⚠️", f"Average sleep this week is {avg_sleep_w:.1f}h — below the 7h minimum.", "#FAC775"))
    if avg_stress_w >= 55:
        insights.append(("🔴", f"Weekly average stress is {avg_stress_w:.0f} — in the High range. Consider adjusting workload.", "#F09595"))
    if exercise_days_w == 0:
        insights.append(("🏃", "No exercise logged this week. Even a 20-min walk helps reduce cortisol.", "#97C459"))
    if avg_study_w > 9:
        insights.append(("📚", f"Averaging {avg_study_w:.1f}h of study/day — above sustainable levels. Plan recovery blocks.", "#FAC775"))
    if not insights:
        insights.append(("✅", "Good week overall! All key metrics are within healthy ranges.", "#97C459"))

    for icon, msg, clr in insights:
        render_html(
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:0.6rem 0.9rem;'
            f'background:{clr}11;border-left:3px solid {clr};border-radius:0 8px 8px 0;margin-bottom:6px;">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span style="font-size:0.85rem;color:var(--text-primary);">{msg}</span></div>')


def _render_month_section(month_df, today):
    render_html('<div class="section-header">🗓️ This Month</div>')
    st.caption(f"{today.strftime('%B %Y')} · {len(month_df)} entries logged")

    if month_df.empty:
        st.info("No entries this month yet.")
        return

    avg_stress_m = month_df["stress_score"].mean()
    avg_sleep_m = month_df["sleep"].mean()
    avg_study_m = month_df["study"].mean()
    mex = month_df.groupby("date")["exercise"].max()
    days_in_month = len(mex)
    exercise_days_m = int((mex.clip(0, 1) > 0).sum())
    avg_screen_m = month_df["screen"].mean() if "screen" in month_df.columns else 0
    avg_anxiety_m = month_df["anxiety"].mean() if "anxiety" in month_df.columns else 0
    dominant_m = month_df["stress_level"].mode()[0] if "stress_level" in month_df.columns else "—"
    dc_m = LEVEL_COLOR.get(dominant_m, "#888")
    best_day = month_df.loc[month_df["stress_score"].idxmin()]
    worst_day = month_df.loc[month_df["stress_score"].idxmax()]

    value_html = (
        f'<h1 style="margin:0;font-size:26px;color:var(--text-primary);font-weight:500;">'
        f'{avg_stress_m:.0f}<span style="font-size:15px;color:var(--text-secondary);font-weight:400;"> / 100</span></h1>'
        f'<p style="margin:2px 0 0;font-size:0.85rem;color:{dc_m};">Dominant: {dominant_m}</p>'
    )
    render_html(aurora_hero_card("Monthly average stress", value_html, progress_ring_svg(avg_stress_m, dc_m),
                                  blob_colors=("aurora-blob-teal", "aurora-blob-purple")))
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    month_cards = [
        ("Monthly Avg Stress", f"{avg_stress_m:.0f}", f"Dominant: {dominant_m}", dc_m, "ti ti-gauge"),
        ("Avg Sleep/Night", f"{avg_sleep_m:.1f}h", "Target: 7–9h", "var(--accent-purple)", "ti ti-moon"),
        ("Exercise Days", f"{exercise_days_m}/{days_in_month}", f"{exercise_days_m} of {days_in_month} logged days",
         "#97C459" if days_in_month > 0 and exercise_days_m / days_in_month >= 0.5 else "#FAC775", "ti ti-run"),
        ("Avg Screen Time", f"{avg_screen_m:.1f}h", "Target: ≤4h/day", "var(--accent-purple)" if avg_screen_m <= 4 else "#F09595", "ti ti-device-mobile"),
    ]
    for col, (label, val, sub, clr, icon) in zip([mc1, mc2, mc3, mc4], month_cards):
        col.markdown(summary_card(label, val, sub, clr, icon_class=icon), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    bd_col, wd_col = st.columns(2)
    best_date_str = pd.to_datetime(best_day["timestamp"]).strftime("%A, %d %b")
    worst_date_str = pd.to_datetime(worst_day["timestamp"]).strftime("%A, %d %b")
    with bd_col:
        render_html(
            f'<div style="background:rgba(99,153,34,0.12);border:1px solid #63992244;'
            f'border-left:4px solid #639922;border-radius:0 12px 12px 0;padding:0.8rem 1rem;">'
            f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">🌿 Best Day This Month</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:#97C459;">{best_date_str}</div>'
            f'<div style="font-size:0.85rem;color:#aaa;">Stress score: {best_day["stress_score"]:.0f} · {best_day.get("stress_level","Low")}</div>'
            f'</div>')
    with wd_col:
        worst_color = LEVEL_COLOR.get(str(worst_day.get("stress_level", "High")), "#E24B4A")
        render_html(
            f'<div style="background:{worst_color}18;border:1px solid {worst_color}44;'
            f'border-left:4px solid {worst_color};border-radius:0 12px 12px 0;padding:0.8rem 1rem;">'
            f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">🔴 Hardest Day This Month</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{worst_color};">{worst_date_str}</div>'
            f'<div style="font-size:0.85rem;color:#aaa;">Stress score: {worst_day["stress_score"]:.0f} · {worst_day.get("stress_level","High")}</div>'
            f'</div>')

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    m_sorted = month_df.sort_values("entry_dt")
    if len(m_sorted) > 1:
        render_html('<div class="section-header">Monthly Trend</div>')
        mfig = make_subplots(specs=[[{"secondary_y": True}]])
        mfig.add_trace(go.Scatter(
            x=m_sorted["date"].apply(lambda d: d.strftime("%d %b")), y=m_sorted["stress_score"],
            mode="lines+markers", name="Stress Score",
            line=dict(color="#AFA9EC", width=2.5, shape="spline"),
            marker=dict(size=7, color=[LEVEL_COLOR.get(l, "#888") for l in m_sorted.get("stress_level", ["Low"] * len(m_sorted))]),
            fill="tozeroy", fillcolor="rgba(83,74,183,0.08)"
        ), secondary_y=False)
        mfig.add_trace(go.Scatter(
            x=m_sorted["date"].apply(lambda d: d.strftime("%d %b")), y=m_sorted["sleep"],
            mode="lines+markers", name="Sleep (hrs)",
            line=dict(color="#97C459", width=2, shape="spline", dash="dot"),
            marker=dict(size=6, color="#97C459")
        ), secondary_y=True)
        mfig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#aaa", size=10), bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
            margin=dict(t=30, b=10, l=10, r=50), height=260,
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#777", size=10)),
            yaxis2=dict(title="Sleep hrs", tickfont=dict(color="#97C459", size=10))
        )
        st.plotly_chart(mfig, use_container_width=True)

    if "stress_level" in month_df.columns:
        dist_col, metric_col = st.columns([1, 1.4])
        with dist_col:
            render_html('<div class="section-header">Level Distribution</div>')
            lc_m = month_df["stress_level"].value_counts()
            fig_dm = go.Figure(go.Pie(
                labels=lc_m.index, values=lc_m.values, hole=0.55,
                marker_colors=[LEVEL_COLOR.get(l, "#888") for l in lc_m.index],
                textfont=dict(size=11),
                hovertemplate='%{label}: %{value} days (%{percent})<extra></extra>'
            ))
            fig_dm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#aaa", size=9), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
                margin=dict(t=10, b=35, l=10, r=10), height=240,
                annotations=[dict(text=f"{len(month_df)}<br>entries", font=dict(size=11, color="#AFA9EC"), showarrow=False)]
            )
            st.plotly_chart(fig_dm, use_container_width=True)

        with metric_col:
            render_html('<div class="section-header">Monthly Averages</div>')
            metrics_data = [
                ("😴 Sleep", f"{avg_sleep_m:.1f}h/night", avg_sleep_m >= 7, "Target ≥7h"),
                ("📚 Study", f"{avg_study_m:.1f}h/day", avg_study_m <= 8, "Target ≤8h"),
                ("📱 Screen", f"{avg_screen_m:.1f}h/day", avg_screen_m <= 4, "Target ≤4h"),
                ("🧘 Anxiety", f"{avg_anxiety_m:.1f}/10", avg_anxiety_m <= 5, "Target ≤5"),
                ("🏃 Exercise days", f"{exercise_days_m}/{days_in_month}",
                 days_in_month > 0 and exercise_days_m / days_in_month >= 0.5, "Target ≥50% of logged days"),
                ("💯 Stress avg", f"{avg_stress_m:.0f}/100", avg_stress_m < 30, "Target <30"),
            ]
            rows_html = ""
            for icon_label, value, good, target in metrics_data:
                clr = "#97C459" if good else "#F09595"
                tick = "✅" if good else "❌"
                rows_html += (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:7px 10px;border-radius:9px;background:rgba(255,255,255,0.03);'
                    f'border:1px solid rgba(255,255,255,0.07);margin-bottom:5px;">'
                    f'<div style="font-size:0.85rem;">{icon_label}</div>'
                    f'<div style="font-size:0.85rem;font-weight:700;color:{clr};">{value}</div>'
                    f'<div style="font-size:0.75rem;color:#555;">{target}</div>'
                    f'<div style="font-size:0.9rem;">{tick}</div>'
                    f'</div>'
                )
            render_html(rows_html)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    m_insights = []
    low_days = len(month_df[month_df["stress_level"] == "Low"]) if "stress_level" in month_df.columns else 0
    crit_days = len(month_df[month_df["stress_level"] == "Critical"]) if "stress_level" in month_df.columns else 0
    if crit_days > 0:
        m_insights.append(("🚨", f"{crit_days} Critical stress day(s) this month. Review what happened on those days.", "#E24B4A"))
    if low_days >= len(month_df) // 2:
        m_insights.append(("🌿", f"{low_days} Low-stress days this month — great resilience!", "#97C459"))
    if avg_sleep_m < 6.5:
        m_insights.append(("💤", f"Monthly average sleep is only {avg_sleep_m:.1f}h — chronic sleep deficit detected.", "#FAC775"))
    ex_rate_m = exercise_days_m / days_in_month if days_in_month > 0 else 0
    if ex_rate_m < 0.5 and days_in_month >= 3:
        m_insights.append(("🏃", f"Exercised on {exercise_days_m} of {days_in_month} logged days ({ex_rate_m*100:.0f}%). Try to hit at least 50% of your logged days.", "#FAC775"))
    if not m_insights:
        m_insights.append(("🎉", "Excellent month! Your wellness indicators are consistently healthy.", "#97C459"))

    for icon, msg, clr in m_insights:
        render_html(
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:0.6rem 0.9rem;'
            f'background:{clr}11;border-left:3px solid {clr};border-radius:0 8px 8px 0;margin-bottom:6px;">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span style="font-size:0.85rem;color:var(--text-primary);">{msg}</span></div>')