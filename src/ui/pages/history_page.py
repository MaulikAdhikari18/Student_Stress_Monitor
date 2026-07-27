"""'History' page — full session history with trend charts and raw data export."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS, CHART_COLORS
from src.ui.components import num_card, render_html, dataframe_html, aurora_hero_card, progress_ring_svg


def render(username: str, history_df: pd.DataFrame):
    st.markdown('<div class="main-title">📈 Stress History</div>', unsafe_allow_html=True)
    st.caption(f"All logged sessions for {username}")

    if history_df.empty:
        st.info("No history yet. Head to ✏️ New Entry to log your first session.")
        return

    hdf = history_df.copy()
    for c in ['stress_score', 'sleep', 'study', 'screen', 'anxiety', 'exercise']:
        hdf[c] = pd.to_numeric(hdf[c], errors='coerce')

    x_labels = hdf['day_label'].tolist()
    level_color_map = CHART_COLORS

    # ════════════════════════════════════════════════════
    # SECTION 1 — Stress Score Over Time
    # ════════════════════════════════════════════════════
    render_html('<div class="section-header">📈 Stress Score Over Time</div>')

    avg_stress = hdf['stress_score'].mean()
    min_stress = hdf['stress_score'].min()
    max_stress = hdf['stress_score'].max()
    last_stress = hdf['stress_score'].iloc[-1]
    trend_val = hdf['stress_score'].iloc[-1] - hdf['stress_score'].iloc[-2] if len(hdf) > 1 else 0
    trend_str = (f"↓ {abs(trend_val):.0f} vs prev" if trend_val < 0
                 else (f"↑ {trend_val:.0f} vs prev" if trend_val > 0 else "→ No change"))
    trend_color = "#97C459" if trend_val < 0 else ("#F09595" if trend_val > 0 else "var(--text-muted)")

    latest_level = hdf.get('stress_level', pd.Series(['Low'])).iloc[-1]
    latest_color = level_color_map.get(latest_level, '#AFA9EC')
    value_html = (
        f'<h1 style="margin:0;font-size:26px;color:var(--text-primary);font-weight:500;">'
        f'{last_stress:.0f}<span style="font-size:15px;color:var(--text-secondary);font-weight:400;"> / 100</span></h1>'
        f'<p style="margin:2px 0 0;font-size:0.85rem;color:{trend_color};">{trend_str}</p>'
    )
    render_html(aurora_hero_card("Latest session", value_html, progress_ring_svg(last_stress, latest_color)))
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    nc1, nc2, nc3, nc4, nc5 = st.columns(5)
    for col, lbl, val, sub, clr, icon in [
        (nc1, "Sessions", str(len(hdf)), "total logged", "var(--accent-purple)", "ti ti-calendar-event"),
        (nc2, "Avg Score", f"{avg_stress:.0f}", "out of 100", "var(--accent-purple)", "ti ti-chart-line"),
        (nc3, "Latest Score", f"{last_stress:.0f}", trend_str, trend_color, "ti ti-activity"),
        (nc4, "Best Score", f"{min_stress:.0f}", "lowest stress", "#97C459", "ti ti-mood-smile"),
        (nc5, "Worst Score", f"{max_stress:.0f}", "highest stress", "#F09595", "ti ti-mood-sad"),
    ]:
        col.markdown(num_card(lbl, val, sub, clr, icon_class=icon), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    marker_colors = [level_color_map.get(l, '#AFA9EC') for l in hdf.get('stress_level', ['Low'] * len(hdf))]
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(
        x=x_labels, y=hdf['stress_score'], mode='lines+markers+text',
        text=hdf['stress_score'].apply(lambda v: f"{v:.0f}"),
        textposition='top center', textfont=dict(size=9, color='#ccc'),
        line=dict(color='#AFA9EC', width=2.5, shape='spline'),
        marker=dict(color=marker_colors, size=9, line=dict(width=1.5, color='rgba(0,0,0,0.3)')),
        fill='tozeroy', fillcolor='rgba(83,74,183,0.10)',
        hovertemplate='<b>%{x}</b><br>Score: %{y}<extra></extra>'
    ))
    for threshold, color, label in [(30, '#639922', 'Low'), (55, '#BA7517', 'High'), (75, '#A32D2D', 'Critical')]:
        fig_stress.add_hline(y=threshold, line_dash='dot', line_color=color, opacity=0.45,
                              annotation_text=label, annotation_position='right',
                              annotation_font_color=color, annotation_font_size=10)
    fig_stress.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 115], gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777', size=10)),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#777', size=10)),
        margin=dict(t=20, b=10, l=10, r=60), height=270, showlegend=False
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 2 — Sleep & Study Hours
    # ════════════════════════════════════════════════════
    render_html('<div class="section-header">🛏 Sleep & Study Hours</div>')

    avg_sleep = hdf['sleep'].mean()
    min_sleep = hdf['sleep'].min()
    avg_study = hdf['study'].mean()
    max_study = hdf['study'].max()
    nights_ok = int((hdf['sleep'] >= 7).sum())
    days_ok_st = int((hdf['study'] <= 8).sum())

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    for col, lbl, val, sub, clr, icon in [
        (sc1, "Avg Sleep", f"{avg_sleep:.1f}h", "per night", "var(--accent-purple)", "ti ti-moon"),
        (sc2, "Lowest Sleep", f"{min_sleep:.1f}h", "worst night", "#F09595", "ti ti-alert-triangle"),
        (sc3, "Nights ≥7h", f"{nights_ok}", f"of {len(hdf)} logged", "#97C459", "ti ti-check"),
        (sc4, "Avg Study", f"{avg_study:.1f}h", "per day", "var(--accent-purple)", "ti ti-book"),
        (sc5, "Max Study", f"{max_study:.1f}h", "heaviest day", "#FAC775", "ti ti-alert-triangle"),
        (sc6, "Days ≤8h Study", f"{days_ok_st}", f"of {len(hdf)} logged", "#97C459", "ti ti-check"),
    ]:
        col.markdown(num_card(lbl, val, sub, clr, icon_class=icon), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(
        x=x_labels, y=hdf['sleep'], name='Sleep (hrs)',
        marker_color='rgba(83,74,183,0.6)', marker_line_width=0,
        text=hdf['sleep'].apply(lambda v: f"{v:.1f}"), textposition='outside',
        textfont=dict(size=9, color='#AFA9EC'),
        hovertemplate='Sleep: %{y}h<extra></extra>'
    ), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=x_labels, y=hdf['study'], mode='lines+markers+text', name='Study (hrs)',
        text=hdf['study'].apply(lambda v: f"{v:.1f}"), textposition='top center',
        textfont=dict(size=9, color='#D4537E'),
        line=dict(color='#D4537E', width=2.5, shape='spline'),
        marker=dict(size=7, color='#D4537E'),
        hovertemplate='Study: %{y}h<extra></extra>'
    ), secondary_y=True)
    fig_dual.add_hline(y=7, line_dash='dot', line_color='#AFA9EC', opacity=0.35,
                        annotation_text='7h target', annotation_font_color='#AFA9EC',
                        annotation_font_size=9)
    fig_dual.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#aaa', size=10), bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
        margin=dict(t=30, b=10, l=10, r=50), height=270,
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777', size=10)),
        yaxis2=dict(tickfont=dict(color='#D4537E', size=10))
    )
    st.plotly_chart(fig_dual, use_container_width=True)
    st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 3 — Lifestyle Trends
    # ════════════════════════════════════════════════════
    if all(c in hdf.columns for c in ['screen', 'anxiety', 'exercise']):
        render_html('<div class="section-header">📉 Lifestyle Trends</div>')

        avg_screen = hdf['screen'].mean()
        avg_anxiety = hdf['anxiety'].mean()
        ex_days = int((hdf['exercise'].clip(0, 1) > 0).sum())
        hi_anxiety = int((hdf['anxiety'] >= 7).sum())
        hi_screen = int((hdf['screen'] > 4).sum())

        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        for col, lbl, val, sub, clr, icon in [
            (lc1, "Avg Screen Time", f"{avg_screen:.1f}h", "per day", "#FAC775" if avg_screen > 4 else "#97C459", "ti ti-device-mobile"),
            (lc2, "Days Screen >4h", f"{hi_screen}", f"of {len(hdf)} days", "#FAC775", "ti ti-alert-triangle"),
            (lc3, "Avg Anxiety", f"{avg_anxiety:.1f}", "out of 10", "#F09595" if avg_anxiety >= 6 else "#97C459", "ti ti-brain"),
            (lc4, "Days Anxiety ≥7", f"{hi_anxiety}", "high-anxiety days", "#F09595", "ti ti-alert-triangle"),
            (lc5, "Exercise Days", f"{ex_days}", f"of {len(hdf)} logged",
             "#97C459" if ex_days / max(1, len(hdf)) >= 0.5 else "#F09595", "ti ti-run"),
        ]:
            col.markdown(num_card(lbl, val, sub, clr, icon_class=icon), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        fig_multi = go.Figure()
        for col_, color_, label_ in [
            ('screen', '#FAC775', 'Screen (hrs)'),
            ('anxiety', '#F09595', 'Anxiety (/10)'),
            ('exercise', '#97C459', 'Exercise (0/1)'),
        ]:
            if col_ in hdf.columns:
                fig_multi.add_trace(go.Scatter(
                    x=x_labels, y=hdf[col_], mode='lines+markers+text',
                    text=hdf[col_].apply(lambda v: f"{v:.0f}"),
                    textposition='top center', textfont=dict(size=8, color=color_),
                    name=label_, line=dict(color=color_, width=2, shape='spline'),
                    marker=dict(size=6, color=color_),
                    hovertemplate=f'{label_}: %{{y}}<extra></extra>'
                ))
        fig_multi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#aaa', size=10), bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
            margin=dict(t=30, b=10, l=10, r=10), height=270,
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777', size=10)),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#777', size=10))
        )
        st.plotly_chart(fig_multi, use_container_width=True)
        st.divider()

    # ════════════════════════════════════════════════════
    # SECTION 4 — Distribution & Spread
    # ════════════════════════════════════════════════════
    if 'stress_level' in hdf.columns:
        render_html('<div class="section-header">🍩 Stress Level Distribution & Spread</div>')

        level_counts = hdf['stress_level'].value_counts()
        LEVEL_ICONS = {'Low': 'ti ti-mood-smile', 'Moderate': 'ti ti-mood-neutral',
                       'High': 'ti ti-mood-sad', 'Critical': 'ti ti-alert-triangle'}

        dcols = st.columns(4)
        for i, lvl in enumerate(['Low', 'Moderate', 'High', 'Critical']):
            cnt = int(level_counts.get(lvl, 0))
            pct = cnt / len(hdf) * 100
            clr = COLORS.get(lvl, '#888')
            lvl_data = hdf[hdf['stress_level'] == lvl]['stress_score'].dropna()
            avg_lvl = f"{lvl_data.mean():.0f}" if not lvl_data.empty else "—"
            dcols[i].markdown(
                num_card(f"{lvl} Sessions", f"{cnt}", f"{pct:.0f}% · avg score {avg_lvl}", clr, clr,
                         icon_class=LEVEL_ICONS[lvl]),
                unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        d_pie, d_box = st.columns(2)
        with d_pie:
            fig_donut = go.Figure(go.Pie(
                labels=level_counts.index, values=level_counts.values, hole=0.52,
                marker_colors=[COLORS.get(l, '#888') for l in level_counts.index],
                textinfo='percent+value', textfont=dict(size=11),
                hovertemplate='%{label}: %{value} sessions (%{percent})<extra></extra>'
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#ccc', size=10), bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=10, l=10, r=10), height=280,
                annotations=[dict(text=f"{len(hdf)}<br>sessions",
                                   font=dict(size=13, color='#AFA9EC'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with d_box:
            fig_box = go.Figure()
            box_fill = {'Low': 'rgba(99,153,34,0.25)', 'Moderate': 'rgba(186,117,23,0.25)',
                        'High': 'rgba(153,60,29,0.25)', 'Critical': 'rgba(163,45,45,0.25)'}
            for lvl in ['Low', 'Moderate', 'High', 'Critical']:
                lvl_data = hdf[hdf['stress_level'] == lvl]['stress_score'].dropna()
                if not lvl_data.empty:
                    fig_box.add_trace(go.Box(
                        y=lvl_data, name=lvl,
                        marker_color=COLORS.get(lvl, '#888'),
                        line_color=COLORS.get(lvl, '#888'),
                        fillcolor=box_fill.get(lvl, 'rgba(128,128,128,0.25)'),
                        boxmean=True, boxpoints='all', jitter=0.4, pointpos=-1.6,
                        hovertemplate='Score: %{y}<extra></extra>'
                    ))
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0, 105], gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#777', size=10)),
                xaxis=dict(tickfont=dict(color='#aaa', size=10)),
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📋 View raw data"):
        render_html(dataframe_html(hdf.drop(columns=['id', 'user_id'], errors='ignore')))
        csv = hdf.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, f"{username}_stress_history.csv", "text/csv")