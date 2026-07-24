"""
'Planner' page — task list, stress-adaptive break/Pomodoro recommendation,
a working study timer (via session_state), auto-generated weekly schedule,
and upcoming deadline countdowns.
"""

import datetime
import time

import pandas as pd
import streamlit as st

from src.db.tasks import add_task, load_tasks, toggle_task, delete_task
from src.services.planner import generate_weekly_schedule, get_break_schedule
from src.ui.components import render_html


def render(user_id: int, stress_score: float, saved_goals: dict):
    tasks_df = load_tasks(user_id)

    st.markdown("#### 📅 Study Planner")
    st.caption("Add subjects and deadlines — the planner builds your week automatically, "
               "adjusting daily hours based on your current stress level.")

    block_min, break_min, break_label = get_break_schedule(stress_score)
    render_html(f"""
    <div class="break-box">
        <div style="font-size:0.78rem;font-weight:600;color:#AFA9EC;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
            ⏱ Recommended study rhythm for today
        </div>
        <div>
            <span class="break-stat">{block_min}m</span>
            <span style="font-size:0.9rem;color:inherit;opacity:0.7;">study block</span>
            &nbsp;→&nbsp;
            <span class="break-stat">{break_min}m</span>
            <span style="font-size:0.9rem;color:inherit;opacity:0.7;">break</span>
        </div>
        <div style="font-size:0.85rem;opacity:0.7;margin-top:4px;">{break_label}</div>
    </div>
    """)

    _render_study_timer(block_min, break_min)

    st.divider()
    with st.expander("➕ Add a new task", expanded=tasks_df.empty):
        with st.form("add_task_form", clear_on_submit=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                t_subject = st.text_input("Subject / Course", placeholder="e.g. Mathematics")
                t_topic = st.text_input("Topic (optional)", placeholder="e.g. Integration by parts")
                t_deadline = st.date_input("Deadline",
                                            value=datetime.date.today() + datetime.timedelta(days=3),
                                            min_value=datetime.date.today())
            with fc2:
                t_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
                t_duration = st.slider("Estimated hours needed", 0.5, 8.0, 1.5, 0.5)
                st.markdown("")
                st.markdown("")
                submitted = st.form_submit_button("Add task →", use_container_width=True, type="primary")
            if submitted:
                if not t_subject.strip():
                    st.error("Please enter a subject name.")
                else:
                    add_task(user_id, t_subject.strip(), t_topic.strip(),
                              t_deadline.strftime("%Y-%m-%d"), t_priority, t_duration)
                    st.success(f"✅ Task added: {t_subject}")
                    st.rerun()

    if tasks_df.empty:
        st.info("No tasks yet — add your first task above to generate your study plan.")
        return

    st.divider()
    pc1, pc2, pc3 = st.columns([2, 2, 2])
    with pc1:
        daily_limit = st.slider("Max study hours per day", 1.0, 12.0,
                                 float(saved_goals.get("goal_study", 8.0)), 0.5,
                                 help="Stress level may reduce this automatically")
    with pc2:
        show_done = st.toggle("Show completed tasks", value=False)
    with pc3:
        st.metric("Total tasks", str(len(tasks_df)))
        st.metric("Pending", str(len(tasks_df[tasks_df['completed'] == 0])))

    st.divider()
    _render_task_list(tasks_df, show_done)

    st.divider()
    _render_weekly_schedule(tasks_df, stress_score, daily_limit)

    st.divider()
    _render_deadlines(tasks_df)


def _render_study_timer(block_min: int, break_min: int):
    st.markdown("#### ⏱ Study Timer")

    ss = st.session_state
    ss.setdefault('timer_running', False)
    ss.setdefault('timer_phase', 'study')
    ss.setdefault('timer_remaining', block_min * 60)
    ss.setdefault('timer_start_time', None)
    ss.setdefault('timer_elapsed', 0)
    ss.setdefault('sessions_done', 0)
    ss.setdefault('timer_log', [])
    ss.setdefault('custom_study_min', block_min)
    ss.setdefault('custom_break_min', break_min)

    t_cfg1, t_cfg2, t_cfg3 = st.columns([2, 2, 2])
    with t_cfg1:
        timer_mode = st.selectbox("Timer Mode", ["🍅 Pomodoro (stress-adjusted)", "⚙️ Custom"], key="timer_mode_sel")
    with t_cfg2:
        if "Custom" in timer_mode:
            c_study = st.number_input("Study block (min)", 5, 120, ss['custom_study_min'], 5, key="custom_study_inp")
            ss['custom_study_min'] = c_study
        else:
            render_html(
                f'<div style="padding:0.6rem 0.8rem;background:rgba(83,74,183,0.15);'
                f'border-radius:8px;font-size:0.88rem;margin-top:1.6rem;">'
                f'📋 Using <strong>{block_min}m/{break_min}m</strong>'
                f' based on your stress level</div>')
    with t_cfg3:
        if "Custom" in timer_mode:
            c_break = st.number_input("Break (min)", 1, 30, ss['custom_break_min'], 1, key="custom_break_inp")
            ss['custom_break_min'] = c_break

    active_study_min = ss['custom_study_min'] if "Custom" in timer_mode else block_min
    active_break_min = ss['custom_break_min'] if "Custom" in timer_mode else break_min

    if not ss['timer_running']:
        if ss['timer_phase'] == 'study':
            ss['timer_remaining'] = active_study_min * 60 - ss['timer_elapsed']
        else:
            ss['timer_remaining'] = active_break_min * 60 - ss['timer_elapsed']

    if ss['timer_running'] and ss['timer_start_time']:
        elapsed_now = int((datetime.datetime.now() - ss['timer_start_time']).total_seconds()) + ss['timer_elapsed']
        total_phase = (active_study_min if ss['timer_phase'] == 'study' else active_break_min) * 60
        remaining = max(0, total_phase - elapsed_now)
    else:
        remaining = ss['timer_remaining']
        elapsed_now = ss['timer_elapsed']

    mins, secs = divmod(int(remaining), 60)
    phase_cls = "timer-phase-study" if ss['timer_phase'] == 'study' else "timer-phase-break"
    phase_icon = "📖" if ss['timer_phase'] == 'study' else "☕"
    phase_lbl = "Study Block" if ss['timer_phase'] == 'study' else "Break Time"

    total_secs = (active_study_min if ss['timer_phase'] == 'study' else active_break_min) * 60
    pct_done = max(0, min(100, int((1 - remaining / max(1, total_secs)) * 100)))

    render_html(f"""
    <div class="timer-container {phase_cls}">
        <div class="timer-label">{phase_icon} {phase_lbl} — Session #{ss['sessions_done'] + 1}</div>
        <div class="timer-display">{mins:02d}:{secs:02d}</div>
        <div style="margin-top:1rem;background:rgba(255,255,255,0.1);
                    border-radius:8px;height:8px;overflow:hidden;">
            <div style="width:{pct_done}%;height:100%;
                        background:{'#639922' if ss['timer_phase']=='study' else '#534AB7'};
                        border-radius:8px;transition:width 0.5s ease;"></div>
        </div>
        <div style="font-size:0.8rem;color:#aaa;margin-top:6px;">{pct_done}% complete</div>
    </div>""")

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("▶ Start" if not ss['timer_running'] else "⏸ Pause",
                     use_container_width=True, type="primary"):
            if not ss['timer_running']:
                ss['timer_running'] = True
                ss['timer_start_time'] = datetime.datetime.now()
            else:
                ss['timer_running'] = False
                ss['timer_elapsed'] = elapsed_now
                ss['timer_remaining'] = remaining
            st.rerun()

    with btn2:
        if st.button("⏭ Skip Phase", use_container_width=True):
            now_str = datetime.datetime.now().strftime("%H:%M")
            if ss['timer_phase'] == 'study':
                ss['sessions_done'] += 1
                ss['timer_log'].append(
                    f"✅ {now_str} — Study block #{ss['sessions_done']} ({active_study_min}m)")
                ss['timer_phase'] = 'break'
                next_secs = active_break_min * 60
            else:
                ss['timer_log'].append(f"☕ {now_str} — Break complete")
                ss['timer_phase'] = 'study'
                next_secs = active_study_min * 60
            ss['timer_running'] = False
            ss['timer_remaining'] = next_secs
            ss['timer_elapsed'] = 0
            ss['timer_start_time'] = None
            st.rerun()

    with btn3:
        if st.button("🔄 Reset", use_container_width=True):
            ss['timer_running'] = False
            ss['timer_phase'] = 'study'
            ss['timer_remaining'] = active_study_min * 60
            ss['timer_elapsed'] = 0
            ss['timer_start_time'] = None
            st.rerun()

    with btn4:
        total_study_done = ss['sessions_done'] * active_study_min
        render_html(
            f'<div style="text-align:center;padding:0.4rem;background:rgba(83,74,183,0.15);'
            f'border-radius:8px;font-size:0.82rem;">'
            f'🔥 <strong>{ss["sessions_done"]}</strong> sessions<br>'
            f'<span style="color:#AFA9EC;">{total_study_done}m studied</span></div>')

    if ss['timer_running']:
        time.sleep(1)
        st.rerun()

    if ss['timer_log']:
        with st.expander(f"📋 Session log ({len(ss['timer_log'])} entries)", expanded=False):
            for entry in reversed(ss['timer_log']):
                render_html(f'<div class="session-log-row">⏺ {entry}</div>')
            if st.button("🗑 Clear log"):
                ss['timer_log'] = []
                st.rerun()


def _render_task_list(tasks_df: pd.DataFrame, show_done: bool):
    st.markdown("#### 🔢 Priority task list")
    st.caption("Sorted by deadline then priority — tackle from the top.")

    pending_tasks = tasks_df[tasks_df['completed'] == 0].copy()
    done_tasks = tasks_df[tasks_df['completed'] == 1].copy()

    pri_weight = {'High': 0, 'Medium': 1, 'Low': 2}
    pending_tasks['pri_w'] = pending_tasks['priority'].map(pri_weight).fillna(1)
    pending_tasks['deadline_dt'] = pd.to_datetime(pending_tasks['deadline'], errors='coerce')
    pending_tasks = pending_tasks.sort_values(['deadline_dt', 'pri_w'])

    def days_left_str(deadline_str):
        try:
            dl = datetime.date.fromisoformat(str(deadline_str)[:10])
            diff = (dl - datetime.date.today()).days
            if diff < 0:
                return "⚠️ Overdue"
            if diff == 0:
                return "🔥 Due today"
            if diff == 1:
                return "⏰ Due tomorrow"
            return f"📅 {diff} days left"
        except Exception:
            return ""

    display_tasks = pd.concat([pending_tasks, done_tasks]) if show_done else pending_tasks

    for _, row in display_tasks.iterrows():
        task_id = int(row['id'])
        is_done = int(row['completed']) == 1
        pri_cls = f"pri-{row['priority'].lower()}"
        dl_str = days_left_str(row['deadline'])
        topic_str = f" — {row['topic']}" if row['topic'] else ""
        dur_str = f"{row['duration_h']}h"

        col_chk, col_info, col_del = st.columns([0.5, 8, 0.8])
        with col_chk:
            if st.button("✅" if is_done else "⬜", key=f"chk_{task_id}", help="Toggle complete"):
                toggle_task(task_id, is_done)
                st.rerun()
        with col_info:
            done_style = "opacity:0.45;text-decoration:line-through;" if is_done else ""
            render_html(f"""
            <div class="task-row" style="{done_style}">
                <span class="{pri_cls}">{row['priority']}</span>
                <span style="font-weight:600;">{row['subject']}</span>
                <span style="opacity:0.65;">{topic_str}</span>
                <span style="margin-left:auto;opacity:0.55;font-size:0.82rem;">
                    {dur_str} &nbsp;|&nbsp; {dl_str}
                </span>
            </div>""")
        with col_del:
            if st.button("🗑", key=f"del_{task_id}", help="Delete task"):
                delete_task(task_id)
                st.rerun()


def _render_weekly_schedule(tasks_df: pd.DataFrame, stress_score: float, daily_limit: float):
    st.markdown("#### 🗓 Your 7-day study schedule")

    pending_only = tasks_df[tasks_df['completed'] == 0].copy()
    schedule, effective_cap = generate_weekly_schedule(pending_only, stress_score, daily_limit)

    if stress_score >= 55:
        st.info(f"⚠️ Your stress score is **{stress_score}/100** — daily study cap "
                f"has been reduced to **{effective_cap:.1f}h/day** to protect your wellbeing.")

    days_per_row = 4
    day_items = list(schedule.items())

    for row_start in range(0, 7, days_per_row):
        row_days = day_items[row_start:row_start + days_per_row]
        cols = st.columns(len(row_days))
        for col, (day_str, day_tasks) in zip(cols, row_days):
            with col:
                total_h = sum(t['duration_h'] for t in day_tasks)
                load_color = ("#A32D2D" if total_h >= effective_cap * 0.9
                              else "#BA7517" if total_h >= effective_cap * 0.6
                              else "#639922")
                render_html(f"""
                <div class="day-card">
                    <div class="day-header">{day_str}</div>
                    <div style="font-size:0.78rem;color:{load_color};
                                margin-bottom:8px;font-weight:600;">
                        {total_h:.1f}h / {effective_cap:.1f}h
                    </div>""")

                if day_tasks:
                    for t in day_tasks:
                        pri_cls = f"pri-{t['priority'].lower()}"
                        render_html(f"""
                        <div style="font-size:0.82rem;padding:3px 0;
                                    border-bottom:0.5px solid rgba(255,255,255,0.06);">
                            <span class="{pri_cls}">{t['priority'][0]}</span>
                            &nbsp;<strong>{t['subject']}</strong>
                            <span style="opacity:0.55;"> {t['duration_h']}h</span>
                        </div>""")
                else:
                    render_html('<div style="font-size:0.82rem;opacity:0.4;padding:4px 0;">Rest day 🌿</div>')

                render_html("</div>")


def _render_deadlines(tasks_df: pd.DataFrame):
    st.markdown("#### ⏳ Upcoming deadlines")
    today = datetime.date.today()
    upcoming = tasks_df[tasks_df['completed'] == 0].copy()
    upcoming['deadline_dt'] = pd.to_datetime(upcoming['deadline'], errors='coerce')
    upcoming = upcoming.dropna(subset=['deadline_dt'])
    upcoming['days_left'] = upcoming['deadline_dt'].apply(lambda x: (x.date() - today).days)
    upcoming = upcoming.sort_values('days_left').head(6)

    if upcoming.empty:
        st.success("🎉 No upcoming deadlines — you're all caught up!")
        return

    dcols = st.columns(min(3, len(upcoming)))
    for i, (_, row) in enumerate(upcoming.iterrows()):
        dl = int(row['days_left'])
        color = "#A32D2D" if dl <= 1 else "#BA7517" if dl <= 3 else "#639922"
        label = "⚠️ Overdue" if dl < 0 else "🔥 Today" if dl == 0 else f"{dl}d left"
        with dcols[i % 3]:
            render_html(f"""
            <div class="day-card" style="text-align:center;">
                <div style="font-size:2rem;font-weight:700;color:{color};">
                    {label}
                </div>
                <div style="font-weight:600;margin-top:4px;">
                    {row['subject']}
                </div>
                <div style="font-size:0.8rem;opacity:0.55;margin-top:2px;">
                    {row.get('topic', '') or ''}
                </div>
                <div style="font-size:0.78rem;opacity:0.45;margin-top:4px;">
                    Due: {str(row['deadline'])[:10]}
                </div>
            </div>""")
