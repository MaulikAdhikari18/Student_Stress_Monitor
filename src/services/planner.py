"""
Planner scheduling logic — distributing pending tasks across the next 7
days based on deadline urgency, priority, and a stress-adjusted daily
study cap. Ported unchanged from dashboard.py's planner page.
"""

import datetime
import pandas as pd


def generate_weekly_schedule(tasks_df: pd.DataFrame, stress_score: float, daily_study_limit: float):
    """
    Returns (schedule: dict[day_label -> list[task_row]], effective_cap: float)
    """
    today = datetime.date.today()
    days = [(today + datetime.timedelta(days=i)) for i in range(7)]

    # Stress-adjusted daily cap
    if stress_score >= 90:
        cap = max(1.5, daily_study_limit * 0.5)
    elif stress_score >= 75:
        cap = max(2.0, daily_study_limit * 0.7)
    else:
        cap = daily_study_limit

    pending = tasks_df[tasks_df['completed'] == 0].copy()
    if pending.empty:
        return {d.strftime("%a %d %b"): [] for d in days}, cap

    pri_weight = {'High': 0, 'Medium': 1, 'Low': 2}
    pending['pri_w'] = pending['priority'].map(pri_weight).fillna(1)
    pending['deadline_dt'] = pd.to_datetime(pending['deadline'], errors='coerce')
    pending = pending.sort_values(['deadline_dt', 'pri_w'])

    schedule = {d.strftime("%a %d %b"): [] for d in days}
    daily_used = {d.strftime("%a %d %b"): 0.0 for d in days}

    for _, task in pending.iterrows():
        try:
            dl = pd.to_datetime(task['deadline']).date()
        except Exception:
            dl = today + datetime.timedelta(days=6)

        assigned = False
        for day in days:
            day_str = day.strftime("%a %d %b")
            if day > dl:
                break
            if daily_used[day_str] + task['duration_h'] <= cap:
                schedule[day_str].append(task)
                daily_used[day_str] += task['duration_h']
                assigned = True
                break

        if not assigned:
            for day in days:
                day_str = day.strftime("%a %d %b")
                if daily_used[day_str] < cap:
                    schedule[day_str].append(task)
                    daily_used[day_str] += task['duration_h']
                    break

    return schedule, cap


def get_break_schedule(stress_score: float):
    """Returns (study_block_minutes, break_minutes, label) based on stress."""
    if stress_score >= 90:
        return 20, 10, "🔴 Critical stress — short blocks, frequent breaks"
    elif stress_score >= 75:
        return 25, 8, "🟠 High stress — Pomodoro 25/8 recommended"
    elif stress_score >= 30:
        return 35, 7, "🟡 Moderate stress — 35 min focus, 7 min break"
    else:
        return 50, 10, "🟢 Low stress — deep work 50/10 recommended"
