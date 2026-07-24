"""Streak counting and weekly goal-progress percentages for the Goals page."""

import pandas as pd


def compute_streaks(hdf: pd.DataFrame, g_sleep, g_study, g_exercise, g_screen) -> dict:
    """Consecutive most-recent sessions (working backwards) meeting each goal."""
    streaks = {"sleep": 0, "study": 0, "exercise": 0, "screen": 0}
    if hdf.empty:
        return streaks

    for col, goal_val, direction, key in [
        ("sleep", g_sleep, "gte", "sleep"),
        ("study", g_study, "lte", "study"),
        ("exercise", g_exercise, "gte", "exercise"),
        ("screen", g_screen, "lte", "screen"),
    ]:
        if col not in hdf.columns:
            continue
        vals = pd.to_numeric(hdf[col], errors='coerce').dropna().tolist()
        s = 0
        for v in reversed(vals):
            if (v >= goal_val if direction == "gte" else v <= goal_val):
                s += 1
            else:
                break
        streaks[key] = s
    return streaks


def week_progress(hdf: pd.DataFrame, g_sleep, g_study, g_exercise, g_screen) -> dict:
    """Percent of the last 7 sessions where each goal was met."""
    pct = {"sleep": 0, "study": 0, "exercise": 0, "screen": 0}
    if hdf.empty:
        return pct

    recent = hdf.tail(7)
    for col, goal_val, direction, key in [
        ("sleep", g_sleep, "gte", "sleep"),
        ("study", g_study, "lte", "study"),
        ("exercise", g_exercise, "gte", "exercise"),
        ("screen", g_screen, "lte", "screen"),
    ]:
        if col not in recent.columns:
            continue
        vals = pd.to_numeric(recent[col], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        met = (vals >= goal_val) if direction == "gte" else (vals <= goal_val)
        pct[key] = int(met.sum() / len(vals) * 100)
    return pct
