"""
The interpretable rule-based stress score (0-100).

This exact formula previously existed three times: once in app.py, once in
the "New Entry" page of dashboard.py, and once in dashboard.py's shared
"latest session" block — each with slightly different variable scaling for
`exercise` (0-7 weekly vs 0/1 daily), which was a real bug risk. This module
is now the only place the formula lives.

`exercise` here is expected in "weekly days" units (0-7), matching how the
ML model was trained (see config.FEATURES). If you only have a daily
yes/no flag, convert it first with `daily_exercise_to_weekly()`.
"""

import numpy as np


def daily_exercise_to_weekly(exercised_today: bool) -> float:
    """
    Convert a single day's yes/no exercise flag into the weekly-days scale
    the formula and ML model expect. Mirrors the original app's assumption
    that one exercised day ≈ 5 out of 7 weekly days.
    """
    return 5.0 if exercised_today else 0.0


def compute_raw_score(
    study_hours: float,
    assignments_pending: float,
    exam_pressure: float,
    academic_performance: float,
    sleep_hours: float,
    exercise_days_per_week: float,
    social_interactions_per_week: float,
    screen_time_hours: float,
    anxiety_level: float,
    financial_stress: float,
    family_support: float,
    peer_pressure: float,
    extracurricular_activities: int,
    relationship_issues: int,
) -> float:
    """Interpretable weighted score before clipping to 0-100."""
    raw = (
        max(0, study_hours - 8) * 3.5
        + assignments_pending * 2.5
        + (exam_pressure - 1) * 5.0
        + max(0, 7 - sleep_hours) * 4.0
        + max(0, 5 - exercise_days_per_week) * 2.0
        + max(0, 8 - social_interactions_per_week) * 1.5
        + max(0, screen_time_hours - 4) * 2.0
        + (anxiety_level - 1) * 4.5
        + (financial_stress - 1) * 3.0
        - (family_support - 1) * 2.5
        - (academic_performance - 1) * 2.0
        + (peer_pressure - 1) * 2.5
        + (5 if extracurricular_activities == 0 else 0)
        + (8 if relationship_issues == 2 else 0)
    )
    return raw


def compute_stress_score(**kwargs) -> int:
    """compute_raw_score(...) clipped to an integer 0-100."""
    raw = compute_raw_score(**kwargs)
    return int(np.clip(raw, 0, 100))


def factor_breakdown(
    study_hours, assignments_pending, exam_pressure,
    sleep_hours, anxiety_level, financial_stress,
    family_support, social_interactions_per_week,
    peer_pressure, screen_time_hours, exercise_days_per_week,
) -> dict:
    """
    Per-factor contribution scores (0-100 each) used in the
    "Factor Analysis" tab / bar chart.
    """
    return {
        "Academic load": min(100, int(
            study_hours / 16 * 50 + assignments_pending / 15 * 30 + exam_pressure / 10 * 20
        )),
        "Sleep deficit": min(100, int(max(0, 8 - sleep_hours) / 6 * 100)),
        "Anxiety": int(anxiety_level / 10 * 100),
        "Financial strain": int(financial_stress / 10 * 100),
        "Social isolation": min(100, int(
            max(0, 10 - family_support) / 9 * 70 + max(0, 8 - social_interactions_per_week) / 8 * 30
        )),
        "Peer pressure": int(peer_pressure / 10 * 100),
        "Screen overuse": min(100, int(max(0, screen_time_hours - 4) / 12 * 100)),
        "Exercise deficit": min(100, int(max(0, 5 - exercise_days_per_week) / 5 * 100)),
    }
