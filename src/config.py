"""
Central configuration for the Student Stress Monitor.

Everything here was previously copy-pasted across app.py and app/dashboard.py
(LABELS, COLORS, EMOJIS, FEATURES, model/db paths). Import from here instead
of redefining it locally — if a label or color needs to change, it changes
in exactly one place.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH    = os.path.join(DATA_DIR, 'stress_monitor.db')

MODEL_PATH  = os.path.join(MODELS_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
META_PATH   = os.path.join(MODELS_DIR, 'meta.pkl')

TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'student_stress_data.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Stress levels ───────────────────────────────────────────────────────

LABELS = ['Low', 'Moderate', 'High', 'Critical']

COLORS = {
    'Low': '#639922',
    'Moderate': '#BA7517',
    'High': '#993C1D',
    'Critical': '#A32D2D',
}

EMOJIS = {
    'Low': '😊',
    'Moderate': '😐',
    'High': '😟',
    'Critical': '😰',
}

# Lighter palette used for some chart traces (donut/box plots) in dashboard.py
CHART_COLORS = {
    'Low': '#639922',
    'Moderate': '#EF9F27',
    'High': '#D85A30',
    'Critical': '#E24B4A',
}

# ─── ML feature schema ───────────────────────────────────────────────────
# Order matters — must match the column order used in train_model.py

FEATURES = [
    'study_hours', 'assignments_pending', 'exam_pressure',
    'academic_performance', 'sleep_hours', 'exercise_days_per_week',
    'social_interactions_per_week', 'screen_time_hours',
    'anxiety_level', 'financial_stress', 'family_support',
    'peer_pressure', 'extracurricular_activities', 'relationship_issues',
]

TARGET = 'stress_level'

# Human-readable names for feature-importance plots
FEATURE_DISPLAY_NAMES = {
    'study_hours': 'Study hours/day',
    'assignments_pending': 'Assignments pending',
    'exam_pressure': 'Exam pressure',
    'academic_performance': 'Academic performance',
    'sleep_hours': 'Sleep hours/night',
    'exercise_days_per_week': 'Exercise days/week',
    'social_interactions_per_week': 'Social interactions',
    'screen_time_hours': 'Screen time hours',
    'anxiety_level': 'Anxiety level',
    'financial_stress': 'Financial stress',
    'family_support': 'Family support',
    'peer_pressure': 'Peer pressure',
    'extracurricular_activities': 'Extracurriculars',
    'relationship_issues': 'Relationship issues',
}

# ─── Rule-based classification thresholds ────────────────────────────────
# Used both as a fallback (no model) and to sanity-check/correct ML predictions

STRESS_THRESHOLDS = {
    'critical': 90,
    'high': 75,
    'moderate': 30,
}


def classify_score(score: float) -> str:
    """Map a 0-100 stress score to a label using the calibrated thresholds."""
    if score >= STRESS_THRESHOLDS['critical']:
        return 'Critical'
    if score >= STRESS_THRESHOLDS['high']:
        return 'High'
    if score >= STRESS_THRESHOLDS['moderate']:
        return 'Moderate'
    return 'Low'
