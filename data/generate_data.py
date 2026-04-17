import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

study_hours      = np.random.uniform(2, 14, n)
assignments      = np.random.randint(0, 15, n)
exam_pressure    = np.random.randint(1, 11, n)
performance      = np.random.randint(1, 11, n)
sleep_hours      = np.random.uniform(3, 10, n)
exercise_days    = np.random.randint(0, 8, n)
social_interactions = np.random.randint(0, 20, n)
screen_time      = np.random.uniform(1, 14, n)
anxiety_level    = np.random.randint(1, 11, n)
financial_stress = np.random.randint(1, 11, n)
family_support   = np.random.randint(1, 11, n)
peer_pressure    = np.random.randint(1, 11, n)
extracurricular  = np.random.randint(0, 3, n)
relationship_issues = np.random.randint(0, 3, n)

score = (
    np.clip(study_hours - 8, 0, None) * 3.5
    + assignments * 2.5
    + (exam_pressure - 1) * 5.0
    + np.clip(7 - sleep_hours, 0, None) * 4.0
    + np.clip(5 - exercise_days, 0, None) * 2.0
    + np.clip(8 - social_interactions, 0, None) * 1.5
    + np.clip(screen_time - 4, 0, None) * 2.0
    + (anxiety_level - 1) * 4.5
    + (financial_stress - 1) * 3.0
    - (family_support - 1) * 2.5
    - (performance - 1) * 2.0
    + (peer_pressure - 1) * 2.5
    + (extracurricular == 0).astype(int) * 5
    + (relationship_issues == 2).astype(int) * 8
    + np.random.normal(0, 5, n)
)
score = np.clip(score, 0, 100)

def label(s):
    if s < 30:  return 0  # Low
    if s < 55:  return 1  # Moderate
    if s < 75:  return 2  # High
    return 3               # Critical

stress_level = np.array([label(s) for s in score])

df = pd.DataFrame({
    'study_hours': np.round(study_hours, 1),
    'assignments_pending': assignments,
    'exam_pressure': exam_pressure,
    'academic_performance': performance,
    'sleep_hours': np.round(sleep_hours, 1),
    'exercise_days_per_week': exercise_days,
    'social_interactions_per_week': social_interactions,
    'screen_time_hours': np.round(screen_time, 1),
    'anxiety_level': anxiety_level,
    'financial_stress': financial_stress,
    'family_support': family_support,
    'peer_pressure': peer_pressure,
    'extracurricular_activities': extracurricular,
    'relationship_issues': relationship_issues,
    'stress_score': np.round(score, 1),
    'stress_level': stress_level,
})

df.to_csv('student_stress_data.csv', index=False)
print(f"Dataset saved: {len(df)} rows")
print(df['stress_level'].value_counts().sort_index().rename({0:'Low',1:'Moderate',2:'High',3:'Critical'}))
