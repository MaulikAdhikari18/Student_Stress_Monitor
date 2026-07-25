"""
Synthetic training data generator for the Student Stress Monitor.

REWRITTEN. Two real bugs in the original version, discovered while
investigating why the trained model was stuck at ~83% "accuracy" that
was actually just a majority-class guesser (0% recall on Low, 42% on
Moderate, 95% on Critical):

1. Its own label() cutoffs (<30 Low, <55 Moderate, <75 High, else
   Critical) never matched config.classify_score()'s actual thresholds
   (<30 Low, <75 Moderate, <90 High, else Critical) -- the thing the
   live app and the ML-correction logic actually use. Training labels
   and the app's rule-based logic were never speaking the same language.

2. Under the shared scoring formula, ~71-80% of randomly generated
   student profiles land in "Critical" (the formula's natural output
   distribution centers well above 90) -- so *even with correct*
   thresholds, uniform random sampling produces a heavily skewed dataset.

Fix: generate a large pool of candidate profiles, score and label every
one using the *exact same* functions the live app uses
(src.ml.scoring.compute_raw_score + src.config.classify_score -- not a
second copy of the formula), then subsample down to an equal count per
class. This guarantees training labels can never drift out of sync with
the app's own logic again, since there's only one formula and one set of
thresholds, imported, not duplicated.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import FEATURES, LABELS, classify_score
from src.ml.scoring import compute_raw_score

np.random.seed(42)

N_CANDIDATES = 300_000   # big pool so even the rarest class (~1.4% "Low") has plenty to draw from
TARGET_PER_CLASS = 2500  # -> 10000 rows total, perfectly balanced (bumped from 2000 -- more data helped)


def _sample_candidates(n):
    return pd.DataFrame({
        'study_hours': np.round(np.random.uniform(2, 14, n), 1),
        'assignments_pending': np.random.randint(0, 15, n),
        'exam_pressure': np.random.randint(1, 11, n),
        'academic_performance': np.random.randint(1, 11, n),
        'sleep_hours': np.round(np.random.uniform(3, 10, n), 1),
        'exercise_days_per_week': np.random.randint(0, 8, n),
        'social_interactions_per_week': np.random.randint(0, 20, n),
        'screen_time_hours': np.round(np.random.uniform(1, 14, n), 1),
        'anxiety_level': np.random.randint(1, 11, n),
        'financial_stress': np.random.randint(1, 11, n),
        'family_support': np.random.randint(1, 11, n),
        'peer_pressure': np.random.randint(1, 11, n),
        'extracurricular_activities': np.random.randint(0, 3, n),
        'relationship_issues': np.random.randint(0, 3, n),
    })


def _score_and_label(df: pd.DataFrame) -> pd.DataFrame:
    scores = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        raw = compute_raw_score(
            study_hours=row.study_hours,
            assignments_pending=row.assignments_pending,
            exam_pressure=row.exam_pressure,
            academic_performance=row.academic_performance,
            sleep_hours=row.sleep_hours,
            exercise_days_per_week=row.exercise_days_per_week,
            social_interactions_per_week=row.social_interactions_per_week,
            screen_time_hours=row.screen_time_hours,
            anxiety_level=row.anxiety_level,
            financial_stress=row.financial_stress,
            family_support=row.family_support,
            peer_pressure=row.peer_pressure,
            extracurricular_activities=row.extracurricular_activities,
            relationship_issues=row.relationship_issues,
        )
        scores[i] = raw
    # NOISE_STD lowered from 5 -> 2.5 (verified empirically: this alone was
    # worth ~+2.5-3.5 points of cross-validated accuracy at every dataset
    # size tested, with no downside -- going lower still (1.0, 0.0) gave
    # almost no further gain, since the features themselves are coarse
    # integers, hitting a natural ceiling on separability). This isn't
    # about making the data "less realistic" -- day-to-day noise in a real
    # student's stress is real, but +/-5 points was large enough to make a
    # meaningful fraction of the *labels themselves* closer to a coin flip
    # than a reflection of the input features, which no amount of model
    # tuning or extra data can fix.
    scores = np.clip(scores + np.random.normal(0, 2.5, len(df)), 0, 100)
    df = df.copy()
    df['stress_score'] = np.round(scores, 1)
    df['stress_level'] = [LABELS.index(classify_score(s)) for s in scores]
    return df


def main():
    print(f"Generating {N_CANDIDATES:,} candidate profiles...")
    candidates = _sample_candidates(N_CANDIDATES)
    candidates = _score_and_label(candidates)

    raw_counts = candidates['stress_level'].value_counts().sort_index()
    print("\nRaw candidate pool distribution (before balancing):")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:10s}: {raw_counts.get(i, 0):7d}  ({raw_counts.get(i, 0) / len(candidates) * 100:.2f}%)")

    balanced_parts = []
    for i, lbl in enumerate(LABELS):
        available = candidates[candidates['stress_level'] == i]
        take = min(TARGET_PER_CLASS, len(available))
        if take < TARGET_PER_CLASS:
            print(f"\n⚠️  Only {take} '{lbl}' candidates available (wanted {TARGET_PER_CLASS}). "
                  f"Increase N_CANDIDATES if you need the full target.")
        balanced_parts.append(available.sample(n=take, random_state=42))

    df = pd.concat(balanced_parts, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df = df[FEATURES + ['stress_score', 'stress_level']]  # consistent column order

    df.to_csv('student_stress_data.csv', index=False)

    print(f"\nDataset saved: {len(df)} rows -> data/student_stress_data.csv")
    print("\nFinal balanced distribution:")
    final_counts = df['stress_level'].value_counts().sort_index()
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:10s}: {final_counts.get(i, 0)}")

    print("\nScore range by class (sanity check -- should match config.STRESS_THRESHOLDS):")
    summary = df.groupby('stress_level')['stress_score'].agg(['min', 'max', 'mean'])
    summary.index = [LABELS[i] for i in summary.index]
    print(summary)


if __name__ == '__main__':
    main()