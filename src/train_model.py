"""
Student Stress Monitor — Model Training
Trains XGBoost + Random Forest, saves best model and scaler.
Run from project root: python src/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

BASE      = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(BASE, 'data', 'student_stress_data.csv')
MODELS_DIR= os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    'study_hours', 'assignments_pending', 'exam_pressure',
    'academic_performance', 'sleep_hours', 'exercise_days_per_week',
    'social_interactions_per_week', 'screen_time_hours',
    'anxiety_level', 'financial_stress', 'family_support',
    'peer_pressure', 'extracurricular_activities', 'relationship_issues',
]
TARGET = 'stress_level'
LABELS = ['Low', 'Moderate', 'High', 'Critical']


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES], df[TARGET]


def train():
    print("=" * 55)
    print("  Student Stress Monitor — Model Training")
    print("=" * 55)
    print(f"XGBoost : {'available' if HAS_XGB else 'not installed'}")
    print(f"SMOTE   : {'available' if HAS_SMOTE else 'not installed'}\n")

    X, y = load_data()
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features")
    for k, v in y.value_counts().sort_index().items():
        print(f"  {LABELS[k]:10s}: {v}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"\nSMOTE applied → {len(X_train)} training samples")
    else:
        print("\nSMOTE not found — using class_weight='balanced'")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {}

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14,
        class_weight=None if HAS_SMOTE else 'balanced',
        random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    models['RandomForest'] = (rf, accuracy_score(y_test, rf.predict(X_test_sc)))
    print(f"  Accuracy: {models['RandomForest'][1]:.4f}")

    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train_sc, y_train)
    models['GradBoost'] = (gb, accuracy_score(y_test, gb.predict(X_test_sc)))
    print(f"  Accuracy: {models['GradBoost'][1]:.4f}")

    if HAS_XGB:
        print("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric='mlogloss', random_state=42)
        xgb.fit(X_train_sc, y_train)
        models['XGBoost'] = (xgb, accuracy_score(y_test, xgb.predict(X_test_sc)))
        print(f"  Accuracy: {models['XGBoost'][1]:.4f}")

    best_name  = max(models, key=lambda k: models[k][1])
    best_model = models[best_name][0]
    best_acc   = models[best_name][1]
    print(f"\n✓ Best model: {best_name}  (accuracy={best_acc:.4f})")

    y_pred = best_model.predict(X_test_sc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump({'features': FEATURES, 'labels': LABELS,
                     'best_model': best_name, 'accuracy': round(best_acc, 4)}, f)

    print("Saved → models/model.pkl | scaler.pkl | meta.pkl")
    _plot_confusion(y_test, y_pred)
    _plot_importance(best_model, best_name)
    print("\nDone! Run the app: streamlit run app.py")


def _plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, pad=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=130)
    plt.close()


def _plot_importance(model, name):
    if not hasattr(model, 'feature_importances_'):
        return
    nice_names = {
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
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.index = [nice_names.get(i, i) for i in imp.index]
    colors = ['#534AB7' if v > imp.median() else '#AFA9EC' for v in imp.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color=colors)
    ax.set_title(f'Feature Importance — {name}', fontsize=14, pad=12)
    ax.set_xlabel('Importance score')
    ax.axvline(imp.median(), ls='--', lw=1, color='#888', label='median')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'), dpi=130)
    plt.close()


if __name__ == '__main__':
    train()
