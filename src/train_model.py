"""
Student Stress Monitor -- Model Training
Trains RandomForest + GradientBoosting (+ XGBoost if installed), picks the
best one via stratified cross-validation, tunes its hyperparameters, then
saves the final model and scaler.

Run from project root: python src/train_model.py

CHANGED from the original version:
- Model selection is now based on 5-fold stratified cross-validation
  (mean +/- std accuracy) instead of a single train/test split, which is
  noisy -- especially with a 4-class problem and a modest dataset. A
  single split can make a mediocre model look artificially good (or bad)
  by chance.
- After picking the best-performing model *type* via CV, that one model
  gets a RandomizedSearchCV pass to tune its hyperparameters, rather than
  using fixed guessed values for all three algorithms.
- meta.pkl now records both the CV accuracy and the final held-out test
  accuracy, plus the tuned hyperparameters, so it's clear which number
  means what.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import (
    FEATURES, TARGET, LABELS, TRAINING_DATA_PATH, MODELS_DIR,
    MODEL_PATH, SCALER_PATH, META_PATH, FEATURE_DISPLAY_NAMES,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

CV_FOLDS = 5

# Hyperparameter search space for the final tuning pass. Kept modest
# (n_iter=20, cv=3) so this stays a "run it and get coffee" script, not
# an overnight job -- feel free to widen these ranges if you want to
# spend more compute chasing the last percentage point.
PARAM_DISTRIBUTIONS = {
    'RandomForest': {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [8, 12, 16, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    },
    'GradBoost': {
        'n_estimators': [150, 200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.85, 1.0],
    },
    'XGBoost': {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [4, 5, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
    },
}


def load_data():
    df = pd.read_csv(TRAINING_DATA_PATH)
    return df[FEATURES], df[TARGET]


def _make_base_estimators():
    estimators = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=14,
            class_weight=None if HAS_SMOTE else 'balanced',
            random_state=42, n_jobs=-1),
        'GradBoost': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    if HAS_XGB:
        estimators['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric='mlogloss', random_state=42)
    return estimators


def _cv_pipeline(estimator):
    """
    Wraps the estimator in a SMOTE + classifier pipeline for
    cross-validation, so SMOTE is only ever applied to each fold's
    training portion (never the validation portion) -- avoiding the
    classic SMOTE-before-split data leakage bug that inflates CV scores.
    Falls back to the bare estimator if imbalanced-learn isn't installed.
    """
    if HAS_SMOTE:
        return ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', estimator)])
    return estimator


def train():
    print("=" * 55)
    print("  Student Stress Monitor -- Model Training")
    print("=" * 55)
    print(f"XGBoost : {'available' if HAS_XGB else 'not installed'}")
    print(f"SMOTE   : {'available' if HAS_SMOTE else 'not installed'}\n")

    X, y = load_data()
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features")
    for k, v in y.value_counts().sort_index().items():
        print(f"  {LABELS[k]:10s}: {v}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Step 1: pick the best model TYPE via stratified cross-validation ──
    print(f"\n{'-'*55}\nStep 1: {CV_FOLDS}-fold stratified cross-validation\n{'-'*55}")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    estimators = _make_base_estimators()
    cv_results = {}
    for name, est in estimators.items():
        pipeline = _cv_pipeline(est)
        scores = cross_val_score(pipeline, X_train_sc, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        print(f"  {name:12s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k][0])
    best_cv_mean, best_cv_std = cv_results[best_name]
    print(f"\n✓ Best model type by CV: {best_name}  ({best_cv_mean:.4f} +/- {best_cv_std:.4f})")

    # ── Step 2: tune the winning model's hyperparameters ──
    print(f"\n{'-'*55}\nStep 2: Hyperparameter tuning for {best_name}\n{'-'*55}")
    base_estimator = estimators[best_name]
    search_space = PARAM_DISTRIBUTIONS[best_name]

    if HAS_SMOTE:
        # Prefix param names for the pipeline step
        tune_pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', base_estimator)])
        prefixed_space = {f'clf__{k}': v for k, v in search_space.items()}
        search = RandomizedSearchCV(
            tune_pipeline, prefixed_space, n_iter=20, cv=3,
            scoring='accuracy', random_state=42, n_jobs=-1)
        search.fit(X_train_sc, y_train)
        best_params = {k.replace('clf__', ''): v for k, v in search.best_params_.items()}
    else:
        search = RandomizedSearchCV(
            base_estimator, search_space, n_iter=20, cv=3,
            scoring='accuracy', random_state=42, n_jobs=-1)
        search.fit(X_train_sc, y_train)
        best_params = search.best_params_

    print(f"  Best params: {best_params}")
    print(f"  Best tuning CV score: {search.best_score_:.4f}")

    # ── Step 3: fit the final tuned model on the full training set ──
    final_estimator = estimators[best_name].__class__(**{
        **estimators[best_name].get_params(), **best_params
    })
    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_train_final, y_train_final = sm.fit_resample(X_train_sc, y_train)
    else:
        X_train_final, y_train_final = X_train_sc, y_train

    final_estimator.fit(X_train_final, y_train_final)
    test_acc = accuracy_score(y_test, final_estimator.predict(X_test_sc))
    print(f"\n✓ Final held-out test accuracy: {test_acc:.4f}")

    y_pred = final_estimator.predict(X_test_sc)
    print("\nClassification Report (held-out test set):")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_estimator, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(META_PATH, 'wb') as f:
        pickle.dump({
            'features': FEATURES, 'labels': LABELS,
            'best_model': best_name,
            'cv_accuracy': round(best_cv_mean, 4),
            'cv_accuracy_std': round(best_cv_std, 4),
            'test_accuracy': round(test_acc, 4),
            'accuracy': round(test_acc, 4),  # kept for backward-compat with app.py's meta.get('accuracy')
            'best_params': best_params,
        }, f)

    print("\nSaved -> models/model.pkl | scaler.pkl | meta.pkl")
    _plot_confusion(y_test, y_pred)
    _plot_importance(final_estimator, best_name)
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
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.index = [FEATURE_DISPLAY_NAMES.get(i, i) for i in imp.index]
    colors = ['#534AB7' if v > imp.median() else '#AFA9EC' for v in imp.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color=colors)
    ax.set_title(f'Feature Importance -- {name}', fontsize=14, pad=12)
    ax.set_xlabel('Importance score')
    ax.axvline(imp.median(), ls='--', lw=1, color='#888', label='median')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'), dpi=130)
    plt.close()


if __name__ == '__main__':
    train()