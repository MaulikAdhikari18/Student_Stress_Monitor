"""
Loads the trained model/scaler/meta and produces corrected predictions.

The "correction" logic here — shifting probability mass from the ML class
down to the rule-based class whenever the ML model predicts something
*more* severe than the calibrated score thresholds — previously lived twice
in dashboard.py (once in the New Entry page, once in the shared "latest
session" block), with identical logic copy-pasted. It's now one function:
`predict_stress()`.
"""

import pickle
import numpy as np

from src.config import MODEL_PATH, SCALER_PATH, META_PATH, LABELS, classify_score


def load_model():
    """
    Returns (model, scaler, meta) or (None, None, None) if the model
    hasn't been trained yet (run `python src/train_model.py` first).
    """
    import os
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    return model, scaler, meta


def predict_stress(model, scaler, feature_row: list, stress_score: int):
    """
    Run the ML model and apply the rule-based correction.

    feature_row must already be in FEATURES order (see config.FEATURES).
    stress_score is the interpretable 0-100 score from ml.scoring, used to
    compute the "expected" class via config.classify_score() as a ceiling
    on how severe the ML prediction is allowed to be.

    Returns (level_name: str, proba: np.ndarray, pred_class: int)
    """
    inp = np.array([feature_row])
    inp_scaled = scaler.transform(inp)

    pred_class = int(model.predict(inp_scaled)[0])
    proba = model.predict_proba(inp_scaled)[0].astype(float)

    rule_label = classify_score(stress_score)
    rule_class = LABELS.index(rule_label)

    # The ML model is systematically biased ~1 class higher than the
    # calibrated rule thresholds (training data distribution). Never let
    # it report *more* severe than the rule-based class — shift the
    # probability mass instead of just overwriting, so the displayed
    # confidence bars stay consistent with the corrected label.
    if pred_class > rule_class:
        corrected = proba.copy()
        corrected[rule_class] += corrected[pred_class]
        corrected[pred_class] = 0.0
        proba = corrected
        pred_class = rule_class

    return LABELS[pred_class], proba, pred_class


def predict_or_fallback(model, scaler, feature_row: list, stress_score: int):
    """
    Convenience wrapper: if no model is loaded, fall back to the pure
    rule-based classification with no probability distribution.
    Returns (level_name, proba_or_None, pred_class_or_None).
    """
    if model is None or scaler is None:
        label = classify_score(stress_score)
        return label, None, LABELS.index(label)
    return predict_stress(model, scaler, feature_row, stress_score)
