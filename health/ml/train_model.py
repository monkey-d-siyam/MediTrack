"""
MediTrack — ML Training Pipeline (Upgraded)
============================================

Model        : GradientBoostingClassifier (ensemble, handles non-linearity well)
Features     : heart_rate, sleep_hours, steps, bmi
Labels       : 0=Low Risk, 1=Medium Risk, 2=High Risk

Outputs saved to health/ml/:
  risk_model.pkl       — trained Pipeline (scaler + classifier)
  evaluation.json      — accuracy, per-class metrics, confusion matrix
  feature_importance.json — ranked feature importances

Usage:
    python health/ml/train_model.py
"""

import os
import json
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR             = os.path.dirname(__file__)
MODEL_PATH         = os.path.join(ML_DIR, 'risk_model.pkl')
EVAL_PATH          = os.path.join(ML_DIR, 'evaluation.json')
IMPORTANCE_PATH    = os.path.join(ML_DIR, 'feature_importance.json')

FEATURE_NAMES = ['Heart Rate (bpm)', 'Sleep Hours', 'Daily Steps', 'BMI']
LABEL_NAMES   = ['Low Risk', 'Medium Risk', 'High Risk']


# ─── Synthetic Dataset Generator ──────────────────────────────────────────────

def generate_dataset(n_samples: int = 5000, seed: int = 42) -> tuple:
    """
    Generate a realistic synthetic health dataset.

    Profile distributions are based on published population health norms:
      - Low (40%):    Ideal ranges for all four metrics
      - Medium (35%): Slightly out-of-range on 1–2 metrics
      - High (25%):   Notably out-of-range on 2–4 metrics

    Gaussian noise is added to prevent the model from learning
    perfectly clean boundaries.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        risk = rng.choice([0, 1, 2], p=[0.40, 0.35, 0.25])

        if risk == 0:   # ── Low Risk ──────────────────────────────────────────
            hr    = rng.uniform(62, 98)
            sleep = rng.uniform(7.0, 9.0)
            steps = int(rng.integers(8000, 15000))
            bmi   = rng.uniform(18.8, 24.5)

        elif risk == 1: # ── Medium Risk ────────────────────────────────────────
            # Pick 1–2 metrics to be borderline
            hr    = rng.uniform(52, 112)
            sleep = rng.uniform(5.5, 7.2)
            steps = int(rng.integers(3500, 8000))
            bmi   = rng.uniform(24.8, 30.5)

        else:           # ── High Risk ─────────────────────────────────────────
            # Multiple metrics severely out of range
            hr    = rng.choice([
                float(rng.uniform(32, 55)),
                float(rng.uniform(112, 185))
            ])
            sleep = rng.uniform(1.0, 5.0)
            steps = int(rng.integers(0, 3500))
            bmi   = rng.choice([
                float(rng.uniform(10.0, 17.0)),
                float(rng.uniform(31.0, 52.0))
            ])

        # Add realistic Gaussian jitter
        hr    = float(np.clip(hr    + rng.normal(0, 2.5),  28,  200))
        sleep = float(np.clip(sleep + rng.normal(0, 0.25),  0,   24))
        bmi   = float(np.clip(bmi   + rng.normal(0, 0.4),   9,   60))

        X.append([hr, sleep, steps, bmi])
        y.append(risk)

    return np.array(X, dtype=float), np.array(y, dtype=int)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_and_save():
    print("\n🧠  MediTrack — ML Training Pipeline")
    print("=" * 52)

    # 1. Generate data
    X, y = generate_dataset(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Dataset  : {len(X)} samples  (train={len(X_train)}, test={len(X_test)})")
    print(f"  Features : {FEATURE_NAMES}")
    print(f"  Classes  : {LABEL_NAMES}\n")

    # 2. Build pipeline
    # GradientBoosting chosen for:
    #   - Strong accuracy on tabular data
    #   - Native feature importance
    #   - No need for one-hot encoding
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            min_samples_leaf=10,
            random_state=42,
        ))
    ])

    # 3. Cross-validation (5-fold, stratified)
    print("  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 4. Fit on full training set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 5. Evaluation metrics
    acc      = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm       = confusion_matrix(y_test, y_pred).tolist()
    report   = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)

    print(f"\n  Test Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1 Score  : {f1_macro:.4f}")
    print(f"\n  Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"    {LABEL_NAMES[i]:<14} → {row}")

    print(f"\n  Per-Class Report:")
    for label in LABEL_NAMES:
        r = report[label]
        print(f"    {label:<14}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    # 6. Feature importance (from the GBM estimator, after pipeline)
    clf          = pipeline.named_steps['clf']
    raw_imp      = clf.feature_importances_
    # Sort descending
    sorted_idx   = np.argsort(raw_imp)[::-1]
    importance   = [
        {
            'feature': FEATURE_NAMES[i],
            'importance': round(float(raw_imp[i]), 6),
            'percentage': round(float(raw_imp[i]) * 100, 2),
        }
        for i in sorted_idx
    ]
    print(f"\n  Feature Importances:")
    for item in importance:
        bar = '█' * int(item['percentage'] / 2)
        print(f"    {item['feature']:<22} {item['percentage']:5.1f}%  {bar}")

    # 7. Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n  ✅  Model saved  → {MODEL_PATH}")

    # 8. Save evaluation JSON (used by the insights view)
    per_class = []
    for i, label in enumerate(LABEL_NAMES):
        r = report[label]
        per_class.append({
            'label': label,
            'precision': round(r['precision'], 4),
            'recall':    round(r['recall'], 4),
            'f1':        round(r['f1-score'], 4),
            'support':   int(r['support']),
        })

    evaluation = {
        'accuracy':        round(acc, 4),
        'f1_macro':        round(f1_macro, 4),
        'cv_mean':         round(float(cv_scores.mean()), 4),
        'cv_std':          round(float(cv_scores.std()), 4),
        'confusion_matrix': cm,
        'class_labels':    LABEL_NAMES,
        'per_class':       per_class,
        'n_train':         len(X_train),
        'n_test':          len(X_test),
        'model_type':      'GradientBoostingClassifier',
    }

    with open(EVAL_PATH, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"  ✅  Evaluation   → {EVAL_PATH}")

    with open(IMPORTANCE_PATH, 'w') as f:
        json.dump(importance, f, indent=2)
    print(f"  ✅  Importance   → {IMPORTANCE_PATH}")

    print("\n  ✅  All done! Set USE_ML_MODEL = True in health/services.py to activate.")
    print("=" * 52)


if __name__ == '__main__':
    train_and_save()
