"""
Score the logistic regression baseline against the XGBoost churn model.

Both are trained on the same customers and the same split. The baseline sees
the raw generator columns only; the model sees the engineered features.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_GEN_CONFIG, PROJECT_ROOT
from .data_generator import ChurnDataGenerator
from .features import FeatureEngineer
from .models import XGBoostChurnClassifier

TARGET = "is_churned"
NON_FEATURES = ["customer_id", "timestamp"]
EVIDENCE_PATH = PROJECT_ROOT / "evidence" / "baseline_metrics.json"


def _drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in NON_FEATURES if c in df.columns])


def _classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_churn": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }


def score_baseline(seed: int, n_samples: int, test_size: float) -> dict:
    """Train the baseline and the model on one split and return both scores."""
    raw = ChurnDataGenerator(seed=seed, n_samples=n_samples).generate()
    engineer = FeatureEngineer()

    raw_features = _drop_non_features(
        engineer.pipeline(raw.copy(), create_temporal=False, create_interactions=False)
    )
    full_features = _drop_non_features(engineer.pipeline(raw.copy()))

    train_idx, test_idx = train_test_split(
        raw.index, test_size=test_size, random_state=seed, stratify=raw[TARGET]
    )
    y_train = raw.loc[train_idx, TARGET]
    y_test = raw.loc[test_idx, TARGET]

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    x_raw = raw_features.drop(columns=[TARGET])
    baseline.fit(x_raw.loc[train_idx], y_train)
    baseline_scores = _classification_metrics(
        y_test,
        baseline.predict(x_raw.loc[test_idx]),
        baseline.predict_proba(x_raw.loc[test_idx])[:, 1],
    )

    model = XGBoostChurnClassifier()
    x_full = full_features.drop(columns=[TARGET])
    model.train(x_full.loc[train_idx], y_train)
    model_scores = _classification_metrics(
        y_test,
        model.predict(x_full.loc[test_idx]),
        model.predict_proba(x_full.loc[test_idx])[:, 1],
    )

    return {
        "seed": seed,
        "n_samples": n_samples,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "baseline": {
            "name": "logistic regression on raw features",
            "n_features": int(x_raw.shape[1]),
            **baseline_scores,
        },
        "model": {
            "name": "xgboost on engineered features",
            "n_features": int(x_full.shape[1]),
            **model_scores,
        },
    }


def write_evidence(scores: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2) + "\n")


def main() -> None:
    scores = score_baseline(
        seed=DATA_GEN_CONFIG["random_seed"],
        n_samples=DATA_GEN_CONFIG["n_samples"],
        test_size=DATA_GEN_CONFIG["test_size"],
    )
    write_evidence(scores, EVIDENCE_PATH)
    print(json.dumps(scores, indent=2))
    print(f"Written to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
