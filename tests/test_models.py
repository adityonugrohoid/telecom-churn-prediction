"""Tests for model training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_prediction.data_generator import ChurnDataGenerator
from churn_prediction.features import FeatureEngineer
from churn_prediction.models import BaseModel, XGBoostChurnClassifier

TARGET = "is_churned"
NON_FEATURES = ["customer_id", "timestamp"]

# The README reports AUROC 0.86 on 10,000 customers. The floor here is lower
# because the test trains on 3,000 to stay fast; a drop below it means the
# signal in the generator or the feature pipeline has broken.
AUROC_FLOOR = 0.75


@pytest.fixture(scope="module")
def features():
    """Seeded data run through the project's own feature pipeline."""
    raw = ChurnDataGenerator(seed=42, n_samples=3000).generate()
    features = FeatureEngineer().pipeline(raw)
    features = features.drop(columns=[c for c in NON_FEATURES if c in features.columns])
    return features


@pytest.fixture(scope="module")
def trained(features):
    """The model as the project uses it: prepare_data, then train."""
    model = XGBoostChurnClassifier()
    X_train, _, y_train, _ = model.prepare_data(features, target_col=TARGET)
    model.train(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def split(features):
    return XGBoostChurnClassifier().prepare_data(features, target_col=TARGET)


class TestTraining:
    def test_untrained_model_refuses_to_predict(self, split):
        _, X_test, _, _ = split
        with pytest.raises(ValueError):
            XGBoostChurnClassifier().predict(X_test)

    def test_base_model_has_no_training(self, split):
        X_train, _, y_train, _ = split
        with pytest.raises(NotImplementedError):
            BaseModel().train(X_train, y_train)

    def test_training_marks_the_model_trained(self, trained):
        assert trained.is_trained

    def test_training_is_reproducible(self, split, trained):
        X_train, X_test, y_train, _ = split
        again = XGBoostChurnClassifier()
        again.train(X_train, y_train)
        np.testing.assert_allclose(
            trained.predict_proba(X_test), again.predict_proba(X_test), rtol=0, atol=1e-7
        )


class TestEvaluation:
    def test_probabilities_are_valid(self, split, trained):
        _, X_test, _, _ = split
        proba = trained.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_metrics_are_complete(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test, task_type="classification")
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0 <= value <= 1 for value in metrics.values())

    def test_auroc_stays_above_the_floor(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test, task_type="classification")
        assert metrics["roc_auc"] >= AUROC_FLOOR, f"AUROC fell to {metrics['roc_auc']:.3f}"

    def test_model_beats_always_predicting_no_churn(self, split, trained):
        _, X_test, _, y_test = split
        majority_accuracy = 1 - y_test.mean()
        metrics = trained.evaluate(X_test, y_test, task_type="classification")
        assert metrics["accuracy"] >= majority_accuracy

    def test_feature_importance_covers_every_feature(self, split, trained):
        X_train, _, _, _ = split
        importance = trained.get_feature_importance()
        assert sorted(importance["feature"]) == sorted(X_train.columns)
        assert importance["importance"].sum() == pytest.approx(1.0, abs=1e-5)


class TestPersistence:
    def test_saved_model_predicts_the_same(self, split, trained, tmp_path):
        _, X_test, _, _ = split
        path = tmp_path / "churn.pkl"
        trained.save(path)
        restored = XGBoostChurnClassifier()
        restored.load(path)
        np.testing.assert_array_equal(trained.predict(X_test), restored.predict(X_test))

    def test_untrained_model_refuses_to_save(self, tmp_path):
        with pytest.raises(ValueError):
            XGBoostChurnClassifier().save(tmp_path / "churn.pkl")
