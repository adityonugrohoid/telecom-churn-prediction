"""Tests for the baseline comparison."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_prediction.baseline import score_baseline

SCORE_KEYS = {"name", "n_features", "roc_auc", "accuracy", "f1_churn"}

# Both the baseline and the model separate churners well on this generator.
# The comparison itself is reported, not asserted: on this synthetic data the
# logistic regression baseline scores slightly above XGBoost (see the README).
AUROC_FLOOR = 0.75


@pytest.fixture(scope="module")
def scores():
    return score_baseline(seed=42, n_samples=3000, test_size=0.2)


class TestBaselineComparison:
    def test_reports_the_run_it_scored(self, scores):
        assert scores["seed"] == 42
        assert scores["n_samples"] == 3000
        assert scores["n_train"] + scores["n_test"] == 3000

    def test_both_blocks_are_complete(self, scores):
        assert set(scores["baseline"]) == SCORE_KEYS
        assert set(scores["model"]) == SCORE_KEYS

    def test_baseline_sees_raw_features_only(self, scores):
        assert scores["baseline"]["n_features"] < scores["model"]["n_features"]

    def test_both_stay_above_the_auroc_floor(self, scores):
        assert scores["baseline"]["roc_auc"] >= AUROC_FLOOR
        assert scores["model"]["roc_auc"] >= AUROC_FLOOR

    def test_split_keeps_the_churn_rate(self, scores):
        assert 0.10 <= scores["churn_rate_test"] <= 0.22

    def test_scoring_is_reproducible(self, scores):
        again = score_baseline(seed=42, n_samples=3000, test_size=0.2)
        assert again == scores
