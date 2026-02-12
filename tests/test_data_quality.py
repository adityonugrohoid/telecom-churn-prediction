"""Tests for data quality and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_prediction.data_generator import ChurnDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    generator = ChurnDataGenerator(seed=42, n_samples=1000)
    return generator.generate()


class TestDataQuality:

    def test_no_missing_values(self, sample_data):
        critical_cols = ["customer_id", "tenure_months", "avg_sinr_db", "avg_qoe_mos", "is_churned"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_numeric_dtype(sample_data["tenure_months"])
        assert pd.api.types.is_numeric_dtype(sample_data["avg_sinr_db"])
        assert pd.api.types.is_numeric_dtype(sample_data["is_churned"])

    def test_value_ranges(self, sample_data):
        assert sample_data["avg_sinr_db"].min() >= -5
        assert sample_data["avg_sinr_db"].max() <= 25
        assert sample_data["avg_qoe_mos"].min() >= 1
        assert sample_data["avg_qoe_mos"].max() <= 5
        assert sample_data["tenure_months"].min() >= 1
        assert sample_data["tenure_months"].max() <= 72
        assert set(sample_data["is_churned"].unique()).issubset({0, 1})

    def test_categorical_values(self, sample_data):
        assert set(sample_data["network_type"].unique()).issubset({"4G", "5G"})
        assert set(sample_data["device_class"].unique()).issubset({"low", "mid", "high"})
        assert set(sample_data["contract_type"].unique()).issubset({"month-to-month", "one-year", "two-year"})

    def test_sample_size(self, sample_data):
        assert len(sample_data) == 1000

    def test_churn_rate_realistic(self, sample_data):
        churn_rate = sample_data["is_churned"].mean()
        assert 0.05 < churn_rate < 0.35, f"Churn rate {churn_rate:.2f} outside realistic range"


class TestDataGenerator:

    def test_generator_reproducibility(self):
        gen1 = ChurnDataGenerator(seed=42, n_samples=100)
        gen2 = ChurnDataGenerator(seed=42, n_samples=100)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_sinr_generation(self):
        gen = ChurnDataGenerator(seed=42, n_samples=100)
        sinr = gen.generate_sinr(1000)
        assert len(sinr) == 1000
        assert sinr.min() >= -5
        assert sinr.max() <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
