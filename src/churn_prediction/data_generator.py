"""
Domain-informed synthetic data generator for Telecom Churn Prediction.

This module generates realistic telecom data using domain knowledge rather than
off-the-shelf synthetic data tools.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(self, congestion: np.ndarray, base_latency_ms: float = 20) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion ** 2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(self, throughput_mbps: np.ndarray, latency_ms: np.ndarray, packet_loss_pct: np.ndarray, app_type: np.ndarray) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class ChurnDataGenerator(TelecomDataGenerator):
    """Generate synthetic customer-level churn data for telecom subscribers.

    Creates a dataset with customer demographics, contract details, network
    quality metrics (SINR, throughput, latency), QoE scores, and churn labels.
    The churn rate targets ~15%, influenced by QoE, tenure, and contract type.
    """

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 10_000,
        churn_rate: float = 0.15,
        observation_window_days: int = 30,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.churn_rate = churn_rate
        self.observation_window_days = observation_window_days

    def generate(self) -> pd.DataFrame:
        """Generate a customer-level churn dataset.

        Returns:
            pd.DataFrame with one row per customer, including network KPIs,
            QoE scores, and a binary ``is_churned`` label.
        """
        n = self.n_samples

        # --------------------------------------------------------------------
        # Customer identifiers and timestamps
        # --------------------------------------------------------------------
        customer_id = np.arange(1, n + 1)

        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=self.observation_window_days)
        random_days = self.rng.integers(0, self.observation_window_days, size=n)
        random_seconds = self.rng.integers(0, 86400, size=n)
        timestamp = pd.to_datetime(
            start_date
            + pd.to_timedelta(random_days, unit="D")
            + pd.to_timedelta(random_seconds, unit="s")
        )

        # --------------------------------------------------------------------
        # Contract and payment attributes
        # --------------------------------------------------------------------
        tenure_months = self.rng.integers(1, 73, size=n)

        contract_type = self.rng.choice(
            ["month-to-month", "one-year", "two-year"],
            size=n,
            p=[0.5, 0.3, 0.2],
        )

        payment_method = self.rng.choice(
            ["electronic", "bank_transfer", "credit_card", "mail_check"],
            size=n,
            p=[0.4, 0.3, 0.2, 0.1],
        )

        # Monthly charges depend on contract type (longer contracts = higher
        # base but better value perception).
        base_charge = np.where(
            contract_type == "month-to-month",
            self.rng.uniform(30, 90, size=n),
            np.where(
                contract_type == "one-year",
                self.rng.uniform(50, 110, size=n),
                self.rng.uniform(70, 120, size=n),
            ),
        )
        monthly_charges = np.round(base_charge, 2)

        # --------------------------------------------------------------------
        # Network and device attributes
        # --------------------------------------------------------------------
        network_type = self.rng.choice(["4G", "5G"], size=n, p=[0.6, 0.4])
        device_class = self.rng.choice(
            ["low", "mid", "high"], size=n, p=[0.2, 0.5, 0.3]
        )

        # --------------------------------------------------------------------
        # Network quality KPIs (per-customer averages)
        # --------------------------------------------------------------------
        avg_sinr_db = self.generate_sinr(n)
        avg_throughput_mbps = self.sinr_to_throughput(avg_sinr_db, network_type)

        # Use timestamps for congestion pattern
        timestamps_idx = pd.DatetimeIndex(timestamp)
        congestion = self.generate_congestion_pattern(timestamps_idx)
        avg_latency_ms = self.congestion_to_latency(congestion)

        # Packet loss correlates with congestion
        avg_packet_loss_pct = np.clip(
            congestion * 3 + self.rng.normal(0, 0.5, n), 0, 10
        )

        # QoE MOS -- use a generic app type to get a balanced score
        app_types = self.rng.choice(
            ["web_browsing", "video_streaming", "gaming", "voip"],
            size=n,
            p=[0.4, 0.3, 0.15, 0.15],
        )
        avg_qoe_mos = self.compute_qoe_mos(
            avg_throughput_mbps, avg_latency_ms, avg_packet_loss_pct, app_types
        )

        # --------------------------------------------------------------------
        # Usage and support
        # --------------------------------------------------------------------
        total_tickets = self.rng.poisson(2, size=n)
        total_sessions = self.rng.integers(50, 501, size=n)
        # Longer-tenured customers tend to have more sessions
        total_sessions = np.clip(
            total_sessions + (tenure_months * 3).astype(int), 50, 500
        )

        # --------------------------------------------------------------------
        # Churn label (~15% overall, influenced by risk factors)
        # --------------------------------------------------------------------
        base_prob = self.churn_rate

        churn_prob = np.full(n, base_prob)

        # Low QoE (< 2.5) triples the churn probability
        low_qoe_mask = avg_qoe_mos < 2.5
        churn_prob = np.where(low_qoe_mask, churn_prob * 3.0, churn_prob)

        # Short tenure (< 6 months) doubles the churn probability
        short_tenure_mask = tenure_months < 6
        churn_prob = np.where(short_tenure_mask, churn_prob * 2.0, churn_prob)

        # Month-to-month contracts increase churn by 1.5x
        mtm_mask = contract_type == "month-to-month"
        churn_prob = np.where(mtm_mask, churn_prob * 1.5, churn_prob)

        # Cap probabilities at 1.0
        churn_prob = np.clip(churn_prob, 0, 1.0)

        # Calibrate so overall mean is close to self.churn_rate
        # Scale probabilities proportionally
        current_mean = churn_prob.mean()
        if current_mean > 0:
            scale_factor = self.churn_rate / current_mean
            churn_prob = np.clip(churn_prob * scale_factor, 0, 1.0)

        is_churned = (self.rng.random(n) < churn_prob).astype(int)

        # --------------------------------------------------------------------
        # Assemble DataFrame
        # --------------------------------------------------------------------
        df = pd.DataFrame(
            {
                "customer_id": customer_id,
                "timestamp": timestamp,
                "tenure_months": tenure_months,
                "contract_type": contract_type,
                "payment_method": payment_method,
                "monthly_charges": monthly_charges,
                "network_type": network_type,
                "device_class": device_class,
                "avg_sinr_db": np.round(avg_sinr_db, 2),
                "avg_throughput_mbps": np.round(avg_throughput_mbps, 2),
                "avg_latency_ms": np.round(avg_latency_ms, 2),
                "avg_packet_loss_pct": np.round(avg_packet_loss_pct, 2),
                "avg_qoe_mos": np.round(avg_qoe_mos, 2),
                "total_tickets": total_tickets,
                "total_sessions": total_sessions,
                "is_churned": is_churned,
            }
        )

        print(f"Generated {len(df):,} customers")
        print(f"Churn rate: {df['is_churned'].mean():.2%}")
        return df


def main() -> None:
    """Generate and save the churn prediction dataset using project config."""
    config = DATA_GEN_CONFIG
    use_case = config.get("use_case_params", {})

    generator = ChurnDataGenerator(
        seed=config.get("random_seed", 42),
        n_samples=config.get("n_samples", 10_000),
        churn_rate=use_case.get("churn_rate", 0.15),
        observation_window_days=use_case.get("observation_window_days", 30),
    )

    df = generator.generate()
    generator.save(df, "churn_data")


if __name__ == "__main__":
    main()
