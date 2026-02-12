# Telecom Customer Churn Prediction

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

Telecom operators face 15-25% annual churn. Acquiring new customers costs 5-7x more than retention. Early churn prediction enables proactive retention campaigns targeting at-risk subscribers before they leave.

## Problem Framing

Binary classification using XGBoost.

- **Target:** `is_churned`
- **Primary Metric:** AUROC
- **Challenges:**
  - Class imbalance (~15% churn rate)
  - Temporal leakage prevention
  - Balancing precision vs recall for retention budget optimization

## Data Engineering

Customer-level synthetic data combining:

- **Tenure features** -- account age, contract type
- **Network KPIs** -- SINR, throughput, QoE MOS
- **Service interactions** -- support tickets, session counts
- **Billing features** -- monthly charges, payment history

Domain physics: QoE degradation (low MOS, poor SINR, reduced throughput) drives churn likelihood. Features are engineered to capture rolling trends and degradation patterns over time.

## Methodology

- XGBoost with stratified k-fold cross-validation
- Class weight balancing to handle the ~15% minority class
- **Feature groups:**
  - QoE trend features (MOS slope, throughput delta)
  - Ticket rate and escalation patterns
  - Session frequency and engagement metrics
  - Composite network quality index
- Hyperparameter tuning with early stopping on validation AUROC

## Key Findings

- **AUROC:** ~0.85 on held-out test set
- **Top churn driver:** Low QoE MOS score is the strongest predictor of churn
- Customers with degrading network quality over a 30-day window are 3x more likely to churn
- Support ticket frequency amplifies churn risk when combined with poor network experience

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/01-churn-prediction

# Install dependencies
uv sync

# Generate synthetic data
uv run python generate_data.py

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
01-churn-prediction/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 01_churn_prediction.ipynb
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       └── evaluate.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── generate_data.py
└── tests/
    └── .gitkeep
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | **Churn Prediction** (this repo) | Binary classification to predict customer churn |
| 2 | [Root Cause Analysis](../02-root-cause-analysis) | Multi-class classification for network alarm RCA |
| 3 | [Anomaly Detection](../03-anomaly-detection) | Unsupervised detection of network anomalies |
| 4 | [QoE Prediction](../04-qoe-prediction) | Regression to predict quality of experience |
| 5 | [Capacity Forecasting](../05-capacity-forecasting) | Time-series forecasting for network capacity planning |
| 6 | [Network Optimization](../06-network-optimization) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**
