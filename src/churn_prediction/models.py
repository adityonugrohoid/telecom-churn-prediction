"""
ML model training and evaluation for Telecom Churn Prediction.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .config import MODEL_CONFIG, PROCESSED_DATA_DIR


class BaseModel:
    """Base class for ML models."""

    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, task_type="classification"):
        y_pred = self.predict(X_test)
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            if len(np.unique(y_test)) == 2:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = 1 - (
                np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            )
        return metrics

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class XGBoostChurnClassifier(BaseModel):
    """XGBoost classifier for telecom customer churn prediction."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        import xgboost

        self.xgb = xgboost

    def train(self, X_train, y_train):
        params = self.config.get("hyperparameters", {})
        self.model = self.xgb.XGBClassifier(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            objective=params.get("objective", "binary:logistic"),
            eval_metric=params.get("eval_metric", "auc"),
            use_label_encoder=params.get("use_label_encoder", False),
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("XGBoost churn classifier trained successfully.")

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)


def cross_validate_model(model, X, y, cv_folds=5, scoring="accuracy"):
    scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scoring)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def print_metrics(metrics, title="Model Performance"):
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:8.4f}")
    print(f"{'=' * 50}\n")


def main():
    """Example usage of the churn prediction model."""
    print("Telecom Churn Prediction - Model Training")
    print("-" * 40)

    # Load processed data
    data_path = PROCESSED_DATA_DIR / "churn_features.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run the feature engineering pipeline first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Initialize and train model
    model = XGBoostChurnClassifier()
    X_train, X_test, y_train, y_test = model.prepare_data(df, target_col="churn")
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test, task_type="classification")
    print_metrics(metrics, title="Churn Prediction Results")

    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train, scoring="roc_auc")
    print(
        f"Cross-validation ROC AUC: {cv_results['mean_score']:.4f} "
        f"(+/- {cv_results['std_score']:.4f})"
    )

    # Feature importance
    importance_df = model.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
