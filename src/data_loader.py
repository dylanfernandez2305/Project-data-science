"""Data loading, preprocessing, and shared utility functions.

Moved out of the former `src/main.py` to keep a clean project entry point.

Public API (used across the codebase):
- time_of_day
- get_optimal_threshold_f1
- load_and_prepare_data
"""

from __future__ import annotations

import os
from typing import Any, Sequence
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

# Optional imports (only if packages are available)
try:
    import kagglehub

    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print(
        "Warning: kagglehub not installed. Dataset download will be skipped if creditcard.csv exists."
    )

warnings.filterwarnings("ignore")


def time_of_day(hour: int) -> str:
    """Classify hour into a time-of-day category."""
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    if 18 <= hour < 24:
        return "Evening"
    return "Night"


def get_optimal_threshold_f1(
    anomaly_scores: np.ndarray | Sequence[float],
    y_true: np.ndarray | Sequence[int],
) -> float:
    """Return the threshold that maximizes F1-score.

    IMPORTANT: This function should be called on VALIDATION data only,
    not on the test set, to prevent data leakage.

    Parameters
    ----------
    anomaly_scores : array-like
        Scores/probabilities (higher = more likely fraud / more anomalous).
    y_true : array-like
        True labels (0: normal, 1: fraud)
    """

    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[optimal_idx])


def load_and_prepare_data() -> dict[str, Any]:
    """Load and prepare the credit card fraud dataset.

    Returns
    -------
    dict
        A dictionary containing train/val/test splits, resampled variants,
        scaler, feature names, etc.
    """

    # Project root is the parent directory of `src/`
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "data" / "creditcard.csv"
    data_dir = project_root / "data"

    if not dataset_path.exists() and KAGGLEHUB_AVAILABLE:
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print("Path to dataset files:", path)

        os.makedirs(data_dir, exist_ok=True)

        # Copy all dataset files to the data folder
        for file_name in os.listdir(path):
            src = os.path.join(path, file_name)
            dst = data_dir / file_name
            shutil.copy(src, dst)

        print("Dataset copied to:", data_dir)
    elif not dataset_path.exists():
        raise FileNotFoundError(
            f"creditcard.csv not found at {dataset_path}. "
            "Please either install kagglehub or manually download the dataset to data/ folder."
        )
    else:
        print(f"Using existing dataset at: {dataset_path}")

    # ===== DATA LOADING =====
    card_data = pd.read_csv(dataset_path)

    # ===== DATA PREPROCESSING & FEATURE ENGINEERING =====
    card_data = card_data.dropna()

    duplicates = card_data.duplicated()
    print("Number of duplicate rows:", duplicates.sum())
    card_data = card_data.drop_duplicates()

    # Reduce skew in transaction amounts
    card_data["LogAmount"] = np.log1p(card_data["Amount"])

    # Temporal features
    card_data["Hour"] = (card_data["Time"] // 3600) % 24
    card_data["Minute"] = (card_data["Time"] // 60) % 60
    card_data["Second"] = card_data["Time"] % 60
    card_data["Minute_Ratio_in_Hour"] = card_data["Minute"] / 60.0
    card_data["Second_Ratio_in_Minute"] = card_data["Second"] / 60.0

    card_data = card_data.sort_values("Time")
    card_data["Time_Since_Last_Transaction"] = card_data["Time"].diff()

    # Cyclical encodings
    card_data["Hour_sin"] = np.sin(2 * np.pi * card_data["Hour"] / 24)
    card_data["Hour_cos"] = np.cos(2 * np.pi * card_data["Hour"] / 24)
    card_data["Minute_sin"] = np.sin(2 * np.pi * card_data["Minute"] / 60)
    card_data["Minute_cos"] = np.cos(2 * np.pi * card_data["Minute"] / 60)

    # Time-of-day categories
    card_data["TimeOfDay"] = card_data["Hour"].apply(time_of_day)

    # Transaction frequency features
    card_data["Transactions_per_Hour"] = card_data.groupby("Hour")["Time"].transform("count")
    card_data["Transactions_per_TimeOfDay"] = card_data.groupby("TimeOfDay")["Time"].transform("count")

    # Amount binning frequency
    bins = [0, 50, 100, 500, 1000, 5000, np.inf]
    card_data["AmountBin"] = pd.cut(
        card_data["Amount"], bins=bins, labels=False, include_lowest=True, right=False
    )
    bin_counts = card_data["AmountBin"].value_counts().to_dict()
    card_data["Amount_frequency"] = card_data["AmountBin"].map(bin_counts)
    card_data = card_data.drop("AmountBin", axis=1)

    # More ratios / interaction features
    card_data["Amount_per_Hour"] = card_data["Amount"] / (card_data["Transactions_per_Hour"] + 1)

    card_data["Amount_Ratio_to_Hour_Mean"] = card_data["Amount"] / (
        card_data["Amount_per_Hour"] + 1e-8
    )
    card_data["Hour_Activity_Ratio"] = card_data["Transactions_per_Hour"] / (
        card_data["Transactions_per_TimeOfDay"] + 1e-8
    )
    card_data["Amount_Intensity"] = card_data["Amount"] / (
        card_data["Transactions_per_Hour"] + 1e-8
    )
    card_data["Amount_to_Frequency_Ratio"] = card_data["Amount"] / (
        card_data["Amount_frequency"] + 1e-8
    )

    global_avg_hour_freq = card_data["Transactions_per_Hour"].mean()
    card_data["Normalized_Hour_Frequency"] = card_data["Transactions_per_Hour"] / global_avg_hour_freq
    card_data["Temporal_Amount_Intensity"] = card_data["Amount"] * card_data["Normalized_Hour_Frequency"]

    # Z-score of amount within hour
    hour_stats = card_data.groupby("Hour")["Amount"].agg(["mean", "std"]).reset_index()
    hour_stats.columns = ["Hour", "Hour_Amount_Mean", "Hour_Amount_Std"]
    card_data = card_data.merge(hour_stats, on="Hour", how="left")
    card_data["Amount_Hour_ZScore"] = (card_data["Amount"] - card_data["Hour_Amount_Mean"]) / (
        card_data["Hour_Amount_Std"] + 1e-8
    )
    card_data = card_data.drop(["Hour_Amount_Mean", "Hour_Amount_Std"], axis=1)

    # Repeat data cleaning steps in case NaNs introduced by feature engineering
    card_data = card_data.dropna()

    duplicates = card_data.duplicated()
    print("Number of duplicate rows after feature engineering:", duplicates.sum())
    card_data = card_data.drop_duplicates()

    card_data = card_data.sort_values(by="Time").reset_index(drop=True)

    X = card_data.drop("Class", axis=1)
    y = card_data["Class"]

    X = pd.get_dummies(X, columns=["TimeOfDay"], drop_first=True)

    # Drop Time column (after sorting)
    X = X.drop("Time", axis=1)

    feature_names = X.columns.tolist()

    # Chronological split (64% train, 16% validation, 20% test)
    train_split = int(0.64 * len(card_data))
    val_split = int(0.80 * len(card_data))

    X_train = X.iloc[:train_split]
    X_val = X.iloc[train_split:val_split]
    X_test = X.iloc[val_split:]

    y_train = y.iloc[:train_split]
    y_val = y.iloc[train_split:val_split]
    y_test = y.iloc[val_split:]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    small_tscv = TimeSeriesSplit(n_splits=2)

    # Semi-supervised: normal-only
    X_train_normal = X_train[y_train == 0]

    fraud_ratio = float((y == 1).sum() / len(y))

    # SMOTE resampling (used upstream for anomaly-detection pipelines in this project)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Original training data shape:", X_train.shape)
    print("Resampled training data shape:", X_train_res.shape)
    print(f"Total features: {X_train.shape[1]}")

    X_train_res_normal = X_train_res[y_train_res == 0]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_normal": X_train_normal,
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_train_res_normal": X_train_res_normal,
        "fraud_ratio": fraud_ratio,
        "small_tscv": small_tscv,
        "scaler": scaler,
        "feature_names": feature_names,
    }
