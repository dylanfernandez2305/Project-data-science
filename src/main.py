# Main script for fraud detection pipeline
# This file handles data loading, preprocessing, and feature engineering

# ===== IMPORTS =====
import sys
sys.dont_write_bytecode = True  # Prevent .pyc file creation

import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE

# Optional imports (only if packages are available)
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not installed. Dataset download will be skipped if creditcard.csv exists.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP analysis will be skipped.")

warnings.filterwarnings('ignore')

# ===== UTILITY FUNCTIONS =====
def time_of_day(hour):
    """Classify hour into time of day category"""
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


def get_optimal_threshold_f1(anomaly_scores, y_true):
    """
    Get optimal threshold by maximizing F1-score

    Parameters:
    -----------
    anomaly_scores : array-like
        Anomaly scores (higher = more anomalous)
    y_true : array-like
        True labels (0: normal, 1: fraud)

    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes F1-score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx]


def load_and_prepare_data():
    """
    Load and prepare the credit card fraud dataset
    Returns preprocessed data and necessary variables
    """
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent

    # Dataset is in data/ folder
    dataset_path = project_root / "data" / "creditcard.csv"
    data_dir = project_root / "data"

    if not dataset_path.exists() and KAGGLEHUB_AVAILABLE:
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print("Path to dataset files:", path)

        # Create data directory if it doesn't exist
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
    # Removes rows with missing values
    card_data = card_data.dropna()

    # Removes dupplicated rows
    duplicates = card_data.duplicated()
    print("Number of duplicate rows:", duplicates.sum())
    card_data = card_data.drop_duplicates()

    # Reduces skew in transaction amounts
    card_data['LogAmount'] = np.log1p(card_data['Amount'])

    # Captures temporal patterns
    card_data['Hour'] = (card_data['Time'] // 3600) % 24
    card_data['Minute'] = (card_data['Time'] // 60) % 60
    card_data['Second'] = card_data['Time'] % 60
    card_data['Minute_Ratio_in_Hour'] = card_data['Minute'] / 60.0
    card_data['Second_Ratio_in_Minute'] = card_data['Second'] / 60.0
    card_data = card_data.sort_values('Time')
    card_data['Time_Since_Last_Transaction'] = card_data['Time'].diff()

    # Captures Cyclical temporal patterns
    card_data['Hour_sin'] = np.sin(2 * np.pi * card_data['Hour'] / 24)
    card_data['Hour_cos'] = np.cos(2 * np.pi * card_data['Hour'] / 24)
    card_data['Minute_sin'] = np.sin(2 * np.pi * card_data['Minute'] / 60)
    card_data['Minute_cos'] = np.cos(2 * np.pi * card_data['Minute'] / 60)

    # Creates Parts of the day classes
    card_data['TimeOfDay'] = card_data['Hour'].apply(time_of_day)

    # Number of transactions per hour
    hour_freq = card_data.groupby('Hour')['Time'].transform('count')
    card_data['Transactions_per_Hour'] = hour_freq

    # Number of transactions per TimeOfDay
    tod_freq = card_data.groupby('TimeOfDay')['Time'].transform('count')
    card_data['Transactions_per_TimeOfDay'] = tod_freq

    # Count frequency of transactions in each bin
    bins = [0, 50, 100, 500, 1000, 5000, np.inf]
    card_data['AmountBin'] = pd.cut(card_data['Amount'], bins=bins, labels=False, include_lowest=True, right=False)
    bin_counts = card_data['AmountBin'].value_counts().to_dict()
    card_data['Amount_frequency'] = card_data['AmountBin'].map(bin_counts)
    card_data = card_data.drop('AmountBin', axis=1)

    # Highlights unusual large transactions relative to normal frequency.
    card_data['Amount_per_Hour'] = card_data['Amount'] / (card_data['Transactions_per_Hour'] + 1)

    ## Ratio features
    # Transaction size vs hourly pattern
    card_data['Amount_Ratio_to_Hour_Mean'] = card_data['Amount'] / (card_data['Amount_per_Hour'] + 1e-8)
    # How busy this hour is vs time of day pattern
    card_data['Hour_Activity_Ratio'] = card_data['Transactions_per_Hour'] / (card_data['Transactions_per_TimeOfDay'] + 1e-8)
    # Transaction concentration in busy periods
    card_data['Amount_Intensity'] = card_data['Amount'] / (card_data['Transactions_per_Hour'] + 1e-8)
    # How common is this transaction amount
    card_data['Amount_to_Frequency_Ratio'] = card_data['Amount'] / (card_data['Amount_frequency'] + 1e-8)
    # Normalized transaction frequency by hour
    global_avg_hour_freq = card_data['Transactions_per_Hour'].mean()
    card_data['Normalized_Hour_Frequency'] = card_data['Transactions_per_Hour'] / global_avg_hour_freq
    # Combined temporal and amount pattern
    card_data['Temporal_Amount_Intensity'] = card_data['Amount'] * card_data['Normalized_Hour_Frequency']
    # Z-score of amount within hour (statistical outlier)
    hour_stats = card_data.groupby('Hour')['Amount'].agg(['mean', 'std']).reset_index()
    hour_stats.columns = ['Hour', 'Hour_Amount_Mean', 'Hour_Amount_Std']
    card_data = card_data.merge(hour_stats, on='Hour', how='left')
    card_data['Amount_Hour_ZScore'] = (card_data['Amount'] - card_data['Hour_Amount_Mean']) / (card_data['Hour_Amount_Std'] + 1e-8)
    card_data = card_data.drop(['Hour_Amount_Mean', 'Hour_Amount_Std'], axis=1)
    # Position within hour as ratio
    card_data['Minute_Ratio_in_Hour'] = card_data['Minute'] / 60.0

    ## Repeat data cleaning steps in case NaNs introduced by feature engineering
    # Removes rows with missing values
    card_data = card_data.dropna()

    # Removes dupplicated rows
    duplicates = card_data.duplicated()
    print("Number of duplicate rows after feature engineering:", duplicates.sum())
    card_data = card_data.drop_duplicates()

    card_data = card_data.sort_values(by='Time').reset_index(drop=True)

    X = card_data.drop('Class', axis=1)
    y = card_data['Class']
    X = pd.get_dummies(X, columns=['TimeOfDay'], drop_first=True)

    # Drop Time column (after sorting)
    X = X.drop('Time', axis=1)

    # Save feature names AFTER dropping Time column
    feature_names = X.columns.tolist()

    # Chronological split (80% train, 20% test)
    ## Avoids temporal leakage and ensures enough fraudulent data in training data
    split_index = int(0.8 * len(card_data))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    small_tscv = TimeSeriesSplit(n_splits=2)

    # For Semi supervised training (Feed statistical models only with "non-fraud" data)
    X_train_normal = X_train[y_train == 0]

    # Fraud ratio from the dataset
    fraud_ratio = len(y[y==1]) / len(y)

    # SMOTE resampling only used for unsupervised samples and semi-supervised models
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Original training data shape:", X_train.shape)
    print("Resampled training data shape:", X_train_res.shape)
    print(f"Total features: {X_train.shape[1]}")

    # For semi-supervised models with resampled data
    X_train_res_normal = X_train_res[y_train_res == 0]

    # Return all necessary variables
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_normal': X_train_normal,
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
        'X_train_res_normal': X_train_res_normal,
        'fraud_ratio': fraud_ratio,
        'small_tscv': small_tscv,
        'scaler': scaler,
        'feature_names': feature_names
    }


def main():
    """Main function to run the data preparation pipeline"""
    print("="*60)
    print("FRAUD DETECTION PIPELINE - DATA PREPARATION")
    print("="*60)

    data = load_and_prepare_data()

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    frauds_train = data['y_train'].sum()
    total_train = len(data['y_train'])
    print(f"Training set: {frauds_train} frauds out of {total_train} transactions")

    frauds_test = data['y_test'].sum()
    total_test = len(data['y_test'])
    print(f"Test set: {frauds_test} frauds out of {total_test} transactions")
    print(f"Fraud ratio: {data['fraud_ratio']:.4f}")
    print("="*60)

    return data


if __name__ == "__main__":
    data = main()
