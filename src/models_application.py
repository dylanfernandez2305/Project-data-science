# Model application and prediction
# This file contains code for making predictions and evaluating models

# ===== IMPORTS =====
import sys
sys.dont_write_bytecode = True  # Prevent .pyc file creation

# Allow running this file directly (e.g., `python src/models_application.py`) while
# still using absolute imports like `from src...`.
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
from src.data_loader import get_optimal_threshold_f1
import pandas as pd
import time


def apply_models(data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """
    Apply all trained models to test data and generate predictions
    Safely handles missing class metrics for imbalanced datasets.
    Now includes time tracking and detailed classification reports!

    IMPORTANT: Uses validation set for threshold selection and ensemble calibration
    to avoid data leakage from test set.

    Parameters:
    -----------
    data : dict
        Dictionary containing validation and test data
    models : dict
        Dictionary containing trained models

    Returns:
    --------
    results : dict
        Dictionary containing all predictions, evaluation metrics, summary, and feature importance
    """
    # Extract test and validation sets
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']

    predictions = {}
    model_reports = {}
    optimal_thresholds = {}
    model_timings = {}  # Track time for each model

    # Store all probability/scores arrays for consistency
    all_probabilities = {}
    # Store validation scores for ensemble selection (prevent test leakage)
    validation_f1_scores = {}

    # Create mapping from model display names to model keys
    model_name_to_key = {
        'Logistic Regression (Supervised)': 'best_lr',
        'Random Forest (Supervised)': 'best_rf',
        'Isolation Forest (Unsupervised)': 'best_iso',
        'Local Outlier Factor (Unsupervised)': 'best_lof',
        'Gaussian Mixture Models (Unsupervised)': 'best_gmm',
        'Isolation Forest (Semi-Supervised)': 'best_iso_semi',
        'Local Outlier Factor (Semi-Supervised)': 'best_lof_semi',
        'Gaussian Mixture Model (Semi-Supervised)': 'best_gmm_semi'
    }

    # --- Helper function to safely get classification metrics ---

    # ===== SUPERVISED MODELS =====
    for model_name, model_key in [
        ('Logistic Regression (Supervised)', 'best_lr'),
        ('Random Forest (Supervised)', 'best_rf')
    ]:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        if model_key not in models:
            print(f"[WARNING] Model {model_key} not found, skipping...")
            continue

        # Start timing
        start_time = time.time()

        # Get probabilities on validation set to determine optimal threshold (NO TEST LEAKAGE)
        y_prob_val = models[model_key].predict_proba(X_val)[:, 1]
        optimal_threshold = get_optimal_threshold_f1(y_prob_val, y_val)

        # Apply threshold to test set predictions
        y_prob = models[model_key].predict_proba(X_test)[:, 1]
        y_pred = np.where(y_prob >= optimal_threshold, 1, 0)

        # Calculate F1 on validation for ensemble selection later
        y_pred_val = np.where(y_prob_val >= optimal_threshold, 1, 0)
        validation_f1_scores[model_name] = f1_score(y_val, y_pred_val, zero_division=0)

        predictions[model_name] = y_pred
        all_probabilities[model_name] = y_prob

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob)

        model_reports[model_name] = {
            'classification_report': report_dict,
            'roc_auc': roc_auc,
            'y_prob': y_prob
        }
        optimal_thresholds[model_name] = optimal_threshold

        # End timing
        elapsed_time = time.time() - start_time
        model_timings[model_name] = elapsed_time

        # Print detailed results
        print(f"\n[TIME]  Evaluation Time: {elapsed_time:.2f}s")
        print(f"[THRESHOLD] Optimal Threshold (from validation): {optimal_threshold:.4f}")
        print(f"[VALIDATION F1] F1-Score on Validation: {validation_f1_scores[model_name]:.4f}")
        print(f"[ROC-AUC] ROC-AUC Score (test): {roc_auc:.4f}")
        print(f"\n[REPORT] Classification Report (test):")
        print(classification_report(y_test, y_pred))

    # ===== UNSUPERVISED & SEMI-SUPERVISED MODELS =====
    for model_name, model_key in [
        ('Isolation Forest (Unsupervised)', 'best_iso'),
        ('Local Outlier Factor (Unsupervised)', 'best_lof'),
        ('Gaussian Mixture Models (Unsupervised)', 'best_gmm'),
        ('Isolation Forest (Semi-Supervised)', 'best_iso_semi'),
        ('Local Outlier Factor (Semi-Supervised)', 'best_lof_semi'),
        ('Gaussian Mixture Model (Semi-Supervised)', 'best_gmm_semi')
    ]:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")

        if model_key not in models:
            print(f"[WARNING] Model {model_key} not found, skipping...")
            continue

        model = models[model_key]

        try:
            # Start timing
            start_time = time.time()

            # Get scores on VALIDATION set to determine optimal threshold (NO TEST LEAKAGE)
            if 'Gaussian' in model_name:
                scores_val = model.score_samples(X_val)
                y_scores_val = -scores_val  # Convert to anomaly scores (higher = more anomalous)
                scores = model.score_samples(X_test)
                y_scores = -scores
            else:
                y_scores_val = -model.decision_function(X_val)  # Convert to anomaly scores
                y_scores = -model.decision_function(X_test)

            # Determine optimal threshold on validation set
            optimal_threshold = get_optimal_threshold_f1(y_scores_val, y_val)

            # Apply threshold to test set
            y_pred = np.where(y_scores >= optimal_threshold, 1, 0)

            # Calculate F1 on validation for ensemble selection later
            y_pred_val = np.where(y_scores_val >= optimal_threshold, 1, 0)
            validation_f1_scores[model_name] = f1_score(y_val, y_pred_val, zero_division=0)

            predictions[model_name] = y_pred
            all_probabilities[model_name] = y_scores

            # Ensure scores have the same length as y_test (safety check)
            if len(y_scores) != len(y_test):
                print(f"[WARNING] Score length mismatch: {len(y_scores)} vs {len(y_test)}")
                min_len = min(len(y_scores), len(y_test))
                y_scores = y_scores[:min_len]
                y_pred = y_pred[:min_len]
                y_test_adjusted = y_test[:min_len]
            else:
                y_test_adjusted = y_test

            report_dict = classification_report(y_test_adjusted, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test_adjusted, y_scores)

            model_reports[model_name] = {
                'classification_report': report_dict,
                'roc_auc': roc_auc,
                'y_prob': y_scores
            }
            optimal_thresholds[model_name] = optimal_threshold

            # End timing
            elapsed_time = time.time() - start_time
            model_timings[model_name] = elapsed_time

            # Print detailed results
            print(f"\n[TIME]  Evaluation Time: {elapsed_time:.2f}s")
            print(f"[THRESHOLD] Optimal Threshold (from validation): {optimal_threshold:.4f}")
            print(f"[VALIDATION F1] F1-Score on Validation: {validation_f1_scores[model_name]:.4f}")
            print(f"[ROC-AUC] ROC-AUC Score (test): {roc_auc:.4f}")
            print(f"\n[REPORT] Classification Report (test):")
            print(classification_report(y_test_adjusted, y_pred))

        except Exception as e:
            print(f"[ERROR] Error with {model_name}: {e}")
            continue

    # ===== ENSEMBLE TOP 3 (F1-WEIGHTED) =====
    print(f"\n{'='*60}\nF1-Weighted Ensemble (Top 3 Models)\n{'='*60}")

    # Check if we have at least 3 models with probabilities
    available_proba_models = [name for name in all_probabilities.keys()
                             if len(all_probabilities[name]) == len(y_test)]

    if len(available_proba_models) >= 3:
        # Start timing
        start_time = time.time()

        # IMPORTANT: Use F1-scores from VALIDATION set (NO TEST LEAKAGE)
        # Sort models by validation F1-score (descending) and select top 3
        sorted_models = sorted(validation_f1_scores.items(), key=lambda x: x[1], reverse=True)
        top_3_models = sorted_models[:3]

        # Extract model names and F1 scores (from validation)
        ensemble_models = [name for name, _ in top_3_models]
        f1_scores = [f1 for _, f1 in top_3_models]

        # Calculate normalized weights (sum to 1.0)
        total_f1 = sum(f1_scores)
        if total_f1 > 0:
            weights = [f1 / total_f1 for f1 in f1_scores]
        else:
            # Fallback to equal weights if all F1 scores are 0
            weights = [1.0 / len(ensemble_models)] * len(ensemble_models)

        # Print selected models and weights
        print(f"Selected top 3 models by validation F1-score (NO TEST LEAKAGE):")
        for i, (model_name, f1, weight) in enumerate(zip(ensemble_models, f1_scores, weights), 1):
            print(f"  {i}. {model_name}")
            print(f"     Validation F1-Score: {f1:.4f} | Weight: {weight:.2%}")

        # Calculate ensemble predictions on VALIDATION first to get optimal threshold
        ensemble_proba_val = np.zeros(len(y_val))
        for model_name, weight in zip(ensemble_models, weights):
            # Get the model key from the mapping
            model_key = model_name_to_key.get(model_name)
            if model_key is None:
                print(f"[WARNING] Model key not found for '{model_name}', skipping...")
                continue

            # Verify the model exists
            if model_key not in models:
                print(f"[WARNING] Model '{model_key}' not found in models dictionary, skipping...")
                continue

            try:
                # Get validation probabilities/scores for each model
                if 'Gaussian' in model_name or 'Isolation' in model_name or 'Outlier' in model_name:
                    # For unsupervised/semi-supervised models
                    if 'Gaussian' in model_name:
                        scores_val = -models[model_key].score_samples(X_val)
                    else:
                        scores_val = -models[model_key].decision_function(X_val)
                    ensemble_proba_val += weight * scores_val
                else:
                    # For supervised models
                    proba_val = models[model_key].predict_proba(X_val)[:, 1]
                    ensemble_proba_val += weight * proba_val
            except Exception as e:
                print(f"[ERROR] Error computing validation scores for {model_name}: {e}")
                continue

        # Get optimal threshold from validation set (NO TEST LEAKAGE)
        optimal_threshold = get_optimal_threshold_f1(ensemble_proba_val, y_val)

        # Now apply to test set
        ensemble_proba = np.zeros(len(y_test))
        for model_name, weight in zip(ensemble_models, weights):
            ensemble_proba += weight * all_probabilities[model_name]

        y_pred_ensemble = np.where(ensemble_proba >= optimal_threshold, 1, 0)

        predictions['F1-Weighted Ensemble (Top 3)'] = y_pred_ensemble
        report_dict = classification_report(y_test, y_pred_ensemble, output_dict=True)
        roc_auc = roc_auc_score(y_test, ensemble_proba)

        model_reports['F1-Weighted Ensemble (Top 3)'] = {
            'classification_report': report_dict,
            'roc_auc': roc_auc,
            'y_prob': ensemble_proba
        }
        optimal_thresholds['F1-Weighted Ensemble (Top 3)'] = optimal_threshold

        # End timing
        elapsed_time = time.time() - start_time
        model_timings['F1-Weighted Ensemble (Top 3)'] = elapsed_time

        # Print detailed results
        print(f"\n[TIME]  Evaluation Time: {elapsed_time:.2f}s")
        print(f"[THRESHOLD] Optimal Threshold (from validation): {optimal_threshold:.4f}")
        print(f"[ROC-AUC] ROC-AUC Score (test): {roc_auc:.4f}")
        print(f"\n[REPORT] Classification Report (test):")
        print(classification_report(y_test, y_pred_ensemble))
    else:
        print("[WARNING] Not enough models for ensemble, skipping...")

    # ===== TIMING SUMMARY =====
    if model_timings:
        print(f"\n{'='*60}\n[TIME]  Model Evaluation Time Summary\n{'='*60}")
        timing_df = pd.DataFrame([
            {'Model': model_name, 'Evaluation Time (s)': f"{time_s:.2f}"}
            for model_name, time_s in model_timings.items()
        ])
        print(timing_df.to_string(index=False))
        total_time = sum(model_timings.values())
        print(f"\n{'='*60}")
        print(f"Total Evaluation Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*60}\n")

    # ===== SUMMARY TABLE =====
    summary_data = []
    for model_name, report_dict in model_reports.items():
        report = report_dict['classification_report']

        # Safely extract Class 1 metrics
        if '1' in report and isinstance(report['1'], dict):
            precision_1 = report['1']['precision']
            recall_1 = report['1']['recall']
            f1_1 = report['1']['f1-score']
        else:
            precision_1 = recall_1 = f1_1 = 0.0

        # Safely extract Class 0 metrics
        if '0' in report and isinstance(report['0'], dict):
            precision_0 = report['0']['precision']
            recall_0 = report['0']['recall']
            f1_0 = report['0']['f1-score']
        else:
            precision_0 = recall_0 = f1_0 = 0.0

        # Safely get macro averages
        macro_avg = report.get('macro avg', {})
        if isinstance(macro_avg, dict):
            macro_precision = macro_avg.get('precision', 0)
            macro_recall = macro_avg.get('recall', 0)
            macro_f1 = macro_avg.get('f1-score', 0)
        else:
            macro_precision = macro_recall = macro_f1 = 0

        summary_data.append({
            'Model': model_name,
            'Precision (Class 0)': precision_0,
            'Recall (Class 0)': recall_0,
            'F1-Score (Class 0)': f1_0,
            'Precision (Class 1)': precision_1,
            'Recall (Class 1)': recall_1,
            'F1-Score (Class 1)': f1_1,
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1-Score': macro_f1,
            'ROC-AUC': report_dict.get('roc_auc', 0)
        })

    summary_df = pd.DataFrame(summary_data)

    # Print comprehensive summary table
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*80}\n")

    # ===== FEATURE IMPORTANCE =====
    feature_importance = {}

    # Random Forest feature importance
    if 'best_rf' in models:
        rf_model = models['best_rf']
        feature_names = data.get('feature_names', [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))])

        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance['random_forest'] = rf_importance

    # Logistic Regression feature importance (absolute coefficient values)
    if 'best_lr' in models:
        lr_model = models['best_lr']
        feature_names = data.get('feature_names', [f'Feature_{i}' for i in range(len(lr_model.coef_[0]))])

        lr_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.abs(lr_model.coef_[0])
        }).sort_values('coefficient', ascending=False)

        feature_importance['logistic_regression'] = lr_importance

        # Also store signed coefficients for positive/negative analysis
        lr_importance_signed = pd.DataFrame({
            'feature': feature_names,
            'coefficient': lr_model.coef_[0]  # Keep the sign
        }).sort_values('coefficient', key=abs, ascending=False)

        feature_importance['logistic_regression_signed'] = lr_importance_signed

    return {
        "predictions": predictions,
        "model_reports": model_reports,
        "summary_df": summary_df,
        "optimal_thresholds": optimal_thresholds,
        "feature_importance": feature_importance,
        "model_timings": model_timings,
    }

def main() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Main function to run model application"""
    from src.data_loader import load_and_prepare_data
    from src.models_calibration import calibrate_models

    print("="*60)
    print("FRAUD DETECTION PIPELINE - MODEL APPLICATION")
    print("="*60)

    data = load_and_prepare_data()
    models = calibrate_models(data)
    results = apply_models(data, models)

    return results, data, models


if __name__ == "__main__":
    results, data, models = main()