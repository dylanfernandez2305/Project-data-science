# Performance visualization
# This file contains code for visualizing model performance

# ===== IMPORTS =====
import sys
sys.dont_write_bytecode = True  # Prevent .pyc file creation

# Allow running this file directly (e.g., `python src/performance_visualization.py`) while
# still using absolute imports like `from src...`.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import os
from datetime import datetime


def visualize_performance(data, results, save_plots=True, show_plots=False, output_dir='output'):
    """
    Create visualizations for model performance

    Parameters:
    -----------
    data : dict
        Dictionary containing test data
    results : dict
        Dictionary containing predictions and reports
    save_plots : bool, default=True
        Whether to save plots to files
    show_plots : bool, default=False
        Whether to display plots interactively
    output_dir : str, default='output'
        Directory to save plots to

    Returns:
    --------
    output_path : Path or None
        Path to output directory if plots were saved, None otherwise
    """
    y_test = data['y_test']
    predictions = results['predictions']
    model_reports = results['model_reports']

    labels = ['Legitimate', 'Fraud']

    # Create output directory if saving plots
    output_path = None
    if save_plots:
        # If output_dir is relative, make it relative to project root (parent of src)
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            # Get project root (parent directory of src)
            project_root = Path(__file__).parent.parent
            output_path = project_root / output_dir

        output_path.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Saving plots to: {output_path.absolute()}")
        print(f"Timestamp: {timestamp}\n")

    # ===== DATA DISTRIBUTION PLOTS =====
    # Load raw data for distribution plots
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "creditcard.csv"

    if dataset_path.exists():
        print("Creating data distribution plots...")
        raw_data = pd.read_csv(dataset_path)

        # Plot 1: Class Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        fraud_counts = raw_data['Class'].value_counts()
        labels_dist = ['Legitimate', 'Fraud']
        colors = ['#2ecc71', '#e74c3c']

        # Pie chart
        axes[0].pie(fraud_counts, labels=labels_dist, autopct='%1.2f%%',
                    colors=colors, startangle=90, explode=(0, 0.1))
        axes[0].set_title('Class Distribution: Fraud vs Legitimate Transactions',
                          fontsize=12, fontweight='bold', pad=20)

        # Bar chart
        axes[1].bar(['Legitimate\n(Class 0)', 'Fraud\n(Class 1)'],
                    fraud_counts, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Number of Transactions', fontsize=11)
        axes[1].set_title('Transaction Counts by Class', fontsize=12, fontweight='bold', pad=20)
        axes[1].grid(axis='y', alpha=0.3)

        # Add count labels on bars
        for i, (idx, count) in enumerate(fraud_counts.items()):
            axes[1].text(i, count + 5000, f'{count:,}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add imbalance ratio
        ratio = fraud_counts[0] / fraud_counts[1]
        fig.text(0.5, 0.02, f'Class Imbalance Ratio: 1:{ratio:.1f} (Fraud:Legitimate)',
                 ha='center', fontsize=11, style='italic',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_plots:
            filepath = output_path / '0_class_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filepath.name}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Plot 2: Amount Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall amount distribution
        axes[0].hist(raw_data['Amount'], bins=100, color='steelblue',
                     edgecolor='black', alpha=0.7, linewidth=0.5)
        axes[0].set_xlabel('Transaction Amount ($)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Overall Transaction Amount Distribution',
                          fontsize=12, fontweight='bold', pad=20)
        axes[0].set_yscale('log')
        axes[0].grid(axis='y', alpha=0.3)

        # Add statistics
        mean_amt = raw_data['Amount'].mean()
        median_amt = raw_data['Amount'].median()
        max_amt = raw_data['Amount'].max()
        axes[0].text(0.98, 0.97,
                     f'Mean: ${mean_amt:.2f}\nMedian: ${median_amt:.2f}\nMax: ${max_amt:.2f}',
                     transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Amount distribution by class
        fraud_amounts = raw_data[raw_data['Class'] == 1]['Amount']
        legit_amounts = raw_data[raw_data['Class'] == 0]['Amount']

        axes[1].hist([legit_amounts, fraud_amounts], bins=50,
                     color=['#2ecc71', '#e74c3c'],
                     label=['Legitimate', 'Fraud'],
                     edgecolor='black', alpha=0.7, linewidth=0.5)
        axes[1].set_xlabel('Transaction Amount ($)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Transaction Amount Distribution by Class',
                          fontsize=12, fontweight='bold', pad=20)
        axes[1].set_yscale('log')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

        # Add comparison statistics
        axes[1].text(0.98, 0.97,
                     f'Fraud - Mean: ${fraud_amounts.mean():.2f}, Median: ${fraud_amounts.median():.2f}\n'
                     f'Legit - Mean: ${legit_amounts.mean():.2f}, Median: ${legit_amounts.median():.2f}',
                     transform=axes[1].transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_plots:
            filepath = output_path / '0_amount_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filepath.name}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        print()  # Add blank line for spacing

    # ===== CONFUSION MATRICES =====
    print("Creating confusion matrices...")

    # Determine grid size based on number of models
    n_models = len(predictions)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.ravel()

    plot_idx = 0

    # Plot all models
    for title in predictions.keys():
        cm = confusion_matrix(y_test, predictions[title])
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[plot_idx])
        axes[plot_idx].set_title(title, fontsize=10)
        axes[plot_idx].set_xlabel('Predicted')
        axes[plot_idx].set_ylabel('Actual')
        plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_plots:
        filepath = output_path / '1_confusion_matrices.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== PERFORMANCE METRICS BAR CHARTS =====
    print("Creating performance metrics charts...")

    # Recreate the results dictionary with shorter model names
    shorter_model_names = {
        "LR (Sup)": "Logistic Regression (Supervised)",
        "RF (Sup)": "Random Forest (Supervised)",
        "IF (Unsup)": "Isolation Forest (Unsupervised)",
        "LOF (Unsup)": "Local Outlier Factor (Unsupervised)",
        "GMM (Unsup)": "Gaussian Mixture Models (Unsupervised)",
        "IF (SemSup)": "Isolation Forest (Semi-Supervised)",
        "LOF (SemSup)": "Local Outlier Factor (Semi-Supervised)",
        "GMM (SemSup)": "Gaussian Mixture Model (Semi-Supervised)",
        "Ensemble": "F1-Weighted Ensemble (Top 3)"
    }

    # Recreate results dictionary with shorter names
    results_dict = {}
    for short_name, full_name in shorter_model_names.items():
        preds = predictions[full_name]
        report = model_reports[full_name]['classification_report']
        results_dict[short_name] = {
            'Class 1': {
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1-score': report['1']['f1-score']
            },
            'Macro Avg': {
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1-score': report['macro avg']['f1-score']
            }
        }

    # Prepare data for plotting
    plot_data_list = []
    for model_name, metrics in results_dict.items():
        for class_type, scores in metrics.items():
            for metric, value in scores.items():
                plot_data_list.append({
                    'Model': model_name,
                    'Class': class_type,
                    'Metric': metric,
                    'Score': value
                })

    plot_df = pd.DataFrame(plot_data_list)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Class 1 (Fraud)
    class_1_data = plot_df[plot_df['Class'] == 'Class 1']
    sns.barplot(data=class_1_data, x='Model', y='Score', hue='Metric', ax=axes[0], palette='viridis')
    axes[0].set_title('Fraud Detection Performance (Class 1)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('Model')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Metric')
    axes[0].set_ylim(0, 1)

    # Add value annotations for Class 1
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Macro Average
    macro_data = plot_df[plot_df['Class'] == 'Macro Avg']
    sns.barplot(data=macro_data, x='Model', y='Score', hue='Metric', ax=axes[1], palette='viridis')
    axes[1].set_title('Overall Performance (Macro Average)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].set_xlabel('Model')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Metric')
    axes[1].set_ylim(0, 1)

    # Add value annotations for Macro Avg
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    if save_plots:
        filepath = output_path / '2_performance_metrics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== F1-SCORE COMPARISON =====
    print("Creating F1-score comparison chart...")

    f1_data = plot_df[plot_df['Metric'] == 'f1-score']

    plt.figure(figsize=(12, 8))
    sns.barplot(data=f1_data, x='Model', y='Score', hue='Class', palette='viridis')
    plt.title('F1-Score Comparison: Fraud Detection vs Overall Performance', fontsize=16, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(title='Performance Type', loc='upper right')

    # Add value annotations
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.2f}',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_plots:
        filepath = output_path / '3_f1_score_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== ROC CURVES =====
    print("Creating ROC curves...")

    plt.figure(figsize=(15, 9))

    for model_name in predictions.keys():
        y_prob = model_reports[model_name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)', alpha=0.8)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Fraud Detection Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_plots:
        filepath = output_path / '4_roc_curves_all_models.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== TOP PERFORMING MODELS ROC CURVES =====
    print("Creating top performers ROC curves...")

    top_models = ["Logistic Regression (Supervised)", "Random Forest (Supervised)", "F1-Weighted Ensemble (Top 3)"]
    plt.figure(figsize=(15, 9))

    for model_name in top_models:
        y_prob = model_reports[model_name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=3)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.500)', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Top Performing Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_plots:
        filepath = output_path / '5_roc_curves_top_performers.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== PRECISION-RECALL CURVES =====
    print("Creating Precision-Recall curves...")

    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(15, 9))

    for model_name in predictions.keys():
        y_prob = model_reports[model_name]['y_prob']
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AP={ap_score:.3f})', linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Fraud Detection Models', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_plots:
        filepath = output_path / '6_precision_recall_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ===== FEATURE IMPORTANCE - RANDOM FOREST =====
    print("Creating feature importance plot (Random Forest)...")

    if 'feature_importance' in results and 'random_forest' in results['feature_importance']:
        rf_importance = results['feature_importance']['random_forest']

        # Plot top 20 features
        top_n = min(20, len(rf_importance))
        plt.figure(figsize=(12, 10))
        plt.barh(range(top_n), rf_importance['importance'].head(top_n), color='steelblue')
        plt.yticks(range(top_n), rf_importance['feature'].head(top_n))
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance - Random Forest (Top 20)', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_plots:
            filepath = output_path / '7_feature_importance_rf.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filepath.name}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    # ===== FEATURE IMPORTANCE - LOGISTIC REGRESSION =====
    print("Creating feature importance plot (Logistic Regression)...")

    if 'feature_importance' in results and 'logistic_regression' in results['feature_importance']:
        lr_importance = results['feature_importance']['logistic_regression']

        # Plot top 20 features
        top_n = min(20, len(lr_importance))
        plt.figure(figsize=(12, 10))
        plt.barh(range(top_n), lr_importance['coefficient'].head(top_n), color='coral')
        plt.yticks(range(top_n), lr_importance['feature'].head(top_n))
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance - Logistic Regression (Top 20)', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_plots:
            filepath = output_path / '8_feature_importance_lr.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filepath.name}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    # ===== FEATURE COEFFICIENTS - LOGISTIC REGRESSION (SIGNED) =====
    print("Creating signed coefficients plot (Logistic Regression)...")

    if 'feature_importance' in results and 'logistic_regression_signed' in results['feature_importance']:
        lr_importance_signed = results['feature_importance']['logistic_regression_signed']

        # Plot top 20 features by absolute value, but show the sign
        top_n = min(20, len(lr_importance_signed))

        features = lr_importance_signed['feature'].head(top_n).values
        coefficients = lr_importance_signed['coefficient'].head(top_n).values

        # Color bars based on sign: red for positive (fraud), blue for negative (legitimate)
        colors = ['#d62728' if c > 0 else '#1f77b4' for c in coefficients]

        plt.figure(figsize=(12, 10))
        plt.barh(range(top_n), coefficients, color=colors)
        plt.yticks(range(top_n), features)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Logistic Regression Coefficients - Fraud Indicators (Top 20)', fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='Positive (→ Fraud)'),
            Patch(facecolor='#1f77b4', label='Negative (→ Legitimate)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_plots:
            filepath = output_path / '9_lr_coefficients_signed.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filepath.name}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    print("\n" + "="*70)
    print("All visualizations complete!")
    if save_plots:
        print(f"\n[SUCCESS] All plots saved to: {output_path.absolute()}")
        print(f"   Total files: 9")
    print("="*70)

    return output_path if save_plots else None


def main():
    """Main function to run performance visualization"""
    from src.data_loader import load_and_prepare_data
    from models_calibration import load_models
    from models_application import apply_models

    print("="*60)
    print("FRAUD DETECTION PIPELINE - PERFORMANCE VISUALIZATION")
    print("="*60)
    print("\nThis will load saved models and generate visualizations.\n")

    # Load data
    print("Step 1/3: Loading and preparing data...")
    data = load_and_prepare_data()

    # Load saved models instead of training
    print("\nStep 2/3: Loading saved models...")
    try:
        models = load_models()
        print(f"[OK] Loaded {len(models)} models from saved_models/trained_models.pkl")
    except FileNotFoundError:
        print("\n[ERROR] No saved models found!")
        print("Please train models first using one of these methods:")
        print("  1. Run menu.py and select option [4] - Train & Save Models")
        print("  2. Run menu.py and select option [1] - Complete Pipeline")
        print("  3. Run models_calibration.py directly")
        return

    # Evaluate models
    print("\nStep 3/3: Evaluating models and generating visualizations...")
    results = apply_models(data, models)

    # Generate visualizations
    print()
    visualize_performance(data, results, save_plots=True, show_plots=False)

    print("\n" + "="*60)
    print("[SUCCESS] Visualizations complete!")
    print("="*60)
    print("\nCheck the output/ folder for all generated plots.")


if __name__ == "__main__":
    main()
