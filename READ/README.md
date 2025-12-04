# Credit Card Fraud Detection

## Project Overview

This project implements a comprehensive fraud detection system using machine learning techniques to identify fraudulent credit card transactions. The system employs multiple supervised, unsupervised, and semi-supervised learning approaches with Bayesian hyperparameter optimization.

## Quick Start

**Get started in 3 steps:**

```bash
# 1. Create conda environment from environment.yml
conda env create -f environment.yml
conda activate data_science_project

# 2. Launch interactive menu
cd src
python menu.py

# 3. Select option [2] to train models, or [3] if pre-trained models exist
```

**Alternative with pip:**
```bash
# Install dependencies directly with pip
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn joblib optuna kagglehub shap
cd src
python menu.py
```

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Implemented Models](#implemented-models)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Time Estimates](#time-estimates)
- [Output Files](#output-files)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)
- [Dataset Information](#dataset-information)
- [Common Workflows](#common-workflows)
- [Next Steps](#next-steps)
- [Author and Contact](#author-and-contact)

## Dataset

**Source**: Kaggle Credit Card Fraud Detection Dataset
**Size**: 284,807 transactions
**Features**: 30 numerical features (PCA-transformed V1-V28, Time, Amount)
**Target**: Binary classification (0=Legitimate, 1=Fraud)
**Class Imbalance**: ~0.172% fraud rate (492 fraudulent transactions)

## Project Structure

```
Project_data_science_source/
├── src/                                          # Source code
│   ├── main.py                      (180 lines)  # Data loading & preprocessing
│   ├── models_calibration.py        (521 lines)  # Hyperparameter optimization
│   ├── models_application.py        (236 lines)  # Model evaluation
│   ├── performance_visualization.py (410 lines)  # Results & data visualization
│   └── menu.py                      (524 lines)  # Interactive menu system
│
├── data/                                         # Dataset storage
│   └── creditcard.csv                            # Kaggle dataset (auto-downloaded)
│
├── saved_models/                                 # Trained model storage
│   └── trained_models.pkl                        # All 8 models + ensemble
│
├── output/                                       # Generated visualizations
│   ├── 0_class_distribution.png
│   ├── 0_amount_distribution.png
│   ├── 1_confusion_matrices.png
│   ├── 2_performance_metrics.png
│   ├── 3_f1_score_comparison.png
│   ├── 4_roc_curves_all_models.png
│   ├── 5_roc_curves_top_performers.png
│   ├── 6_precision_recall_curves.png
│   ├── 7_feature_importance_rf.png
│   ├── 8_feature_importance_lr.png
│   └── 9_lr_coefficients_signed.png
│
├── READ/                                         # Documentation
│   ├── README.md                                 # This file - complete guide
│   └── Proposal.md                   (69 lines)  # Project proposal
│
└── environment.yml                               # Conda environment specification
```

**Total Code Lines**: 1,871 lines (excluding blank lines and comments)

## Implemented Models

### Supervised Learning
1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble tree-based classifier

### Unsupervised Learning
3. **Isolation Forest** - Anomaly detection based on data isolation
4. **Local Outlier Factor (LOF)** - Density-based outlier detection
5. **Gaussian Mixture Model (GMM)** - Probabilistic clustering

### Semi-Supervised Learning
6. **Isolation Forest (Semi-Supervised)** - Trained only on legitimate transactions
7. **LOF (Semi-Supervised)** - Trained only on legitimate transactions
8. **GMM (Semi-Supervised)** - Trained only on legitimate transactions

### Ensemble Model
9. **F1-Weighted Ensemble** - Combines top 3 models weighted by F1-scores

## Key Features

### Advanced Hyperparameter Optimization
- Optuna Bayesian Optimization with TPE sampler
- 3-4x faster than GridSearchCV
- Reproducible results with seeded optimization (seed=42)
- 20 trials per model with intelligent search space exploration

### Comprehensive Feature Engineering (54 features)
- Temporal features: Hour, Minute, Second, Time_Since_Last_Transaction
- Cyclical encoding: Hour_sin, Hour_cos, Minute_sin, Minute_cos
- Transaction patterns: Transactions_per_Hour, Transactions_per_TimeOfDay
- Amount features: LogAmount, Amount_per_Hour, Amount_frequency
- Statistical features: Amount_Hour_ZScore, Amount_Ratio_to_Hour_Mean
- Interaction features: Temporal_Amount_Intensity, Hour_Activity_Ratio

### Robust Data Preprocessing
- RobustScaler normalization (resistant to outliers)
- Time-series split (80/20 train/test)
- SMOTE oversampling for class imbalance
- Dataset reduction for distance-based models (30% chronological sampling)

### Model Evaluation
- F1-score optimized thresholds
- Comprehensive classification reports
- ROC-AUC analysis
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis

### Visualization Suite
- Performance comparison across all models
- ROC curves (all models and top performers)
- Precision-Recall curves
- Confusion matrices with normalized views
- Feature importance plots
- Logistic Regression coefficient analysis

## Technical Stack

### Core Libraries
- Python 3.8+
- scikit-learn >= 1.0.0: ML algorithms and preprocessing
- optuna >= 3.0.0: Bayesian hyperparameter optimization
- imbalanced-learn >= 0.12.0: SMOTE resampling
- numpy >= 1.21.0, <2.0.0: Numerical computations
- pandas >= 1.3.0: Data manipulation

### Visualization
- matplotlib >= 3.4.0: Plot generation
- seaborn >= 0.11.0: Statistical visualizations

### Additional Libraries
- scipy >= 1.7.0: Scientific computing
- joblib >= 1.0.0: Model persistence (cross-version compatible)
- kagglehub >= 0.1.0: Dataset download (optional)
- shap >= 0.40.0: Model interpretability (optional)

## Performance Metrics

Typical results on test set (56,745 transactions, 74 frauds):

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Random Forest | 0.96 | 0.70 | 0.81 | 0.979 |
| Logistic Regression | 0.73 | 0.76 | 0.74 | 0.979 |
| F1-Weighted Ensemble | 0.96 | 0.68 | 0.79 | 0.973 |
| ISO Forest (Semi-Sup) | 0.14 | 0.41 | 0.21 | 0.934 |
| GMM (Semi-Sup) | 0.21 | 0.46 | 0.29 | 0.932 |

**Note**: Performance may vary depending on random seed and Optuna trial results.

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda (Anaconda/Miniconda) **recommended** or pip package manager

### Method 1: Conda Environment (Recommended)

1. **Navigate to project directory:**
```bash
cd /path/to/Project_data_science_source
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate data_science_project
```

3. **Verify installation:**
```bash
python -c "import sklearn, optuna, pandas; print('Installation successful')"
```

### Method 2: pip Installation

1. **Navigate to project directory:**
```bash
cd /path/to/Project_data_science_source
```

2. **Install dependencies:**
```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn joblib optuna kagglehub shap
```

3. **Verify installation:**
```bash
python -c "import sklearn, optuna, pandas; print('Installation successful')"
```

### Important Version Notes
- **NumPy**: Must be version <2.0 (numpy 1.26.4 recommended)
- If you have NumPy 2.x, downgrade with: `pip install 'numpy<2' --force-reinstall`
- The conda environment.yml file automatically ensures correct versions

## How to Run

### Option 1: Interactive Menu (Recommended)

The easiest way to use the system:

```bash
cd src
python menu.py
```

**Menu Options:**

**Individual Steps:**
- **[1]** Data Preparation Only - Load and preprocess data
- **[2]** Train & Save Models - Train all 8 models and save them (15-30 min) - no need if already saved
- **[3]** Evaluate Models - Evaluate using saved models
- **[4]** Show Visualizations - Generate all performance plots

**Utilities:**
- **[5]** Check Environment & Dependencies - Verify installation
- **[6]** Check Saved Models Status - View saved model information
- **[7]** View Project Information - Display project details

**[0]** Exit

### Option 2: Direct Script Execution

Run individual scripts manually:

```bash
cd src

# 1. Data preparation and feature engineering
python main.py

# 2. Train models (optional if using pre-trained models)
python models_calibration.py

# 3. Evaluate models
python models_application.py

# 4. Generate visualizations
python performance_visualization.py
```

### Using Pre-Trained Models

**Note:** Pre-trained models may be included in `saved_models/trained_models.pkl`. If available, use menu options **[3]** (Evaluate Models) and **[4]** (Show Visualizations) to skip the 15-30 minute training step and immediately evaluate and visualize results.

## Time Estimates

Expected duration for each operation:

| Task | Duration |
|------|----------|
| Data loading & preparation | 10-30 seconds |
| Train single model | 30 sec - 2 minutes |
| Train all 8 models (option [2]) | 15-30 minutes |
| Evaluate all models | 1-2 minutes |
| Generate visualizations | 30-60 seconds |
| View summary | Instant |

## Output Files

### Generated Visualizations

The system generates publication-quality visualizations in the `output/` directory:

**Data Distribution Plots:**
1. **0_class_distribution.png** - Class distribution (pie chart and bar chart)
2. **0_amount_distribution.png** - Transaction amount distributions by class

**Model Performance Plots:**
3. **1_confusion_matrices.png** - 3×3 grid of confusion matrices
4. **2_performance_metrics.png** - Bar charts comparing all metrics across models
5. **3_f1_score_comparison.png** - F1-score comparison by class
6. **4_roc_curves_all_models.png** - ROC curves for all 9 models
7. **5_roc_curves_top_performers.png** - ROC curves for top 3 models
8. **6_precision_recall_curves.png** - Precision-Recall curves with AP scores

**Feature Importance Plots:**
9. **7_feature_importance_rf.png** - Random Forest feature importance ranking
10. **8_feature_importance_lr.png** - Logistic Regression feature importance
11. **9_lr_coefficients_signed.png** - Signed LR coefficients (fraud indicators)

All plots are automatically generated by `performance_visualization.py`.

### Saved Models

Models are saved to `saved_models/trained_models.pkl` with metadata:
- All 8 trained models plus ensemble
- Training timestamp
- Optuna optimization details
- Model performance metrics

## Understanding the Output

### Classification Report Example

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56671  ← Legitimate transactions
           1       0.96      0.70      0.81        74  ← Fraud transactions

    accuracy                           1.00     56745
   macro avg       0.98      0.85      0.91     56745
weighted avg       1.00      1.00      1.00     56745
```

**Metrics Explained:**
- **Precision (Class 1)**: Of all flagged frauds, what percentage were actually fraud?
- **Recall (Class 1)**: Of all actual frauds, what percentage did we catch?
- **F1-Score**: Harmonic mean balancing precision and recall
- **Support**: Number of actual instances in each class

### F1-Weighted Ensemble

The ensemble automatically combines the top 3 models by F1-score with proportional weighting:

```
Selected top 3 models by F1-score:
  1. Random Forest (Supervised)
     F1-Score: 0.8100 | Weight: 41.75%
  2. Logistic Regression (Supervised)
     F1-Score: 0.7400 | Weight: 38.14%
  3. Local Outlier Factor (Semi-Supervised)
     F1-Score: 0.3900 | Weight: 20.10%
```

Weights are normalized by performance, giving more influence to better-performing models.

## Troubleshooting

### Error: "Model file incompatible with current numpy version"

**Cause:** `trained_models.pkl` created with different numpy version

**Solution:** Retrain models using menu option [2]

```bash
cd src
python menu.py
# Select [2] - Train & Save Models
```

### Error: "No saved models found"

**Cause:** Models haven't been trained yet

**Solution:** Train models first:
- Use menu option [2] to train and save models

### Error: "FileNotFoundError: creditcard.csv not found"

**Cause:** Dataset not downloaded

**Solutions:**
1. Install kagglehub for automatic download: `pip install kagglehub`
2. Manual download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Place `creditcard.csv` in `data/` directory

### Slow Training Time

**Normal:** 15-30 minutes for all 8 models

**If significantly slower:**
- Check CPU usage and close other applications
- Reduce Optuna trials: Edit `models_calibration.py` line 69, change `N_TRIALS = 20` to `N_TRIALS = 10`
- Train models individually using menu options

### Out of Memory Errors

**Symptoms:** Python crashes during training

**Solutions:**
1. Close memory-intensive applications
2. Train models individually using direct script execution
3. Reduce dataset size for IF/LOF (already reduced to 30% by default)
4. Increase system swap space

### ImportError with NumPy 2.x

**Cause:** scikit-learn incompatible with NumPy 2.x

**Solution:** Downgrade NumPy:
```bash
pip install 'numpy<2' --force-reinstall
pip install pandas scikit-learn imbalanced-learn --force-reinstall
```

Or use menu option [5] to automatically check and fix dependencies.

## Dataset Information

### Automatic Download
The dataset is automatically downloaded from Kaggle when you first run the pipeline (requires `kagglehub` package).

### Manual Download
If automatic download fails:
1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the `data/` directory
3. Expected file size: ~144 MB

## Common Workflows

### First-Time Setup

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate data_science_project
cd src
python3 menu.py
# Select [5] to verify environment
# Select [2] to train models
```

### Quick Evaluation (with pre-trained models)

```bash
cd src
python3 menu.py
# Select [3] to evaluate saved models
# Select [4] to generate visualizations
```

### Custom Experiments

```bash
cd src
# Modify hyperparameters in models_calibration.py
python3 menu.py
# Select [2] to retrain with new settings
# Select [3] to evaluate
# Select [4] to visualize
```

## Next Steps

After running the complete pipeline:
1. **Review visualizations** in the `output/` folder
2. **Analyze performance** using the summary output
3. **Study feature importance** plots (files 7-9) to understand model decisions
4. **Compare ROC curves** (files 4-5) to assess model discrimination
5. **Examine confusion matrices** (files 1-3) for prediction patterns

## Author and Contact

**Dylan Fernandez**
University of Lausanne
Dylan.Fernandez@unil.ch
