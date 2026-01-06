# Credit Card Fraud Detection

![Python 3.11-3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)
![UNIL](https://img.shields.io/badge/University-UNIL-7B1FA2.svg)

## Table of Contents

- [Project Overview](#project-overview)
- [Research Question](#research-question)
- [Performance Highlights](#performance-highlights)
- [Quick Start](#quick-start)
  - [First Time Setup](#first-time-setup)
  - [Updating Existing Environment](#updating-existing-environment)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [How to Run](#how-to-run)
- [Time Estimates](#time-estimates)
- [Output Files](#output-files)
- [Understanding the Output](#understanding-the-output)
- [Using Pre-Trained Models](#using-pre-trained-models)
- [Interactive Dashboard](#interactive-dashboard-streamlit)
- [Dataset Information](#dataset-information)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Author & Contact](#author--contact)
- [AI Tools Disclosure](#ai-tools-disclosure)

---

## Project Overview

This project implements a comprehensive fraud detection system using machine learning techniques to identify fraudulent credit card transactions. The system employs multiple supervised, unsupervised, and semi-supervised learning approaches with Bayesian hyperparameter optimization.

---

## Research Question
How do different machine learning paradigms compare for credit card fraud detection under extreme class imbalance, and which methodological strategies optimize model performance and reliability?

---

## Quick Start

### First Time Setup

```bash
# 1) Clone the repository
git clone https://github.com/dylanfernandez2305/Project-data-science.git
cd Project-data-science

# 2) Create and activate the Conda environment
conda env create -f environment.yml
conda activate data_science_project

# 3) Verify environment (optional but recommended)
python test_environment.py

# 4) Run the project
python main.py
```

**Recommended workflow:**
1. **[5]** Check Environment & Dependencies
2. **[1]** Prepare Data (auto-downloads dataset, ~30 seconds)
3. **[6]** Check Saved Models Status (pre-trained models available)
4. **[3]** Evaluate Saved Models (~4 min)
5. **[4]** Generate Visualizations (~3 min)

**Note:** Training new models ([2]) takes 15-30 minutes and is not required for evaluation.

---

### Updating Existing Environment

If you already have the environment and need to update it:

```bash
conda env update -n data_science_project -f environment.yml --prune
conda activate data_science_project
```

---

## Dataset

**Source**: Kaggle Credit Card Fraud Detection Dataset  
**Size**: 284,807 transactions  
**Features**: 30 numerical features (PCA-transformed V1-V28, Time, Amount)  
**Target**: Binary classification (0=Legitimate, 1=Fraud)  
**Class Imbalance**: ~0.172% fraud rate (492 fraudulent transactions)

---

## Project Structure

```
Project-data-science/
├── README.md                                     # Project overview, setup & usage instructions
├── PROPOSAL.md                       (70 lines)  # Project proposal 
├── main.py                          (162 lines)  # Entry point (launches menu)
├── environment.yml                               # Conda environment specification
├── test_environment.py                           # Environment verification script
│
├── src/                                          # Source code
│   ├── __init__.py                               # Package initializer
│   ├── data_loader.py               (249 lines)  # Data loading & preprocessing
│   ├── models_calibration.py        (710 lines)  # Hyperparameter optimization
│   ├── models_application.py        (431 lines)  # Model evaluation and threshold optimization
│   ├── performance_visualization.py (596 lines)  # Results & data visualization
│   └── menu.py                      (631 lines)  # Interactive command-line interface
│
├── data/                                         # Dataset storage
│   └── creditcard.csv                            # Kaggle dataset (auto-downloaded, 144 MB)
│
├── saved_models/                                 # Trained model storage
│   └── trained_models.pkl                        # All 9 models (see Google Drive)
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
├── dashboard/                                    # Interactive dashboard
│   └── dashboard.py                 (1573 lines) # Streamlit web interface (8 pages)
│
└── run_dashboard.py                              # Dashboard launcher with dependency checking
```

**Total codebase:** ~2,700 lines of documented, type-hinted Python code

---

## Technical Stack

**Compatible with:** Python 3.11-3.12  
**Not compatible with:** Python 3.13+

### Core Dependencies

**Data Science & Machine Learning:**
- numpy (1.26-2.x): numerical computing
- pandas (2.1+): data manipulation
- scipy (1.11+): scientific computing
- scikit-learn (1.4+): ML algorithms and preprocessing
- imbalanced-learn (0.12+): SMOTE resampling for class imbalance
- optuna (3.x-4.x): hyperparameter optimization
- joblib (1.3+): model persistence

**Visualization:**
- matplotlib (3.7+): static plots
- seaborn (0.13+): statistical visualizations
- plotly (5.17+): interactive charts
- streamlit (1.28+): web dashboard

**Utilities:**
- tqdm (4.65+): progress bars
- kagglehub (0.1+): automatic dataset download

**Important Version Notes:**
- NumPy 1.26-2.x supported (automatically managed by environment.yml)
- NumPy 3.x is not yet supported by all dependencies
- All version constraints are enforced in `environment.yml` for reproducibility

---

## Performance Highlights

Test set results (56,746 transactions, 98 frauds - showing Class 1 fraud detection performance):

| Model | Fraud Precision | Fraud Recall | Fraud F1 | ROC-AUC |
|-------|----------------|--------------|----------|---------|
| Random Forest (Supervised) | 0.85 | 0.72 | **0.78** | 0.962 |
| Logistic Regression (Supervised) | 0.78 | 0.64 | 0.70 | 0.980 |
| Ensemble (Top 3) | 0.45 | 0.64 | 0.53 | 0.933 |
| GMM (Semi-Supervised) | 0.26 | 0.36 | 0.31 | 0.958 |
| Isolation Forest (Semi-Supervised) | 0.13 | 0.47 | 0.20 | 0.939 |
| LOF (Semi-Supervised) | 0.12 | 0.23 | 0.16 | 0.884 |
| LOF (Unsupervised) | 0.07 | 0.18 | 0.10 | 0.890 |
| Isolation Forest (Unsupervised) | 0.02 | 0.43 | 0.04 | 0.870 |
| GMM (Unsupervised) | 0.01 | 0.30 | 0.02 | 0.764 |

**Key findings:**
- Random Forest (Supervised) achieved highest F1-score (0.78) for fraud detection
- Logistic Regression achieved best ROC-AUC (0.980) showing excellent ranking ability
- All models handled extreme class imbalance (0.172% fraud rate)
- Supervised methods vastly outperformed unsupervised approaches
- Unsupervised methods showed poor precision, generating many false positives

---

## How to Run

```bash
python main.py
```

**Menu options:**
- **[1]** Prepare Data (downloads dataset if needed)
- **[2]** Train & Save Models (15-30 minutes)
- **[3]** Evaluate Saved Models
- **[4]** Generate Visualizations
- **[5]** Check Environment & Dependencies
- **[6]** Check Saved Models Status
- **[7]** View Project Information

---

## Time Estimates

Expected duration for each operation:

| Task | Duration |
|------|----------|
| Data loading & preparation | 10-30 seconds |
| Train single model | 30 sec - 2 minutes |
| Train all 9 models (option [2]) | 15-30 minutes |
| Evaluate all models | 3-5 minutes |
| Generate visualizations | 2-3 minutes |
| View summary | Instant |

---

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

---

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

---

## Using Pre-Trained Models

**Note:** Pre-trained models may be included in `saved_models/trained_models.pkl`. If available, use menu options **[3]** (Evaluate Models) and **[4]** (Generate Visualizations) to skip the 15-30 minute training step and immediately evaluate results.

### Download Pre-trained Models

**Link:** [Download trained_models.pkl (313 MB)](https://drive.google.com/file/d/12NoDJmpz_FEDE8H-3dUD2BpS8pz9gvJf/view?usp=sharing)

**Installation:**
1. Download from Google Drive
2. Place in `saved_models/` folder:
   ```bash
   mv ~/Downloads/trained_models.pkl saved_models/
   ls -lh saved_models/trained_models.pkl  # Verify ~313 MB
   ```
3. Run:
   ```bash
   python main.py
   # Select [6] Check Saved Models Status
   # Select [3] Evaluate Models
   ```

### Alternative: Train Models Yourself

```bash
python main.py
# Select [2] Train & Save Models (~15-30 min)
```

---

## Interactive Dashboard (Streamlit)

Launch the interactive web dashboard with automatic dependency checking:

```bash
python run_dashboard.py
```

**Alternative (direct launch):**
```bash
streamlit run dashboard/dashboard.py
```

**Note:** If dependencies are missing, `run_dashboard.py` will display installation instructions.

---

## Dataset Information

### Automatic Download
The dataset is automatically downloaded from Kaggle when you first run the pipeline (requires `kagglehub` package, included in environment).

### Manual Download
If automatic download fails:
1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the `data/` directory
3. Expected file size: ~144 MB

---

## Troubleshooting

### **Environment Verification**

Run the verification script to check all dependencies:

```bash
conda activate data_science_project
python test_environment.py
```

**Expected output:**
```
============================================================
FRAUD DETECTION - ENVIRONMENT CHECK
============================================================
Python version: 3.11.5
✓ Python version OK

✓ Correct environment: data_science_project

============================================================
TESTING CORE PACKAGES
============================================================
  ✓ Standard library
  ✓ numpy       (2.4.0)
  ✓ pandas      (2.3.3)
  ✓ scipy       (1.16.3)
  ✓ sklearn     (1.7.2)

============================================================
TESTING ML PACKAGES
============================================================
  ✓ optuna      (4.6.0)
  ✓ imblearn    (0.14.0)
  ✓ joblib      (1.5.2)

============================================================
TESTING VISUALIZATION PACKAGES
============================================================
  ✓ matplotlib  (3.10.7)
  ✓ seaborn     (0.13.2)
  ✓ plotly      (6.5.0)

============================================================
TESTING DASHBOARD & UTILITIES
============================================================
  ✓ streamlit   (1.52.2)
  ✓ tqdm        (4.67.1)
  ✓ kagglehub

============================================================
TESTING SKLEARN SUBMODULES
============================================================
  ✓ sklearn submodules

============================================================
TESTING PROJECT MODULES
============================================================
  ✓ src.data_loader
  ✓ src.models_calibration
  ✓ src.models_application
  ✓ src.performance_visualization
  ✓ src.menu

============================================================
VALIDATION CHECKS
============================================================
  ✓ Key sklearn classes are callable
  ✓ NumPy and Pandas basics work

============================================================
SUMMARY
============================================================

✓ ALL CORE PACKAGES INSTALLED!

✓ ALL PROJECT MODULES AVAILABLE!

You can now run:
  python main.py                # Interactive menu
  python run_dashboard.py       # Launch dashboard

============================================================
```

---

### **Missing Package Errors**

If you see `ModuleNotFoundError` for any package:

```bash
conda activate data_science_project
python test_environment.py  # Identify missing packages
```

**Common fixes:**

**Missing optuna:**
```bash
pip install optuna
# Or: conda install -c conda-forge optuna -y
```

**Missing any package:**
```bash
# Update environment from yml file
conda env update -f environment.yml

# Or install individually
pip install <package-name>
```

---

### **Dataset Issues**

**Error: Dataset file not found**

The dataset should auto-download via `kagglehub`. If this fails:

1. **Manual download:**
   - Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Download `creditcard.csv`
   - Place in `data/` directory

2. **Verify:**
   ```bash
   ls -lh data/creditcard.csv  # Should show ~144 MB
   ```

---

### **Environment Not Activated**

If imports fail, ensure the environment is activated:

```bash
conda activate data_science_project
python -c "import sys; print(sys.executable)"
# Should show: .../envs/data_science_project/bin/python
```

---

### **Wrong Python Version**

If you see Python 3.13+ or Python 3.10-:

```bash
# Remove old environment
conda env remove -n data_science_project

# Recreate with correct Python version
conda env create -f environment.yml
conda activate data_science_project
```

---

### **NumPy Version Issues**

If you encounter NumPy 3.x related errors:

```bash
pip install 'numpy>=1.26,<3' --force-reinstall
```

The `environment.yml` automatically ensures correct versions, so this should only be needed if packages were manually installed.

---

## Documentation

- **Academic Report**: Detailed methodology and results (PDF)
- **Proposal**: [PROPOSAL.md](PROPOSAL.md)
- **AI Usage Transparency**: [AI-USAGE.md](AI-USAGE.md)

---

## Author & Contact

**Dylan Fernandez** (20428967)  
University of Lausanne (HEC Lausanne)  
Dylan.Fernandez@unil.ch

---

## AI Tools Disclosure

This project used AI tools (ChatGPT, Claude) for code review, documentation, and report writing assistance. 

**Full transparency report**: See [AI-USAGE.md](AI-USAGE.md) for detailed disclosure of AI tool usage.
