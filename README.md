# Credit Card Fraud Detection

![Python 3.11-3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)
![UNIL](https://img.shields.io/badge/University-UNIL-7B1FA2.svg)

## Research Question
How do different machine learning paradigms compare for credit card fraud detection under extreme class imbalance, and which methodological strategies optimize model performance and reliability?

---

## Quick Start

```bash
# 0) Clone the repository
git clone https://github.com/dylanfernandez2305/Project-data-science.git
cd Project-data-science

# 1) Create and activate the Conda environment
conda env create -f environment.yml
conda activate data_science_project

# 2) Run the project
python main.py
```

**Recommended workflow:**
1. **[5]** Check Environment
2. **[6]** Check Saved Models (pre-trained models included)
3. **[3]** Evaluate Saved Models (~5 min)
4. **[4]** Generate Visualizations (~3 min)

**Note:** Training new models ([2]) takes 15-30 minutes and is not required for evaluation.

---

## Project Structure

```
Project_data_science_source/
├── README.md                                     # Project overview, setup & usage instructions
├── PROPOSAL.md                       (70 lines)  # Project proposal 
├── main.py                          (162 lines)  # Entry point (launches menu)
├── environment.yml                               # Conda environment specification
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
│   └── trained_models.pkl                        # All 9 models  (see Google Drive)
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

**Tested on:** Python 3.11-3.12 (see `environment.yml`)

### Core (required)
- scikit-learn: ML algorithms + preprocessing
- optuna: hyperparameter optimization
- imbalanced-learn: SMOTE resampling
- numpy, pandas, scipy: data + numerical computing
- joblib: model persistence

### Visualization
- matplotlib, seaborn: static plots
- plotly, streamlit: interactive dashboard

### Optional
- kagglehub: dataset download helper
- shap: model interpretability

## How to Run

```bash
python main.py
```

**Menu options:**
- [1] Data Preparation Only
- [2] Train & Save Models
- [3] Evaluate Models
- [4] Show Visualizations
- [5] Check Environment & Dependencies
- [6] Check Saved Models Status
- [7] View Project Information

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

### Using Pre-Trained Models

**Note:** Pre-trained models may be included in `saved_models/trained_models.pkl`. If available, use menu options **[3]** (Evaluate Models) and **[4]** (Show Visualizations) to skip the 15-30 minute training step and immediately evaluate and visualize results.

### Download Pre-trained Models

**Link:** [Download trained_models.pkl](https://drive.google.com/file/d/12NoDJmpz_FEDE8H-3dUD2BpS8pz9gvJf/view?usp=sharing)

### Installation

1. Download from Google Drive
2. Place in `saved_models/` folder:
   ```bash
   mv ~/Downloads/trained_models.pkl saved_models/
   ls -lh saved_models/trained_models.pkl  # Verify ~300 MB
   ```
3. Run:
   ```bash
   python main.py
   # Select [3] Evaluate Models
   ```

**If the file is incompatible** (library versions changed), retrain using [2].

### Alternative: Train Yourself

```bash
python main.py
# Select [2] Train & Save Models (~15-30 min)
```

---

## Interactive Dashboard (Streamlit)

**Recommended** (after conda install):
```bash
python run_dashboard.py
```

**Alternative:**
```bash
streamlit run dashboard/dashboard.py
```

**Note:** The launcher can offer a pip fallback if dependencies are missing (conda is recommended for reproducibility).

---

## Dataset Information

### Automatic Download
The dataset is automatically downloaded from Kaggle when you first run the pipeline (requires `kagglehub` package).

### Manual Download
If automatic download fails:
1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the `data/` directory
3. Expected file size: ~144 MB

## Troubleshooting (common)

- **Dataset missing:** put `creditcard.csv` in `data/` (or enable auto-download via `kagglehub`)
- **Model file incompatible:** retrain with [2]
- **Dependency issues:** use menu option [5] to diagnose versions (NumPy 2.x supported; avoid 3.x)

---

## Documentation

- Academic report (PDF)
- Proposal (PROPOSAL.md)
- AI usage transparency (AI-USAGE.md)

---

## Author & Contact

**Dylan Fernandez** (20428967)  
University of Lausanne (HEC Lausanne)  
Dylan.Fernandez@unil.ch
