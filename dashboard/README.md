# Interactive Dashboard - Credit Card Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)]()
[![UNIL](https://img.shields.io/badge/University-UNIL-7B1FA2.svg)](https://www.unil.ch/)

## Dashboard Overview

This interactive Streamlit dashboard provides a professional interface for exploring the Credit Card Fraud Detection project results. It visualizes model performance, dataset characteristics, and technical methodology across 8 comprehensive pages.

**Key Features:**
- 8 interactive pages with Plotly visualizations
- Real-time performance comparisons
- Interactive model comparison tool
- Feature engineering details (54 features)
- Model evaluation metrics and insights

## Quick Launch Dashboard

**Fastest way to launch (from project root):**

```bash
# Option 1: Using Python launcher (recommended)
python run_dashboard.py

# Option 2: Direct Streamlit command
streamlit run dashboard/dashboard.py

# Option 3: From dashboard folder
cd dashboard
streamlit run dashboard.py
```

**The dashboard will automatically open in your browser at:** `http://localhost:8501`

**To stop the dashboard:** Press `Ctrl + C` in the terminal

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dashboard Content](#dashboard-content)
- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)
- [VS Code Integration](#vs-code-integration)
- [Dashboard Structure](#dashboard-structure)
- [Technologies Used](#technologies-used)
- [Verification Checklist](#verification-checklist)
- [Author and Contact](#author-and-contact)

## Prerequisites

Ensure the following packages are installed:

```bash
pip install streamlit pandas plotly numpy
```

**Or use the main project environment:**
```bash
conda env create -f environment.yml
conda activate data_science_project
```

## Dashboard Content

### 8 Interactive Pages

#### 1. Home
- Project title and elegant subtitle
- Professional introduction to the fraud detection system

#### 2. Executive Summary
- Best performing model: Random Forest (F1=0.78)
- Key findings and insights
- Production deployment recommendations
- Project limitations and perspectives

#### 3. Dataset Overview
- Dataset statistics (283,726 transactions, 0.17% fraud rate)
- Class imbalance visualization
- **Key features table** showcasing 6 main engineered features
- **Expandable feature list** with all 54 features (30 original + 24 engineered)

#### 4. Methodology
- 3 research questions
- Validation protocol explanation
- Chronological data split strategy (64/16/20)
- Hyperparameter optimization approach

#### 5. Models Evaluated
- 9 models across 3 learning paradigms
- F1-weighted ensemble composition (pie chart)
- Model descriptions and training strategies

#### 6. Performance Results
- F1-score comparison charts
- Precision-Recall trade-off analysis
- ROC-AUC interpretation for imbalanced datasets
- Interactive performance tables
- Call-to-action to Interactive Comparison page

#### 7. Interactive Comparison ⭐ NEW
- **Bonus interactive feature** for detailed model comparisons
- Select any 2 models from dropdown menus (🔵 Model A vs 🟢 Model B)
- Side-by-side metrics comparison table with "Winner" column
- Confusion matrices comparison (color-coded: blue vs green)
- Performance radar chart overlay (7 metrics)
- Automated comparison summary with KPI cards
- Key insights based on priorities (F1-Score, Precision, Recall)

#### 8. Technical Details
- Code implementation snippets
- Project structure overview
- Key technical references

## Features

### Interactive Visualizations
- **Plotly charts**: Zoom, pan, hover for detailed inspection
- **Bar charts**: Model performance comparisons
- **Pie charts**: Ensemble composition breakdown
- **Scatter plots**: Precision-Recall trade-offs
- **Tables**: Feature lists and performance metrics

### Navigation
- **Sidebar menu**: Quick access to all 8 pages
- **Quick Stats**: Key metrics always visible
- **Project card**: Author and course information

### Feature Engineering Details
- **Summary table**: 6 key engineered features with descriptions
- **Complete list**: Expandable view of all 54 features
  - 30 original features (Time, Amount, V1-V28)
  - 24 engineered features (Temporal, Amount-based, Interactions)

## Installation

**From project root:**

The dashboard is included in the main project. Simply ensure Streamlit is installed:

```bash
conda activate data_science_project
pip list | grep streamlit  # Verify installation
```

**If Streamlit is missing:**
```bash
pip install streamlit --upgrade
```

## How to Run

### Method 1: Python Launcher (Recommended)
```bash
python run_dashboard.py
```

### Method 2: Direct Command
```bash
streamlit run dashboard/dashboard.py
```

### Method 3: From Dashboard Folder
```bash
cd dashboard
streamlit run dashboard.py
```

**Expected behavior:**
1. Terminal displays "You can now view your Streamlit app in your browser"
2. Browser opens automatically to `http://localhost:8501`
3. Dashboard loads with Home page displayed

**To stop:** Press `Ctrl + C` in the terminal where Streamlit is running

## Troubleshooting

### Dashboard won't start

**Issue**: Command not recognized or module errors

**Solution 1** - Verify Streamlit installation:
```bash
conda activate data_science_project
pip list | grep streamlit
```

**Solution 2** - Reinstall Streamlit:
```bash
pip install streamlit --upgrade
```

**Solution 3** - Check working directory:
```bash
pwd                      # Should show project root
ls dashboard/dashboard.py  # Should exist
```

### Port already in use

**Issue**: Error message about port 8501 being occupied

**Solution** - Use alternative port:
```bash
streamlit run dashboard/dashboard.py --server.port 8502
```

### Missing module errors

**Issue**: ImportError for pandas, plotly, or numpy

**Solution** - Install missing packages:
```bash
pip install streamlit pandas plotly numpy
```

### Browser doesn't open automatically

**Issue**: Dashboard runs but browser doesn't launch

**Solution** - Manually open browser and navigate to:
```
http://localhost:8501
```

## VS Code Integration

### Method 1: Integrated Terminal

1. Open VS Code in project root
2. Open integrated terminal: `Ctrl + ù` (Windows) or `Cmd + J` (Mac)
3. Run: `python run_dashboard.py`

### Method 2: F5 Quick Launch

Create `.vscode/launch.json` in project root:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch Dashboard",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "${workspaceFolder}/dashboard/dashboard.py"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

**Then press `F5`** to launch the dashboard instantly.

## Dashboard Structure

```
dashboard/
├── dashboard.py          # Single-file dashboard (~1400 lines)
└── README.md             # This file
```

**The dashboard is self-contained** and includes:
- All performance data (embedded in code)
- All visualization functions
- All page navigation logic
- All CSS styling and configurations

**Total size:** ~60 KB, ~1700 lines of Python code

## Technologies Used

- **Streamlit** (v1.28+): Python web framework for data applications
- **Plotly** (v5.17+): Interactive visualization library
- **Pandas** (v2.0+): Data manipulation and analysis
- **NumPy** (v1.24+): Numerical computing

## Verification Checklist

After launching the dashboard, verify the following:

**Basic Functionality:**
- [ ] Browser opens automatically to `http://localhost:8501`
- [ ] Home page displays with blue italic subtitle
- [ ] Sidebar shows 8 page navigation options
- [ ] Quick Stats section displays key metrics

**Page Content:**
- [ ] Executive Summary shows best model card (Random Forest, F1=0.78)
- [ ] Dataset Overview displays class distribution chart
- [ ] Text "99.83%" is visible (white text, inside green bar)
- [ ] Key features table with 6 rows is present
- [ ] "View All 54 Features" expander works and shows grouped features
- [ ] Models Evaluated page shows pie chart with 3 colors
- [ ] Performance Results displays comparison charts
- [ ] Interactive Comparison page allows selecting 2 different models
- [ ] Comparison page shows confusion matrices in blue and green
- [ ] Radar chart displays with both model overlays

**Visual Elements:**
- [ ] All charts are interactive (zoom, pan, hover)
- [ ] Sidebar has dark background with white text
- [ ] Footer displays "University of Lausanne | HEC Lausanne"

**Expected exploration time:** 10-15 minutes for complete review

## Important Notes

### Dashboard Purpose

This dashboard **visualizes results** generated by the main ML pipeline. It does not train models or process data.

**For training code:** See `/src/main.py`, `/src/models_calibration.py`  
**For evaluation logic:** See `/src/models_application.py`  
**For static charts:** See `/output/` folder (10 PNG files)

### Data Source

All performance metrics, confusion matrices, and model comparisons displayed in the dashboard are derived from the test set evaluation performed by the main pipeline.

### Reproducibility

To regenerate the underlying results shown in the dashboard:
1. Run the complete pipeline: `python main.py`
2. Select option [2] to train models (~15-30 minutes)
3. Models and results will be saved to `saved_models/trained_models.pkl`
4. The dashboard will reflect these results when launched

## Author and Contact

**Dylan Fernandez**  
**Student ID**: [20428967]  
University of Lausanne  
Dylan.Fernandez@unil.ch

**Course:** Advanced Programming for Data Science  
**Date:** December 15, 2025  
**Institution:** HEC Lausanne

---

**For questions about the main project:** See [README.md](../READ/README.md) in the project root
