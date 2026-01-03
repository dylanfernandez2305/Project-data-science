# Interactive Dashboard

This folder contains a Streamlit dashboard (1,573 lines) for interactive exploration of fraud detection results.

---

## Quick Start

**From project root:**

```bash
# Recommended (launcher with dependency check)
python run_dashboard.py

# Alternative (direct)
streamlit run dashboard/dashboard.py
```

The dashboard opens automatically at: `http://localhost:8501`  
Stop it with: `Ctrl + C`

---

## Requirements

**If using the main project environment:**

```bash
conda env create -f environment.yml
conda activate data_science_project
```

**OR if using pip only:**

```bash
pip install streamlit plotly pandas numpy
```

## First-Time Setup

When running Streamlit for the first time, you'll be prompted for an email address. 
This is optional—simply press `Enter` to skip and proceed to the dashboard.

---

## Dashboard Features

**8 interactive pages:**
- Home
- Executive Summary
- Dataset Overview
- Methodology
- Models evaluated
- Performance Results
- Interactive Comparison
- Technical details

**Interactive visualizations** powered by Plotly.

**Model Comparison:**
- Select multiple models from the sidebar to compare performance
- Toggle between fraud-class (Class 1) and macro-averaged metrics
- Interactive hover tooltips show exact values
- All metrics reflect validation-calibrated thresholds on test data

---

## Notes

- Dashboard is for **visualization only** (does not train models)
- Launcher (`run_dashboard.py`) checks dependencies and offers pip fallback
- Conda environment recommended for reproducibility

---

## Author & Contact

**Dylan Fernandez** (20428967)  
University of Lausanne (HEC Lausanne)  
Dylan.Fernandez@unil.ch
