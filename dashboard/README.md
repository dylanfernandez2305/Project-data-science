# Interactive Dashboard - Credit Card Fraud Detection

**Author:** Dylan Fernandez  
**Course:** Advanced Programming for Data Science  
**Date:** December 15, 2025  
**Institution:** University of Lausanne | HEC Lausanne

---

## 🚀 Quick Start

### From the dashboard folder:

```bash
streamlit run final_Dashboard.py
```

### From the project root:

```bash
streamlit run dashboard/final_Dashboard.py
```

---

## 📦 Prerequisites

Ensure the following packages are installed:

```bash
pip install streamlit pandas plotly numpy
```

Or use the `environment.yml` file from the main project.

---

## 📊 Dashboard Content

### 7 Interactive Pages

#### 1. Home
Project introduction with title and elegant subtitle.

#### 2. Executive Summary
- Best model (Random Forest, F1=0.78)
- Production recommendations
- Limitations and perspectives

#### 3. Dataset Overview
- Dataset statistics (283,726 transactions, 0.17% fraud rate)
- Extreme imbalance visualization
- **Key features table** (6 main features)
- **Expandable view of all 54 features** (30 original + 24 engineered)

#### 4. Methodology
- 3 research questions
- Validation protocol
- Chronological split (64/16/20)

#### 5. Models Evaluated
- 9 models compared (3 paradigms)
- F1-weighted ensemble pie chart
- Model descriptions

#### 6. Performance Results
- F1-score comparison
- Precision-Recall trade-off
- Why ROC-AUC is misleading

#### 7. Technical Details
- Code snippets
- Project structure
- References

---

## 🎨 Features

### Visualizations
- **Interactive Plotly charts**: zoom, pan, hover
- **Pie chart**: Ensemble composition
- **Bar charts**: Model comparison
- **Tables**: Features and results

### Navigation
- **Sidebar**: Page navigation menu
- **Quick Stats**: Key metrics always visible
- **Project card**: Project and author information

### Feature Engineering
- **Table of 6 key features** with type and description
- **Expandable "📋 View All 54 Features"**:
  - 30 original features (Time, Amount, V1-V28)
  - 24 engineered features (Temporal, Amount, Interactions)

---

## 🛑 Stopping the Dashboard

In the terminal where the dashboard is running:
```bash
Ctrl + C
```

---

## 📖 Troubleshooting

### Dashboard won't start

**1. Check Streamlit installation:**
```bash
pip list | grep streamlit
```

**2. Reinstall if necessary:**
```bash
pip install streamlit --upgrade
```

**3. Verify you're in the correct directory:**
```bash
pwd
ls final_Dashboard.py
```

---

### Port already in use

If port 8501 is already in use:
```bash
streamlit run final_Dashboard.py --server.port 8502
```

---

### Missing module error

```bash
pip install streamlit pandas plotly numpy
```

---

## 💻 Using VS Code

### Option 1: Integrated Terminal

1. Open the `dashboard/` folder in VS Code
2. Open `final_Dashboard.py` to view the code
3. Open Terminal: `Ctrl + ù` (or `Cmd + J` on Mac)
4. Type: `streamlit run final_Dashboard.py`

### Option 2: Run Button (F5)

Create `.vscode/launch.json` in the dashboard folder:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Dashboard",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "final_Dashboard.py"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

Then press `F5` to launch.

---

## 📁 Structure

```
dashboard/
├── final_Dashboard.py    ← The dashboard (single file)
└── README.md             ← This file
```

The dashboard is a single file (~50 KB, ~1400 lines) containing:
- All data (performance table)
- All visualization functions
- All navigation logic
- All styles and configurations

---

## 💡 Important Note

This dashboard **visualizes** the results from the main project.

- **For training code**: see `/src/main.py`
- **For models**: see `/src/models_calibration.py`
- **For evaluations**: see `/src/models_application.py`
- **For PNG charts**: see `/output/`

The dashboard is an **interactive** interface that presents results professionally.

---

## 🎓 For Reviewers

### Viewing the code
Open `final_Dashboard.py` in VS Code or any text editor.

### Launching the dashboard
```bash
streamlit run final_Dashboard.py
```

### Exploring
- 7 pages accessible via sidebar
- All visualizations are interactive
- Features are detailed in "Dataset Overview"

### Exploration time
~10-15 minutes to view all pages and interactions.

---

## 📚 Technologies Used

- **Streamlit**: Python web framework for data apps
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

---

## ✅ Verification Checklist

After launching, verify that:

- [ ] Browser opens automatically
- [ ] "Home" page displays with blue subtitle
- [ ] Sidebar contains 7 pages + Quick Stats
- [ ] Class distribution chart in "Dataset Overview" is visible
- [ ] Text "99.83%" is visible (white, inside the bar)
- [ ] Features table is present
- [ ] Expander "📋 View All 54 Features" works
- [ ] Pie chart in "Models Evaluated" displays
- [ ] Footer contains "University of Lausanne | HEC Lausanne"

---

## 🔗 Additional Information

For questions about the main project, see the README in the project root.

---

**Enjoy exploring the dashboard!** 🎓✨
