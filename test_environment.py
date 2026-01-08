#!/usr/bin/env python3
"""
Test all imports before running fraud detection pipeline.

Usage:
    conda activate data_science_project
    python test_environment.py
    
Expected output:
    ✓ ALL IMPORTS SUCCESSFUL!

If packages are missing, run:
    conda env create -f environment.yml
    conda activate data_science_project
"""

import sys
import os

print("="*60)
print("FRAUD DETECTION - ENVIRONMENT CHECK")
print("="*60)

# =============================================================================
# 1) CHECK PYTHON VERSION
# =============================================================================
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"Python version: {python_version}")

if sys.version_info < (3, 11):
    print("⚠️  WARNING: Python 3.11+ recommended. You have:", python_version)
else:
    print("✓ Python version OK")

# =============================================================================
# 2) CHECK CONDA ENVIRONMENT
# =============================================================================
conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
expected_env = "data_science_project"

print()
if conda_env is None:
    print("⚠️  WARNING: No conda environment detected!")
    print("   Run: conda activate data_science_project")
elif conda_env != expected_env:
    print(f"⚠️  WARNING: Wrong environment active!")
    print(f"   Active:   {conda_env}")
    print(f"   Expected: {expected_env}")
    print(f"   Run: conda activate {expected_env}")
else:
    print(f"✓ Correct environment: {conda_env}")

print()
print("="*60)
print("TESTING CORE PACKAGES")
print("="*60)

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import argparse
from pathlib import Path
from datetime import datetime
print("  ✓ Standard library")

# =============================================================================
# THIRD-PARTY PACKAGES
# =============================================================================
# If any import fails, the error message will tell you which package is missing.
# To install missing packages:
#   conda activate data_science_project
#   conda env update -f environment.yml
# Or individually:
#   pip install <package_name>

failed_imports = []

try:
    import numpy as np
    print(f"  ✓ numpy ({np.__version__})")
except ImportError:
    print("  ✗ numpy - Install: pip install numpy")
    failed_imports.append("numpy")

try:
    import pandas as pd
    print(f"  ✓ pandas ({pd.__version__})")
except ImportError:
    print("  ✗ pandas - Install: pip install pandas")
    failed_imports.append("pandas")

try:
    import scipy
    print(f"  ✓ scipy ({scipy.__version__})")
except ImportError:
    print("  ✗ scipy - Install: pip install scipy")
    failed_imports.append("scipy")

try:
    import sklearn
    print(f"  ✓ scikit-learn ({sklearn.__version__})")
except ImportError:
    print("  ✗ scikit-learn - Install: pip install scikit-learn")
    failed_imports.append("scikit-learn")

print()
print("="*60)
print("TESTING ML PACKAGES")
print("="*60)

try:
    import optuna
    print(f"  ✓ optuna ({optuna.__version__})")
except ImportError:
    print("  ✗ optuna - Install: pip install optuna")
    failed_imports.append("optuna")

try:
    import imblearn
    print(f"  ✓ imbalanced-learn ({imblearn.__version__})")
except ImportError:
    print("  ✗ imbalanced-learn - Install: pip install imbalanced-learn")
    failed_imports.append("imbalanced-learn")

try:
    import joblib
    print(f"  ✓ joblib ({joblib.__version__})")
except ImportError:
    print("  ✗ joblib - Install: pip install joblib")
    failed_imports.append("joblib")

print()
print("="*60)
print("TESTING VISUALIZATION PACKAGES")
print("="*60)

try:
    import matplotlib
    print(f"  ✓ matplotlib ({matplotlib.__version__})")
except ImportError:
    print("  ✗ matplotlib - Install: pip install matplotlib")
    failed_imports.append("matplotlib")

try:
    import seaborn
    print(f"  ✓ seaborn ({seaborn.__version__})")
except ImportError:
    print("  ✗ seaborn - Install: pip install seaborn")
    failed_imports.append("seaborn")

try:
    import plotly
    print(f"  ✓ plotly ({plotly.__version__})")
except ImportError:
    print("  ✗ plotly - Install: pip install plotly")
    failed_imports.append("plotly")

print()
print("="*60)
print("TESTING DASHBOARD & UTILITIES")
print("="*60)

try:
    import streamlit
    print(f"  ✓ streamlit ({streamlit.__version__})")
except ImportError:
    print("  ✗ streamlit - Install: pip install streamlit")
    failed_imports.append("streamlit")

try:
    import tqdm
    print(f"  ✓ tqdm ({tqdm.__version__})")
except ImportError:
    print("  ✗ tqdm - Install: pip install tqdm")
    failed_imports.append("tqdm")

try:
    import kagglehub
    print(f"  ✓ kagglehub")
except ImportError:
    print("  ✗ kagglehub - Install: pip install kagglehub")
    failed_imports.append("kagglehub")

print()
print("="*60)
print("TESTING SKLEARN SUBMODULES")
print("="*60)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, f1_score, precision_recall_curve
    )
    print("  ✓ sklearn submodules")
except ImportError as e:
    print(f"  ✗ sklearn submodules - {e}")
    failed_imports.append("sklearn-submodules")

print()
print("="*60)
print("TESTING PROJECT MODULES")
print("="*60)

# =============================================================================
# PROJECT MODULES (optional - may not exist yet)
# =============================================================================

project_modules_ok = True

try:
    from src.data_loader import load_and_prepare_data, get_optimal_threshold_f1
    print("  ✓ src.data_loader")
except ImportError as e:
    print(f"  ⚠️  src.data_loader - {e}")
    project_modules_ok = False

try:
    from src.models_calibration import calibrate_models, load_models
    print("  ✓ src.models_calibration")
except ImportError as e:
    print(f"  ⚠️  src.models_calibration - {e}")
    project_modules_ok = False

try:
    from src.models_application import apply_models
    print("  ✓ src.models_application")
except ImportError as e:
    print(f"  ⚠️  src.models_application - {e}")
    project_modules_ok = False

try:
    from src.performance_visualization import visualize_performance
    print("  ✓ src.performance_visualization")
except ImportError as e:
    print(f"  ⚠️  src.performance_visualization - {e}")
    project_modules_ok = False

try:
    from src.menu import main, print_menu
    print("  ✓ src.menu")
except ImportError as e:
    print(f"  ⚠️  src.menu - {e}")
    project_modules_ok = False

if not project_modules_ok:
    print()
    print("  Note: Some project modules not available.")
    print("  This is OK if you just cloned the repository.")
    print("  Make sure you're in the project root directory.")

print()
print("="*60)
print("VALIDATION CHECKS")
print("="*60)

# Verify key components if imports succeeded
if not failed_imports:
    # Verify sklearn classes are accessible
    assert callable(LogisticRegression), "LogisticRegression should be callable"
    assert callable(RandomForestClassifier), "RandomForestClassifier should be callable"
    print("  ✓ Key sklearn classes are callable")
    
    # Verify numpy/pandas basics
    test_array = np.array([1, 2, 3])
    test_df = pd.DataFrame({'a': [1, 2, 3]})
    assert len(test_array) == 3, "NumPy array creation should work"
    assert len(test_df) == 3, "Pandas DataFrame creation should work"
    print("  ✓ NumPy and Pandas basics work")

print()
print("="*60)
print("SUMMARY")
print("="*60)

if failed_imports:
    print()
    print("❌ MISSING PACKAGES:")
    for pkg in failed_imports:
        print(f"   - {pkg}")
    print()
    print("To install missing packages:")
    print("   conda activate data_science_project")
    print("   conda env update -f environment.yml")
    print()
    print("Or install individually:")
    print(f"   pip install {' '.join(failed_imports)}")
    print()
    sys.exit(1)
else:
    print()
    print("✓ ALL CORE PACKAGES INSTALLED!")
    print()
    
    if project_modules_ok:
        print("✓ ALL PROJECT MODULES AVAILABLE!")
        print()
        print("You can now run:")
        print("  python main.py                # Interactive menu")
        print("  python run_dashboard.py       # Launch dashboard")
    else:
        print("⚠️  Some project modules not available yet.")
        print("   Make sure you're in the project root directory.")
        print()
        print("Core packages are ready. You can:")
        print("  python main.py                # Try running the menu")
    
    print()
    print("="*60)
    sys.exit(0)
