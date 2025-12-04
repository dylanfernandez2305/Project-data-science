#!/usr/bin/env python3
"""
Interactive Terminal Menu for Fraud Detection Pipeline
This script provides a central command center to run all project components
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\033[35m'
    ORANGE = '\033[38;5;214m'


def print_header():
    """Print the application header"""
    print("\n" + "="*70)
    print(Colors.BOLD + Colors.CYAN +
          "           FRAUD DETECTION PIPELINE - CONTROL CENTER" +
          Colors.ENDC)
    print("="*70 + "\n")


def print_menu():
    """Display the main menu"""
    print(Colors.BOLD + "Main Menu:" + Colors.ENDC)
    print()
    print(Colors.BOLD + "  Individual Steps:" + Colors.ENDC)
    print(Colors.CYAN + "  [1]" + Colors.ENDC + " Data Preparation Only")
    print(Colors.CYAN + "  [2]" + Colors.ENDC + " Train & Save Models (no need if trained_models.pkl)")
    print(Colors.CYAN + "  [3]" + Colors.ENDC + " Evaluate Models (requires saved models)")
    print(Colors.CYAN + "  [4]" + Colors.ENDC + " Show Visualizations")
    print()
    print(Colors.BOLD + "  Utilities:" + Colors.ENDC)
    print(Colors.CYAN + "  [5]" + Colors.ENDC + " Check Environment & Dependencies")
    print(Colors.CYAN + "  [6]" + Colors.ENDC + " Check Saved Models Status")
    print(Colors.CYAN + "  [7]" + Colors.ENDC + " View Project Information")
    print()
    print(Colors.RED + "  [0]" + Colors.ENDC + " Exit")
    print()


def run_complete_pipeline():
    """Run the complete fraud detection pipeline"""
    print("\n" + Colors.CYAN + "="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70 + Colors.ENDC)
    print("\nThis will run all 4 steps:")
    print("  1. Data Preparation")
    print("  2. Model Calibration")
    print("  3. Model Application")
    print("  4. Visualizations")

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import calibrate_models, save_models
        from models_application import apply_models
        from performance_visualization import visualize_performance

        print("\n" + Colors.GREEN + "Starting pipeline..." + Colors.ENDC + "\n")

        # Step 1: Data Preparation
        print(Colors.CYAN + "Step 1/4: Data Preparation" + Colors.ENDC)
        data = load_and_prepare_data()

        # Step 2: Model Training
        print("\n" + Colors.CYAN + "Step 2/4: Model Training" + Colors.ENDC)
        models = calibrate_models(data)
        save_models(models)

        # Step 3: Model Evaluation
        print("\n" + Colors.CYAN + "Step 3/4: Model Evaluation" + Colors.ENDC)
        results = apply_models(data, models)

        # Step 4: Visualizations
        print("\n" + Colors.CYAN + "Step 4/4: Creating Visualizations" + Colors.ENDC)
        visualize_performance(data, results, save_plots=True, show_plots=False)

        print("\n" + Colors.GREEN + "[SUCCESS] Pipeline completed successfully!" + Colors.ENDC)

    except KeyboardInterrupt:
        print("\n" + Colors.YELLOW + "[WARNING] Pipeline interrupted by user" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def run_data_preparation():
    """Run data preparation only"""
    print("\n" + Colors.CYAN + "="*70)
    print("DATA PREPARATION")
    print("="*70 + Colors.ENDC)

    try:
        from main import load_and_prepare_data
        print("\n" + Colors.GREEN + "Loading and preparing data..." + Colors.ENDC + "\n")
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "[SUCCESS] Data preparation complete!" + Colors.ENDC)
        print(f"\nData dictionary contains {len(data)} variables:")
        for key in data.keys():
            print(f"  - {key}")

    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
        print("\n" + Colors.YELLOW + "Fix NumPy: pip install 'numpy<2' --force-reinstall" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def run_pipeline_with_saved_models():
    """Run pipeline using pre-trained models from file"""
    print("\n" + Colors.CYAN + "="*70)
    print("RUN PIPELINE WITH SAVED MODELS (FAST MODE)")
    print("="*70 + Colors.ENDC)
    print("\n" + Colors.GREEN + "This uses pre-trained models - much faster!" + Colors.ENDC)
    print("Skips model training (saves 10-30 minutes)")

    # Check if models file exists
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    models_file = project_root / "saved_models" / "trained_models.pkl"

    if not models_file.exists():
        print("\n" + Colors.RED + "[ERROR] No saved models found!" + Colors.ENDC)
        print("\nYou need to train models first using option [4]")
        input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)
        return

    print(f"\n[OK] Found saved models: {models_file.name}")
    print(f"  File size: {models_file.stat().st_size / (1024*1024):.2f} MB")

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import load_models
        from models_application import apply_models
        from performance_visualization import visualize_performance

        print("\n" + Colors.GREEN + "Step 1/4: Loading data..." + Colors.ENDC)
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "Step 2/4: Loading pre-trained models..." + Colors.ENDC)
        models = load_models()

        print("\n" + Colors.GREEN + "Step 3/4: Evaluating models..." + Colors.ENDC)
        results = apply_models(data, models)

        print("\n" + Colors.GREEN + "Step 4/4: Creating visualizations..." + Colors.ENDC)
        output_path = visualize_performance(data, results, save_plots=True, show_plots=False)

        print("\n" + Colors.GREEN + "[SUCCESS] Pipeline completed successfully using saved models!" + Colors.ENDC)
        print("\nSummary:")
        print(results['summary_df'].to_string(index=False))

    except FileNotFoundError as e:
        print(Colors.RED + f"\n[ERROR] {e}" + Colors.ENDC)
    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def train_and_save_models():
    """Train models and save them to file"""
    print("\n" + Colors.CYAN + "="*70)
    print("TRAIN & SAVE MODELS")
    print("="*70 + Colors.ENDC)
    print("\n" + Colors.YELLOW + "[WARNING] WARNING: This step takes 10-30 minutes!" + Colors.ENDC)
    print("\nThis will:")
    print("  1. Train all 8 models")
    print("  2. Save them to 'trained_models.pkl'")
    print("  3. Allow you to skip training in future runs")

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import calibrate_models, save_models

        print("\n" + Colors.GREEN + "Step 1/3: Loading data..." + Colors.ENDC)
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "Step 2/3: Training models (this takes time)..." + Colors.ENDC)
        models = calibrate_models(data)

        print("\n" + Colors.GREEN + "Step 3/3: Saving models to file..." + Colors.ENDC)
        save_models(models)

        print("\n" + Colors.GREEN + f"[SUCCESS] Training complete! {len(models)} models saved." + Colors.ENDC)
        print("\n" + Colors.CYAN + "Next time, use option [2] to skip training!" + Colors.ENDC)

    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def evaluate_saved_models():
    """Evaluate models loaded from file"""
    print("\n" + Colors.CYAN + "="*70)
    print("EVALUATE MODELS (Using Saved Models)")
    print("="*70 + Colors.ENDC)

    # Check if models file exists
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    models_file = project_root / "saved_models" / "trained_models.pkl"

    if not models_file.exists():
        print("\n" + Colors.RED + "[ERROR] No saved models found!" + Colors.ENDC)
        print("\nYou need to train models first using option [4]")
        input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)
        return

    print(f"\n[OK] Found saved models: {models_file.name}")

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import load_models
        from models_application import apply_models

        print("\n" + Colors.GREEN + "Step 1/3: Loading data..." + Colors.ENDC)
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "Step 2/3: Loading pre-trained models..." + Colors.ENDC)
        models = load_models()

        print("\n" + Colors.GREEN + "Step 3/3: Evaluating models..." + Colors.ENDC)
        results = apply_models(data, models)

        print("\n" + Colors.GREEN + "[SUCCESS] Evaluation complete!" + Colors.ENDC)
        print("\nSummary:")
        print(results['summary_df'].to_string(index=False))

    except FileNotFoundError as e:
        print(Colors.RED + f"\n[ERROR] {e}" + Colors.ENDC)
    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def run_model_application():
    """Run model application and evaluation"""
    print("\n" + Colors.CYAN + "="*70)
    print("MODEL APPLICATION (EVALUATION)")
    print("="*70 + Colors.ENDC)
    print("\n" + Colors.YELLOW + "Note: This requires data preparation and model training." + Colors.ENDC)

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import calibrate_models
        from models_application import apply_models

        print("\n" + Colors.GREEN + "Step 1/3: Loading data..." + Colors.ENDC)
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "Step 2/3: Training models..." + Colors.ENDC)
        models = calibrate_models(data)

        print("\n" + Colors.GREEN + "Step 3/3: Evaluating models..." + Colors.ENDC)
        results = apply_models(data, models)

        print("\n" + Colors.GREEN + "[SUCCESS] Evaluation complete!" + Colors.ENDC)
        print("\nSummary:")
        print(results['summary_df'].to_string(index=False))

    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def run_visualizations():
    """Run visualizations only"""
    print("\n" + Colors.CYAN + "="*70)
    print("PERFORMANCE VISUALIZATIONS")
    print("="*70 + Colors.ENDC)

    # Check if models file exists
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    models_file = project_root / "saved_models" / "trained_models.pkl"

    if not models_file.exists():
        print("\n" + Colors.RED + "[ERROR] No saved models found!" + Colors.ENDC)
        print("\nYou need to train models first using option [4]")
        input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)
        return

    print(f"\n[OK] Found saved models: {models_file.name}")
    print("\n" + Colors.GREEN + "This will load pre-trained models and generate visualizations." + Colors.ENDC)

    confirm = input("\n" + Colors.YELLOW + "Continue? (y/n): " + Colors.ENDC).lower()
    if confirm != 'y':
        print(Colors.RED + "Cancelled." + Colors.ENDC)
        return

    try:
        from main import load_and_prepare_data
        from models_calibration import load_models
        from models_application import apply_models
        from performance_visualization import visualize_performance

        print("\n" + Colors.GREEN + "Step 1/4: Loading data..." + Colors.ENDC)
        data = load_and_prepare_data()

        print("\n" + Colors.GREEN + "Step 2/4: Loading pre-trained models..." + Colors.ENDC)
        models = load_models()

        print("\n" + Colors.GREEN + "Step 3/4: Evaluating models..." + Colors.ENDC)
        results = apply_models(data, models)

        print("\n" + Colors.GREEN + "Step 4/4: Creating visualizations..." + Colors.ENDC)
        output_path = visualize_performance(data, results, save_plots=True, show_plots=False)

        print("\n" + Colors.GREEN + "[SUCCESS] Visualizations complete!" + Colors.ENDC)

    except FileNotFoundError as e:
        print(Colors.RED + f"\n[ERROR] {e}" + Colors.ENDC)
    except ImportError as e:
        print(Colors.RED + f"\n[ERROR] Import Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"\n[ERROR] Error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def check_saved_models_status():
    """Check status of saved models"""
    print("\n" + Colors.CYAN + "="*70)
    print("SAVED MODELS STATUS")
    print("="*70 + Colors.ENDC + "\n")

    from pathlib import Path
    project_root = Path(__file__).parent.parent
    models_file = project_root / "saved_models" / "trained_models.pkl"

    if models_file.exists():
        size_mb = models_file.stat().st_size / (1024 * 1024)
        from datetime import datetime
        modified_time = datetime.fromtimestamp(models_file.stat().st_mtime)

        print(Colors.GREEN + "[OK] Saved models found!" + Colors.ENDC)
        print(f"\nFile: {models_file.name}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Try to load and show details
        try:
            from models_calibration import load_models
            print("\nAttempting to load models...")
            models = load_models()
            print(Colors.GREEN + "\n[OK] Models loaded successfully!" + Colors.ENDC)
            print(f"Total models: {len(models)}")
            print("\nAvailable models:")
            for key in sorted(models.keys()):
                if 'best_' in key:
                    print(f"  - {key}")
        except Exception as e:
            print(Colors.YELLOW + f"\n[WARNING] Could not load models: {e}" + Colors.ENDC)

        print("\n" + Colors.CYAN + "You can use option [2] to run pipeline with these models!" + Colors.ENDC)

    else:
        print(Colors.YELLOW + "[WARNING] No saved models found" + Colors.ENDC)
        print(f"\nExpected location: {models_file}")
        print("\nTo create saved models:")
        print("  • Use option [4] - Train & Save Models")
        print("  • Or use option [1] - Complete Pipeline (auto-saves)")

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def check_environment():
    """Check environment and dependencies"""
    print("\n" + Colors.CYAN + "="*70)
    print("ENVIRONMENT & DEPENDENCIES CHECK")
    print("="*70 + Colors.ENDC + "\n")

    # Check Python version
    print(Colors.BOLD + "Python Version:" + Colors.ENDC)
    print(f"  {sys.version}\n")

    # Check packages
    packages = [
        ('numpy', 'NumPy', 'numpy<2'),
        ('pandas', 'Pandas', 'pandas'),
        ('sklearn', 'Scikit-learn', 'scikit-learn'),
        ('imblearn', 'Imbalanced-learn', 'imbalanced-learn'),
        ('matplotlib', 'Matplotlib', 'matplotlib'),
        ('seaborn', 'Seaborn', 'seaborn'),
        ('scipy', 'SciPy', 'scipy'),
    ]

    missing_packages = []
    numpy_needs_downgrade = False

    print(Colors.BOLD + "Required Packages:" + Colors.ENDC)
    for module_name, display_name, pip_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')

            # Special check for NumPy
            if module_name == 'numpy' and version.startswith('2.'):
                print(f"  {display_name:20} {version:15} " +
                      Colors.RED + "[WARNING] Version 2.x (should be <2.0)" + Colors.ENDC)
                numpy_needs_downgrade = True
            else:
                print(f"  {display_name:20} {version:15} " + Colors.GREEN + "[OK]" + Colors.ENDC)
        except ImportError:
            print(f"  {display_name:20} " + Colors.RED + "[NOT INSTALLED]" + Colors.ENDC)
            missing_packages.append((display_name, pip_name))

    # Optional packages
    print("\n" + Colors.BOLD + "Optional Packages:" + Colors.ENDC)
    optional = [('kagglehub', 'KaggleHub'), ('shap', 'SHAP')]
    for module_name, display_name in optional:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {display_name:20} {version:15} " + Colors.GREEN + "[OK]" + Colors.ENDC)
        except ImportError:
            print(f"  {display_name:20} " + Colors.YELLOW + "Not installed (optional)" + Colors.ENDC)

    # Check for dataset
    print("\n" + Colors.BOLD + "Dataset:" + Colors.ENDC)
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "creditcard.csv"
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  creditcard.csv       {size_mb:.1f} MB        " + Colors.GREEN + "[OK] Found" + Colors.ENDC)
    else:
        print(f"  creditcard.csv       " + Colors.RED + "[NOT FOUND]" + Colors.ENDC)

    # Check project files
    print("\n" + Colors.BOLD + "Project Files:" + Colors.ENDC)
    files = [
        'main.py',
        'models_calibration.py',
        'models_application.py',
        'performance_visualization.py',
        'menu.py'
    ]

    for file in files:
        if Path(file).exists():
            print(f"  {file:30} " + Colors.GREEN + "[OK]" + Colors.ENDC)
        else:
            print(f"  {file:30} " + Colors.RED + "[MISSING]" + Colors.ENDC)

    # Recommendations and auto-install
    print("\n" + Colors.BOLD + "Recommendations:" + Colors.ENDC)

    try:
        import numpy
        if numpy.__version__.startswith('2.'):
            print(Colors.YELLOW + "  [WARNING] NumPy version is 2.x - please downgrade:" + Colors.ENDC)
            print("    pip install 'numpy<2' --force-reinstall")
            print("    pip install pandas scikit-learn imbalanced-learn --force-reinstall")
    except:
        pass

    if not dataset_path.exists():
        print(Colors.YELLOW + "  [WARNING] Dataset not found - download it or install kagglehub" + Colors.ENDC)

    # Auto-install missing packages
    if missing_packages or numpy_needs_downgrade:
        print("\n" + Colors.CYAN + "="*70 + Colors.ENDC)
        if missing_packages:
            print(Colors.YELLOW + f"\n{len(missing_packages)} package(s) need to be installed." + Colors.ENDC)
        if numpy_needs_downgrade:
            print(Colors.YELLOW + "NumPy needs to be downgraded to version <2.0" + Colors.ENDC)

        install = input("\n" + Colors.BOLD + "Would you like to install/fix them now? (y/n): " + Colors.ENDC).lower()

        if install == 'y':
            print("\n" + Colors.GREEN + "Installing packages..." + Colors.ENDC + "\n")

            try:
                # Fix NumPy first if needed
                if numpy_needs_downgrade:
                    print(Colors.CYAN + "Downgrading NumPy..." + Colors.ENDC)
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2', '--force-reinstall'])
                    print(Colors.GREEN + "[OK] NumPy downgraded successfully\n" + Colors.ENDC)

                # Install missing packages
                for display_name, pip_name in missing_packages:
                    print(Colors.CYAN + f"Installing {display_name}..." + Colors.ENDC)
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
                    print(Colors.GREEN + f"[OK] {display_name} installed successfully\n" + Colors.ENDC)

                print(Colors.GREEN + "\n[SUCCESS] All packages installed successfully!" + Colors.ENDC)
                print(Colors.YELLOW + "\nNote: You may need to restart the menu for changes to take effect." + Colors.ENDC)

            except subprocess.CalledProcessError as e:
                print(Colors.RED + f"\n[ERROR] Installation failed: {e}" + Colors.ENDC)
            except Exception as e:
                print(Colors.RED + f"\n[ERROR] Unexpected error: {e}" + Colors.ENDC)

    input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def view_project_info():
    """Display project information"""
    print("\n" + Colors.CYAN + "="*70)
    print("PROJECT INFORMATION")
    print("="*70 + Colors.ENDC + "\n")

    info = """
PROJECT: Fraud Detection Pipeline
DESCRIPTION: Complete machine learning pipeline for credit card fraud detection

MODULES:
  • main.py                     - Data loading and preprocessing
  • models_calibration.py       - Model training (8 models)
  • models_application.py       - Prediction and evaluation
  • performance_visualization.py - Performance charts
  • run_pipeline.py             - Master orchestrator
  • menu.py                     - This interactive menu

MODELS TRAINED:
  Supervised:
    • Logistic Regression
    • Random Forest

  Unsupervised:
    • Isolation Forest
    • Local Outlier Factor (LOF)
    • Gaussian Mixture Model (GMM)

  Semi-Supervised:
    • Isolation Forest (trained on normal data)
    • LOF (trained on normal data)
    • GMM (trained on normal data)

WORKFLOW:
  1. Data Preparation    → Feature engineering, SMOTE, scaling
  2. Model Calibration   → GridSearchCV with TimeSeriesSplit
  3. Model Application   → Predictions with threshold optimization
  4. Visualization       → Confusion matrices, ROC curves, metrics

DOCUMENTATION:
  • RESTRUCTURED_README.md    - Complete usage guide
  • VERIFICATION_REPORT.md    - Code verification results
  • RESTRUCTURE_SUMMARY.md    - Summary of changes
  • requirements.txt          - Python dependencies

For detailed information, see RESTRUCTURED_README.md
"""

    print(info)
    input(Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    """Main menu loop"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input(Colors.BOLD + "Enter your choice [0-7]: " + Colors.ENDC).strip()

        if choice == '1':
            run_data_preparation()
        elif choice == '2':
            train_and_save_models()
        elif choice == '3':
            evaluate_saved_models()
        elif choice == '4':
            run_visualizations()
        elif choice == '5':
            check_environment()
        elif choice == '6':
            check_saved_models_status()
        elif choice == '7':
            view_project_info()
        elif choice == '0':
            print("\n" + Colors.GREEN + "Thank you for using the Fraud Detection Pipeline!" + Colors.ENDC)
            print("Goodbye!\n")
            sys.exit(0)
        else:
            print(Colors.RED + "\n✗ Invalid choice. Please select 0-7." + Colors.ENDC)
            input("\n" + Colors.BLUE + "Press Enter to continue..." + Colors.ENDC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + Colors.YELLOW + "Menu interrupted by user. Goodbye!" + Colors.ENDC + "\n")
        sys.exit(0)
    except Exception as e:
        print("\n" + Colors.RED + f"Fatal error: {e}" + Colors.ENDC)
        import traceback
        traceback.print_exc()
        sys.exit(1)
