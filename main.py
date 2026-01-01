#!/usr/bin/env python3
"""Credit Card Fraud Detection Pipeline - Command-line Entry Point.

This script serves as the main entry point for the fraud detection system.
Following the separation of concerns principle, all business logic is
implemented in specialized modules within the src/ package:
    - src/menu.py: Interactive CLI
    - src/data_loader.py: Data preparation
    - src/models_calibration.py: Model training
    - src/models_application.py: Model evaluation
    - src/performance_visualization.py: Results visualization

Usage:
    python main.py              # Launch interactive menu
    python main.py --debug      # Enable debug tracebacks in fallback mode

Project Structure:
    main.py                     # This file (minimal entry point)
    src/                        # Core modules
    data/                       # Dataset storage
    saved_models/               # Trained models
    output/                     # Generated visualizations

Author: Dylan Fernandez
Course: Advanced Programming 2025, HEC Lausanne
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Execute the fraud detection pipeline.
    
    Attempts to launch the interactive menu system. If unavailable,
    falls back to a basic data preparation sanity check.
    
    Returns
    -------
    int
        Exit code: 0 (success), 1 (failure), 130 (user interrupt)
    """
    # Environment Setup
    # Ensure project root is in sys.path for absolute imports
    # Allows running from any directory (e.g., for tests)
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Interactive Menu Launch
    # Attempt to import and run the interactive menu
    try:
        from src.menu import main as menu_main
    except (ModuleNotFoundError, ImportError) as exc:
        # Menu unavailable - provide helpful guidance
        print(
            "╔════════════════════════════════════════════════════════════════╗\n"
            "║  Could not import the interactive menu system                 ║\n"
            "╚════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "Common causes:\n"
            "  • Missing dependencies (run: conda env create -f environment.yml)\n"
            "  • Environment not activated (run: conda activate data_science_project)\n"
            "  • Missing src/menu.py file\n"
            f"\nTechnical details: {exc}\n"
            "\nRunning fallback data preparation check...\n"
        )
        return _run_fallback_check()
    
    # Menu imported successfully - execute it
    try:
        menu_main()
        return 0
    except KeyboardInterrupt:
        print("\n\n" + "─" * 70)
        print("Operation cancelled by user (Ctrl+C)")
        print("─" * 70)
        return 130


def _run_fallback_check() -> int:
    """Run basic data preparation check when menu unavailable.
    
    Returns
    -------
    int
        Exit code: 0 (success), 1 (failure)
    """
    # Import Data Loader Module
    try:
        from src.data_loader import load_and_prepare_data
    except ImportError as exc:
        print(
            "╔════════════════════════════════════════════════════════════════╗\n"
            "║  Error: Cannot import data_loader module                      ║\n"
            "╚════════════════════════════════════════════════════════════════╝\n"
            f"\nTechnical details: {exc}\n"
            "\nPlease verify:\n"
            "  • You are in the project root directory\n"
            "  • The src/ folder exists with all required files\n"
            "  • Dependencies are installed (see environment.yml)\n"
        )
        return 1
    
    # Load and Prepare Dataset
    try:
        data = load_and_prepare_data()
    except FileNotFoundError as exc:
        print(
            "╔════════════════════════════════════════════════════════════════╗\n"
            "║  Error: Credit card dataset not found                        ║\n"
            "╚════════════════════════════════════════════════════════════════╝\n"
            f"\nTechnical details: {exc}\n"
            "\nTo obtain the dataset:\n"
            "  1. Install kagglehub: pip install kagglehub\n"
            "  2. Or download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  3. Place creditcard.csv in data/ folder\n"
        )
        return 1
    except Exception as exc:
        print(
            "╔════════════════════════════════════════════════════════════════╗\n"
            "║  Error during data preparation                                ║\n"
            "╚════════════════════════════════════════════════════════════════╝\n"
        )
        
        # Show full traceback in debug mode
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        else:
            print(f"\nAn unexpected error occurred: {exc}\n")
        
        print(
            "Possible causes:\n"
            "  • Corrupted dataset file\n"
            "  • Incompatible package versions\n"
            "  • Insufficient memory\n"
            "\nTry: python main.py --debug (for full traceback)\n"
        )
        return 1
    
   # Display Success Summary
    try:
        print("\n" + "═" * 70)
        print("DATA PREPARATION SANITY CHECK - SUCCESS ✓")
        print("═" * 70)
        print(f"\n  Training set:    {int(data['y_train'].sum()):>5} frauds / {len(data['y_train']):>6} total")
        print(f"  Validation set:  {int(data['y_val'].sum()):>5} frauds / {len(data['y_val']):>6} total")
        print(f"  Test set:        {int(data['y_test'].sum()):>5} frauds / {len(data['y_test']):>6} total")
        print(f"\n  Overall fraud ratio: {data['fraud_ratio']:.4%}")
        print(f"  Total features:      {len(data['feature_names'])}")
        print("\n" + "═" * 70)
        print("\nEnvironment verified. Run 'python main.py' to launch the menu.")
    except KeyError as exc:
        print(f"\n[WARNING] Data bundle missing key: {exc}")
        print("Data preparation succeeded, but summary could not be displayed.\n")
    
    return 0


# Script Entry Point
if __name__ == "__main__":
    sys.exit(main())
