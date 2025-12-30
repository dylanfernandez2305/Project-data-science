#!/usr/bin/env python3
"""
Quick Launcher for Credit Card Fraud Detection Dashboard

This script provides a simple way to launch the interactive Streamlit dashboard.
It automatically installs required packages (streamlit, plotly) if they are missing.

Usage:
    python run_dashboard.py

The dashboard will automatically open in your default web browser at http://localhost:8501

To stop the dashboard, press Ctrl+C in this terminal.

Author: Dylan Fernandez
Course: Advanced Programming for Data Science
Date: December 15, 2025
Institution: University of Lausanne | HEC Lausanne
"""

# ===== IMPORTS =====
import subprocess
import sys
from pathlib import Path


def check_and_install_package(package_name, import_name=None):
    """
    Check if a package is installed, and install it if missing.
    
    Args:
        package_name: Name of the package to install (e.g., 'streamlit')
        import_name: Name to use for import check (e.g., 'streamlit'). 
                     If None, uses package_name.
    
    Returns:
        bool: True if package is available (or successfully installed)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"⚠️  {package_name} is not installed. Installing now...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, "-q"
            ])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False


def main():
    """Launch the Streamlit dashboard"""
    # Get project root directory
    project_root = Path(__file__).parent
    dashboard_file = project_root / "dashboard" / "dashboard.py"
    
    # Verify dashboard file exists
    if not dashboard_file.exists():
        print("=" * 70)
        print("ERROR: Dashboard file not found!")
        print("=" * 70)
        print(f"\nExpected location: {dashboard_file}")
        print("\nPlease ensure the dashboard/ folder exists with dashboard.py")
        print("=" * 70)
        sys.exit(1)
    
    # Check and install required packages
    print("\n" + "=" * 70)
    print("Checking required packages...")
    print("=" * 70)
    
    required_packages = [
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_and_install_package(package_name, import_name):
            all_installed = False
    
    if not all_installed:
        print("\n" + "=" * 70)
        print("ERROR: Failed to install required packages")
        print("=" * 70)
        print("\nPlease install manually:")
        print("  pip install streamlit plotly")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    print("✅ All required packages are installed")
    
    # Print launch information
    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION - INTERACTIVE DASHBOARD")
    print("=" * 70)
    print("\nLaunching Streamlit dashboard...")
    print("Dashboard will open automatically in your browser")
    print("\nURL: http://localhost:8501")
    print("\nTo stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 70 + "\n")
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Dashboard stopped successfully")
        print("=" * 70 + "\n")
    except FileNotFoundError:
        print("\n" + "=" * 70)
        print("ERROR: Streamlit module not found!")
        print("=" * 70)
        print("\nThis should not happen after installation.")
        print("Please try manually:")
        print("  pip install streamlit plotly")
        print("  python run_dashboard.py")
        print("=" * 70 + "\n")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"ERROR: Failed to launch dashboard")
        print("=" * 70)
        print(f"\nError details: {e}")
        print("\nPlease check:")
        print("  1. Streamlit is installed: pip list | grep streamlit")
        print("  2. Dashboard file exists: ls dashboard/dashboard.py")
        print("  3. Python environment is active")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
