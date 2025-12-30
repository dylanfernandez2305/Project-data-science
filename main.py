#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Main Entry Point

This is the main entry point required for project submission.
Launches the interactive menu system for full pipeline functionality.

Author: Dylan Fernandez
Date: December 30, 2025
Course: Advanced Programming for Data Science
Institution: University of Lausanne | HEC Lausanne
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Main entry point - launches the interactive menu system
    
    The menu provides options for:
    - Data loading and preprocessing
    - Model training with hyperparameter optimization
    - Model evaluation on test set
    - Visualization generation
    """
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 70)
    print("\nLaunching interactive menu system...")
    print("\nFor direct module execution, see README.md")
    print("For dashboard visualization: python run_dashboard.py")
    print("=" * 70 + "\n")
    
    # Save original directory and change to src/
    original_dir = os.getcwd()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    
    try:
        # Change to src directory so menu.py can find all modules
        os.chdir(src_dir)
        
        # Import and run the menu
        from menu import main as menu_main
        menu_main()
        
    except ImportError as e:
        print(f"Error: Could not import menu module: {e}")
        print("\nPlease ensure you are running from the project root directory.")
        print("Expected structure: main.py at root, src/menu.py exists")
        os.chdir(original_dir)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        os.chdir(original_dir)
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        os.chdir(original_dir)
        sys.exit(1)
    finally:
        # Always restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
