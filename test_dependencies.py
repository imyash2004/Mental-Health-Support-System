#!/usr/bin/env python3
import sys
import os
import importlib
import subprocess

def check_dependency(module_name):
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Check dependencies and create a report."""
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check core dependencies
    dependencies = {
        "streamlit": "Streamlit",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "plotly": "Plotly",
        "nltk": "NLTK",
        "transformers": "Transformers",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "sqlalchemy": "SQLAlchemy"
    }
    
    for module, name in dependencies.items():
        if check_dependency(module):
            print(f"{name} is installed")
        else:
            print(f"{name} is NOT installed")
    
    # Check if key files exist
    files_to_check = [
        "run.py",
        "src/app.py",
        "data"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"{file} exists")
        else:
            print(f"{file} does NOT exist")
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")
    
    # Create src/data directory if it doesn't exist
    if not os.path.exists("src/data"):
        os.makedirs("src/data")
        print("Created src/data directory")

if __name__ == "__main__":
    main()
