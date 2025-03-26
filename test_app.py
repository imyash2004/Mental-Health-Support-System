import os
import sys

def main():
    """
    Test script to diagnose issues with the main application.
    """
    # Open a file to write the output
    with open("test_output.txt", "w") as f:
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Current working directory: {os.getcwd()}\n")
        
        # Check for required packages
        try:
            import streamlit
            f.write(f"Streamlit is installed: {streamlit.__version__}\n")
        except ImportError:
            f.write("Streamlit is NOT installed\n")
        
        try:
            import pandas
            f.write(f"Pandas is installed: {pandas.__version__}\n")
        except ImportError:
            f.write("Pandas is NOT installed\n")
        
        try:
            import nltk
            f.write(f"NLTK is installed: {nltk.__version__}\n")
        except ImportError:
            f.write("NLTK is NOT installed\n")
        
        try:
            import transformers
            f.write(f"Transformers is installed: {transformers.__version__}\n")
        except ImportError:
            f.write("Transformers is NOT installed\n")
        
        try:
            import torch
            f.write(f"PyTorch is installed: {torch.__version__}\n")
        except ImportError:
            f.write("PyTorch is NOT installed\n")
        
        # Check if the run.py file exists
        if os.path.exists("run.py"):
            f.write("run.py exists\n")
        else:
            f.write("run.py does NOT exist\n")
        
        # Check if the app.py file exists
        if os.path.exists("src/app.py"):
            f.write("src/app.py exists\n")
        else:
            f.write("src/app.py does NOT exist\n")
        
        # Check if the data directory exists
        if os.path.exists("data"):
            f.write("data directory exists\n")
        else:
            f.write("data directory does NOT exist\n")
    
    print("Test completed. Results written to test_output.txt")

if __name__ == "__main__":
    main()
