#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)
    print(f"Python version: {sys.version.split()[0]} ✓")

def install_dependencies(simple_mode=False):
    """Install required dependencies."""
    print("Installing dependencies...")
    
    if simple_mode:
        # Install only core dependencies for simple mode
        dependencies = [
            "streamlit>=1.0.0",
            "pandas>=1.0.0",
            "numpy>=1.18.0",
            "matplotlib>=3.0.0",
            "plotly>=4.0.0",
            "sqlalchemy>=1.4.0",
            "python-dotenv>=0.19.0"
        ]
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
            print("Core dependencies installed successfully ✓")
        except subprocess.CalledProcessError:
            print("Error: Failed to install core dependencies.")
            sys.exit(1)
    else:
        # Install all dependencies from requirements.txt
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("All dependencies installed successfully ✓")
        except subprocess.CalledProcessError:
            print("Error: Failed to install dependencies from requirements.txt.")
            print("You can try running with --simple flag to install only core dependencies.")
            sys.exit(1)

def setup_nltk_data():
    """Download required NLTK data."""
    if not os.environ.get('SIMPLE_MODE'):
        print("Downloading NLTK data...")
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            print("NLTK data downloaded successfully ✓")
        except Exception as e:
            print(f"Warning: Failed to download NLTK data: {e}")
            print("Some NLP features may not work properly.")

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    os.makedirs("data", exist_ok=True)
    print("Directories created ✓")

def initialize_database():
    """Initialize the database with sample data."""
    print("Initializing database with sample data...")
    try:
        subprocess.check_call([sys.executable, "src/init_db.py"])
        print("Database initialized successfully ✓")
    except subprocess.CalledProcessError:
        print("Error: Failed to initialize database.")
        sys.exit(1)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup the Mental Health Support System')
    parser.add_argument('--simple', action='store_true', help='Install only core dependencies for simple mode')
    parser.add_argument('--no-db', action='store_true', help='Skip database initialization')
    args = parser.parse_args()
    
    print("Setting up the Mental Health Support System...")
    
    # Set environment variable for simple mode
    if args.simple:
        os.environ['SIMPLE_MODE'] = 'true'
        print("Running in simple mode.")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies(args.simple)
    
    # Setup NLTK data
    if not args.simple:
        setup_nltk_data()
    
    # Create directories
    create_directories()
    
    # Initialize database
    if not args.no_db and not args.simple:
        initialize_database()
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    if args.simple:
        print("  python run.py --simple")
    else:
        print("  python run.py")
    
    print("\nFor more information, please refer to the README.md file.")

if __name__ == "__main__":
    main()
