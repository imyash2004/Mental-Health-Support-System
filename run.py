#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def main():
    """
    Main entry point for running the Mental Health Support System.
    """
    parser = argparse.ArgumentParser(description='Run the Mental Health Support System')
    parser.add_argument('--init-db', action='store_true', help='Initialize the database with sample data')
    parser.add_argument('--simple', action='store_true', help='Run in simple mode without database or NLP dependencies')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check for required packages
    try:
        import streamlit
    except ImportError:
        print("Streamlit is required to run this application.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Initialize database if requested
    if args.init_db and not args.simple:
        print("Initializing database with sample data...")
        try:
            subprocess.run([sys.executable, 'src/init_db.py'])
            print("Database initialization complete!")
        except Exception as e:
            print(f"Database initialization failed: {e}")
            print("Running in simple mode instead.")
            args.simple = True
    
    # Set environment variables for simple mode
    if args.simple:
        os.environ['SIMPLE_MODE'] = 'true'
        print("Running in simple mode without database or NLP dependencies.")
    
    # Run the Streamlit app
    print("Starting the Mental Health Support System...")
    try:
        subprocess.run(['streamlit', 'run', 'src/app.py'])
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
