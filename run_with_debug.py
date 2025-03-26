#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_output.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mental_health_app")

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    try:
        import streamlit
        logger.info("Streamlit is installed")
    except ImportError:
        logger.error("Streamlit is not installed. Please install it with: pip install streamlit")
        return False
    
    try:
        import pandas
        logger.info("Pandas is installed")
    except ImportError:
        logger.warning("Pandas is not installed. Some features may not work properly.")
    
    try:
        import nltk
        logger.info("NLTK is installed")
    except ImportError:
        logger.warning("NLTK is not installed. NLP features will use simple fallbacks.")
    
    try:
        import transformers
        logger.info("Transformers is installed")
        
        # Check for deep learning frameworks
        dl_framework = None
        try:
            import torch
            dl_framework = "PyTorch"
        except ImportError:
            try:
                import tensorflow as tf
                if tf.__version__ >= "2.0.0":
                    dl_framework = "TensorFlow"
            except ImportError:
                try:
                    from jax import numpy as jnp
                    import flax
                    dl_framework = "Flax"
                except ImportError:
                    pass
        
        if dl_framework:
            logger.info(f"Deep learning framework found: {dl_framework}")
        else:
            logger.warning("No deep learning framework found. NLP models will not be available.")
    except ImportError:
        logger.warning("Transformers is not installed. Advanced NLP features will not be available.")
    
    return True

def setup_environment():
    """Set up the environment for running the application."""
    logger.info("Setting up environment...")
    
    # Set environment variables for simple mode
    os.environ['SIMPLE_MODE'] = 'true'
    logger.info("Running in simple mode")
    
    # Create data directories if they don't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        logger.info("Created data directory")
    
    if not os.path.exists("src/data"):
        os.makedirs("src/data")
        logger.info("Created src/data directory")

def main():
    """Main entry point for running the Mental Health Support System with debugging."""
    logger.info("Starting Mental Health Support System with debugging...")
    
    parser = argparse.ArgumentParser(description='Run the Mental Health Support System with debugging')
    parser.add_argument('--init-db', action='store_true', help='Initialize the database with sample data')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database with sample data...")
        try:
            subprocess.run([sys.executable, 'src/init_db.py'], check=True)
            logger.info("Database initialization complete!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Database initialization failed: {e}")
    
    # Run the Streamlit app
    logger.info("Starting the Streamlit app...")
    try:
        subprocess.run(['streamlit', 'run', 'src/app.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
