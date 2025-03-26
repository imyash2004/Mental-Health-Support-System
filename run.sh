#!/bin/bash

# Run script for Mental Health Support System

# Display help message
show_help() {
    echo "Mental Health Support System Runner"
    echo ""
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -s, --setup      Run the setup script before starting the application"
    echo "  -i, --init-db    Initialize the database with sample data"
    echo "  -m, --simple     Run in simple mode without NLP dependencies"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                   Run the application normally"
    echo "  ./run.sh --setup           Run setup and then start the application"
    echo "  ./run.sh --simple          Run in simple mode"
    echo "  ./run.sh --setup --simple  Run setup in simple mode and start the application"
    echo ""
}

# Parse command line arguments
SETUP=false
INIT_DB=false
SIMPLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--setup)
            SETUP=true
            shift
            ;;
        -i|--init-db)
            INIT_DB=true
            shift
            ;;
        -m|--simple)
            SIMPLE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make sure the script is executable
chmod +x run.py
if [ -f "setup.py" ]; then
    chmod +x setup.py
fi

# Run setup if requested
if [ "$SETUP" = true ]; then
    echo "Running setup..."
    if [ "$SIMPLE" = true ]; then
        python setup.py --simple
    else
        python setup.py
    fi
    
    # Check if setup was successful
    if [ $? -ne 0 ]; then
        echo "Setup failed. Please check the error messages above."
        exit 1
    fi
fi

# Run the application
echo "Starting the Mental Health Support System..."
if [ "$SIMPLE" = true ]; then
    python run.py --simple
elif [ "$INIT_DB" = true ]; then
    python run.py --init-db
else
    python run.py
fi
