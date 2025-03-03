#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv soln_architect
source soln_architect/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting Cloud Architecture Recommendation System..."
streamlit run app.py