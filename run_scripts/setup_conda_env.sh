#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment from yml file
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "To activate the environment, run:"
echo "conda activate carlisle-env"

# Instructions for running the code
echo ""
echo "After activating the environment, you can run your code with:"
echo "python main.py [your arguments]"
echo ""
echo "The environment is configured with GDAL environment variables preset."
