#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Create required directories
mkdir -p uploads temp outputs model_cache
chmod -R 777 uploads temp outputs model_cache

# Install Python dependencies with exact versions
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir
