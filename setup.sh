#!/bin/bash

# Create required directories
mkdir -p uploads temp outputs model_cache

# Install FFmpeg (required by pydub)
apt-get update && apt-get install -y ffmpeg

# Set proper permissions
chmod -R 777 uploads temp outputs model_cache