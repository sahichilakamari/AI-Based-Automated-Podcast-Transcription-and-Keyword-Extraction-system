#!/bin/bash

# Download FFmpeg static build (includes ffprobe)
mkdir -p ffmpeg
cd ffmpeg

# Use a smaller build if available to avoid exceeding HF limits
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extract the contents
tar -xf ffmpeg-release-amd64-static.tar.xz --strip-components=1

# Add it to PATH for runtime
echo "export PATH=$(pwd):\$PATH" >> ~/.bashrc
