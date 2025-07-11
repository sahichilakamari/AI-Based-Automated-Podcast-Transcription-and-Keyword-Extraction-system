#!/bin/bash

echo "🔧 Setting up FFmpeg and FFprobe..."

# Set working directory
mkdir -p ffmpeg
cd ffmpeg

# Download small static build
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extract only ffmpeg and ffprobe
tar -xf ffmpeg-release-amd64-static.tar.xz --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe'

# Make binaries executable
chmod +x ffmpeg ffprobe

# Cleanup archive
rm -f ffmpeg-release-amd64-static.tar.xz

echo "✅ FFmpeg and FFprobe setup complete!"
