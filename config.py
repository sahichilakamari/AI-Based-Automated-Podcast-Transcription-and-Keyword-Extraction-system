"""Configuration settings for the podcast transcription system."""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    CHUNK_LENGTH_MS: int = 30000  # 30 seconds chunks
    OVERLAP_MS: int = 2000  # 2 seconds overlap
    SAMPLE_RATE: int = 16000
    MAX_FILE_SIZE_MB: int = 500
    SUPPORTED_FORMATS: tuple = ("mp3", "wav", "m4a", "flac", "ogg", "aac")

@dataclass
class TranscriptionConfig:
    MODEL_SIZE: str = "small"  # Start with smaller model
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"
    DOWNLOAD_ROOT: str = os.path.normpath(os.path.join(os.getcwd(), "model_cache"))
    BEAM_SIZE: int = 5
    VAD_FILTER: bool = False  # Disabled by default due to dependency issues
    VAD_PARAMETERS: Dict = None
    
@dataclass
class ProcessingConfig:
    """Processing configuration."""
    MAX_KEYWORDS: int = 15
    MAX_SUMMARY_LENGTH: int = 200
    MIN_SUMMARY_LENGTH: int = 50
    MAX_TOPICS: int = 5
    BATCH_SIZE: int = 4

# Global configuration instances
AUDIO_CONFIG = AudioConfig()
TRANSCRIPTION_CONFIG = TranscriptionConfig()
PROCESSING_CONFIG = ProcessingConfig()

# Directories
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
OUTPUT_DIR = "outputs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR, TRANSCRIPTION_CONFIG.DOWNLOAD_ROOT]:
    os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory not writable: {directory}")