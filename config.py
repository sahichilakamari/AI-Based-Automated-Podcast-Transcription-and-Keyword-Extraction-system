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
    """Transcription model configuration."""
    MODEL_SIZE: str = "base"  # small, base, large-v2, large-v3
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"
    BEAM_SIZE: int = 5
    VAD_FILTER: bool = True
    VAD_PARAMETERS: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.VAD_PARAMETERS is None:
            self.VAD_PARAMETERS = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float("inf"),
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            }

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

# Create directories
for directory in [UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)