"""Enhanced audio processing utilities with chunking support."""

import os
import logging
from typing import List, Tuple, Generator
from pydub import AudioSegment
from pydub.utils import which

# Explicitly set the converter and ffprobe paths
AudioSegment.converter = which("ffmpeg") or os.path.abspath("bin/ffmpeg")
AudioSegment.ffprobe = which("ffprobe") or os.path.abspath("bin/ffprobe")

# Optional logging to confirm
print("FFmpeg path used:", AudioSegment.converter)
print("FFprobe path used:", AudioSegment.ffprobe)

import numpy as np
from config import AUDIO_CONFIG, TEMP_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Enhanced audio processor with chunking capabilities."""
    
    def __init__(self):
        self.config = AUDIO_CONFIG
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate audio file size and format."""
        try:
            if not os.path.exists(file_path):
                return False, "File not found"
                
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.config.MAX_FILE_SIZE_MB:
                return False, f"File too large: {file_size_mb:.1f}MB (max: {self.config.MAX_FILE_SIZE_MB}MB)"
            
            # Try to load a small portion to validate format
            audio = AudioSegment.from_file(file_path)[:1000]  # First second
            return True, "Valid audio file"
        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False, f"Invalid audio file: {str(e)}"
    
    def convert_to_wav(self, input_path: str, output_path: str) -> str:
        """Convert audio file to WAV format with optimal settings."""
        try:
            logger.info(f"Converting {input_path} to WAV format...")
            audio = AudioSegment.from_file(input_path)
            
            # Normalize audio settings
            audio = audio.set_frame_rate(self.config.SAMPLE_RATE)
            audio = audio.set_channels(1)  # Mono for better transcription
            
            # Export with optimal settings
            audio.export(
                output_path,
                format="wav",
                parameters=["-ac", "1", "-ar", str(self.config.SAMPLE_RATE)]
            )
            
            logger.info(f"Successfully converted to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Conversion error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio conversion failed: {str(e)}")
    
    def get_audio_info(self, file_path: str) -> dict:
        """Get detailed audio file information."""
        try:
            audio = AudioSegment.from_file(file_path)
            return {
                "duration_seconds": len(audio) / 1000,
                "duration_formatted": self._format_duration(len(audio) / 1000),
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "estimated_chunks": self._estimate_chunks(len(audio))
            }
        except Exception as e:
            logger.error(f"Audio info error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not get audio info: {str(e)}")
    
    def create_audio_chunks(self, audio_path: str) -> Generator[Tuple[str, int, int], None, None]:
        """Create overlapping audio chunks for processing."""
        try:
            audio = AudioSegment.from_file(audio_path)
            total_length = len(audio)
            
            chunk_start = 0
            chunk_index = 0
            
            while chunk_start < total_length:
                chunk_end = min(chunk_start + self.config.CHUNK_LENGTH_MS, total_length)
                
                # Extract chunk with overlap
                if chunk_start > 0:
                    actual_start = max(0, chunk_start - self.config.OVERLAP_MS)
                else:
                    actual_start = chunk_start
                
                chunk = audio[actual_start:chunk_end]
                
                # Save chunk to temporary file
                chunk_filename = f"chunk_{chunk_index:04d}.wav"
                chunk_path = os.path.join(TEMP_DIR, chunk_filename)
                
                chunk.export(chunk_path, format="wav")
                
                yield chunk_path, chunk_start, chunk_end
                
                chunk_start += self.config.CHUNK_LENGTH_MS
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"Chunking error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio chunking failed: {str(e)}")
    
    def cleanup_temp_files(self, pattern: str = "chunk_*.wav"):
        """Clean up temporary chunk files."""
        import glob
        temp_files = glob.glob(os.path.join(TEMP_DIR, pattern))
        for file_path in temp_files:
            try:
                os.remove(file_path)
                logger.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {str(e)}")
    
    def _estimate_chunks(self, audio_length_ms: int) -> int:
        """Estimate number of chunks needed."""
        return max(1, (audio_length_ms + self.config.CHUNK_LENGTH_MS - 1) // self.config.CHUNK_LENGTH_MS)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
