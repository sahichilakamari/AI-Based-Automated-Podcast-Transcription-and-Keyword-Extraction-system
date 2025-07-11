"""Enhanced transcription engine with chunking and progress tracking."""

import logging
import os
from typing import List, Tuple, Dict, Any
from faster_whisper import WhisperModel
from config import TRANSCRIPTION_CONFIG
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """Enhanced transcription engine with chunking support."""
    
    def __init__(self):
        self.config = TRANSCRIPTION_CONFIG
        self.model = self._load_model()
        self.vad_available = self._check_vad_support()
    
    def _check_vad_support(self) -> bool:
        """Check if VAD filtering is available."""
        if not self.config.VAD_FILTER:
            return False
            
        try:
            import onnxruntime
            return True
        except ImportError:
            logger.warning("ONNX Runtime not available - VAD filtering will be disabled")
            return False
        except Exception as e:
            logger.warning(f"VAD check failed: {str(e)} - VAD filtering will be disabled")
            return False
    
    def _load_model(self):
        """Load the Whisper model with error handling."""
        try:
            logger.info(f"Loading Whisper model: {self.config.MODEL_SIZE}")
            
            # Use the public model from guillaumekln
            model_name = f"guillaumekln/faster-whisper-{self.config.MODEL_SIZE}"
            
            # Load the model directly from Hugging Face Hub
            model = WhisperModel(
                model_name,
                device=self.config.DEVICE,
                compute_type=self.config.COMPUTE_TYPE,
                download_root=self.config.DOWNLOAD_ROOT
            )
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}", exc_info=True)
            st.error("Failed to load Whisper model. Please check your internet connection and try again.")
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def transcribe_chunk(self, audio_path: str) -> Tuple[str, str, float]:
        """Transcribe a single audio chunk."""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return "", "unknown", 0.0
                
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=self.config.BEAM_SIZE,
                vad_filter=self.vad_available and self.config.VAD_FILTER,
                vad_parameters=self.config.VAD_PARAMETERS if self.vad_available else None
            )
            
            transcription = " ".join(segment.text.strip() for segment in segments)
            language = info.language
            confidence = getattr(info, 'language_probability', 0.0)
            
            return transcription, language, confidence
        except Exception as e:
            logger.error(f"Chunk transcription error: {str(e)}", exc_info=True)
            return "", "unknown", 0.0
    
    # [Rest of the existing methods remain unchanged...]
    
    def transcribe_audio_chunks(self, chunk_generator, total_chunks: int) -> Dict[str, Any]:
        """Transcribe audio using chunks with progress tracking."""
        transcriptions = []
        languages = []
        confidences = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, (chunk_path, start_ms, end_ms) in enumerate(chunk_generator):
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i + 1}/{total_chunks} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)")
                
                transcription, language, confidence = self.transcribe_chunk(chunk_path)
                
                if transcription.strip():
                    transcriptions.append({
                        'text': transcription,
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'language': language,
                        'confidence': confidence
                    })
                    languages.append(language)
                    confidences.append(confidence)
                
                try:
                    os.remove(chunk_path)
                except Exception as e:
                    logger.warning(f"Could not remove chunk file: {str(e)}")
            
            full_transcription = self._combine_transcriptions(transcriptions)
            detected_language = max(set(languages), key=languages.count) if languages else "unknown"
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            progress_bar.empty()
            status_text.empty()
            
            return {
                'transcription': full_transcription,
                'language': detected_language,
                'confidence': avg_confidence,
                'chunks_processed': len(transcriptions),
                'chunk_details': transcriptions
            }
            
        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}", exc_info=True)
            progress_bar.empty()
            status_text.empty()
            raise RuntimeError(f"Audio transcription failed: {str(e)}")
    
    def _combine_transcriptions(self, transcriptions: List[Dict]) -> str:
        """Intelligently combine transcriptions from chunks."""
        if not transcriptions:
            return ""
        
        combined_text = []
        
        for i, chunk in enumerate(transcriptions):
            text = chunk['text'].strip()
            
            if i == 0:
                combined_text.append(text)
            else:
                text = self._remove_overlap(combined_text[-1], text)
                if text:
                    combined_text.append(text)
        
        return " ".join(combined_text)
    
    def _remove_overlap(self, prev_text: str, current_text: str, max_overlap_words: int = 10) -> str:
        """Remove overlapping text between chunks."""
        prev_words = prev_text.split()
        current_words = current_text.split()
        
        for overlap_len in range(min(max_overlap_words, len(prev_words), len(current_words)), 0, -1):
            if prev_words[-overlap_len:] == current_words[:overlap_len]:
                return " ".join(current_words[overlap_len:])
        
        return current_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_size': self.config.MODEL_SIZE,
            'device': self.config.DEVICE,
            'compute_type': self.config.COMPUTE_TYPE,
            'vad_enabled': self.config.VAD_FILTER
        }
