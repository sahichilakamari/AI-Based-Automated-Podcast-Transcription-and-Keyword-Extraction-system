"""Enhanced transcription engine with chunking and progress tracking."""

import logging
from typing import List, Tuple, Dict, Any
from faster_whisper import WhisperModel
from config import TRANSCRIPTION_CONFIG
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """Enhanced transcription engine with chunking support."""
    
    def __init__(self):
        self.config = TRANSCRIPTION_CONFIG
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model with error handling."""
        try:
            logger.info(f"Loading Whisper model: {self.config.MODEL_SIZE}")
            self.model = WhisperModel(
                self.config.MODEL_SIZE,
                device=self.config.DEVICE,
                compute_type=self.config.COMPUTE_TYPE
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def transcribe_chunk(self, audio_path: str) -> Tuple[str, str, float]:
        """Transcribe a single audio chunk."""
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=self.config.BEAM_SIZE,
                vad_filter=self.config.VAD_FILTER,
                vad_parameters=self.config.VAD_PARAMETERS
            )
            
            transcription = " ".join(segment.text.strip() for segment in segments)
            language = info.language
            confidence = getattr(info, 'language_probability', 0.0)
            
            return transcription, language, confidence
        except Exception as e:
            logger.error(f"Error transcribing chunk {audio_path}: {str(e)}")
            return "", "unknown", 0.0
    
    def transcribe_audio_chunks(self, chunk_generator, total_chunks: int) -> Dict[str, Any]:
        """Transcribe audio using chunks with progress tracking."""
        transcriptions = []
        languages = []
        confidences = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, (chunk_path, start_ms, end_ms) in enumerate(chunk_generator):
                # Update progress
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i + 1}/{total_chunks} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)")
                
                # Transcribe chunk
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
                
                # Clean up chunk file
                try:
                    import os
                    os.remove(chunk_path)
                except:
                    pass
            
            # Combine transcriptions
            full_transcription = self._combine_transcriptions(transcriptions)
            detected_language = max(set(languages), key=languages.count) if languages else "unknown"
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Clear progress indicators
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
            logger.error(f"Error in chunk transcription: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            raise
    
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
                # Remove potential overlap by checking for repeated phrases
                prev_text = combined_text[-1]
                text = self._remove_overlap(prev_text, text)
                if text:
                    combined_text.append(text)
        
        return " ".join(combined_text)
    
    def _remove_overlap(self, prev_text: str, current_text: str, max_overlap_words: int = 10) -> str:
        """Remove overlapping text between chunks."""
        prev_words = prev_text.split()
        current_words = current_text.split()
        
        # Check for overlap at the end of prev_text and beginning of current_text
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