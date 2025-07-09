"""Enhanced analysis engine for keywords, summarization, and topic modeling."""

import logging
import os
import shutil
from typing import List, Dict, Any
import streamlit as st
from keybert import KeyBERT
from transformers import pipeline
from bertopic import BERTopic
from config import PROCESSING_CONFIG
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Enhanced analysis engine with better error handling."""
    
    def __init__(self):
        self.config = PROCESSING_CONFIG
        self.keyword_model = self._load_keyword_model()
        self.summarizer = self._load_summarizer()
        self.topic_model = self._load_topic_model()
    
    def _load_keyword_model(self):
        """Load KeyBERT model with error handling."""
        try:
            logger.info("Loading KeyBERT model...")
            model = KeyBERT()
            logger.info("KeyBERT model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load KeyBERT: {str(e)}", exc_info=True)
            return None
    
    def _load_summarizer(self):
        """Load summarization model with error handling."""
        try:
            logger.info("Loading summarization model...")
            
            # Set alternative cache location
            os.environ['HF_HOME'] = os.path.normpath(os.path.join(os.getcwd(), "model_cache"))
            
            # Try smaller model first
            model = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # CPU
            )
            logger.info("Summarization model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load summarizer: {str(e)}", exc_info=True)
            
            # Fallback to even smaller model
            try:
                model = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1
                )
                logger.info("Loaded fallback summarization model")
                return model
            except Exception as fallback_error:
                logger.error(f"Fallback summarizer failed: {str(fallback_error)}")
                return None
    
    def _load_topic_model(self):
        """Load topic model with error handling."""
        try:
            logger.info("Loading topic modeling...")
            model = BERTopic(verbose=False)
            logger.info("Topic model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load topic model: {str(e)}", exc_info=True)
            return None
    
    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords with scores and confidence."""
        if not self.keyword_model:
            logger.warning("Keyword model not available")
            return []
            
        if not text or not text.strip():
            logger.warning("No text provided for keyword extraction")
            return []
        
        try:
            keywords = self.keyword_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=self.config.MAX_KEYWORDS,
                use_mmr=True,
                diversity=0.5
            )
            
            return [
                {
                    'keyword': keyword,
                    'score': float(score),
                    'confidence': 'high' if score > 0.5 else 'medium' if score > 0.3 else 'low'
                }
                for keyword, score in keywords
            ]
        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}", exc_info=True)
            return []
    
    def summarize_text(self, text: str) -> Dict[str, Any]:
        """Generate summary with multiple approaches."""
        if not self.summarizer:
            logger.warning("Summarizer model not available")
            return self._extractive_summary(text)
            
        if not text or not text.strip():
            logger.warning("No text provided for summarization")
            return {'summary': '', 'bullet_points': [], 'method': 'none'}
        
        try:
            max_chunk_length = 1024
            text_chunks = self._chunk_text(text, max_chunk_length)
            
            summaries = []
            for chunk in text_chunks:
                if len(chunk.split()) < 20:
                    continue
                
                try:
                    result = self.summarizer(
                        chunk,
                        max_length=self.config.MAX_SUMMARY_LENGTH // len(text_chunks),
                        min_length=self.config.MIN_SUMMARY_LENGTH // len(text_chunks),
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Summarization chunk error: {str(e)}")
                    continue
            
            if summaries:
                combined_summary = " ".join(summaries)
                bullet_points = self._create_bullet_points(combined_summary)
                
                return {
                    'summary': combined_summary,
                    'bullet_points': bullet_points,
                    'method': 'transformer',
                    'chunks_processed': len(summaries)
                }
            else:
                return self._extractive_summary(text)
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}", exc_info=True)
            return self._extractive_summary(text)
    
    def detect_topics(self, text: str) -> List[Dict[str, Any]]:
        """Detect topics with detailed information."""
        if not self.topic_model:
            logger.warning("Topic model not available")
            return []
            
        if not text or not text.strip():
            logger.warning("No text provided for topic detection")
            return []
        
        try:
            sentences = self._split_into_sentences(text)
            
            if len(sentences) < 3:
                logger.warning("Not enough sentences for topic detection")
                return []
            
            topics, probabilities = self.topic_model.fit_transform(sentences)
            topic_info = self.topic_model.get_topic_info()
            
            detected_topics = []
            for topic_id in set(topics):
                if topic_id == -1:  # Skip outlier topic
                    continue
                
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words:
                    topic_data = {
                        'topic_id': topic_id,
                        'words': [word for word, _ in topic_words[:5]],
                        'scores': [float(score) for _, score in topic_words[:5]],
                        'document_count': sum(1 for t in topics if t == topic_id),
                        'representative_docs': self._get_representative_docs(sentences, topics, topic_id)
                    }
                    detected_topics.append(topic_data)
            
            detected_topics.sort(key=lambda x: x['document_count'], reverse=True)
            return detected_topics[:self.config.MAX_TOPICS]
            
        except Exception as e:
            logger.error(f"Topic detection error: {str(e)}", exc_info=True)
            return []
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for processing."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_length:
                if len(current_chunk) > 1:
                    current_chunk.pop()
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                else:
                    chunks.append(word)
                    current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_bullet_points(self, summary: str) -> List[str]:
        """Convert summary into bullet points."""
        sentences = summary.split('. ')
        return [sentence.strip() + '.' for sentence in sentences if sentence.strip()]
    
    def _extractive_summary(self, text: str) -> Dict[str, Any]:
        """Fallback extractive summarization."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 5:
            selected = sentences
        else:
            selected = sentences[:3] + sentences[-2:]
        
        summary = " ".join(selected)
        bullet_points = selected
        
        return {
            'summary': summary,
            'bullet_points': bullet_points,
            'method': 'extractive',
            'chunks_processed': 1
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_representative_docs(self, sentences: List[str], topics: List[int], topic_id: int) -> List[str]:
        """Get representative documents for a topic."""
        topic_sentences = [sentences[i] for i, t in enumerate(topics) if t == topic_id]
        return topic_sentences[:3]