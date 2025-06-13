"""Enhanced Streamlit app for podcast transcription and analysis."""

import os
import streamlit as st
import logging
from pathlib import Path

# Import our enhanced modules
from utils.audio_processor import AudioProcessor
from utils.transcription_engine import TranscriptionEngine
from utils.analysis_engine import AnalysisEngine
from utils.ui_components import (
    display_audio_info, display_transcription_results,
    display_keywords, display_summary, display_topics,
    display_export_options, show_processing_animation
)
from config import AUDIO_CONFIG, UPLOAD_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎙️ Advanced Podcast Transcription & Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Advanced Podcast Transcription & Analysis</h1>
        <p>Upload audio files and get AI-powered transcription, keyword extraction, summarization, and topic modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This advanced tool provides:
        - **Smart Chunking**: Handles long audio files efficiently
        - **High Accuracy**: Uses Whisper AI for transcription
        - **Keyword Extraction**: AI-powered keyword identification
        - **Summarization**: Automatic content summarization
        - **Topic Modeling**: Discover main themes
        - **Export Options**: Multiple output formats
        """)
        
        st.header("📋 Supported Formats")
        formats = ", ".join(AUDIO_CONFIG.SUPPORTED_FORMATS).upper()
        st.write(formats)
        
        st.header("⚙️ Settings")
        model_size = st.selectbox(
            "Transcription Model",
            ["tiny", "base", "small", "medium", "large-v2"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        max_file_size = st.slider(
            "Max File Size (MB)",
            min_value=50,
            max_value=1000,
            value=AUDIO_CONFIG.MAX_FILE_SIZE_MB,
            help="Maximum allowed file size"
        )
    
    # File upload section
    st.header("📁 Upload Audio Files")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=list(AUDIO_CONFIG.SUPPORTED_FORMATS),
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(AUDIO_CONFIG.SUPPORTED_FORMATS).upper()}"
    )
    
    if uploaded_files:
        # Initialize processors
        audio_processor = AudioProcessor()
        transcription_engine = TranscriptionEngine()
        analysis_engine = AnalysisEngine()
        
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"🎵 Processing: {uploaded_file.name}")
            
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate file
            is_valid, message = audio_processor.validate_audio_file(file_path)
            if not is_valid:
                st.error(f"❌ {message}")
                continue
            
            # Get audio information
            with st.spinner("📊 Analyzing audio file..."):
                audio_info = audio_processor.get_audio_info(file_path)
            
            if audio_info:
                st.success("✅ Audio file validated successfully!")
                display_audio_info(audio_info)
                
                # Convert to WAV if needed
                if not file_path.lower().endswith('.wav'):
                    with st.spinner("🔄 Converting to WAV format..."):
                        wav_path = os.path.join(UPLOAD_DIR, f"{Path(uploaded_file.name).stem}.wav")
                        wav_path = audio_processor.convert_to_wav(file_path, wav_path)
                else:
                    wav_path = file_path
                
                # Process with chunking
                if st.button(f"🚀 Start Processing {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    
                    # Transcription
                    st.header("🎯 Transcription")
                    with st.spinner("🎙️ Transcribing audio (this may take a while for long files)..."):
                        try:
                            # Create chunks
                            chunk_generator = audio_processor.create_audio_chunks(wav_path)
                            total_chunks = audio_info.get('estimated_chunks', 1)
                            
                            # Transcribe
                            transcription_results = transcription_engine.transcribe_audio_chunks(
                                chunk_generator, total_chunks
                            )
                            
                            if transcription_results['transcription']:
                                st.success("✅ Transcription completed!")
                                display_transcription_results(transcription_results)
                                
                                # Store results
                                st.session_state.results[uploaded_file.name] = {
                                    'transcription': transcription_results,
                                    'audio_info': audio_info
                                }
                                
                                # Analysis tabs
                                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                    "📝 Transcript", "🔑 Keywords", "📋 Summary", "🧠 Topics", "💾 Export"
                                ])
                                
                                with tab1:
                                    st.text_area(
                                        "Full Transcript:",
                                        transcription_results['transcription'],
                                        height=400,
                                        help="Complete transcription of the audio file"
                                    )
                                
                                with tab2:
                                    with st.spinner("🔍 Extracting keywords..."):
                                        keywords = analysis_engine.extract_keywords(
                                            transcription_results['transcription']
                                        )
                                    display_keywords(keywords)
                                
                                with tab3:
                                    with st.spinner("📋 Generating summary..."):
                                        summary = analysis_engine.summarize_text(
                                            transcription_results['transcription']
                                        )
                                    display_summary(summary)
                                
                                with tab4:
                                    with st.spinner("🧠 Detecting topics..."):
                                        topics = analysis_engine.detect_topics(
                                            transcription_results['transcription']
                                        )
                                    display_topics(topics)
                                
                                with tab5:
                                    display_export_options(
                                        transcription_results['transcription'],
                                        keywords,
                                        summary,
                                        topics
                                    )
                                
                            else:
                                st.error("❌ Transcription failed. Please check your audio file.")
                                
                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")
                            logger.error(f"Processing error: {str(e)}")
                        
                        finally:
                            # Cleanup
                            audio_processor.cleanup_temp_files()
            
            else:
                st.error("❌ Could not analyze audio file.")
    
    else:
        # Show welcome message and features
        st.markdown("""
        <div class="feature-box">
            <h3>🚀 Key Features</h3>
            <ul>
                <li><strong>Smart Processing:</strong> Handles files up to 1GB with intelligent chunking</li>
                <li><strong>High Accuracy:</strong> Uses OpenAI's Whisper model for transcription</li>
                <li><strong>Multi-language:</strong> Automatic language detection and support</li>
                <li><strong>Advanced Analysis:</strong> Keywords, summaries, and topic modeling</li>
                <li><strong>Export Ready:</strong> Multiple output formats for your workflow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👆 Upload one or more audio files to get started!")

if __name__ == "__main__":
    main()