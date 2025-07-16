"""Enhanced Streamlit app for podcast transcription and analysis."""

# 1. BASE IMPORTS (No dependencies)
import os
import sys
import logging
from pathlib import Path

# Verify Python version
if sys.version_info >= (3, 11) or sys.version_info < (3, 10):
    sys.stderr.write("Error: Python 3.10 required (current: %s)\n" % sys.version)
    sys.exit(1)

# Set cache directories
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'model_cache')
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'model_cache')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 2. CONFIGURE LOGGING FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 3. IMPORT STREAMLIT AND AUDIO DEPENDENCIES
import streamlit as st
from pydub import AudioSegment
import imageio_ffmpeg

# 4. CONFIGURE FFMPEG (with proper error handling)
# 4. CONFIGURE FFMPEG (with proper error handling)
try:
    # Define ffmpeg_dir
    ffmpeg_dir = Path(__file__).parent.parent / "ffmpeg"
    # Resolve full binary paths
    ffmpeg_path = str(ffmpeg_dir / "ffmpeg")
    ffprobe_path = str(ffmpeg_dir / "ffprobe")

    # Optional: Make sure they're executable (especially in Linux)
    os.chmod(ffmpeg_path, 0o755)
    os.chmod(ffprobe_path, 0o755)

    # Set Pydub config manually
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
    if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
        raise FileNotFoundError("ffmpeg or ffprobe not found in ./ffmpeg directory")

    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    os.environ["PATH"] += os.pathsep + ffmpeg_dir  # Add ffmpeg folder to PATH

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    logger.info("FFmpeg and FFprobe configured successfully")
except Exception as e:
    logger.error(f"FFmpeg configuration failed: {str(e)}")
    st.error("Audio processing requires FFmpeg and FFprobe. Please ensure both are available in the 'ffmpeg' folder.")
    st.stop()


# 5. IMPORT APPLICATION MODULES
from utils.audio_processor import AudioProcessor
from utils.transcription_engine import TranscriptionEngine
from utils.analysis_engine import AnalysisEngine
from utils.ui_components import (
    display_audio_info, display_transcription_results,
    display_keywords, display_summary, display_topics,
    display_export_options
)
from config import AUDIO_CONFIG, TRANSCRIPTION_CONFIG, PROCESSING_CONFIG, UPLOAD_DIR, OUTPUT_DIR  # Added TRANSCRIPTION_CONFIG here

def check_system_requirements():
    """Check if system meets requirements."""
    if sys.version_info < (3, 8):
        st.error("❌ Python 3.8 or higher is required")
        st.stop()
    
    try:
        import torch
        import pydub
    except ImportError as e:
        st.error(f"❌ Missing required package: {str(e)}")
        st.stop()

def initialize_directories():
    """Ensure required directories exist."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR, TRANSCRIPTION_CONFIG.DOWNLOAD_ROOT]:
        os.makedirs(directory, exist_ok=True)
        if not os.access(directory, os.W_OK):
            st.error(f"❌ Directory not writable: {directory}")
            st.stop()

# Page configuration
st.set_page_config(
    page_title="🎙️ Advanced Podcast Transcription & Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .stAlert {
        border-radius: 8px;
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
    check_system_requirements()
    initialize_directories()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Advanced Podcast Transcription & Analysis</h1>
        <p>Upload audio files and get AI-powered transcription, keyword extraction, summarization, and topic modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This advanced tool provides:
        - **Smart Processing:** Handles files up to 1GB with intelligent chunking
        - **High Accuracy:** Uses OpenAI's Whisper model for transcription
        - **Keyword Extraction:** AI-powered keyword identification
        - **Summarization:** Automatic content summarization
        - **Topic Modeling:** Discover main themes
        - **Export Options:** Multiple output formats
        """)
        
        st.header("📋 Supported Formats")
        formats = ", ".join(AUDIO_CONFIG.SUPPORTED_FORMATS).upper()
        st.write(formats)
        
        st.header("⚙️ Settings")
        model_size = st.selectbox(
            "Transcription Model",
            ["tiny", "base", "small", "medium", "large-v2"],
            index=2,  # Default to small
            help="Larger models are more accurate but slower"
        )
        
        if model_size == "large-v2":
            st.warning("Large models require significant memory and may crash on systems with less than 16GB RAM")
        
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
        try:
            # Initialize processors
            audio_processor = AudioProcessor()
            
            try:
                # Create a local copy of the config to modify
                from config import TRANSCRIPTION_CONFIG as config_copy
                config_copy.MODEL_SIZE = model_size
                
                transcription_engine = TranscriptionEngine()
                
                if not transcription_engine.model:
                    st.error("❌ Failed to initialize Whisper model. Please check your internet connection and try again.")
                    return
                    
                analysis_engine = AnalysisEngine()
                
                if not all([analysis_engine.keyword_model, analysis_engine.summarizer, analysis_engine.topic_model]):
                    st.warning("⚠️ Some analysis features may not be available due to model loading issues")
                
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
                        try:
                            audio_info = audio_processor.get_audio_info(file_path)
                            if not audio_info:
                                st.error("❌ Could not analyze audio file")
                                continue
                                
                            st.success("✅ Audio file validated successfully!")
                            display_audio_info(audio_info)
                            
                            # Convert to WAV if needed
                            if not file_path.lower().endswith('.wav'):
                                with st.spinner("🔄 Converting to WAV format..."):
                                    wav_path = os.path.join(UPLOAD_DIR, f"{Path(uploaded_file.name).stem}.wav")
                                    wav_path = audio_processor.convert_to_wav(file_path, wav_path)
                            else:
                                wav_path = file_path
                            
                            # Process button
                            if st.button(f"🚀 Start Processing {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
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
                                        
                                        if not transcription_results['transcription']:
                                            st.error("❌ Transcription failed. No text was generated.")
                                            continue
                                            
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
                                        
                                    except Exception as e:
                                        st.error(f"❌ Processing error: {str(e)}")
                                        logger.error(f"Processing error: {str(e)}", exc_info=True)
                                    
                                    finally:
                                        # Cleanup
                                        audio_processor.cleanup_temp_files()
                                        try:
                                            os.remove(wav_path)
                                        except:
                                            pass
                        
                        except Exception as e:
                            st.error(f"❌ Error processing file: {str(e)}")
                            logger.error(f"File processing error: {str(e)}", exc_info=True)
                            continue
                            
            except RuntimeError as e:
                st.error(f"❌ Failed to initialize transcription engine: {str(e)}")
                logger.error(f"Transcription engine initialization error: {str(e)}", exc_info=True)
                return
                
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            logger.error(f"System initialization error: {str(e)}", exc_info=True)
    
    else:
        # Welcome message
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
