import os
import streamlit as st
from audio_utils import convert_to_wav
from transcription import transcribe_audio
from keyword_extraction import extract_keywords
from summarization import summarize_text
from topic_modeling import detect_topics

# Create uploads folder if not exists
os.makedirs("uploads", exist_ok=True)

st.set_page_config(page_title="🎙️ Podcast Transcription and Analysis", layout="centered")
st.title("🎙️ AI-Based Podcast Transcription and Keyword Extraction")
st.write("Upload audio files (MP3, WAV, M4A) — get Transcription, Keywords, Summarization, and Topics!")

uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")

        with st.spinner("Saving and converting audio..."):
            # Save uploaded file
            uploaded_file_path = os.path.join("uploads", uploaded_file.name)
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convert to WAV
            wav_output_path = os.path.join("uploads", uploaded_file.name.split('.')[0] + ".wav")
            wav_audio = convert_to_wav(uploaded_file_path, wav_output_path)

        with st.spinner("Transcribing audio..."):
            transcription, detected_language = transcribe_audio(wav_audio)
            st.success(f"Transcription completed! Language Detected: **{detected_language.upper()}**")

        # Tab for Transcription
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcription", "🔑 Keywords", "📋 Summary", "🧠 Topics"])

        with tab1:
            st.text_area("Transcript:", transcription, height=300)

        with tab2:
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(transcription)
            st.write(", ".join(keywords))

        with tab3:
            with st.spinner("Summarizing podcast..."):
                summary = summarize_text(transcription)
            for point in summary:
                st.markdown(f"- {point}")

        with tab4:
            with st.spinner("Detecting topics..."):
                topics = detect_topics(transcription)
            for topic in topics:
                st.markdown(f"- {topic}")

        st.divider()
