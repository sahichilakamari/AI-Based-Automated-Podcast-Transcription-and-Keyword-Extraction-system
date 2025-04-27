from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")

def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    transcription = " ".join(segment.text for segment in segments)
    language = info.language
    return transcription, language
