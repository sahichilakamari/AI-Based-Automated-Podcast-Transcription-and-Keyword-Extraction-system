from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarized = summarizer(text, max_length=150, min_length=40, do_sample=False)
    summary_points = summarized[0]['summary_text'].split(". ")
    return summary_points
