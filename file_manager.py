import os
import json
import csv

def save_to_json(transcript, keywords, output_path):
    data = {
        "transcript": transcript,
        "keywords": keywords
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_to_csv(transcript, keywords, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Transcript", "Keywords"])
        writer.writerow([transcript, ", ".join(keywords)])
