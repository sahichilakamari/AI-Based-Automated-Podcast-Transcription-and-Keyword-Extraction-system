from pydub import AudioSegment

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path
