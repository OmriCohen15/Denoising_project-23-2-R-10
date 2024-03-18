import time
import pyaudio
import wave
import math
import os


def create_directory_structure():
    # Create 'results' directory within the current script directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a subdirectory named with the current date and time
    current_time = time.strftime("%d_%m_%Y_%H_%M", time.localtime())
    recording_dir = os.path.join(results_dir, current_time)
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)

    # Create a 'chunks' subdirectory within the recording directory
    chunks_dir = os.path.join(recording_dir, 'chunks')
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    return recording_dir, chunks_dir


def split_audio(input_filename, interval_seconds, label_prefix):
    # Only the chunks directory is needed here
    _, chunks_dir = create_directory_structure()
    # Ensure input file is located correctly
    input_path = os.path.join(input_filename)

    with wave.open(input_path, 'rb') as wav:
        length = wav.getnframes()
        rate = wav.getframerate()
        duration = length / rate
        chunks = math.ceil(duration / interval_seconds)

        # results_folder = os.path.join(os.path.dirname(__file__), "results")
        for i in range(chunks):
            wav.setpos(int(i * interval_seconds * rate))
            chunk_data = wav.readframes(int(interval_seconds * rate))
            output_filename = f"{label_prefix}_{i}.wav"
            # Save in the chunks directory
            output_path = os.path.join(chunks_dir, output_filename)

            with wave.open(output_path, 'wb') as chunk_wav:
                chunk_wav.setnchannels(wav.getnchannels())
                chunk_wav.setsampwidth(wav.getsampwidth())
                chunk_wav.setframerate(rate)
                chunk_wav.writeframes(chunk_data)
            print(f"Created {output_filename}")


input_filename = input("Enter the path to the input file: ")
interval_seconds = float(
    input("Enter the interval (in seconds) for splitting the audio: "))
label_prefix = input(
    "Enter the prefix for the labels of the split audio files: ")
split_audio(input_filename, interval_seconds, label_prefix)
