
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


def record_audio(output_filename, record_seconds):
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 2
    RATE = 48000  # Sample rate
    CHUNK = 1024  # Buffer size

    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    # Calculate the total number of chunks needed
    total_chunks = math.ceil((RATE * record_seconds) / CHUNK)
    for i in range(total_chunks):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Only the recording directory is needed here
    recording_dir, _ = create_directory_structure()
    output_path = os.path.join(recording_dir, output_filename)

    # Save the recorded data as a WAV file
    waveFile = wave.open(output_path, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def split_audio(input_filename, interval_seconds, label_prefix):
    # Only the chunks directory is needed here
    _, chunks_dir = create_directory_structure()
    # Ensure input file is located correctly
    input_path = os.path.join(create_directory_structure()[0], input_filename)

    with wave.open(input_path, 'rb') as wav:
        num_of_frames = wav.getnframes()
        rate = wav.getframerate()
        duration = num_of_frames / rate
        print("Rate= " + str(rate) + ", Number of frames= " +
              str(num_of_frames) + ", Duration= " + str(duration))
        chunks = math.ceil(duration / interval_seconds)

        # results_folder = os.path.join(os.path.dirname(__file__), "results")
        for wav_idx in range(chunks):
            start_frame = int(wav_idx * interval_seconds * rate)
            end_frame = int((wav_idx + 1) * interval_seconds * rate)

            # Ensure not to read past the end of the file
            # Check if this is the last chunk
            if wav_idx == chunks - 1:
                num_frames = num_of_frames - start_frame  # Only the remaining frames
            else:
                num_frames = end_frame - start_frame

            wav.setpos(start_frame)
            chunk_data = wav.readframes(num_frames)

            output_filename = f"{label_prefix}_{wav_idx}.wav"
            output_path = os.path.join(chunks_dir, output_filename)

            with wave.open(output_path, 'wb') as chunk_wav:
                chunk_wav.setnchannels(wav.getnchannels())
                chunk_wav.setsampwidth(wav.getsampwidth())
                chunk_wav.setframerate(rate)
                chunk_wav.writeframes(chunk_data)

            print(f"Created {output_filename}")
