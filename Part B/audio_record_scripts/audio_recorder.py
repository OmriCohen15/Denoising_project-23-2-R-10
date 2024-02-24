
import pyaudio
import wave
import math
import os


def record_audio(output_filename, record_seconds):
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 2
    RATE = 44100  # Sample rate
    CHUNK = 1024  # Buffer size
    WAVE_OUTPUT_FILENAME = output_filename

    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def split_audio(input_filename, interval_seconds, label_prefix):
    with wave.open(input_filename, 'rb') as wav:
        length = wav.getnframes()
        rate = wav.getframerate()
        duration = length / rate
        chunks = math.ceil(duration / (interval_seconds))
        try:
            os.mkdir(f"chunks")
        except Exception:
            pass

        for i in range(chunks):
            wav.setpos(int(i * interval_seconds * rate))
            chunk_data = wav.readframes(int(interval_seconds * rate))
            output_filename = f"chunks/{label_prefix}_{i}.wav"

            with wave.open(output_filename, 'wb') as chunk_wav:
                chunk_wav.setnchannels(wav.getnchannels())
                chunk_wav.setsampwidth(wav.getsampwidth())
                chunk_wav.setframerate(rate)
                chunk_wav.writeframes(chunk_data)
            print(f"Created {output_filename}")
