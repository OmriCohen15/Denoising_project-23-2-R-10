import os
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
from tqdm import tqdm
import time


# Increased noise
def walkie_talkie_effect(clean_signal, sampling_freq_hz, noise_amplitude=0.015):
    noisy_signal = clean_signal + noise_amplitude * \
        np.random.randn(len(clean_signal))

    # Pass the signal through a band pass filter that simulates a phone line
    f_low_hz = 300
    f_high_hz = 3000
    filter_order = 4
    b, a = signal.butter(filter_order, [
                         f_low_hz / (sampling_freq_hz / 2), f_high_hz / (sampling_freq_hz / 2)], btype='band')
    filter_signal = signal.filtfilt(b, a, noisy_signal)
    return filter_signal


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_directory(directory_path):
    # Create results directory with date and time
    current_time = time.strftime("%d_%m_%Y_%H_%M", time.localtime())
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, current_time)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log_file_path = os.path.join(results_path, 'processed_files_log.txt')
    with open(log_file_path, 'w') as log_file:
        for filename in tqdm([f for f in os.listdir(directory_path) if f.endswith('.wav')], desc="Processing"):
            clean_file_path = os.path.join(directory_path, filename)
            noisy_file_path = os.path.join(results_path, "noisy_" + filename)

            # Load and process the audio file
            clean_audio_data, sampling_rate = librosa.load(
                clean_file_path, sr=None)
            duration = len(clean_audio_data) / sampling_rate
            walkie_signal = walkie_talkie_effect(
                clean_audio_data, sampling_rate)

            # Write the processed audio
            sf.write(noisy_file_path, walkie_signal, int(sampling_rate))

            # Log file details
            log_file.write(
                f"{filename}, Sample Rate: {sampling_rate} Hz, Duration: {duration:.6f} seconds\n")


if __name__ == "__main__":
    # Replace with the path to your WAV files directory
    directory_path = '/home/ai_lab/git/Denoising_project-23-2-R-10/part_b/noise2noise/Datasets/trainset_target'
    process_directory(directory_path)
    print("Processing completed.")
