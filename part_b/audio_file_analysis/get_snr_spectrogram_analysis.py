import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys


def compute_snr(file_path_signal_noise, file_path_signal):
    """
    Compute the Signal-to-Noise Ratio (SNR) between two audio files.
    Assumes the first file is signal + noise, and the second is the signal (reference).
    """
    # Load the audio files
    y_noise, sr_noise = librosa.load(file_path_signal_noise, sr=None)
    y_signal, sr_signal = librosa.load(file_path_signal, sr=None)

    # Check if sampling rates match
    if sr_noise != sr_signal:
        print("Error: Sampling rates do not match.")
        return

    # Ensure lengths match by trimming the longer one
    min_len = min(len(y_noise), len(y_signal))
    y_noise = y_noise[:min_len]
    y_signal = y_signal[:min_len]

    # Compute the Noise (signal + noise) - (signal)
    y_noise_only = y_noise - y_signal

    # Calculate power of signal and noise
    signal_power = np.mean(y_signal ** 2)
    noise_power = np.mean(y_noise_only ** 2)

    # Compute SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def plot_spectrogram(file_path):
    """
    Plot the spectrogram of an audio file.
    """
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_signal+noise_file.wav> <path_to_signal_file.wav>")
    else:
        file_path_signal_noise = sys.argv[1]
        file_path_signal = sys.argv[2]

        # Compute and print SNR
        snr = compute_snr(file_path_signal_noise, file_path_signal)
        print(f"Signal-to-Noise Ratio (SNR): {snr:.3f} dB")

        # Plot spectrogram of the signal + noise file
        plot_spectrogram(file_path_signal_noise)
        plot_spectrogram(file_path_signal)
