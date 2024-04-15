import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_spectrogram(file_path, folder_path, spectrogram_name):
    """
    Plot the spectrogram of an audio file.
    """
    plt.clf()
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    # plt.show()
    plt.savefig(folder_path+'/'+spectrogram_name+'.png')
