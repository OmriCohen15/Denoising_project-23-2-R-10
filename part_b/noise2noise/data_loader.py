import os
import torchaudio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SpeechDataset(Dataset):
    """
    A dataset class designed for audio processing. It prepares audio data by trimming or padding
    them to a fixed length, applies a Short-Time Fourier Transform (STFT) to convert the audio
    into the frequency domain, and normalizes it for neural network processing.
    """

    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # Initialize dataset with lists of file paths for noisy and clean audio samples.
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # Parameters for STFT
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Total number of samples in the dataset
        self.len_ = len(self.noisy_files)

        # The maximum length to which all audio samples will be padded or cut
        self.max_len = 165000

    def __len__(self):
        # Returns the total number of samples in the dataset
        return self.len_

    def load_sample(self, file):
        # Loads an audio file using torchaudio, returning the waveform and ignoring the sample rate
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        # Retrieves a single pair of clean and noisy audio samples from the dataset at the specified index

        # Load audio files as waveforms
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        # Adjust waveform lengths by padding or cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # Convert waveforms to frequency domain using STFT
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True)

        # Convert complex STFT results to real numbers for compatibility with certain PyTorch layers
        x_noisy_stft = torch.view_as_real(x_noisy_stft)
        x_clean_stft = torch.view_as_real(x_clean_stft)

        return x_noisy_stft, x_clean_stft

    def _prepare_sample(self, waveform):
        # Ensures that all audio waveforms are of the same length
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        # Initialize a zero array of fixed size to hold the waveform
        output = np.zeros((1, self.max_len), dtype='float32')
        
        # Place the waveform in the last part of the output array to match the fixed size
        output[0, -current_len:] = waveform[0, :self.max_len]
        
        # Convert the numpy array back to a PyTorch tensor
        output = torch.from_numpy(output)

        return output
class PreTraining:
    """
    This class prepares the data for training a noise reduction model by setting up directories
    and handling the organization of input and target audio files.

    Args:
        noise_class (str): Specifies the category of noise to use (e.g., "street", "rain").
        training_type (str): Determines the training methodology, e.g., "Noise2Noise" or "Noise2Clean".
    """

    def __init__(self, noise_class, training_type):
        # Initialize the class with noise and training types
        self.noise_class = noise_class
        self.training_type = training_type

        # Display the initialization settings for verification
        print("Noise Class:", self.noise_class)
        print("Training Type:", self.training_type)

    def import_and_create_training_dir(self, noise_class, training_type):
        # Import data according to specified noise class and training type,
        # and create directories for training.
        self.import_data(noise_class, training_type)
        self.create_training_dir(noise_class, training_type)

    def import_data(self, noise_class, training_type):
        # Define base paths for datasets depending on the training type
        # and noise class specified.
        if noise_class != "":
            self.TRAIN_INPUT_DIR = Path('Datasets/trainset_input')

            if training_type == "Noise2Noise":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/trainset_target')
            elif training_type == "Noise2Clean":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/trainset_clean')
            else:
                # Handle invalid training types with an exception
                raise Exception("Enter valid training type")

            self.TEST_NOISY_DIR = Path('Datasets/testset_input')
            self.TEST_CLEAN_DIR = Path('Datasets/testset_clean')

        else:
            self.TRAIN_INPUT_DIR = Path('Datasets/trainset_input')

            if training_type == "Noise2Noise":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/trainset_target')
            elif training_type == "Noise2Clean":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/trainset_clean')
            else:
                raise Exception("Enter valid training type")

            self.TEST_NOISY_DIR = Path('Datasets/testset_input')
            self.TEST_CLEAN_DIR = Path('Datasets/testset_clean')

    def create_training_dir(self, noise_class, training_type):
        # Create directories for weights and samples under a base path
        # constructed from noise class and training type
        self.basepath = str(noise_class) + "_" + training_type
        os.makedirs(self.basepath, exist_ok=True)
        os.makedirs(self.basepath + "/Weights", exist_ok=True)
        os.makedirs(self.basepath + "/Samples", exist_ok=True)

    def save_train_test_files(self):
        # Gather and sort audio file paths from directories
        train_input_files = sorted(list(self.TRAIN_INPUT_DIR.rglob('*.wav')))
        train_target_files = sorted(list(self.TRAIN_TARGET_DIR.rglob('*.wav')))
        test_noisy_files = sorted(list(self.TEST_NOISY_DIR.rglob('*.wav')))
        test_clean_files = sorted(list(self.TEST_CLEAN_DIR.rglob('*.wav')))

        # Log the number of files available for training and testing
        print("No. of Training files:", len(train_input_files))
        print("No. of Testing files:", len(test_noisy_files))

        # Organize train and test files into dictionaries
        train_files = {
            'input': train_input_files,
            'target': train_target_files
        }
        test_files = {
            'input': test_noisy_files,  # noisy
            'target': test_clean_files  # clean
        }
        return train_files, test_files

