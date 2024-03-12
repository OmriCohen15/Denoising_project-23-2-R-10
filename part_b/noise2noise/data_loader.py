import os
import torchaudio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from pathlib import Path


class SpeechDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """

    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 165000

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # Short-time Fourier transform
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft,
                                  hop_length=self.hop_length, normalized=True, return_complex=True)
        x_noisy_stft = torch.view_as_real(x_noisy_stft)
        x_clean_stft = torch.view_as_real(x_clean_stft)

        # x_noisy_stft = torch.unsqueeze(x_noisy_stft, dim=0)
        # x_noisy_stft = x_noisy_stft.to(torch.device("cuda"))

        return x_noisy_stft, x_clean_stft

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output


class PreTraining:
    """
    This class is used to prepare the data for training the model.

    Args:
        noise_class (str): The class of noise to be used for training.
        training_type (str): The type of training to be performed.    
    """

    def __init__(self, noise_class, training_type):
        self.noise_class = noise_class
        self.training_type = training_type

        print("Noise Class:", self.noise_class)
        print("Training Type:", self.training_type)

    def import_and_create_training_dir(self, noise_class, training_type):
        self.import_data(noise_class, training_type)
        self.create_training_dir(noise_class, training_type)

    def import_data(self, noise_class, training_type):
        if noise_class == "white":
            self.TRAIN_INPUT_DIR = Path('Datasets/WhiteNoise_Train_Input')

            if training_type == "Noise2Noise":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/WhiteNoise_Train_Output')
            elif training_type == "Noise2Clean":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/clean_trainset_28spk_wav')
            else:
                raise Exception("Enter valid training type")

            self.TEST_NOISY_DIR = Path('Datasets/WhiteNoise_Test_Input')
            self.TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav')

        else:
            self.TRAIN_INPUT_DIR = Path('Datasets/US_Class' +
                                        str(noise_class)+'_Train_Input')

            if training_type == "Noise2Noise":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/US_Class'+str(noise_class)+'_Train_Output')
            elif training_type == "Noise2Clean":
                self.TRAIN_TARGET_DIR = Path(
                    'Datasets/clean_trainset_28spk_wav')
            else:
                raise Exception("Enter valid training type")

            self.TEST_NOISY_DIR = Path('Datasets/US_Class' +
                                       str(noise_class)+'_Test_Input')
            self.TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav')

    def create_training_dir(self, noise_class, training_type):
        self.basepath = str(noise_class) + "_" + training_type
        os.makedirs(self.basepath, exist_ok=True)
        os.makedirs(self.basepath + "/Weights", exist_ok=True)
        os.makedirs(self.basepath + "/Samples", exist_ok=True)

    def save_train_test_files(self):
        train_input_files = sorted(list(self.TRAIN_INPUT_DIR.rglob('*.wav')))
        train_target_files = sorted(list(self.TRAIN_TARGET_DIR.rglob('*.wav')))
        test_noisy_files = sorted(list(self.TEST_NOISY_DIR.rglob('*.wav')))
        test_clean_files = sorted(list(self.TEST_CLEAN_DIR.rglob('*.wav')))

        print("No. of Training files:", len(train_input_files))
        print("No. of Testing files:", len(test_noisy_files))
        # Save train and test files
        train_files = {
            'input': train_input_files,
            'target': train_target_files
        }
        test_files = {
            'input': test_noisy_files,  # noisy
            'target': test_clean_files  # clean
        }
        return train_files, test_files


# data_loader = CustomDataLoader(noise_class, training_type)
# data_loader.create_training_dir()
