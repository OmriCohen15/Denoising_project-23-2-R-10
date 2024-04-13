import utils.colored_noise_utils as noiser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm

TRAINING_INPUT_PATH = 'Datasets/trainset_input'
TRAINING_OUTPUT_PATH = 'Datasets/trainset_target'
TESTING_INPUT_PATH = 'Datasets/testset_input'
CLEAN_TRAINING_DIR = Path('Datasets/trainset_clean')
CLEAN_TESTING_DIR = Path("Datasets/testset_clean")
clean_training_dir_wav_files = sorted(list(CLEAN_TRAINING_DIR.rglob('*.wav')))
clean_testing_dir_wav_files = sorted(list(CLEAN_TESTING_DIR.rglob('*.wav')))
print("Total training samples:", len(clean_training_dir_wav_files))

# Announce the start of training data generation
print("Generating Training data")

# Check if the directory for training input exists, if not, create it
if not os.path.exists(TRAINING_INPUT_PATH):
    os.makedirs(TRAINING_INPUT_PATH)

# Check if the directory for training output exists, if not, create it
if not os.path.exists(TRAINING_OUTPUT_PATH):
    os.makedirs(TRAINING_OUTPUT_PATH)

# Process each audio file in the list of clean training audio files
for audio_file in tqdm(clean_training_dir_wav_files):
    # Load the clean audio file
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    # Generate a random signal-to-noise ratio (SNR) between 0 and 10
    random_snr = np.random.randint(0, 10)
    # Generate white Gaussian noise on the clean audio with the random SNR
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    # Save the noised audio to the training input path
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TRAINING_INPUT_PATH, audio_file.name))

    # Generate another random SNR for variation
    random_snr = np.random.randint(0, 10)
    # Generate and save another version of the noised audio for the training output
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TRAINING_OUTPUT_PATH, audio_file.name))

# Announce the start of testing data generation
print("Generating Testing data")

# Check if the directory for testing input exists, if not, create it
if not os.path.exists(TESTING_INPUT_PATH):
    os.makedirs(TESTING_INPUT_PATH)

# Process each audio file in the list of clean testing audio files
for audio_file in tqdm(clean_testing_dir_wav_files):
    # Load the clean audio file
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    # Generate a random SNR between 0 and 10
    random_snr = np.random.randint(0, 10)
    # Generate white Gaussian noise on the clean audio with the random SNR
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    # Save the noised audio to the testing input path
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TESTING_INPUT_PATH, audio_file.name))
