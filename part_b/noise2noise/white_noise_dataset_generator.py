import utils.colored_noise_utils as noiser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm


# Define paths for training and testing data directories
TRAINING_INPUT_PATH = 'Datasets/trainset_input'
TRAINING_OUTPUT_PATH = 'Datasets/trainset_target'
TESTING_INPUT_PATH = 'Datasets/testset_input'


# Define paths for directories containing clean audio files for training and testing
CLEAN_TRAINING_DIR = Path('Datasets/trainset_clean')
CLEAN_TESTING_DIR = Path("Datasets/testset_clean")
# Retrieve all .wav files from the clean training directory, sorted alphabetically
clean_training_dir_wav_files = sorted(list(CLEAN_TRAINING_DIR.rglob('*.wav')))
# Retrieve all .wav files from the clean testing directory, sorted alphabetically
clean_testing_dir_wav_files = sorted(list(CLEAN_TESTING_DIR.rglob('*.wav')))
# Print the total number of training samples found
print("Total training samples:", len(clean_training_dir_wav_files))

# Start the process of generating training data
print("Generating Training data")
# Check if the training input directory exists, if not, create it
if not os.path.exists(TRAINING_INPUT_PATH):
    os.makedirs(TRAINING_INPUT_PATH)
# Check if the training output directory exists, if not, create it
if not os.path.exists(TRAINING_OUTPUT_PATH):
    os.makedirs(TRAINING_OUTPUT_PATH)


for audio_file in tqdm(clean_training_dir_wav_files):
    # Load the original, clean audio file
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    # Generate a random signal-to-noise ratio (SNR) between 0 and 10
    random_snr = np.random.randint(0, 10)
    # Add white Gaussian noise to the clean audio with the generated SNR
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    # Save the noised audio as training input
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TRAINING_INPUT_PATH, audio_file.name))

    # Generate another random SNR for output data generation
    random_snr = np.random.randint(0, 10)
    # Add white Gaussian noise to the clean audio again with the new SNR
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    # Save the noised audio as training output
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TRAINING_OUTPUT_PATH, audio_file.name))


print("Generating Testing data")
if not os.path.exists(TESTING_INPUT_PATH):
    os.makedirs(TESTING_INPUT_PATH)

for audio_file in tqdm(clean_testing_dir_wav_files):
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    random_snr = np.random.randint(0, 10)
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(
        file_path=audio_file, snr=random_snr, color='white')
    noiser.save_audio_file(np_array=white_gaussian_noised_audio,
                           file_path='{}/{}'.format(TESTING_INPUT_PATH, audio_file.name))
