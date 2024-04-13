from tqdm import tqdm
from pydub import AudioSegment
import torchaudio
import numpy as np
from scipy import interpolate
from scipy.io import wavfile
import os
import random

import warnings
warnings.filterwarnings("ignore")
np.random.seed(999)

noise_class_dictionary = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}


# Set Audio backend as Sounfile for windows and Sox for Linux
# torchaudio.set_audio_backend("soundfile")


def resample(original, old_rate, new_rate):
    """
    Resample the audio signal to a new rate using linear interpolation.

    This function changes the sample rate of an audio signal from an old rate to a new rate. 
    It uses linear interpolation to estimate the values at the new sample rate.

    Parameters:
    - original (numpy.ndarray): The original audio signal array.
    - old_rate (int): The original sample rate of the audio signal.
    - new_rate (int): The desired sample rate to convert the audio signal to.

    Returns:
    - numpy.ndarray: The resampled audio signal array at the new sample rate.

    If the old and new rates are the same, the original audio signal is returned unchanged.
    """
    if old_rate != new_rate:
        # Calculate the duration of the audio in seconds
        duration = original.shape[0] / old_rate

        # Time points for the original signal
        time_old = np.linspace(0, duration, original.shape[0])

        # Time points for the new signal
        time_new = np.linspace(0, duration, int(
            original.shape[0] * new_rate / old_rate))

        # Create a linear interpolator
        interpolator = interpolate.interp1d(time_old, original.T)

        # Use the interpolator to find new values
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original  # Return the original signal if rates are the same


fold_names = []
for i in range(1, 11):
    fold_names.append("fold"+str(i)+"/")


def diffNoiseType(files, noise_type):
    """
    Filter a list of filenames to include only those not matching a specific noise type.

    This function examines a list of filenames, expected to be in a format where the
    second element (when split by '-') indicates the noise type. It returns a list of
    filenames that do not match the specified noise type.

    Parameters:
    - files (list of str): The list of filenames to be filtered.
    - noise_type (int): The noise type to be excluded from the result list. The noise type
      is expected to be an integer value, which is compared against the second element of
      the filename when split by '-'.

    Returns:
    - list of str: A list containing only the filenames that do not match the specified
      noise type.
    """
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] != str(noise_type):
                result.append(i)
    return result


def oneNoiseType(files, noise_type):
    """
    Filter a list of filenames to include only those matching a specific noise type.

    This function examines a list of filenames, expected to be in a format where the
    second element (when split by '-') indicates the noise type. It returns a list of
    filenames that match the specified noise type.

    Parameters:
    - files (list of str): The list of filenames to be filtered.
    - noise_type (int): The noise type to be included in the result list. The noise type
      is expected to be an integer value, which is compared against the second element of
      the filename when split by '-'.

    Returns:
    - list of str: A list containing only the filenames that match the specified
      noise type.
    """
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] == str(noise_type):
                result.append(i)
    return result


def genNoise(filename, num_per_fold, dest):
    """
    Generate noise-augmented audio files by overlaying original audio with random noise samples.

    This function takes an audio file, selects a specified number of random noise samples from
    each fold of the UrbanSound8K dataset, overlays the noise onto the original audio, and saves
    the resulting audio files to a specified destination directory. Each resulting file is named
    by appending '_noise_' followed by a counter value to the original filename.

    Parameters:
    - filename (str): The name of the original audio file to be augmented with noise.
    - num_per_fold (int): The number of noise samples to overlay from each fold.
    - dest (str): The destination directory where the augmented audio files will be saved.

    Returns:
    - None: This function does not return a value but saves the augmented audio files directly
      to the filesystem.
    """
    true_path = target_folder+"/"+filename
    audio_1 = AudioSegment.from_file(true_path)
    counter = 0
    for fold in fold_names:
        dirname = Urban8Kdir + fold
        dirlist = os.listdir(dirname)
        total_noise = len(dirlist)
        samples = np.random.choice(total_noise, num_per_fold, replace=False)
        for s in samples:
            noisefile = dirlist[s]
            try:
                audio_2 = AudioSegment.from_file(dirname+"/"+noisefile)
                combined = audio_1.overlay(audio_2, times=5)
                target_dest = dest+"/" + \
                    filename[:len(filename)-4]+"_noise_"+str(counter)+".wav"
                combined.export(target_dest, format="wav")
                counter += 1
            except:
                print("Some kind of audio decoding error occurred, skipping this case")


def makeCorruptedFile_singletype(filename, dest, noise_type, snr):
    """
    Generate a noise-augmented audio file with a specific type of noise and signal-to-noise ratio (SNR).

    This function overlays a selected noise sample of a specific type onto the original audio file
    with a given SNR. The noise-augmented audio is then saved to the specified destination directory.
    The process attempts to load and augment the audio file until successful, skipping files that
    cause decoding errors.

    Parameters:
    - filename (str): The name of the original audio file to be augmented.
    - dest (str): The destination directory where the augmented audio file will be saved.
    - noise_type (int): The specific type of noise to overlay on the original audio. The noise type
      corresponds to an index in a predefined dictionary of noise types.
    - snr (int): The desired signal-to-noise ratio (in dB) for the augmented audio file.

    Returns:
    - None: This function does not return a value but saves the augmented audio file directly
      to the filesystem.
    """
    succ = False
    true_path = target_folder+"/"+filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except:
            print("Some kind of audio decoding error occurred for base file... skipping")
            break

        un_noised_file, _ = torchaudio.load(true_path)
        un_noised_file = un_noised_file.numpy()
        un_noised_file = np.reshape(un_noised_file, -1)
        # Calculate the power of the original audio signal in watts and then in decibels
        un_noised_file_watts = un_noised_file ** 2
        un_noised_file_db = 10 * np.log10(un_noised_file_watts)
        # Calculate the average power of the original signal in dB
        un_noised_file_avg_watts = np.mean(un_noised_file_watts)
        un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
        # Determine the average dB level of the added noise based on the desired SNR
        added_noise_avg_db = un_noised_file_avg_db - snr
        try:
            fold = np.random.choice(fold_names, 1, replace=False)
            fold = fold[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = oneNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            samples = np.random.choice(total_noise, 1, replace=False)
            s = samples[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname+"/"+noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file ** 2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            # Adjust the noise file's volume to achieve the desired SNR with the original audio
            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname+"/"+noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest+"/"+filename
            combined.export(target_dest, format="wav")
            succ = True
        except:
            pass  # If an error occurs during processing, the loop continues to retry
            # print("Some kind of audio decoding error occurred for the noise file..retrying")


def makeCorruptedFile_differenttype(filename, dest, noise_type, snr):
    """
    Generate a noise-augmented audio file with a different type of noise and specified SNR.

    This function overlays a randomly selected noise sample of a different type (not matching the specified noise_type)
    onto the original audio file with a given signal-to-noise ratio (SNR). The noise-augmented audio is then saved to
    the specified destination directory. The process is attempted repeatedly until successful, skipping files that
    cause decoding errors.

    Parameters:
    - filename (str): The name of the original audio file to be augmented.
    - dest (str): The destination directory where the augmented audio file will be saved.
    - noise_type (int): The type of noise to be excluded. The function will select a different noise type for augmentation.
    - snr (int): The desired signal-to-noise ratio (in dB) for the augmented audio file.

    Returns:
    - None: This function does not return a value but saves the augmented audio file directly to the filesystem.
    """
    succ = False
    true_path = target_folder+"/"+filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except:
            print("Some kind of audio decoding error occurred for base file... skipping")
            break

        un_noised_file, _ = torchaudio.load(true_path)
        un_noised_file = un_noised_file.numpy()
        un_noised_file = np.reshape(un_noised_file, -1)
        # Calculate the power of the original audio signal in watts and then in decibels
        un_noised_file_watts = un_noised_file ** 2
        un_noised_file_db = 10 * np.log10(un_noised_file_watts)
        # Calculate the average power of the original signal in dB
        un_noised_file_avg_watts = np.mean(un_noised_file_watts)
        un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
        # Determine the average dB level of the added noise based on the desired SNR
        added_noise_avg_db = un_noised_file_avg_db - snr

        try:
            fold = np.random.choice(fold_names, 1, replace=False)
            fold = fold[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = diffNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            samples = np.random.choice(total_noise, 1, replace=False)
            s = samples[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname+"/"+noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file ** 2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            # Adjust the noise file's volume to achieve the desired SNR with the original audio
            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname+"/"+noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest+"/"+filename
            combined.export(target_dest, format="wav")
            succ = True
        except:
            pass  # If an error occurs during processing, the loop continues to retry


Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/trainset_clean"

for key in noise_class_dictionary:
    print("\t{} : {}".format(key, noise_class_dictionary[key]))

# noise_type = int(input("Enter the noise class dataset to generate :\t"))

inp_folder = "Datasets/trainset_input"
op_folder = "Datasets/trainset_target"

print("Generating Training Data..")
print("Making train input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)
print("Making train output folder")
if not os.path.exists(op_folder):
    os.makedirs(op_folder)


counter = 0
# noise_type = 1
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        noise_type = random.randint(0, 9)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        snr = random.randint(0, 10)
        makeCorruptedFile_differenttype(filename, op_folder, noise_type, snr)
        counter += 1


Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/testset_clean"
inp_folder = "Datasets/testset_input"

print("Generating Testing Data..")
print("Making test input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)

counter = 0
# noise type was specified earlier
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        noise_type = random.randint(0, 9)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        counter += 1
