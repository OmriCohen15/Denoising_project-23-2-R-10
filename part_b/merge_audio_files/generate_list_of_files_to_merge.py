# inorder to use FFmpeg, you need to have ffmpeg installed on your computer.
# For Windows:
# link https://www.gyan.dev/ffmpeg/builds/
# A popular source is the gyan.dev FFmpeg builds:
# https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z

# For Linux:
# sudo apt install ffmpeg

import os
import subprocess
from tqdm import tqdm
from natsort import natsorted, ns

def create_file_list(folder_path, list_file_path):
    """
    Generate a text file listing all the .wav files in the folder_path.
    """
    with open(list_file_path, 'w') as list_file:
        # Retrieve and sort the file list case-insensitively
        file_list = os.listdir(folder_path)
        file_list = natsorted(file_list, alg=ns.IGNORECASE)  # Natural and case-insensitive sorting

        for filename in file_list:
            if filename.endswith('.wav'):
                # Write the file path in the required format for FFmpeg
                list_file.write(
                    f"file '{os.path.join(folder_path, filename)}'\n")


def merge_wav_files(folder_path, output_file_with_path):
    """
    Merge all .wav files in the specified folder into a single output file using FFmpeg.
    """
    # Path for the temporary file list
    list_file_path = os.path.join(output_file_path, "wav_files_list.txt")

    # Create the list of files to merge
    print("Generating list of files...")
    create_file_list(folder_path, output_file_with_path)

    # Use FFmpeg to merge the files based on the list
    print("Merging files using FFmpeg...")
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i',
                   list_file_path, '-c', 'copy', output_file], check=True)

    # # Clean up the list file
    # os.remove(list_file_path)
    # print(f"All files have been successfully merged into {output_file_path}")



folder_path = '/home/ai_lab/git/Denoising_project-23-2-R-10/part_b/noise2noise/Datasets/trainset_input'
output_file_path = '/home/ai_lab/git/Denoising_project-23-2-R-10/part_b/noise2noise/Datasets'
output_file_name = 'trainset_input.txt'
output_file = os.path.join(output_file_path, output_file_name)

# Create the list of files to merge
print("Generating list of files...")
create_file_list(folder_path, output_file)

# merge_wav_files(folder_path, output_file)

# Faster option:--- just write the following command in the terminal after generating the list:
# ffmpeg -f concat -safe 0 -i wav_files_list.txt -c copy output.wav

# If you get errors, try using the following command:
# ffmpeg -f concat -safe 0 -i wav_files_list.txt -af asetpts=N/SR/TB output.wav


folder_path = '/home/ai_lab/git/Denoising_project-23-2-R-10/part_b/noise2noise/Datasets/trainset_target'
output_file_path = '/home/ai_lab/git/Denoising_project-23-2-R-10/part_b/noise2noise/Datasets'
output_file_name = 'trainset_target.txt'
output_file = os.path.join(output_file_path, output_file_name)

# Create the list of files to merge
print("Generating list of files...")
create_file_list(folder_path, output_file)

# merge_wav_files(folder_path, output_file)

# Faster option:--- just write the following command in the terminal after generating the list:
# ffmpeg -f concat -safe 0 -i wav_files_list.txt -c copy output.wav
