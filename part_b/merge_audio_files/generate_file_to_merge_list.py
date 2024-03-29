# inorder to use FFmpeg, you need to have ffmpeg installed on your computer.
# link https://www.gyan.dev/ffmpeg/builds/
# A popular source is the gyan.dev FFmpeg builds:
# https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z

import os
import subprocess
from tqdm import tqdm


def create_file_list(folder_path, list_file_path):
    """
    Generate a text file listing all the .wav files in the folder_path.
    """
    with open(list_file_path, 'w') as list_file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                # Write the file path in the required format for FFmpeg
                list_file.write(
                    f"file '{os.path.join(folder_path, filename)}'\n")


def merge_wav_files(folder_path, output_file_path):
    """
    Merge all .wav files in the specified folder into a single output file using FFmpeg.
    """
    # Path for the temporary file list
    list_file_path = os.path.join(folder_path, "wav_files_list.txt")

    # Create the list of files to merge
    print("Generating list of files...")
    create_file_list(folder_path, list_file_path)

    # Use FFmpeg to merge the files based on the list

    # print("Merging files using FFmpeg...")
    # subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i',
    #                list_file_path, '-c', 'copy', output_file], check=True)
    # # Or just write the following command after generating the list:
    # # ffmpeg -f concat -safe 0 -i wav_files_list.txt -c copy output.wav

    # # Clean up the list file
    # os.remove(list_file_path)
    # print(f"All files have been successfully merged into {output_file_path}")


# Example usage
folder_path = 'clean_trainset_56spk_wav'  # Replace with your folder path
# Replace with your desired output file path
output_file_path = ''  # in the same folder as the .wav files
output_file_name = 'merged_trainset_56spk_wav.wav'
output_file = os.path.join(output_file_path, output_file_name)
merge_wav_files(folder_path, output_file)


# Example usage:
# Ensure you replace 'path_to_your_folder' with the actual path to your folder containing the .wav files,
# and 'path_to_output_file.wav' with the desired output file path.
# merge_wav_files_with_conversion_and_progress(
# 'noisy_trainset_56spk_wav', 'noisy_trainset_56spk_wav_merged.wav')
