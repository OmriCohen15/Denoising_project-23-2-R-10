import os
from pydub import AudioSegment
from tqdm import tqdm  # Import tqdm for the progress bar

# NOTE: on large files this can take a long time to complete and may fail.


def merge_wav_files_with_conversion_and_progress(folder_path, output_file_path, target_sample_rate=48000, channels=2):
    # Ensure the output file does not already exist
    if os.path.exists(output_file_path):
        print(
            f"Output file {output_file_path} already exists. Please remove it and try again.")
        return

    # Find all .wav files in the specified folder
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not wav_files:
        print("No .wav files found in the folder.")
        return

    # Initialize an empty AudioSegment for concatenating
    output_segment = AudioSegment.empty()

    # Iterate through each file with a progress bar
    for wav_file in tqdm(wav_files, desc="Merging files"):
        file_path = os.path.join(folder_path, wav_file)
        # Load the current file
        audio_segment = AudioSegment.from_wav(file_path)

        # Convert this file to the target parameters
        audio_segment = audio_segment.set_frame_rate(
            target_sample_rate).set_channels(channels)

        # Append this segment to the output segment
        output_segment += audio_segment

    # Export the concatenated audio segment to a file
    output_segment.export(output_file_path, format="wav")
    print(f"All files have been merged into {output_file_path}.")


# Example usage:
# Ensure you replace 'path_to_your_folder' with the actual path to your folder containing the .wav files,
# and 'path_to_output_file.wav' with the desired output file path.
merge_wav_files_with_conversion_and_progress(
    'path_to_your_folder', 'path_to_output_file.wav')
