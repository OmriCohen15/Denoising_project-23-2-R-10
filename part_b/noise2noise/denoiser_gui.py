import tkinter as tk
from tkinter import filedialog
import os
from imports import *
from scipy.io import wavfile

# The sampling frequency and the selected values for the stft.
SAMPLE_RATE = 48000  # (our project sample rate is 16000)
N_FFT = (SAMPLE_RATE * 64) // 1000
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000


def update_status_label(text):
    status_label.config(text=text)


def select_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)


def denoise_folder():
    folder_path = folder_entry.get()

    DEVICE = torch.device('cpu')

    # update_status_label("Started Denoising!")

    # Load the pre-trained model weights
    model_weights_path = "training_results/urban_Noise2Noise/Weights/dc20_model_4.pth"

    # Load the model
    dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
    optimizer = torch.optim.Adam(dcunet20.parameters())
    checkpoint = torch.load(model_weights_path,
                            map_location=torch.device('cpu')
                            )

    test_noisy_files = sorted(
        list(Path(folder_path).rglob('*.wav')))

    test_dataset = SpeechDataset(
        test_noisy_files, test_noisy_files, N_FFT, HOP_LENGTH)

    # Get an iterator over the test dataset
    test_loader_single_unshuffled = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    # Load the model
    dcunet20.load_state_dict(checkpoint)
    dcunet20.eval()
    test_loader_single_unshuffled_iter = iter(
        test_loader_single_unshuffled)

    results_dir = 'Results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Loop through the files in the directory
    for wav_file_full_path in test_noisy_files:
        filename = Path(wav_file_full_path).stem
        results_path = results_dir + '/' + filename
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        # Print the path to the file
        print(wav_file_full_path)
        x_noisy, _ = next(test_loader_single_unshuffled_iter)
        x_estimated = dcunet20(x_noisy, is_istft=True)

        x_estimated_np = x_estimated[0].view(-1).detach().cpu().numpy()

        x_noisy_np = torch.view_as_complex(
            torch.squeeze(x_noisy[0], 1))

        x_noisy_np = torch.istft(x_noisy_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 normalized=True).view(-1).detach().cpu().numpy()

        # Save the audio files
        save_audio_file(np_array=x_estimated_np, file_path=Path(
            results_path + "/denoised.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)

        update_status_label("Denoising completed!")


# Create the main window
window = tk.Tk()
window.title("Denoiser")

# Set the width of the window
window.geometry("500x120")

# Create a label and an entry for the folder path
folder_label = tk.Label(window, text="Folder Path:")
folder_label.pack()

# Create a wider entry for the folder path
folder_entry = tk.Entry(window, width=80)
folder_entry.pack()

# Create a label for denoising status
status_label = tk.Label(window, text="")
status_label.pack()

# Create a button to select the folder
select_button = tk.Button(window, text="Select Folder", command=select_folder)
select_button.pack()

# Create a button to denoise the folder
denoise_button = tk.Button(window, text="Denoise", command=denoise_folder)
denoise_button.pack()

# Start the main event loop
window.mainloop()
