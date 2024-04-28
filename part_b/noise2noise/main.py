from imports import *


def main():
    """
    Set the random seed and turn on deterministic mode for PyTorch.
    """
    np.random.seed(999)
    torch.manual_seed(999)

    # If running on Cuda set these 2 for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
########################## Set the model assumptions ###########################
################################################################################
    """
    Set the model assumptions for the training and inference stages.

    sample_rate (int): The sample rate of the input audio.
    n_fft (int): The size of the FFT window. 
    hop_length (int): The hop length of the STFT.

    """

    # The sampling frequency and the selected values for the stft.
    SAMPLE_RATE = 48000  # (our project sample rate is 16000)
    N_FFT = (SAMPLE_RATE * 64) // 1000
    HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

################################################################################
################## Choose the mode of training and inference ###################
################################################################################

    mode = "train"
    # mode = "inference"

    if mode == "train":
        """
        Train the model for a given number of epochs.

        model (torch.nn.Module): The model to be trained.
        data_object (PreTraining): An object containing the training and testing data.
        train_loader (DataLoader): The data loader for the training data.
        test_loader (DataLoader): The data loader for the testing data.
        loss_fn (torch.nn.Module): The loss function to be used.
        optimizer (torch.optim): The optimizer to be used.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to be used.
        num_epochs (int): The number of epochs for training.

        During the training process the training log is printed containing the training losses and the testing losses.

        """
        # Code for training mode
        print("Training mode selected")

        # First checking if GPU is available
        train_on_gpu = torch.cuda.is_available()

        if (train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

        training_type = "Noise2Noise"
        noise_class = "walkie-talkie"
        data_object = PreTraining(training_type, noise_class)
        data_object.import_and_create_training_dir(noise_class, training_type)
        train_files, test_files = data_object.save_train_test_files()

        test_dataset = SpeechDataset(
            test_files['input'], test_files['target'], N_FFT, HOP_LENGTH)
        train_dataset = SpeechDataset(
            train_files['input'], train_files['target'], N_FFT, HOP_LENGTH)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        # For testing purpose
        test_loader_single_unshuffled = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        # Clear cache before training
        gc.collect()
        torch.cuda.empty_cache()

        dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(dcunet20.parameters())
        loss_fn = wsdr_fn
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.1)

        # specify paths and uncomment to resume training from a given point
        model_checkpoint = torch.load('/home/ubuntu/git/Denoising_project-23-2-R-10/part_b/noise2noise/training_results/urban_Noise2Noise/Weights/dc20_model_4.pth')
        dcunet20.load_state_dict(model_checkpoint)
        
        opt_checkpoint = torch.load('/home/ubuntu/git/Denoising_project-23-2-R-10/part_b/noise2noise/training_results/urban_Noise2Noise/Weights/dc20_opt_4.pth')
        optimizer.load_state_dict(opt_checkpoint)

        train_losses, test_losses = train(
            dcunet20, data_object, train_loader, test_loader, loss_fn, optimizer, scheduler, 4)

        print("train_losses = " + str(train_losses))
        print("test_losses = " + str(test_losses))
        print("Training complete")

    elif mode == "inference":
        """
        Perform inference on a given model and data object.

        Args:
            model (torch.nn.Module): The model to be used for inference.
            data_object (PreTraining): An object containing the training and testing data.

        Returns:
            None

        """
        # Code for inference mode
        print("Inference mode selected")

        DEVICE = torch.device('cpu')
        print('Training on CPU.')

        # Load the pre-trained model weights
        # model_weights_path = "training_results/white_Noise2Noise_12-03-24_20-20/Weights/dc20_model_3.pth"
        # model_weights_path = "training_results/podcasts_Noise2Noise_19-03-24_21-31/Weights/6hrs_podcasts_traning_dc20_model_3.pth"
        model_weights_path = "training_results/urban_Noise2Noise/Weights/dc20_model_4.pth"

        #
        dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(dcunet20.parameters())
        checkpoint = torch.load(model_weights_path,
                                map_location=torch.device('cpu')
                                )

        test_noisy_files = sorted(
            list(Path("Samples/Sample_Test_Input").rglob('*.wav')))

        test_clean_files = sorted(
            list(Path("Samples/Sample_Test_Target").rglob('*.wav')))

        test_dataset = SpeechDataset(
            test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)

        # Get an iterator over the test dataset
        test_loader_single_unshuffled = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        # Load the model
        dcunet20.load_state_dict(checkpoint)
        dcunet20.eval()
        test_loader_single_unshuffled_iter = iter(
            test_loader_single_unshuffled)

        results_dir = 'Samples/Results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Loop through the files in the directory
        for idx, wav_file_full_path in enumerate(test_clean_files):
            # clean_filename = os.path.basename(wav_file_full_path)
            clean_filename = Path(wav_file_full_path).stem
            noisy_filename = Path(test_noisy_files[idx]).stem
            results_path = results_dir + '/' + noisy_filename
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            original_sample_rate, data = wavfile.read(
                "Samples/Sample_Test_Target/"+clean_filename+".wav")

            # Print the path to the file
            print(wav_file_full_path)
            x_noisy, x_clean = next(test_loader_single_unshuffled_iter)
            x_estimated = dcunet20(x_noisy, is_istft=True)

            x_estimated_np = x_estimated[0].view(-1).detach().cpu().numpy()
            x_clean_np = torch.view_as_complex(
                torch.squeeze(x_clean[0], 1))
            x_noisy_np = torch.view_as_complex(
                torch.squeeze(x_noisy[0], 1))

            x_clean_np = torch.istft(x_clean_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     normalized=True).view(-1).detach().cpu().numpy()
            x_noisy_np = torch.istft(x_noisy_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     normalized=True).view(-1).detach().cpu().numpy()

            # Plot the results as Waveform and save them
            metrics = AudioMetrics(x_clean_np, x_estimated_np, SAMPLE_RATE)
            print("Clean vs. Denoised")
            print(metrics.display())    # Print the metrics
            metrics.save_to_file(
                results_path + "/metrics_clean_vs_denoised.txt")

            # Plot the results as Waveform and save them
            metrics = AudioMetrics(x_clean_np, x_noisy_np, SAMPLE_RATE)
            print("\nClean vs. Noisy")
            print(metrics.display())    # Print the metrics
            metrics.save_to_file(results_path + "/metrics_clean_vs_noisy.txt")

            plt.clf()
            # Noisy audio waveform
            plt.plot(x_noisy_np, label='Noisy', color='#1F77B4')
            plt.xlabel('Sample Number', fontweight='bold')
            plt.ylabel('Amplitude', fontweight='bold')
            plt.title('Noisy')
            plt.savefig(results_path+'/waveform_Noisy.png')
            plt.clf()

            # Clean audio waveform
            plt.plot(x_clean_np, label='Clean', color='#FF7F0E')
            plt.xlabel('Sample Number', fontweight='bold')
            plt.ylabel('Amplitude', fontweight='bold')
            plt.title('Clean')
            plt.savefig(results_path+'/waveform_Clean.png')
            plt.clf()

            # Estimated audio waveform
            plt.plot(x_estimated_np, label='Denoised', color='#2CA02C')
            plt.xlabel('Sample Number', fontweight='bold')
            plt.ylabel('Amplitude', fontweight='bold')
            plt.title('Denoised')
            plt.savefig(results_path+'/waveform_Denoised.png')
            plt.clf()

            plt.plot(x_noisy_np, label='Noisy')
            plt.plot(x_clean_np, label='Clean')
            plt.plot(x_estimated_np, label='Denoised')
            plt.xlabel('Sample Number', fontweight='bold')
            plt.ylabel('Amplitude', fontweight='bold')
            plt.title('Combined Results')
            plt.legend()
            # plt.show()
            plt.savefig(results_path+'/waveform_Combined_results.png')
            plt.clf()

            # Save the audio files
            save_audio_file(np_array=x_noisy_np, file_path=Path(
                results_path + "/noisy.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
            save_audio_file(np_array=x_estimated_np, file_path=Path(
                results_path + "/denoised.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
            save_audio_file(np_array=x_clean_np, file_path=Path(
                results_path + "/clean.wav"), sample_rate=original_sample_rate, bit_precision=16)

            # Plot the results as Spectrogram and save them
            plot_spectrogram(results_path + "/noisy.wav",
                             results_path, "spectrogram_Noisy")

            plot_spectrogram(results_path + "/denoised.wav",
                             results_path, "spectrogram_Denoised")

            plot_spectrogram(results_path + "/clean.wav",
                             results_path, "spectrogram_Clean")


if __name__ == "__main__":
    main()
