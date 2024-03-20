from imports import *


def main():

    np.random.seed(999)
    torch.manual_seed(999)

    # If running on Cuda set these 2 for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
########################## Set the model assumptions ###########################
################################################################################

    # The sampling frequency and the selected values for the stft.
    SAMPLE_RATE = 48000  # (our project sample rate is 16000)
    N_FFT = (SAMPLE_RATE * 64) // 1000
    HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

################################################################################
################## Choose the mode of training and inference ###################
################################################################################

    # mode = input("Enter mode (train or inference): ")

    # mode = "train"
    mode = "inference"

    if mode == "train":
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
        noise_class = "white"
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
        # model_checkpoint = torch.load(path_to_model)
        # opt_checkpoint = torch.load(path_to_opt)
        # dcunet20.load_state_dict(model_checkpoint)
        # optimizer.load_state_dict(opt_checkpoint)

        train_losses, test_losses = train(
            dcunet20, data_object, train_loader, test_loader, loss_fn, optimizer, scheduler, 3)

        print("train_losses = " + str(train_losses))
        print("test_losses = " + str(test_losses))
        print("Training complete")

    elif mode == "inference":
        # Code for inference mode
        print("Inference mode selected")

        DEVICE = torch.device('cpu')
        print('Training on CPU.')

        # Load the pre-trained model weights
        # model_weights_path = "Pretrained_Weights/Noise2Noise/white.pth"
        model_weights_path = "white_Noise2Noise/Weights/dc20_model_3.pth"

        #
        dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(dcunet20.parameters())
        checkpoint = torch.load(model_weights_path,
                                map_location=torch.device('cpu')
                                )

        # Load test files
        sample_index_to_be_test = 0

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

        x_noisy, x_clean = next(test_loader_single_unshuffled_iter)
        for _ in range(sample_index_to_be_test):
            x_noisy, x_clean = next(test_loader_single_unshuffled_iter)

        x_estimated = dcunet20(x_noisy, is_istft=True)

        x_estimated_np = x_estimated[0].view(-1).detach().cpu().numpy()
        x_clean_np = torch.view_as_complex(torch.squeeze(x_clean[0], 1))
        x_noisy_np = torch.view_as_complex(torch.squeeze(x_noisy[0], 1))

        x_clean_np = torch.istft(x_clean_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 normalized=True).view(-1).detach().cpu().numpy()
        x_noisy_np = torch.istft(x_noisy_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 normalized=True).view(-1).detach().cpu().numpy()

        # Plot the results
        metrics = AudioMetrics(x_clean_np, x_estimated_np, SAMPLE_RATE)
        print(metrics.display())    # Print the metrics
        plt.plot(x_noisy_np, label='Noisy')         # Noisy audio waveform
        plt.plot(x_clean_np, label='Clean')         # Clean audio waveform
        plt.plot(x_estimated_np, label='Estimated')  # Estimated audio waveform

        plt.legend()
        plt.show()
        # plt.savefig('/path/to/save/images.png')

        # Save the audio files
        save_audio_file(np_array=x_noisy_np, file_path=Path(
            "Samples/noisy.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
        save_audio_file(np_array=x_estimated_np, file_path=Path(
            "Samples/denoised.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
        save_audio_file(np_array=x_clean_np, file_path=Path(
            "Samples/clean.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)

    else:
        print("Invalid mode entered. Please choose either 'train' or 'inference'.")


if __name__ == "__main__":
    main()
