import numpy as np
import torch
import torchaudio
import noise2noise_model


def main():
    print("running")
    wav_file_name = "speaker_record.wav"
    # model assumptions (copied and pasted from notebook)
    SAMPLE_RATE = 48000  # (our project sample rate is 16000)
    N_FFT = (SAMPLE_RATE * 64) // 1000
    HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

    torchaudio.set_audio_backend("sox_io")
    print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

    waveform, sr = torchaudio.load(wav_file_name)
    # resample the file to the model sample rate
    transform = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=SAMPLE_RATE)
    resampled_waveform = transform(waveform)
    waveform_stft_input = get_waveform_stft(
        resampled_waveform, N_FFT, HOP_LENGTH)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")

    # device = torch.device("cpu")
    # load model with the above parameters
    dcunet20 = noise2noise_model.DCUnet20(N_FFT, HOP_LENGTH).to(device)

    # get pre-trained model weights
    # this is for white noise model (our waveform may not match this  type of noise)
    model_weights_path = "Pretrained_Weights/Noise2Noise/white.pth"
    checkpoint = torch.load(
        model_weights_path, map_location=torch.device('cpu'))

    # load weights to model
    dcunet20.load_state_dict(checkpoint)

    # set the model to evaluation mode (not training)
    dcunet20.eval()

    # run model prediction
    print("executing model")
    with torch.no_grad():
        noisy_signal = waveform_stft_input
        # input is short time fourier transform of the wav signal
        # output is clean wav signal
        # add batch size of 1 as first dimension
        noisy_signal = torch.unsqueeze(noisy_signal, dim=0)
        noisy_signal = noisy_signal.to(device)
        # run the model
        clean_signal = dcunet20(noisy_signal, is_istft=True)
        print("finished processing")

    # save result as wav file
    file_path = wav_file_name + "filtered.wav"
    torch_tensor = clean_signal
    sample_rate = SAMPLE_RATE
    bit_precision = 16
    print("saving filtered file")
    torch_tensor = torch_tensor.detach().cpu()
    torchaudio.save(file_path, torch_tensor, sample_rate,
                    bits_per_sample=bit_precision)


def get_waveform_stft(waveform, n_fft, hop_length):
    x_noisy = _prepare_sample(waveform=waveform)
    # Short-time Fourier transform
    x_noisy_stft = torch.stft(input=x_noisy, n_fft=n_fft,
                              hop_length=hop_length, normalized=True, return_complex=True)
    x_noisy_stft = torch.view_as_real(x_noisy_stft)

    return x_noisy_stft


# copy and paste from the notebook file
def _prepare_sample(waveform):
    waveform = waveform.numpy()
    current_len = waveform.shape[1]
    # paper's magic number that causes output shorter than input (see if we can remove this limitation)
    max_len = 165000
    output = np.zeros((1, max_len), dtype='float32')
    output[0, -current_len:] = waveform[0, :max_len]
    output = torch.from_numpy(output)

    return output


if __name__ == '__main__':
    main()
    print("done")
