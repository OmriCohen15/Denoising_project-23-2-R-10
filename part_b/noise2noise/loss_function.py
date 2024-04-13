from pesq import pesq
from scipy import interpolate
from imports import *

# The sampling frequency and the selected values for the stft.
SAMPLE_RATE = 48000  # (our project sample rate is 16000)
N_FFT = (SAMPLE_RATE * 64) // 1000
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

DEVICE = torch.device('cuda')


def resample(original, old_rate, new_rate):
    """
    Resample audio data from one sample rate to another using linear interpolation.

    This function changes the sample rate of an audio signal by interpolating
    the original signal to match the desired new sample rate. If the original
    sample rate and the new sample rate are the same, the original audio is returned
    without any changes.

    Parameters:
    - original (numpy.ndarray): The original audio signal array.
    - old_rate (int): The original sample rate of the audio signal.
    - new_rate (int): The desired sample rate to convert the audio signal to.

    Returns:
    - numpy.ndarray: The resampled audio signal with the new sample rate.
    """
    if old_rate != new_rate:
        # Calculate the duration of the audio in seconds
        duration = original.shape[0] / old_rate

        # Time points for original audio
        time_old = np.linspace(0, duration, original.shape[0])

        # Time points for resampled audio
        time_new = np.linspace(0, duration, int(
            original.shape[0] * new_rate / old_rate))

        # Create an interpolator function
        interpolator = interpolate.interp1d(time_old, original.T)

        # Interpolate to get the new audio data
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original  # Return the original audio if sample rates are the same


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    """
    Calculate the Weighted Signal-to-Distortion Ratio (wSDR) for evaluating audio quality.

    This function computes the wSDR, a metric used to assess the quality of audio signals,
    especially in tasks like audio source separation or enhancement. It takes into account
    both the target signal and the interference (noise) signal to provide a more comprehensive
    measure of the predicted audio quality compared to traditional SDR.

    Parameters:
    - x_ (Tensor): The mixture input signal from which the target signal is to be separated,
                   in the frequency domain.
    - y_pred_ (Tensor): The predicted target signal by the model, in the frequency domain.
    - y_true_ (Tensor): The ground truth target signal, in the frequency domain.
    - eps (float, optional): A small epsilon value to prevent division by zero. Default is 1e-8.

    Returns:
    - Tensor: The mean weighted Signal-to-Distortion Ratio (wSDR) for the batch of audio samples.

    The function first converts the input signals from the frequency domain to the time domain
    using the inverse short-time Fourier transform (ISTFT). It then computes the SDR for both
    the target signal and the noise (the difference between the mixture and the target signal).
    The final wSDR score is a weighted sum of these two SDR values, where the weights are
    determined based on the energy of the target signal relative to the total energy of the
    target signal and the noise.
    """
    # Convert the complex tensors from frequency to time domain
    y_true_ = torch.squeeze(y_true_, 1)
    y_true_ = torch.view_as_complex(y_true_)
    y_true = torch.istft(y_true_, n_fft=N_FFT,
                         hop_length=HOP_LENGTH, normalized=True)
    x_ = torch.squeeze(x_, 1)
    x_ = torch.view_as_complex(x_)
    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    # Flatten the time-domain signals for SDR calculation
    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        """
        Calculate the Signal-to-Distortion Ratio (SDR).

        Parameters:
        - true (Tensor): The ground truth signal.
        - pred (Tensor): The predicted signal.
        - eps (float): A small epsilon value to prevent division by zero.

        Returns:
        - Tensor: The SDR value for each sample in the batch.
        """
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # Calculate the true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    # Compute the energy ratio of the target signal to the total energy
    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2,
                                                 dim=1) + torch.sum(z_true**2, dim=1) + eps)

    # Calculate the weighted SDR for both the target and noise signals
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


wonky_samples = []


def getMetricsonLoader(loader, net, use_net=True):
    """
    Evaluate audio quality metrics on a dataset loader using a specified network.

    This function iterates over a dataset loader, optionally processes the data through a neural network,
    and computes various audio quality metrics for each sample. The metrics include PESQ (both wideband and narrowband),
    SNR, SSNR, and STOI. The function aggregates these metrics across the dataset to provide an overall assessment
    of the audio quality or the effectiveness of the noise reduction model.

    Parameters:
    - loader (DataLoader): A PyTorch DataLoader containing the dataset to evaluate.
    - net (nn.Module): The neural network model to use for processing the audio data.
    - use_net (bool, optional): Flag to determine whether to process the audio data through the network.
                                If False, evaluates the metrics on the original noisy data. Default is True.

    Returns:
    - dict: A dictionary containing the mean, standard deviation, minimum, and maximum values for each metric.
    """
    net.eval()  # Set the network to evaluation mode
    scale_factor = 32768  # Original scale factor for audio data normalization

    # Names of the metrics to be evaluated
    metric_names = ["PESQ-WB", "PESQ-NB", "SNR", "SSNR", "STOI"]

    # Initialize lists to store metric values for each sample
    overall_metrics = [[] for i in range(5)]

    for i, data in enumerate(loader):
        if (i+1) % 10 == 0:
            end_str = "\n"
        else:
            end_str = ","
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
            continue  # Skip processing for problematic samples

        # Unpack the noisy and clean audio data from the loader
        noisy, clean = data[0], data[1]

        # Process the audio data through the network if specified, otherwise use the original noisy data
        if use_net:  # Forward pass through the network
            x_est = net(noisy.to(DEVICE), is_istft=True)
            # Convert the output to NumPy array
            x_est_np = x_est.view(-1).detach().cpu().numpy()
        else:
            x_est_np = torch.view_as_complex(torch.squeeze(noisy, 1))
            x_est_np = torch.istft(
                x_est_np, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

        # Convert the clean audio data to NumPy array
        x_clean_np = torch.view_as_complex(torch.squeeze(clean, 1))
        x_clean_np = torch.istft(x_clean_np, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 normalized=True).view(-1).detach().cpu().numpy()

        # Compute audio quality metrics
        metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)

        # Resample the audio for PESQ evaluation
        ref_wb, deg_wb = resample(x_clean_np, 48000, 16000), resample(
            x_est_np, 48000, 16000)
        # Compute wideband PESQ
        pesq_wb = pesq.pesq(16000, ref_wb, deg_wb, 'wb')
        ref_nb, deg_nb = resample(
            x_clean_np, 48000, 8000), resample(x_est_np, 48000, 8000)
        # Compute narrowband PESQ
        pesq_nb = pesq.pesq(8000, ref_nb, deg_nb, 'nb')

        # Aggregate the metrics
        overall_metrics[0].append(pesq_wb)
        overall_metrics[1].append(pesq_nb)
        overall_metrics[2].append(metrics.SNR)
        overall_metrics[3].append(metrics.SSNR)
        overall_metrics[4].append(metrics.STOI)

    # Compute and print summary statistics for each metric
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] = np.mean(overall_metrics[i])
        temp["STD"] = np.std(overall_metrics[i])
        temp["Min"] = min(overall_metrics[i])
        temp["Max"] = max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"

    # Print the computed metrics
    print("Metrics on test data", addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(
            overall_metrics[i]), np.std(overall_metrics[i])))
    return results
