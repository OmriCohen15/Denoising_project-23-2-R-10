import torch
import torch.nn as nn

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer. This layer implements convolution separately
    for the real and imaginary components of complex-valued inputs, enabling the use of complex
    numbers in neural networks.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Initialize module parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Real part convolution
        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        # Imaginary part convolution
        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        # Initialize weights using Glorot (Xavier) uniform initialization for both convolutions
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        # Split input into real and imaginary components
        x_real = x[..., 0]  # Real part of the input
        x_im = x[..., 1]    # Imaginary part of the input

        # Perform the complex convolution operation
        c_real = self.real_conv(x_real) - self.im_conv(x_im)  # Compute real part of the output
        c_im = self.im_conv(x_real) + self.real_conv(x_im)    # Compute imaginary part of the output

        # Combine the real and imaginary parts into a complex tensor
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
    Class of complex valued dilation convolutional layer, also known as a transposed convolution
    layer. This layer is used for up-sampling and effectively performs the inverse of a 
    convolution operation for complex-valued inputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        # Initialize module parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        # Real part transposed convolution
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        # Imaginary part transposed convolution
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Initialize weights using Glorot (Xavier) uniform initialization for both convolutions
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        # Split input into real and imaginary components
        x_real = x[..., 0]  # Real part of the input
        x_im = x[..., 1]    # Imaginary part of the input

        # Perform the transposed convolution (dilation convolution) operation
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)  # Compute real part of the output
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)    # Compute imaginary part of the output

        # Combine the real and imaginary parts into a complex tensor
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer. This layer extends the standard batch normalization
    for complex numbers by applying separate normalization to the real and imaginary components
    of the input. This is essential for maintaining the statistical independence and stability
    of complex-valued activations during training.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        # Initialize module parameters
        self.num_features = num_features
        self.eps = eps  # Small value added to the denominator for numerical stability
        self.momentum = momentum  # Momentum for the moving average
        self.affine = affine  # Whether to include learnable affine parameters
        self.track_running_stats = track_running_stats  # Whether to keep track of running averages

        # Real component batch normalization
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)

        # Imaginary component batch normalization
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        # Split input into real and imaginary components
        x_real = x[..., 0]  # Real part of the input
        x_im = x[..., 1]    # Imaginary part of the input

        # Normalize the real and imaginary components separately
        n_real = self.real_b(x_real)  # Normalized real component
        n_im = self.im_b(x_im)        # Normalized imaginary component

        # Combine the normalized real and imaginary parts into a complex tensor
        output = torch.stack([n_real, n_im], dim=-1)
        return output
    
class Encoder(nn.Module):
    """
    Class of upsample block in a neural network architecture, which increases the dimensionality
    of the input data. This class uses complex convolution, batch normalization, and a non-linear
    activation to process the input.
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()
        # Define convolutional block parameters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        # Complex convolution layer
        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        # Complex batch normalization layer
        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        # Non-linear activation function
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Apply convolution, normalization, and activation sequentially
        conved = self.cconv(x)  # Convolution step
        normed = self.cbn(conved)  # Normalization step
        acted = self.leaky_relu(normed)  # Activation function
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block in a neural network architecture, which decreases the dimensionality
    of the input data. This class uses complex transposed convolution, batch normalization, and
    a non-linear activation to process the input, with special processing for the last layer.
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()
        # Define transposed convolutional block parameters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        self.last_layer = last_layer  # Indicates if this is the last layer in the network

        # Complex transposed convolution layer
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)

        # Complex batch normalization layer
        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        # Non-linear activation function
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Apply transposed convolution
        conved = self.cconvt(x)

        if not self.last_layer:
            # Apply normalization and activation if it's not the last layer
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            # Special processing for the last layer to ensure phase and magnitude are handled correctly
            m_phase = conved / (torch.abs(conved) + 1e-8)  # Normalize the phase component
            m_mag = torch.tanh(torch.abs(conved))  # Apply tanh to the magnitude for stability
            output = m_phase * m_mag  # Combine phase and magnitude

        return output

class DCUnet20(nn.Module):
    """
    Deep Complex U-Net model, which is structured to work with complex-valued data, typically used
    for tasks that involve significant amounts of data transformation in the frequency domain,
    such as speech enhancement or other audio processing tasks.
    """

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        # Parameters for the inverse short-time Fourier transform (iSTFT)
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Setting up model complexity and channel configurations
        self.set_size(model_complexity=int(45//1.414),
                      input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 20 // 2  # Model depth divided by two to account for encoder-decoder symmetry

        # Initialize encoder blocks
        for i in range(self.model_length):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], stride_size=self.enc_strides[i], padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        # Initialize decoder blocks
        for i in range(self.model_length):
            # Check if it is the last layer, which has special configurations
            if i != self.model_length - 1:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[
                                     i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[
                                     i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

    def forward(self, x, is_istft=True):
        # Forward propagation through the encoder and decoder blocks
        orig_x = x
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)  # Propagate through each encoder

        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            # Concatenate output of decoder with corresponding encoder output for skip connections
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)

        # Generate mask from the last decoder output
        mask = p

        # Apply the mask to the original input
        output = mask * orig_x
        output = torch.squeeze(output, 1)

        # Convert back to time domain using iSTFT if required
        if is_istft:
            output = torch.view_as_complex(output)
            output = torch.istft(output, n_fft=self.n_fft,
                                 hop_length=self.hop_length, normalized=True)

        return output

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        # Configure model size and parameters based on specified complexity and depth
        if model_depth == 20:
            # List of channel counts for each encoder layer, scales with model complexity
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]
            # Tuple list defining kernel sizes for each encoder layer
            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            # Tuple list defining stride sizes for each encoder layer to control the convolution steps
            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]
            # Tuple list defining padding sizes for each encoder layer to maintain dimensions
            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]
            # List of channel counts for each decoder layer, mirrors the encoder and scales down
            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]
            # Tuple list defining kernel sizes for each decoder layer, similar to encoders but for transposed convolution
            self.dec_kernel_sizes = [(6, 3),
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]
            # Tuple list defining stride sizes for each decoder layer to control the upscaling steps
            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]
            # Tuple list defining padding sizes for each decoder layer to correctly shape output
            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]
            # Tuple list defining output padding for each decoder layer to ensure correct output size after transposed convolutions
            self.dec_output_padding = [(0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
