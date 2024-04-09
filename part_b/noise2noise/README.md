# Speech Denoising Using Noise2Noise: 
Welcome to our project repository! This project showcases the development of an AI-driven speech denoising system, specifically tailored for high-noise environments like military communications. Utilizing the innovative Noise2Noise approach, our system is designed to improve audio quality by filtering out noise without the need for clean reference data. This README provides an overview of our project, including the goals, methodology, and significant outcomes. Developed as part of a capstone project by students at Braude College, this system leverages deep learning techniques to enhance speaker recognition capabilities, making it particularly useful for defense applications where clarity and accuracy are paramount.

## Python Requirements
We recommend using Python 3.8.8. The package versions are in requirements.txt.
Install required Python libraries using the requirements.txt provided. Run:
```
pip install -r requirements.txt 
```

## Prepare Audio Files
Place the noisy audio files you want to denoise into a specific directory, e.g.,
```
./data/noisy/.
```

## Configure Model and Script Settings
Ensure the main.py script is configured to use the correct path for the trained model weights. Update the script or a config file if necessary to include:
```
model_path = 'path/to/trained_model_weights.pth' 
```
Set the input directory where you placed your noisy files in the script:
```
input_directory = './data/noisy/' 
```

## Run the Denoising Process
Execute the main.py script to start the denoising process. Run:
```
python main.py 
```
This will load your model, process each audio file in the input directory, and save the denoised audio to an output directory specified in the script.

## Check the Denoised Audio Files
Go to the output directory set in main.py to review the denoised files. The directory might be set like:
```
output_directory = './data/denoised/' 
```

By following these steps, you should be able to use the denoising model to clean your noisy audio files effectively. Make sure each configuration path and setting is correctly specified in your scripts to avoid any issues.

## Contact Information
Free to contact us with any problem or suggestion.
Omri Cohen: Omri.cohen2@e.braude.ac.il
Eliav Shabat: Eliav.shabat@e.braude.ac.il














## Dataset Generation
We use 2 standard datasets; 'UrbanSound8K'(for real-world noise samples), and 'Voice Bank + DEMAND'(for speech samples). Please download the datasets from [urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html) and [datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791) respectively. Extract and organize into the Datasets folder as shown below:
```
Noise2Noise-audio_denoising_without_clean_training_data
│     README.md
│     speech_denoiser_DCUNet.ipynb
|     ...
│_____Datasets
      |     clean_testset_wav
      |     clean_trainset_28spk_wav
      |     noisy_testset_wav
      |     noisy_trainset_28spk_wav
      |_____UrbanSound8K
            |_____audio
                  |_____fold1
                  ...
                  |_____fold10

```

To train a White noise denoising model, run the script:
```
python white_noise_dataset_generator.py
```

To train a UrbanSound noise class denoising model, run the script, and select the noise class:
```
python urban_sound_noise_dataset_generator.py

0 : air_conditioner
1 : car_horn
2 : children_playing
3 : dog_bark
4 : drilling
5 : engine_idling
6 : gun_shot
7 : jackhammer
8 : siren
9 : street_music
```
The train and test datasets for the specified noise will be generated in the 'Datasets' directory.

## Training a New Model
In the 'speech_denoiser_DCUNet.ipynb' file. Specify the type of noise model you want to train to denoise(You have to generate the specific noise Dataset first). You can choose whether to train using our Noise2Noise approach(using noisy audio for both training inputs and targets), or the conventional approach(using noisy audio as training inputs and the clean audio as training target). If you are using Windows, set 'soundfile' as the torchaudio backend. If you are using Linux, set 'sox' as the torchaudio backend. The weights .pth file is saved for each training epoch in the 'Weights' directory.

## Testing Model Inference on Pretrained Weights
We have trained our model with both the Noise2Noise and Noise2Clean approaches, for all 10(numbered 0-9) UrbanSound noise classes and White Gaussian noise. All of our pre-trained model weights are uploaded in 'Pretrained_Weights' directory under the 'Noise2Noise' and 'Noise2Clean' subdirectories.

In the 'speech_denoiser_DCUNet.ipynb' file. Select the weights .pth file for model to use. Point to the testing folders containing the audio you want to denoise. Audio quality metrics will also be calculated. The noisy, clean and denoised wav files will be saved in the 'Samples' directory.

## Example
### Noisy audio waveform
![./static/noisy_waveform.PNG](./static/noisy_waveform.PNG)
### Model denoised audio waveform
![static/denoised_waveform.PNG](static/denoised_waveform.PNG)
### True clean audio waveform
![static/clean_waveform.PNG](static/clean_waveform.PNG)
## 20-Layered Deep Complex U-Net 20 Model Used
![static/dcunet20.PNG](static/dcunet20.PNG)
## Results
![static/results.PNG](static/results.PNG)

## Special thanks to the following repositories:
* https://github.com/pheepa/DCUnet
* https://github.com/ludlows/python-pesq
* https://github.com/mpariente/pystoi

## References
[1] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature,
vol. 521, no. 7553, pp. 436–444, May 2015.

[2] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras,
M. Aittala, and T. Aila, “Noise2Noise: Learning image restoration
without clean data,” in Proceedings of the 35th International
Conference on Machine Learning, 2018, pp. 2965–2974.

[3] N. Alamdari, A. Azarang, and N. Kehtarnavaz, “Improving
deep speech denoising by noisy2noisy signal mapping,” Applied
Acoustics, vol. 172, p. 107631, 2021.

[4] R. E. Zezario, T. Hussain, X. Lu, H. M.Wang, and Y. Tsao, “Selfsupervised
denoising autoencoder with linear regression decoder
for speech enhancement,” in ICASSP 2020 - 2020 IEEE International
Conference on Acoustics, Speech and Signal Processing
(ICASSP), 2020, pp. 6669–6673.

[5] Y. Shi, W. Rong, and N. Zheng, “Speech enhancement using convolutional
neural network with skip connections,” in 2018 11th
International Symposium on Chinese Spoken Language Processing
(ISCSLP), 2018, pp. 6–10.

[6] Z. Zhao, H. Liu, and T. Fingscheidt, “Convolutional neural networks
to enhance coded speech,” IEEE/ACM Transactions on Audio,
Speech, and Language Processing, vol. 27, no. 4, pp. 663–
678, 2019.

[7] F. G. Germain, Q. Chen, and V. Koltun, “Speech Denoising
with Deep Feature Losses,” in Proc. Interspeech 2019, 2019,
pp. 2723–2727. [Online]. Available: http://dx.doi.org/10.21437/
Interspeech.2019-1924

[8] A. Azarang and N. Kehtarnavaz, “A review of multi-objective
deep learning speech denoising methods,” Speech Communication,
vol. 122, 05 2020.

[9] C. Valentini-Botinhao, “Noisy speech database for training
speech enhancement algorithms and TTS models 2016[sound].”
[Online]. Available: https://doi.org/10.7488/ds/2117

[10] J. Salamon, C. Jacoby, and J. P. Bello, “A dataset and taxonomy
for urban sound research,” in 22nd ACM International Conference
on Multimedia (ACM-MM’14), Orlando, FL, USA, Nov. 2014, pp.
1041–1044.

[11] J. Robert, M. Webbie et al., “Pydub,” 2018. [Online]. Available:
http://pydub.com/

[12] H.-S. Choi, J.-H. Kim, J. Huh, A. Kim, J.-W. Ha, and K. Lee,
“Phase-aware speech enhancement with deep complex u-net,” in
International Conference on Learning Representations, 2018.

[13] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional
networks for biomedical image segmentation,” in International
Conference on Medical image computing and computer-assisted
intervention. Springer, 2015, pp. 234–241.

[14] C. Veaux, J. Yamagishi, and S. King, “The voice bank corpus: Design,
collection and data analysis of a large regional accent speech
database,” in 2013 international conference oriental COCOSDA
held jointly with 2013 conference on Asian spoken language research
and evaluation (O-COCOSDA/CASLRE). IEEE, 2013,
pp. 1–4.

[15] J. Thiemann, N. Ito, and E. Vincent, “The diverse environments
multi-channel acoustic noise database (demand): A database of
multichannel environmental noise recordings,” in Proceedings of
Meetings on Acoustics ICA2013, vol. 19, no. 1. Acoustical Society
of America, 2013, p. 035081.

[16] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi,
“Investigating rnn-based speech enhancement methods for noiserobust
text-to-speech.” in SSW, 2016, pp. 146–152.

[17] S. Kelkar, L. Grigsby, and J. Langsner, “An extension of parseval’s
theorem and its use in calculating transient energy in the
frequency domain,” IEEE Transactions on Industrial Electronics,
no. 1, pp. 42–45, 1983.

[18] C. Trabelsi, O. Bilaniuk, Y. Zhang, D. Serdyuk, S. Subramanian,
J. F. Santos, S. Mehri, N. Rostamzadeh, Y. Bengio, and C. J.
Pal, “Deep complex networks,” in 6th International Conference
on Learning Representations, ICLR 2018.

[19] B. Xu, N. Wang, T. Chen, and M. Li, “Empirical evaluation
of rectified activations in convolutional network,” arXiv preprint
arXiv:1505.00853, 2015.

[20] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra,
“Perceptual evaluation of speech quality (pesq)-a new method for
speech quality assessment of telephone networks and codecs,” in
2001 IEEE International Conference on Acoustics, Speech, and
Signal Processing. Proceedings (Cat. No.01CH37221), vol. 2,
2001, pp. 749–752 vol.2.

[21] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A shorttime
objective intelligibility measure for time-frequency weighted
noisy speech,” in 2010 IEEE International Conference on Acoustics,
Speech and Signal Processing, 2010, pp. 4214–4217.

[22] M. Zhou, T. Liu, Y. Li, D. Lin, E. Zhou, and T. Zhao, “Toward understanding
the importance of noise in training neural networks,”
in Proceedings of the 36th International Conference on Machine
Learning, ser. Proceedings of Machine Learning Research,
K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 09–15
Jun 2019, pp. 7594–7602.

[23] P. Ndajah, H. Kikuchi, M. Yukawa, H. Watanabe, and S. Muramatsu,
“An investigation on the quality of denoised images,” International
Journal of Circuits, Systems and Signal Processing,
vol. 5, no. 4, pp. 423–434, Oct. 2011.
