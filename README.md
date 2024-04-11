# Speech Denoising Using Noise2Noise: 
Welcome to our project repository! This project showcases the development of an AI-driven speech denoising system, specifically tailored for high-noise environments like military communications. Utilizing the innovative Noise2Noise approach, our system is designed to improve audio quality by filtering out noise without the need for clean reference data. This README provides an overview of our project, including the goals, methodology, and significant outcomes. Developed as part of a capstone project by students at Braude College, this system leverages deep learning techniques to enhance speaker recognition capabilities, making it particularly useful for defense applications where clarity and accuracy are paramount.

## Python Requirements
We recommend using Python 3.8.8. The package versions are in requirements.txt.
Install required Python libraries using the requirements.txt provided. Run:
```
pip install -r requirements.txt 
```

## Prepare Audio Files
Place the noisy audio files you want to denoise into the following directory:
```
Samples/Sample_Test_Input
```

## Configure Model and Script Settings
Ensure the `main.py` script is configured to use the correct path for the trained model weights. Update the script or a config file if necessary to include:
```
model_weights_path = 'path/to/trained_model_weights.pth' 
```

## Run the Denoising Process
Execute the main.py script to start the denoising process.  
Run:
```
python main.py 
```
This will load your model, process each audio file in the input directory, and save the denoised audio to an output directory specified in the script.

## Check the Denoised Audio Files
Go to the output directory set in main.py to review the denoised files.  
The directory is set to the following:
```
Samples/Sample_Test_Target
```

By following these steps, you should be able to use the denoising model to clean your noisy audio files effectively. Make sure each configuration path and setting is correctly specified in your scripts to avoid any issues.

## Contact Information
Free to contact us with any problem or suggestion.
Omri Cohen: Omri.cohen2@e.braude.ac.il
Eliav Shabat: Eliav.shabat@e.braude.ac.il


## Dataset Generation
Your git project file should contain the folder 'Dataset' (you need to create it) and place your datasets inside of it as follows:
```
DENOISING_PROJECT-23-2-R-10
|     README.md
|     part_a
|_____part_b
      │_____Datasets
            |_______trainset_input
            |_______trainset_target
            |_______testset_input
                    |_______input1.wav
                    |_______input1.wav
                            ...
                    |_______input100.wav        
            |_______testset_clean
                    |_______clean1.wav
                    |_______clean1.wav
                            ...
                    |_______clean100.wav        
                            
                        

```

## Python scripts for pre-processing and datasets generation:
Our methodology included the creation of Python scripts for data pre-processing, enabling us to prepare the audio samples for training and generate new, denoised samples from each audio file. Specifically, we developed:
- _**audio_recorder.py**_: A script for recording and saving audio files, crucial for accumulating raw data.
- _**audio_recorder_gui.py**_: An interface script providing a graphical user interface for the audio recorder, enhancing user interaction.
- _**generate_file_to_merge_list.py**_: This script was designed to prepare lists of audio files for merging, facilitating batch processing.
- _**get_snr_spectrogram_analysis.py**_: A utility for analyzing signal-to-noise ratio (SNR) and spectrograms, essential for evaluating audio quality.
- _**split_audio_files.py**_: A tool to split audio files into smaller segments, aiding in the creation of a more diverse dataset.
