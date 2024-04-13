# Speech Denoising Using Noise2Noise: 
Welcome to our project repository! This project showcases the development of an AI-driven speech denoising system, specifically tailored for high-noise environments like military communications. Utilizing the innovative Noise2Noise approach, our system is designed to improve audio quality by filtering out noise without the need for clean reference data. This README provides an overview of our project, including the goals, methodology, and significant outcomes. Developed as part of a capstone project by students at Braude College, this system leverages deep learning techniques to enhance speaker recognition capabilities, making it particularly useful for defense applications where clarity and accuracy are paramount.

## Python Requirements
We recommend using Python 3.8.8. The package versions are in requirements.txt.
Install required Python libraries using the requirements.txt provided. Run:
```
pip install -r requirements.txt 
```

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

## How to Run the Denoiser GUI (Inference)

Follow these steps to run the Denoiser GUI:

1. Make sure you have Python installed on your system. If not, you can install it using your package manager. For example, on Ubuntu, you can use `sudo apt install python3`.

2. Install the required dependencies by running the following command in your terminal:
     ```sh
     pip install tkinter scipy torch
     ```

3. Navigate to the directory where the `denoiser_gui.py` file is located. You can use the `cd` command to change directories. For example:
   ```sh
   cd /path/to/directory/
   ```
4. Run the following command to start the GUI:
   ```sh
   python denoiser_gui.py
   ```
5. The GUI window will open. Click on the "Select Folder" button to choose the folder containing the audio files you want to denoise.
6. After selecting the folder, click on the "Denoise" button to start the denoising process. The denoised audio files will be saved in a subdirectory named "Results" within the selected folder.
7. The status label in the GUI will display the progress and completion of the denoising process.
8. You can repeat steps 5-7 to denoise additional folders to denoise.

>Note: Make sure you have all the necessary dependencies installed, such as tkinter, scipy, and torch. If any dependencies are missing, you may need to install them using a package manager like pip.

## How to run denoiser without GUI (Inference)
1. **Prepare Audio Files:**  
   Place the noisy audio files you want to denoise into the following directory:
   ```sh
   Samples/Sample_Test_Input
   ```

   The matching target clean audio files should be placed in the following directory:
   ```sh
   Samples/Sample_Test_Target
   ```

   e.g: for `Samples/Sample_Test_Input/audio_1.wav` the target file should be `Samples/Sample_Test_Target/audio_1.wav`

2. **Configure Model and Script Settings:**  
Ensure the `main.py` script is configured to use the correct path for the trained model weights. Update the script or a config file if necessary to include:
   ```sh
   model_weights_path = 'path/to/trained_model_weights.pth' 
   ```

3. **Run the Denoising Process:**  
   Execute the main.py script to start the denoising process.  
   Run:
   ```sh
   python main.py 
   ```
   This will load your model, process each audio file in the input directory, and save the denoised audio to an output directory specified in the script.

4. **Check the Denoised Audio Files:**  
   Go to the output directory set in main.py to review the denoised files.  
   The directory is set to the following:
   ```sh
   Samples/Sample_Test_Target
   ```
   The clean audio files will be saved to the following directory:
   ```sh
   Samples/Results/
   ```

   For example: after denoising `Samples/Sample_Test_Input/audio_1.wav` the denoised file will be saved in the directory `Samples/Results/audio_1` which will contain the following:
   - `noisy.wav`
   - `denoised.wav`
   - `clean.wav`
   - `Waveform.png`


By following these steps, you should be able to use the denoising model to clean your noisy audio files effectively. Make sure each configuration path and setting is correctly specified in your scripts to avoid any issues.

## Contact Information
Free to contact us with any problem or suggestion.
Omri Cohen: Omri.cohen2@e.braude.ac.il
Eliav Shabat: Eliav.shabat@e.braude.ac.il