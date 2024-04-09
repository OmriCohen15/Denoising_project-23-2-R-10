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

