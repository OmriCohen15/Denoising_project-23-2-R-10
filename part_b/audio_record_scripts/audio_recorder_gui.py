import tkinter as tk
from tkinter import messagebox
import threading
from audio_recorder import record_audio, split_audio


class AudioRecorderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Audio Recorder")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        self.record_duration_var = tk.StringVar()
        self.split_interval_var = tk.StringVar()
        self.label_prefix_var = tk.StringVar()
        self.recording_stop_flag = False

        tk.Label(master, text="Record Duration (sec):").grid(row=0)
        tk.Entry(master, textvariable=self.record_duration_var).grid(
            row=0, column=1)

        tk.Label(master, text="Split Interval (sec):").grid(row=1)
        tk.Entry(master, textvariable=self.split_interval_var).grid(
            row=1, column=1)

        tk.Label(master, text="Label Prefix:").grid(row=2)
        tk.Entry(master, textvariable=self.label_prefix_var).grid(
            row=2, column=1)

        self.start_button = tk.Button(
            master, text="Start Recording", command=self.start_recording_threaded)
        self.stop_button = tk.Button(
            master, text="Stop & Process", command=self.stop_recording, state=tk.DISABLED)

        self.start_button.grid(row=3, column=0)
        self.stop_button.grid(row=3, column=1)

        self.status_label = tk.Label(master, textvariable=self.status_var)
        self.status_label.grid(row=4)

        # Create a new label for the timer
        self.timer_label = tk.Label(master, text="00:00:00")
        self.timer_label.grid(row=5, columnspan=2)

    # Creating a string with the current time in the format 'hh:mm:ss'
    def get_time_string(self, seconds):
        """
        Converts a time duration from seconds to a formatted string.
        This method takes an integer representing a time duration in seconds and converts it into a string formatted as 'hh:mm:ss'. It handles the conversion by dividing the total seconds into hours, minutes, and remaining seconds.
        Parameters:
        - seconds (int): The time duration in seconds to be converted.

        Returns:
        - str: A string representing the formatted time duration as 'hh:mm:ss'.
        """

        # Divide seconds into minutes and remaining seconds
        minutes, seconds = divmod(seconds, 60)
        # Divide minutes into hours and remaining minutes
        hours, minutes = divmod(minutes, 60)
        # Format and return the time string
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    def update_timer(self):
        """
        Updates the timer label in the GUI every second.

        This method increments the timer displayed on the GUI by one second at a time. It first checks if the recording_stop_flag is False, indicating that recording is ongoing. It then parses the current time from the timer_label, increments it by one second, and handles overflow by incrementing minutes and hours as necessary. Finally, it updates the timer_label with the new time and schedules itself to be called again after 1000 milliseconds (1 second) if recording has not been stopped.
        """
        if self.recording_stop_flag == False:
            current_time = list(
                map(int, self.timer_label.cget('text').split(':')))
            current_time[2] += 1  # Increment seconds
            if current_time[2] >= 60:  # Handle minute overflow
                current_time[2] = 0
                current_time[1] += 1
            if current_time[1] >= 60:  # Handle hour overflow
                current_time[1] = 0
                current_time[0] += 1
            self.timer_label.config(
                text=f'{current_time[0]:02d}:{current_time[1]:02d}:{current_time[2]:02d}')
            # Schedule the next update
            self.master.after(1000, self.update_timer)

    def start_recording_threaded(self):
        """
        Starts the recording process in a separate thread to keep the GUI responsive.

        This method initializes and starts a new thread dedicated to handling the audio recording process. It ensures that the main GUI thread remains responsive and can handle user input, such as stopping the recording. Additionally, it resets the recording stop flag to False and initiates the timer update process to visually indicate the recording duration to the user.
        """
        # Initialize the recording thread
        recording_thread = threading.Thread(target=self.start_recording)
        # Start the recording thread
        recording_thread.start()

        # Reset the stop flag to allow recording
        self.recording_stop_flag = False
        # Start the timer to update the recording duration on the GUI
        self.update_timer()

    def start_recording(self):
        """
        Initiates the audio recording process, updates the GUI to reflect the recording state, and handles the recording logic.

        This method sets the application status to "Recording...", disables the 'Start Recording' button to prevent multiple recordings at the same time, and enables the 'Stop & Process' button to allow the user to end the recording. It retrieves the desired recording duration from the GUI, starts the recording process in a separate thread, and upon completion, triggers the stop_recording method to process the recorded audio.
        """
        self.status_var.set("Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Get recording duration from user input
        record_seconds = int(self.record_duration_var.get())
        # Start recording audio for the specified duration
        record_audio(f"output.wav", record_seconds)

        self.recording_stop_flag = True  # Signal to stop updating the timer
        # Disable the 'Stop & Process' button
        self.stop_button.config(state=tk.DISABLED)
        self.stop_recording()  # Process the recorded audio

        # Reset the timer display
        self.timer_label.config(text=self.get_time_string(0))
        # Re-enable the 'Start Recording' button
        self.start_button.config(state=tk.NORMAL)

    def stop_recording(self):
        """
        Handles the stopping of the audio recording process, updates the GUI to reflect the processing state, and initiates audio splitting.

        This method sets the recording stop flag to True to halt the timer update, resets the timer display, updates the application status to "Processing Splitting...", and disables the 'Start Recording' button to prevent new recordings during processing. It retrieves the split interval and label prefix from the GUI, calls the split_audio function to process the recorded audio, and upon completion, updates the GUI to indicate readiness for a new recording session.
        """
        # Signal to halt the timer update
        self.recording_stop_flag = True

        # Reset the timer display
        self.timer_label.config(text=self.get_time_string(0))

        # Update application status
        self.status_var.set("Processing Splitting...")

        # Re-enable the 'Start Recording' button
        self.start_button.config(state=tk.NORMAL)

        # Ensure the 'Stop & Process' button is disabled
        self.stop_button.config(state=tk.DISABLED)

        # Get split interval from user input
        interval_seconds = int(self.split_interval_var.get())
        # Get label prefix from user input
        label_prefix = self.label_prefix_var.get()
        # Process the recorded audio
        split_audio("output.wav", interval_seconds, label_prefix)

        # Update application status to indicate readiness
        self.status_var.set("Ready")
        # Show completion message
        messagebox.showinfo("Process Complete",
                            "Recording and splitting process completed.")


root = tk.Tk()
app = AudioRecorderGUI(root)
root.mainloop()
