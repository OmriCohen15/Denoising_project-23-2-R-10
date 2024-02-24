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
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    def update_timer(self):
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
            self.master.after(1000, self.update_timer)

    def start_recording_threaded(self):
        recording_thread = threading.Thread(target=self.start_recording)
        recording_thread.start()

        self.recording_stop_flag = False
        # Start the timer
        self.update_timer()

    def start_recording(self):
        self.status_var.set("Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Start the timer when the 'Start Recording' button is clicked
        record_seconds = int(self.record_duration_var.get())
        record_audio(f"output.wav", record_seconds)

        # after record raw data is done - split the recorded audio
        self.recording_stop_flag = True  # stop updating the timer
        # disable the 'Stop & Process' button
        self.stop_button.config(state=tk.DISABLED)
        self.stop_recording()  # stop recording and split the audio

        # set the timer back to 00:00:00
        self.timer_label.config(text=self.get_time_string(0))
        # self.status_var.set("Ready")
        # enable the 'Start Recording' button
        self.start_button.config(state=tk.NORMAL)

    def stop_recording(self):
        self.recording_stop_flag = True
        self.timer_label.config(text=self.get_time_string(0))
        self.status_var.set("Processing Splitting...")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        interval_seconds = int(self.split_interval_var.get())
        label_prefix = self.label_prefix_var.get()
        split_audio("output.wav", interval_seconds, label_prefix)

        self.status_var.set("Ready")
        messagebox.showinfo("Process Complete",
                            "Recording and splitting process completed.")


root = tk.Tk()
app = AudioRecorderGUI(root)
root.mainloop()
