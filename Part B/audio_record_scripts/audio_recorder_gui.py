import tkinter as tk
from tkinter import messagebox
from audio_recorder import record_audio, split_audio


class AudioRecorderGUI:
    def __init__(self, master):
        master.title("Audio Recorder")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        self.record_duration_var = tk.StringVar()
        self.split_interval_var = tk.StringVar()
        self.label_prefix_var = tk.StringVar()

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
            master, text="Start Recording", command=self.start_recording)
        self.stop_button = tk.Button(
            master, text="Stop & Process", command=self.stop_recording, state=tk.DISABLED)

        self.start_button.grid(row=3, column=0)
        self.stop_button.grid(row=3, column=1)

        self.status_label = tk.Label(master, textvariable=self.status_var)
        self.status_label.grid(row=4)

    def start_recording(self):
        self.status_var.set("Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        record_seconds = int(self.record_duration_var.get())
        record_audio(f"output.wav", record_seconds)

    def stop_recording(self):
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
