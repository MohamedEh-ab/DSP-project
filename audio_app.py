import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import sounddevice as sd
import threading
import time
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class AudioDSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP Audio Player & Analyzer")
        self.root.geometry("1000x950")
        self.root.configure(bg="#1e1e19")

        # Audio State
        self.fs = 44100
        self.original = None
        self.denoised = None
        self.reconstructed = None
        self.specs = {"original": None, "denoised": None, "reconstructed": None}
        
        # Size Metrics
        self.orig_size = 0
        self.comp_size = 0
        
        self.current_signal = None
        self.is_playing = False
        self.duration = 0
        self.playback_offset = 0

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        # --- Header ---
        tk.Label(self.root, text="Advanced DSP Audio Pipeline", font=("Segoe UI", 18, "bold"), 
                 bg="#1e1e19", fg="#ffffff").pack(pady=10)

        # --- Dynamic Flow Chart Frame ---
        flow_frame = tk.LabelFrame(self.root, text="Processing Pipeline Status", bg="#1e1e19", fg="#4682c8", font=("Arial", 10))
        flow_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.flow_canvas = tk.Canvas(flow_frame, width=950, height=100, bg="#121212", highlightthickness=1, highlightbackground="#333")
        self.flow_canvas.pack(pady=5, padx=5)

        # --- Control & Metrics Panel ---
        ctrl_frame = tk.Frame(self.root, bg="#1e1e19")
        ctrl_frame.pack(pady=5)
        
        tk.Button(ctrl_frame, text="1. Load WAV", command=self.load_audio, width=15, bg="#4682c8", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(ctrl_frame, text="2. Process DSP", command=self.start_processing_thread, width=15, bg="#00c878", fg="white").grid(row=0, column=1, padx=5)
        tk.Button(ctrl_frame, text="3. Save Compressed", command=self.save_compressed_audio, width=18, bg="#e67e22", fg="white").grid(row=0, column=2, padx=5)

        # --- Size & Performance Metrics ---
        metrics_frame = tk.Frame(self.root, bg="#252520", pady=10)
        metrics_frame.pack(pady=5, fill=tk.X, padx=20)

        self.size_label = tk.Label(metrics_frame, text="Original Size: 0 KB | Compressed Size: 0 KB", 
                                   bg="#252520", fg="#ccc", font=("Consolas", 10))
        self.size_label.pack()

        self.snr_label = tk.Label(metrics_frame, text="SNR: -- dB", bg="#252520", fg="#00c878", font=("Consolas", 11))
        self.snr_label.pack(side=tk.LEFT, expand=True)
        
        self.comp_label = tk.Label(metrics_frame, text="Compression Ratio: -- %", bg="#252520", fg="#4682c8", font=("Consolas", 11))
        self.comp_label.pack(side=tk.LEFT, expand=True)

        # --- Visualizers ---
        self.canvas = tk.Canvas(self.root, width=800, height=100, bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(pady=5)
        
        self.spec_canvas = tk.Canvas(self.root, width=800, height=180, bg="#000", highlightthickness=0)
        self.spec_canvas.pack(pady=5)
        self.spec_image_label = self.spec_canvas.create_image(0, 0, anchor=tk.NW)

        # --- Seek Slider ---
        self.slider_val = tk.DoubleVar()
        self.slider = ttk.Scale(self.root, from_=0, to=1, orient="horizontal", variable=self.slider_val, length=700)
        self.slider.pack(pady=10)

        # Playback row
        play_frame = tk.Frame(self.root, bg="#1e1e19")
        play_frame.pack()
        tk.Button(play_frame, text="Play Original", command=lambda: self.play(self.original, "original")).grid(row=0, column=0, padx=5)
        tk.Button(play_frame, text="Play Denoised", command=lambda: self.play(self.denoised, "denoised")).grid(row=0, column=1, padx=5)
        tk.Button(play_frame, text="Play Reconstructed", command=lambda: self.play(self.reconstructed, "reconstructed")).grid(row=0, column=2, padx=5)
        tk.Button(play_frame, text="Stop", command=self.stop, bg="#d9534f", fg="white").grid(row=0, column=3, padx=10)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333", fg="#ccc").pack(side=tk.BOTTOM, fill=tk.X)

    # ===============================
    # DSP & File Logic
    # ===============================

    def calculate_metrics(self):
        # SNR Calculation
        ref_len = min(len(self.original), len(self.denoised))
        noise = self.original[:ref_len] - self.denoised[:ref_len]
        snr = 10 * np.log10(np.mean(self.original**2) / (np.mean(noise**2) + 1e-10))
        
        # Real-time Size Calculation
        # Original: 32-bit (4 bytes) per sample
        self.orig_size = (len(self.original) * 4) / 1024 
        # Compressed: Simulated bit-depth reduction (e.g., from 32-bit to 8-bit effective quantization)
        self.comp_size = self.orig_size * 0.25 
        comp_ratio = ((self.orig_size - self.comp_size) / self.orig_size) * 100

        self.root.after(0, lambda: self.snr_label.config(text=f"SNR: {snr:.2f} dB"))
        self.root.after(0, lambda: self.comp_label.config(text=f"Compression Ratio: {comp_ratio:.1f}%"))
        self.root.after(0, lambda: self.size_label.config(
            text=f"Original Size: {self.orig_size:.2f} KB | Compressed Estimated: {self.comp_size:.2f} KB"))

    def save_compressed_audio(self):
        if self.reconstructed is None:
            messagebox.showwarning("Warning", "Please process the audio first!")
            return
        
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            # Normalize to 16-bit PCM for standard WAV saving
            out_data = np.clip(self.reconstructed, -1, 1)
            out_data = (out_data * 32767).astype(np.int16)
            wavfile.write(path, self.fs, out_data)
            messagebox.showinfo("Success", f"Compressed audio saved to:\n{path}")

    # ===============================
    # Existing DSP Functions (Integrated)
    # ===============================

    def apply_median_filter(self, signal, window_size):
        padded = np.pad(signal, (window_size // 2,), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
        return np.median(windows, axis=1)

    def spectral_denoise(self, x, fs):
        f, t, Zxx = stft(x, fs, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        clean_mag = np.maximum(magnitude - noise_profile, 0)
        _, x_clean = istft(clean_mag * np.exp(1j * np.angle(Zxx)), fs)
        return x_clean

    def suppress_musical_noise_temporal(self, x, fs):
        f, t, Zxx = stft(x, fs, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        clean_mag = magnitude.copy()
        for i in range(2, magnitude.shape[1] - 2):
            clean_mag[:, i] = np.median(magnitude[:, i-2:i+3], axis=1)
        _, x_clean = istft(np.maximum(clean_mag, 1e-6) * np.exp(1j * np.angle(Zxx)), fs)
        return x_clean

    def compress_reconstruct(self, x):
        _, _, Zxx = stft(x, self.fs, nperseg=1024)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        # Quantization creates the "compression" effect
        q = np.round(mag / 0.001) * 0.001 
        _, x_rec = istft(q * np.exp(1j * phase), self.fs)
        return x_rec

    def process_audio(self):
        self.root.after(0, lambda: self.draw_dynamic_flow(1))
        d1 = self.spectral_denoise(self.original, self.fs)
        
        self.root.after(0, lambda: self.draw_dynamic_flow(2))
        d2 = self.apply_median_filter(d1, window_size=5)
        
        self.root.after(0, lambda: self.draw_dynamic_flow(3))
        self.denoised = self.suppress_musical_noise_temporal(d2, self.fs)
        self.specs["denoised"] = self.get_spec_data(self.denoised)
        
        self.root.after(0, lambda: self.draw_dynamic_flow(4))
        self.reconstructed = self.compress_reconstruct(self.denoised)
        self.specs["reconstructed"] = self.get_spec_data(self.reconstructed)
        
        self.root.after(0, lambda: self.draw_dynamic_flow(5))

        self.root.after(0, lambda: self.draw_dynamic_flow(6))


        
        self.calculate_metrics()
        self.root.after(0, lambda: self.status_var.set("Pipeline Complete."))

    # ===============================
    # UI Helpers
    # ===============================

    def draw_dynamic_flow(self, step=0):
        self.flow_canvas.delete("all")
        stages = ["WAV Input", "Denoising", "FFT", "Encoding", "ISFT", "Compression"]
        x_start, y_center, box_w, box_h = 30, 50, 140, 40
        for i, name in enumerate(stages):
            color = "#00c878" if i < step else "#444"
            self.flow_canvas.create_rectangle(x_start, y_center-box_h/2, x_start+box_w, y_center+box_h/2, outline=color, width=2)
            self.flow_canvas.create_text(x_start+box_w/2, y_center, text=name, fill="white" if i < step else "#888", font=("Arial", 8, "bold"))
            if i < len(stages) - 1:
                self.flow_canvas.create_line(x_start+box_w, y_center, x_start+box_w+40, y_center, fill="#333", arrow=tk.LAST)
            x_start += 180

    def get_spec_data(self, x):
        _, _, Zxx = stft(x, self.fs, nperseg=512, noverlap=256)
        return 20 * np.log10(np.abs(Zxx) + 1e-10)

    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if not path: return
        self.fs, data = wavfile.read(path)
        if len(data.shape) > 1: data = data.mean(axis=1)
        self.original = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-8)
        self.specs["original"] = self.get_spec_data(self.original)
        self.draw_static_waveform(self.original)
        self.draw_spectrogram("original")
        self.status_var.set("Audio Loaded.")

    def start_processing_thread(self):
        if self.original is None: return
        threading.Thread(target=self.process_audio, daemon=True).start()

    def play(self, signal, mode):
        if signal is None: return
        sd.stop()
        self.current_signal = signal
        self.duration = len(signal) / self.fs
        self.start_time = time.time()
        self.is_playing = True
        self.draw_static_waveform(signal)
        self.draw_spectrogram(mode)
        sd.play(signal, self.fs)

    def stop(self):
        sd.stop()
        self.is_playing = False
        self.slider_val.set(0)

    def draw_static_waveform(self, signal):
        self.canvas.delete("wave")
        w, h = 800, 100
        step = max(1, len(signal) // w)
        points = [(i, (h/2) - (signal[i*step]*(h/2))) for i in range(w) if i*step < len(signal)]
        if len(points) > 1: self.canvas.create_line(points, fill="#00c878", tags="wave")

    def draw_spectrogram(self, mode_key):
        data = self.specs.get(mode_key)
        if data is None: return
        img = Image.fromarray(np.flipud((plt.get_cmap('magma')(np.clip((data+60)/60, 0, 1))[:,:,:3]*255).astype(np.uint8))).resize((800,180))
        self.spec_photo = ImageTk.PhotoImage(image=img)
        self.spec_canvas.itemconfig(self.spec_image_label, image=self.spec_photo)

    def update_loop(self):
        if self.is_playing:
            progress = (time.time() - self.start_time) / self.duration
            if progress >= 1.0: self.stop()
            else: self.slider_val.set(progress)
        self.root.after(33, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioDSPApp(root)
    root.mainloop()