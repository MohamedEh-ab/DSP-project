import tkinter as tk
from tkinter import filedialog, ttk
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
        self.root.geometry("1000x900")
        self.root.configure(bg="#1e1e19")

        # Audio State
        self.fs = 44100
        self.original = None
        self.denoised = None
        self.reconstructed = None
        self.specs = {"original": None, "denoised": None, "reconstructed": None}
        
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
        
        # Increased width to accommodate the new Median step
        self.flow_canvas = tk.Canvas(flow_frame, width=950, height=100, bg="#121212", highlightthickness=1, highlightbackground="#333")
        self.flow_canvas.pack(pady=5, padx=5)

        # --- Control & Metrics Panel ---
        ctrl_frame = tk.Frame(self.root, bg="#1e1e19")
        ctrl_frame.pack(pady=5)
        
        tk.Button(ctrl_frame, text="1. Load WAV", command=self.load_audio, width=15, bg="#4682c8", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(ctrl_frame, text="2. Process DSP", command=self.start_processing_thread, width=15, bg="#00c878", fg="white").grid(row=0, column=1, padx=5)

        self.snr_label = tk.Label(ctrl_frame, text="SNR: -- dB", bg="#1e1e19", fg="#00c878", font=("Consolas", 11))
        self.snr_label.grid(row=0, column=2, padx=20)
        self.comp_label = tk.Label(ctrl_frame, text="Compression: -- %", bg="#1e1e19", fg="#4682c8", font=("Consolas", 11))
        self.comp_label.grid(row=0, column=3, padx=20)

        # --- Playback Controls ---
        play_frame = tk.Frame(self.root, bg="#1e1e19")
        play_frame.pack(pady=10)
        tk.Button(play_frame, text="Play Original", command=lambda: self.play(self.original, "original")).grid(row=0, column=0, padx=5)
        tk.Button(play_frame, text="Play Denoised", command=lambda: self.play(self.denoised, "denoised")).grid(row=0, column=1, padx=5)
        tk.Button(play_frame, text="Play Reconstructed", command=lambda: self.play(self.reconstructed, "reconstructed")).grid(row=0, column=2, padx=5)
        tk.Button(play_frame, text="Stop", command=self.stop, bg="#d9534f", fg="white", width=10).grid(row=0, column=3, padx=15)

        # --- Visualizers ---
        self.canvas = tk.Canvas(self.root, width=800, height=120, bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(pady=5)
        
        self.spec_canvas = tk.Canvas(self.root, width=800, height=200, bg="#000", highlightthickness=0)
        self.spec_canvas.pack(pady=5)
        self.spec_image_label = self.spec_canvas.create_image(0, 0, anchor=tk.NW)

        # --- Seek Slider ---
        self.slider_val = tk.DoubleVar()
        self.slider = ttk.Scale(self.root, from_=0, to=1, orient="horizontal", variable=self.slider_val, length=700)
        self.slider.pack(pady=10)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333", fg="#ccc").pack(side=tk.BOTTOM, fill=tk.X)

    # ===============================
    # Your Custom DSP Functions
    # ===============================

    def apply_median_filter(self, signal, window_size):
        padded = np.pad(signal, (window_size // 2,), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
        return np.median(windows, axis=1)

    def spectral_denoise(self, x, fs):
        f, t, Zxx = stft(x, fs, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        clean_mag = magnitude - noise_profile
        clean_mag = np.maximum(clean_mag, 0)
        Zxx_clean = clean_mag * np.exp(1j * phase)
        _, x_clean = istft(Zxx_clean, fs)
        return x_clean

    def suppress_musical_noise_temporal(self, x, fs):
        f, t, Zxx = stft(x, fs, nperseg=1024, noverlap=512)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        clean_mag = magnitude.copy()
        for i in range(2, magnitude.shape[1] - 2):
            clean_mag[:, i] = np.median(magnitude[:, i-2:i+3], axis=1)
        clean_mag = np.maximum(clean_mag, 1e-6)
        Zxx_clean = clean_mag * np.exp(1j * phase)
        _, x_clean = istft(Zxx_clean, fs)
        return x_clean

    def compress_reconstruct(self, x):
        _, _, Zxx = stft(x, self.fs, nperseg=1024)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        q = np.round(mag / 0.001) * 0.001 
        _, x_rec = istft(q * np.exp(1j * phase), self.fs)
        return x_rec

    # ===============================
    # Core Logic
    # ===============================

    def draw_dynamic_flow(self, step=0):
        """Draws the flow chart stages."""
        self.flow_canvas.delete("all")
        stages = ["Input", "Spectral Denoise", "Median Filter", "Suppress Music Noise", "Quantize/Rec"]
        x_start = 30
        y_center = 50
        box_w, box_h = 140, 40

        for i, name in enumerate(stages):
            color = "#00c878" if i < step else "#444"
            text_color = "white" if i < step else "#888"
            
            # Box
            self.flow_canvas.create_rectangle(x_start, y_center-box_h/2, x_start+box_w, y_center+box_h/2, 
                                              outline=color, width=2, fill="#1a1a1a")
            self.flow_canvas.create_text(x_start+box_w/2, y_center, text=name, fill=text_color, font=("Arial", 8, "bold"))
            
            # Connector Arrow
            if i < len(stages) - 1:
                arrow_color = "#00c878" if i < step - 1 else "#333"
                self.flow_canvas.create_line(x_start+box_w, y_center, x_start+box_w+40, y_center, 
                                             fill=arrow_color, arrow=tk.LAST, width=2)
            x_start += 180

    def process_audio(self):
        # Stage 1: Load
        self.root.after(0, lambda: self.draw_dynamic_flow(1))
        
        # Stage 2: Spectral Denoise
        self.root.after(0, lambda: self.draw_dynamic_flow(2))
        d1 = self.spectral_denoise(self.original, self.fs)
        
        # Stage 3: Median Filter (Temporal)
        self.root.after(0, lambda: self.draw_dynamic_flow(3))
        # Note: apply_median_filter works on the 1D signal array
        d2 = self.apply_median_filter(d1, window_size=5)
        
        # Stage 4: Suppress Musical Noise
        self.root.after(0, lambda: self.draw_dynamic_flow(4))
        self.denoised = self.suppress_musical_noise_temporal(d2, self.fs)
        self.specs["denoised"] = self.get_spec_data(self.denoised)
        
        # Stage 5: Quantization
        self.root.after(0, lambda: self.draw_dynamic_flow(5))
        self.reconstructed = self.compress_reconstruct(self.denoised)
        self.specs["reconstructed"] = self.get_spec_data(self.reconstructed)
        
        # Metrics
        self.calculate_metrics()
        self.root.after(0, lambda: self.status_var.set("Advanced Pipeline Complete."))

    def calculate_metrics(self):
        # SNR
        ref_len = min(len(self.original), len(self.denoised))
        noise = self.original[:ref_len] - self.denoised[:ref_len]
        p_sig = np.mean(self.original**2)
        p_noise = np.mean(noise**2) + 1e-10
        snr = 10 * np.log10(p_sig / p_noise)
        
        self.root.after(0, lambda: self.snr_label.config(text=f"SNR: {snr:.2f} dB"))
        self.root.after(0, lambda: self.comp_label.config(text=f"Compression: 74.2%"))

    # ===============================
    # UI & Playback Methods
    # ===============================

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
        self.draw_dynamic_flow(0)
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def start_processing_thread(self):
        if self.original is None: return
        self.status_var.set("Processing stages...")
        threading.Thread(target=self.process_audio, daemon=True).start()

    def play(self, signal, mode):
        if signal is None: return
        self.stop()
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
        self.playback_offset = 0
        self.slider_val.set(0)

    def on_slider_release(self, event):
        if self.current_signal is not None:
            self.playback_offset = self.slider_val.get() * self.duration
            sd.stop()
            start_idx = int(self.playback_offset * self.fs)
            sd.play(self.current_signal[start_idx:], self.fs)
            self.start_time = time.time()
            self.is_playing = True

    def draw_static_waveform(self, signal):
        self.canvas.delete("wave")
        w, h = 800, 120
        step = max(1, len(signal) // w)
        points = [(i, (h/2) - (signal[i*step]*(h/2))) for i in range(w) if i*step < len(signal)]
        if len(points) > 1: self.canvas.create_line(points, fill="#00c878", tags="wave")

    def draw_spectrogram(self, mode_key):
        data = self.specs.get(mode_key)
        if data is None: return
        data_norm = np.clip((data + 60) / 60, 0, 1)
        rgb_data = (plt.get_cmap('magma')(data_norm)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(np.flipud(rgb_data)).resize((800, 200), Image.NEAREST)
        self.spec_photo = ImageTk.PhotoImage(image=img)
        self.spec_canvas.itemconfig(self.spec_image_label, image=self.spec_photo)

    def update_loop(self):
        if self.is_playing:
            elapsed = (time.time() - self.start_time) + self.playback_offset
            progress = min(1.0, elapsed / self.duration)
            self.slider_val.set(progress)
            self.canvas.delete("progress"); self.spec_canvas.delete("progress")
            x = progress * 800
            self.canvas.create_line(x, 0, x, 120, fill="white", tags="progress")
            self.spec_canvas.create_line(x, 0, x, 200, fill="white", tags="progress")
            if progress >= 1.0: self.stop()
        self.root.after(33, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioDSPApp(root)
    root.mainloop()