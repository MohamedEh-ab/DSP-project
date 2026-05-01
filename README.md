# DSP Audio Encoder and Player

A Digital Signal Processing (DSP) project that demonstrates audio denoising and compression using spectral analysis techniques. The project includes an interactive GUI player built with Pygame and an exploratory Jupyter notebook for step-by-step analysis and visualization.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A WAV audio file to process

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MohamedEh-ab/DSP-project.git
   cd DSP-project
   ```

2. **Install dependencies for the GUI app**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies for the Jupyter notebook**

   ```bash
   pip install librosa soundfile matplotlib pandas jupyter
   ```

### Running the GUI Application

```bash
python audio_app.py
```

### Running the Jupyter Notebook

```bash
jupyter notebook audio_encoder.ipynb
```

Place your input WAV file at `Input/rec.wav` before running the notebook. Processed outputs will be saved to the `Output/` directory.

---

## Project Overview

This project focuses on the following DSP pipeline applied to WAV audio files:

1. **Loading** — Read a WAV file and convert it to a normalized float array.
2. **Denoising** — Remove background noise using spectral subtraction followed by temporal median filtering to suppress musical noise artifacts.
3. **Compression** — Compress the denoised audio in the frequency domain by splitting it into sub-bands, quantizing each band, and encoding it with Run-Length Encoding (RLE). The audio is then decoded and reconstructed via the Inverse STFT.
4. **Playback** — Listen to the original, denoised, and reconstructed audio signals side by side.

---

## Features

### GUI Application (`audio_app.py`)

- **Load WAV** — Open any WAV file through a file dialog.
- **Process** — Run the full DSP pipeline (denoising + compression) with a single click.
- **Play Original** — Play back the raw input audio.
- **Play Denoised** — Play back the spectrally denoised audio.
- **Play Reconstructed** — Play back the compressed-and-reconstructed audio.
- **Mute** — Stop playback at any time.
- **Waveform Visualizer** — Live animated waveform display that tracks playback position.
- **Seek Slider** — Click or drag the progress bar to jump to any point in the audio.

### Jupyter Notebook (`audio_encoder.ipynb`)

- Load and visualize raw audio waveforms.
- Apply programmable silence regions to simulate missing segments.
- Step-by-step denoising: spectral subtraction → temporal median filtering.
- Full compression pipeline: STFT → band splitting → quantization → RLE encoding → decoding → dequantization → Inverse STFT.
- Compute and display **SNR (Signal-to-Noise Ratio)** and **Compression Ratio**.
- In-notebook audio playback for comparing original, denoised, and compressed signals.
- Export processed audio to `Output/denoised.wav` and `Output/compressed.wav`.

---

## DSP Techniques Used

| Technique | Description |
|---|---|
| **STFT / Inverse STFT** | Short-Time Fourier Transform for time-frequency analysis and reconstruction |
| **Spectral Subtraction** | Estimate and subtract noise profile from each frequency bin |
| **Temporal Median Filtering** | Smooth spectral frames over time to eliminate musical noise |
| **Sub-band Quantization** | Divide the spectrum into 8 frequency bands and apply perceptual quantization steps |
| **Run-Length Encoding (RLE)** | Lossless encoding of repeated quantized values to achieve compression |

---

## Project Structure

```
DSP-project/
├── audio_app.py          # Pygame GUI application
├── audio_encoder.ipynb   # Jupyter notebook for analysis and visualization
├── requirements.txt      # Python dependencies for the GUI app
├── Input/
│   └── rec.wav           # Input audio file
└── Output/
    ├── denoised.wav      # Denoised audio output
    └── compressed.wav    # Compressed and reconstructed audio output
```

---

## Dependencies

### GUI App (`requirements.txt`)

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| pygame | 2.6.1 |
| scipy | 1.17.1 |

### Jupyter Notebook (additional)

| Package |
|---|
| librosa |
| soundfile |
| matplotlib |
| pandas |
| jupyter |
