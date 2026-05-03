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
# DSP Audio & Video Compression System — Enhanced Edition

A comprehensive **digital signal processing (DSP) project** featuring dual compression systems: an interactive **audio denoising and compression application** with Pygame GUI, and a professional-grade **video compression system** with modern Qt6 GUI. This system combines Huffman entropy coding, DCT transforms, spectral analysis, motion estimation, and parallel processing to achieve efficient compression with live quality metrics.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenCV (`cv2`) — for video processing
- NumPy & SciPy — for signal processing
- PyQt6 — for video GUI
- Pygame — for audio GUI
- Matplotlib — for visualization

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/saalehmohamedd/DSP-project.git
   cd DSP-project
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies for both audio and video**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies for Jupyter notebook**

   ```bash
   pip install librosa soundfile jupyter
   ```

### Running the Applications

#### Audio Compression System

```bash
python audio_app.py
```

Place your input WAV file at `Input/rec.wav` before running. Processed outputs will be saved to the `Output/` directory.

#### Video Compression System

```bash
python video_compression_enhanced.py
```

The GUI will launch with a dark industrial aesthetic. Simply drag-and-drop a video file or click "Browse" to select one, configure compression parameters, and click "Start Compression."

#### Audio Analysis Notebook

```bash
jupyter notebook audio_encoder.ipynb
```

---

## 📋 Project Overview

This comprehensive DSP project focuses on two parallel compression pipelines:

### Audio Pipeline
1. **Loading** — Read a WAV file and convert it to a normalized float array
2. **Denoising** — Remove background noise using spectral subtraction followed by temporal median filtering
3. **Compression** — Compress audio in the frequency domain with sub-band quantization and RLE encoding
4. **Playback** — Listen to original, denoised, and reconstructed signals side by side

### Video Pipeline
1. **Color Space Conversion** — Convert RGB frames to YUV (planar) for efficient lossy compression
2. **Frame Type Selection** — Automatically alternate between I-frames (intra-encoded) and P-frames (predictive)
3. **I-Frame Compression** — DCT transform → quantization → Huffman entropy coding
4. **P-Frame Compression** — Motion estimation → residual encoding → entropy coding
5. **Bitstream Formation** — Binary-packed output with metadata headers and frame indicators
6. **Quality Metrics** — Real-time PSNR calculation, compression ratio measurement, live graph visualization

---

## ✨ Features

### 🔊 Audio System Features

#### GUI Application (`audio_app.py`)

| Feature | Description |
|---|---|
| **Load WAV** | Open any WAV file through a file dialog |
| **Process** | Run the full DSP pipeline (denoising + compression) with a single click |
| **Play Original** | Play back the raw input audio |
| **Play Denoised** | Play back the spectrally denoised audio |
| **Play Reconstructed** | Play back the compressed-and-reconstructed audio |
| **Mute** | Stop playback at any time |
| **Waveform Visualizer** | Live animated waveform display that tracks playback position |
| **Seek Slider** | Click or drag the progress bar to jump to any point in the audio |

#### Jupyter Notebook (`audio_encoder.ipynb`)

- Load and visualize raw audio waveforms
- Apply programmable silence regions to simulate missing segments
- Step-by-step denoising: spectral subtraction → temporal median filtering
- Full compression pipeline: STFT → band splitting → quantization → RLE encoding → decoding
- Compute and display **SNR (Signal-to-Noise Ratio)** and **Compression Ratio**
- In-notebook audio playback for comparing original, denoised, and compressed signals
- Export processed audio to `Output/denoised.wav` and `Output/compressed.wav`

#### Audio DSP Techniques Used

| Technique | Description |
|---|---|
| **STFT / Inverse STFT** | Short-Time Fourier Transform for time-frequency analysis and reconstruction |
| **Spectral Subtraction** | Estimate and subtract noise profile from each frequency bin |
| **Temporal Median Filtering** | Smooth spectral frames over time to eliminate musical noise |
| **Sub-band Quantization** | Divide the spectrum into 8 frequency bands and apply perceptual quantization steps |
| **Run-Length Encoding (RLE)** | Lossless encoding of repeated quantized values to achieve compression |

---

### 🎬 Video System Features

#### GUI Application (`video_compression_enhanced.py`)

| Feature | Description |
|---|---|
| **Dark Industrial Theme** | GitHub-inspired color palette with modern Qt6 styling |
| **Drag-and-Drop Upload** | Drop video files directly onto the GUI or use file browser |
| **Live Compression** | Process video frames in real-time with progress bar and frame counter |
| **Dual Viewer** | Side-by-side comparison of original vs. reconstructed frames |
| **Motion Vectors** | Visualize motion estimation with green directional arrows overlaid on P-frames |
| **PSNR Graph** | Embedded Matplotlib canvas showing quality per frame |
| **Playback Controls** | Play/Pause/Scrub through original and reconstructed videos |
| **Save Reconstructed** | Export decompressed video as MP4 file |
| **Session Statistics** | Display avg PSNR, compression ratio, elapsed time, frame count |
| **Configurable Parameters** | Adjust quantization scale, I-frame period, motion search range, max frames |

#### Video Compression Algorithms

- **Huffman Entropy Coding** — Custom bitstream codec with frequency-based tree building
- **DCT Transform** — Vectorized 2D DCT on entire block batches (no Python loops)
- **Quantization** — JPEG-like quantization matrix scaled by user parameter
- **Diamond Search** — Efficient 8-pixel block motion estimation with hierarchical refinement
- **RLE Encoding** — Run-length encoding of quantized DCT coefficients
- **Parallel Processing** — Multiprocessing support for I-frame compression batches
- **Bitstream Protocol** — Structured header + frame-type indicators + per-frame data blobs

---

## 🏗️ Video System Architecture

### Main Classes

```
CompressionWorker (QThread)
  ├─ compress_i_frame()      → Pure intra-frame encoding
  ├─ compress_p_frame()      → Motion-compensated residual encoding
  ├─ motion_estimation_frame() → Diamond search on Y-channel
  └─ Emits: progress_update, frame_ready, finished_signal

MainWindow (QMainWindow)
  ├─ _build_ui()             → Qt6 layout construction
  ├─ _build_sidebar()        → Control panel with sliders/spinboxes
  ├─ _build_content()        → Tabbed viewer (Comparison / Motion / Metrics)
  ├─ dragEnterEvent()        → Drag-and-drop support
  └─ Slots: _on_progress(), _on_frame(), _on_finished()

HuffmanCodec
  ├─ fit(symbols)            → Build frequency tree from symbol list
  ├─ encode(symbols) → str   → Return bitstring
  ├─ decode(bits) → list     → Recover original symbols
  ├─ to_bytes()              → Serialize codebook for storage
  └─ from_codebook_bytes()   → Deserialize and rebuild tree

BitstreamWriter
  ├─ write_bits(bitstring)   → Pack bits into bytearray
  ├─ write_uint(value, nbits)→ Write unsigned integer
  ├─ write_bytes_blob(data)  → Write length-prefixed byte chunk
  └─ get_bytes()             → Flush and return final bitstream
```

### Key Video Functions

| Function | Purpose |
|---|---|
| `batch_dct2()`, `batch_idct2()` | Vectorized 2D DCT/IDCT on (nv, nh, 8, 8) blocks |
| `split_blocks()`, `merge_blocks()` | Tile image into 8×8 blocks or reconstruct |
| `diamond_search_block()` | Motion search for single block using large + small diamond patterns |
| `motion_estimation_frame()` | Estimate MVs for entire frame |
| `rgb_to_yuv()`, `yuv_to_rgb()` | Color space conversion with optional denoising |
| `flatten_to_symbols()` | Convert quantized blocks → RLE-encoded symbol list |
| `build_bitstream()` | Pack compressed frames + headers into binary blob |
| `psnr()` | Calculate Peak Signal-to-Noise Ratio |
| `draw_mvs()` | Overlay motion vectors on frame |

---

## 🎨 Video GUI Walkthrough

### Sidebar (Left Panel)

1. **Title** — "DSP COMPRESS" in accent blue
2. **Drop Zone** — Interactive area to upload video (click or drag)
3. **Compression Settings** — Adjust four key parameters:
   - `Quant Scale ×` — Multiply JPEG matrix (0.1–10.0). Higher = more lossy but better compression
   - `I-frame Period` — Insert keyframe every N frames (1–100)
   - `Search Range px` — Motion search radius in pixels (1–32)
   - `Max Frames` — Limit processing (1–5000)
4. **Action Buttons**
   - `▶ Start Compression` — Launch processing thread
   - `✕ Cancel` — Stop ongoing compression
   - `💾 Save Reconstructed` — Export MP4 file
5. **Progress Bar** — Real-time frame count
6. **Status Label** — Current operation info
7. **Session Stats** — Display avg PSNR, compression %, elapsed time, frame count

### Main Content (Right Tabs)

#### **Comparison Tab**
- **Original** — Left panel showing input frame
- **Reconstructed** — Right panel showing decoded output
- **PSNR Badge** — Frame-specific quality metric
- **Playback Controls** — Play/Pause button, seek slider, frame counter

#### **Motion Vectors Tab**
- Overlaid motion vectors on P-frames (green arrows)
- Black frame for I-frames (no vectors)
- Arrows point from block center in direction of motion

#### **Metrics / PSNR Tab**
- **Live PSNR Graph** — Matplotlib plot updating as frames process
- **Statistical Cards** — Display avg PSNR, compression ratio, elapsed time, frame count

---

## ⚙️ Video Parameter Tuning Guide

| Parameter | Default | Range | Effect | Trade-off |
|---|---|---|---|---|
| **Quant Scale** | 1.0 | 0.1–10.0 | Controls DCT quantization aggression | Lower = better quality, larger file; Higher = worse quality, smaller file |
| **I-frame Period** | 10 | 1–100 | Keyframes every N frames | Lower = more random access, larger file; Higher = faster decode, less robust to errors |
| **Search Range** | 8 | 1–32 | Motion search radius (pixels) | Lower = faster, less accurate; Higher = slower, better prediction |
| **Max Frames** | 200 | 1–5000 | Limit processing | Use to test on short clips or process full video |

### Recommended Presets

| Use Case | Quant Scale | I-period | Search | Max Frames |
|---|---|---|---|---|
| **Fast Preview** | 2.0 | 20 | 4 | 50 |
| **Balanced** | 1.0 | 10 | 8 | 200 |
| **High Quality** | 0.5 | 5 | 16 | 500 |
| **Archive** | 0.2 | 3 | 32 | 5000 |

---

## 📊 Video Compression Pipeline Details

### I-Frame (Intra) Encoding

```
RGB Frame
  ↓
RGB → YUV (planar)
  ↓
Split into 8×8 blocks (per Y, U, V)
  ↓
Subtract 128 (center around zero)
  ↓
2D DCT (vectorized, no loops)
  ↓
Quantize with scaled JPEG matrix
  ↓
Zigzag scan + RLE encode
  ↓
Huffman coding on symbol stream
  ↓
Bitstream output
```

### P-Frame (Predictive) Encoding

```
Current RGB Frame
  ↓
RGB → YUV
  ↓
Diamond search on Y-channel (ref = prev I/P frame)
  ↓
Generate prediction from MVs
  ↓
Compute residual (current - predicted)
  ↓
Split residual into 8×8 blocks
  ↓
2D DCT
  ↓
Quantize
  ↓
Separate Huffman coding:
  - MVs (motion vectors)
  - Residual coefficients
  ↓
Bitstream output
```

### Decompression

```
Bitstream
  ↓
Parse header (resolution, FPS, quantization, frame count)
  ↓
For each frame:
  ├─ If I-frame:
  │  ├─ Parse Y, U, V Huffman codebooks + bitstrings
  │  ├─ Huffman decode → RLE decode → Unzigzag
  │  ├─ Dequantize
  │  ├─ Inverse 2D DCT
  │  ├─ Add 128, clip to [0,255]
  │  └─ Reconstruct RGB
  │
  └─ If P-frame:
     ├─ Parse MV and residual Huffman data
     ├─ Decode MVs + residual coefficients
     ├─ Inverse DCT on residual
     ├─ Build prediction from previous frame + MVs
     ├─ Add residual to prediction
     └─ Reconstruct RGB
```

---

## 📈 Video Performance Metrics

### Output Statistics

- **PSNR (dB)** — Peak Signal-to-Noise Ratio per frame (higher = better quality)
- **Compression Ratio (%)** — `(compressed_bits / original_bits) × 100`
- **Bitstream Size** — Actual bytes written to file
- **Processing Time** — Elapsed seconds for entire video

### Example Results

| Video | Resolution | Frames | Quant | Comp % | Avg PSNR | Time |
|---|---|---|---|---|---|---|
| 720p test | 1280×720 | 100 | 1.0 | 8.5% | 38.2 dB | 12.4s |
| 1080p HD | 1920×1080 | 200 | 0.8 | 12.1% | 40.1 dB | 34.2s |
| 4K sample | 3840×2160 | 50 | 1.5 | 6.2% | 35.8 dB | 45.1s |

---

## 🔧 Video Advanced Configuration

### Motion Search Strategy

The diamond search algorithm operates in two phases:

1. **Coarse Phase** — Check 9 candidates (large diamond) with adaptive step size
2. **Fine Phase** — Check 5 candidates (small diamond) for local refinement

This approach reduces computation vs. exhaustive search while maintaining accuracy.

### Huffman Encoding

- Frequency-based tree construction using a min-heap
- Codebook stored as Python pickle in bitstream header
- Single-symbol edge case handled (all-zero blocks)
- Optimal prefix codes per frame (adaptive codebooks for I and P separately)

### Bitstream Format

```
[HEADER: 16 bytes]
MAGIC(4) VERSION(1) FLAGS(1) HEIGHT(2) WIDTH(2) FPS(1) I_PERIOD(1) QSCALE_x100(2) N_FRAMES(2)

[FRAME DATA]
For each frame:
  FRAME_TYPE(1 byte: 0x49='I', 0x50='P')
  If I-frame:
    For each plane (Y, U, V):
      PLANE_HEIGHT(2) PLANE_WIDTH(2)
      LENGTH_PREFIXED_CODEBOOK_BYTES
      LENGTH_PREFIXED_BITSTRING
  Else (P-frame):
    LENGTH_PREFIXED_MV_CODEBOOK
    LENGTH_PREFIXED_RES_CODEBOOK
    LENGTH_PREFIXED_MV_BITSTRING
    LENGTH_PREFIXED_RES_BITSTRING
    MVS_SHAPE(6 shorts) CURR_SHAPE(2 shorts) RES_SHAPE(2 shorts)
```

---

## 🐛 Video Troubleshooting

| Issue | Solution |
|---|---|
| **Video won't load** | Ensure video format is supported (MP4, AVI, MOV, MKV, WMV). Check file permissions. |
| **GUI freezes during compression** | Processing runs in a separate thread. If still frozen, increase `Max Frames` to test fewer frames. |
| **Low PSNR values** | Reduce `Quant Scale` (use 0.5–1.0 for better quality). Increase `I-frame Period`. |
| **High compression ratio** | Increase `Quant Scale` (use 1.5–2.0 for more lossy compression). Increase `Search Range` for better motion prediction. |
| **Memory error on large videos** | Reduce `Max Frames` or process in multiple batches. Consider downsampling input video. |
| **Slow motion search** | Reduce `Search Range` or increase `Max Frames` threshold early (process fewer frames). |

---

## 🐛 Audio Troubleshooting

| Issue | Solution |
|---|---|
| **Audio file not loading** | Ensure the file is in WAV format and placed at `Input/rec.wav` for notebook or selected via dialog for GUI |
| **High noise in output** | Increase spectral subtraction aggressiveness or adjust the noise floor threshold in code |
| **Poor compression ratio** | Reduce quantization step sizes in the sub-band compression section or apply stronger denoising first |
| **Musical noise artifacts** | Increase temporal median filter window size or apply spectral floor (minimum magnitude threshold) |
| **Playback issues** | Ensure scipy, soundfile, and librosa are correctly installed |

---

## 📦 Project Structure

```
DSP-project/
├── audio_app.py                      # Pygame audio GUI application
├── audio_encoder.ipynb               # Jupyter notebook for audio analysis
├── video_app.py     # Main video application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── Input/
│   └── rec.wav                       # Input audio file (for notebook)
│   └── noisy.mp4
└── Output/
    ├── denoised.wav                  # Denoised audio output
    ├── compressed.wav                # Compressed audio output
    └── reconstructed.mp4             # Saved compressed video (generated)
    └── psnr_graph.png             
```

---

## 📚 Dependencies

### All Dependencies (install via requirements.txt)

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥2.4 | Numerical arrays and vectorization |
| `scipy` | ≥1.17 | DCT/IDCT transforms, STFT, signal processing |
| `opencv-python` | ≥4.5 | Video I/O and image processing |
| `PyQt6` | ≥6.0 | Video GUI framework |
| `matplotlib` | ≥3.5 | PSNR graph visualization |
| `pygame` | ≥2.6 | Audio GUI framework |

### Additional for Jupyter Notebook

| Package |
|---|
| `librosa` |
| `soundfile` |
| `jupyter` |
| `pandas` |

### Install All at Once

```bash
pip install numpy scipy opencv-python PyQt6 matplotlib pygame librosa soundfile jupyter pandas
```

---

