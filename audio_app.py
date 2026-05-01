import pygame
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import tempfile, os, time
import tkinter as tk
from tkinter import filedialog

# ===============================
# INIT
# ===============================
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

screen = pygame.display.set_mode((900, 500))
pygame.display.set_caption("DSP Audio Player")

font = pygame.font.SysFont("Segoe UI", 22)
clock = pygame.time.Clock()

# ===============================
# DSP
# ===============================

def safe_stft(x, fs):
    nperseg = min(1024, len(x))
    noverlap = nperseg // 2
    return stft(x, fs, nperseg=nperseg, noverlap=noverlap)

def spectral_denoise(x, fs):
    _, _, Zxx = safe_stft(x, fs)
    mag, phase = np.abs(Zxx), np.angle(Zxx)

    noise = np.percentile(mag, 20, axis=1, keepdims=True)
    clean = mag - 0.6 * noise
    clean = np.maximum(clean, 0.1 * noise)

    Z = clean * np.exp(1j * phase)
    _, x_clean = istft(Z, fs)
    return x_clean

def suppress_musical_noise_temporal(x, fs):
    _, _, Zxx = safe_stft(x, fs)
    mag, phase = np.abs(Zxx), np.angle(Zxx)

    for i in range(2, mag.shape[1]-2):
        mag[:, i] = np.median(mag[:, i-2:i+3], axis=1)

    Z = mag * np.exp(1j * phase)
    _, x_clean = istft(Z, fs)
    return x_clean

def compress_reconstruct(x, fs):
    _, _, Zxx = safe_stft(x, fs)
    mag, phase = np.abs(Zxx), np.angle(Zxx)

    num_bands = 8
    bands = np.array_split(mag, num_bands, axis=0)

    step = 0.0005
    scales = 1 + 0.2 * np.linspace(0,1,num_bands)

    q = np.zeros_like(mag)

    start = 0
    for i, b in enumerate(bands):
        end = start + b.shape[0]
        step_i = step * scales[i]
        q[start:end] = np.round(mag[start:end] / step_i)
        start = end

    recon = np.zeros_like(mag)

    start = 0
    for i, b in enumerate(bands):
        end = start + b.shape[0]
        step_i = step * scales[i]
        recon[start:end] = q[start:end] * step_i
        start = end

    Z = recon * np.exp(1j * phase)
    _, x_rec = istft(Z, fs)
    return x_rec

# ===============================
# UTIL
# ===============================

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-8)

def to_wav_path(signal, fs):
    signal = normalize(signal)
    signal = (signal * 32767).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    wavfile.write(path, fs, signal)
    return path

# ===============================
# STATE
# ===============================

fs = None
original = None
denoised = None
reconstructed = None

path_original = None
path_denoised = None
path_reconstructed = None

current_signal = None
current_path = None

start_time = 0
duration = 1
is_playing = False

slider_rect = pygame.Rect(100, 380, 700, 8)
slider_dragging = False

status = "Load a WAV file"

# ===============================
# UI
# ===============================

def draw_button(text, rect):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()[0]

    if rect.collidepoint(mouse):
        color = (100, 180, 255)
        if click:
            color = (50, 120, 220)
    else:
        color = (70, 130, 200)

    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, (255,255,255), rect, 1, border_radius=12)

    label = font.render(text, True, (255,255,255))
    screen.blit(label, (rect.x + 15, rect.y + 12))


def draw_text(text, x, y):
    label = font.render(text, True, (200,200,200))
    screen.blit(label, (x, y))


def draw_slider():
    if current_signal is None:
        return

    elapsed = time.time() - start_time
    progress = min(elapsed / duration, 1.0)

    pygame.draw.rect(screen, (60,60,60), slider_rect, border_radius=5)

    filled = int(progress * slider_rect.w)
    pygame.draw.rect(screen, (0,200,120),
                     (slider_rect.x, slider_rect.y, filled, slider_rect.h),
                     border_radius=5)

    handle_x = slider_rect.x + filled
    pygame.draw.circle(screen, (255,255,255),
                       (handle_x, slider_rect.y + 4), 8)


def draw_waveform():
    if current_signal is None:
        return

    width = 700
    height = 120
    x0 = 100
    y0 = 300

    step = max(1, len(current_signal)//width)
    data = current_signal[::step][:width]

    elapsed = time.time() - start_time
    idx = int((elapsed / duration) * len(data))
    idx = min(idx, len(data)-1)

    for i in range(len(data)-1):
        y1 = int(y0 - data[i]*height)
        y2 = int(y0 - data[i+1]*height)

        color = (0,255,150) if i <= idx else (100,100,100)

        pygame.draw.line(screen, color,
                         (x0+i, y1),
                         (x0+i+1, y2), 2)

# ===============================
# ACTIONS
# ===============================

def load_audio():
    global fs, original, path_original, status

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(filetypes=[("WAV","*.wav")])
    if not path:
        return

    fs, data = wavfile.read(path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    data /= (np.max(np.abs(data)) + 1e-8)

    original = data
    path_original = to_wav_path(original, fs)

    status = f"Loaded ({len(data)} samples)"

def process_audio():
    global denoised, reconstructed
    global path_denoised, path_reconstructed, status

    if original is None:
        status = "Load audio first"
        return

    status = "Processing..."

    denoised = spectral_denoise(original, fs)
    denoised = suppress_musical_noise_temporal(denoised, fs)

    reconstructed = compress_reconstruct(denoised, fs)

    path_denoised = to_wav_path(denoised, fs)
    path_reconstructed = to_wav_path(reconstructed, fs)

    status = "Done"

def play(path, signal, start_pos=0):
    global start_time, duration, is_playing, current_signal, current_path

    if path is None:
        return

    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=start_pos)

    start_time = time.time() - start_pos
    duration = len(signal) / fs
    is_playing = True
    current_signal = signal
    current_path = path

def stop():
    global is_playing
    pygame.mixer.music.stop()
    is_playing = False

# ===============================
# BUTTONS
# ===============================

btn_load = pygame.Rect(100, 50, 140, 50)
btn_process = pygame.Rect(260, 50, 140, 50)

btn_orig = pygame.Rect(100, 140, 200, 50)
btn_den = pygame.Rect(330, 140, 200, 50)
btn_rec = pygame.Rect(560, 140, 200, 50)

btn_stop = pygame.Rect(330, 210, 200, 50)

# ===============================
# LOOP
# ===============================

running = True
while running:
    screen.fill((30,30,25))

    draw_text("MP3 Audio Encoder and Player", 350, 10)
    draw_text(status, 100, 460)

    draw_button("Load", btn_load)
    draw_button("Process", btn_process)

    draw_button("Play Original", btn_orig)
    draw_button("Play Denoised", btn_den)
    draw_button("Play Reconstructed", btn_rec)

    draw_button("Mute", btn_stop)

    draw_waveform()
    draw_slider()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if btn_load.collidepoint(event.pos):
                load_audio()

            if btn_process.collidepoint(event.pos):
                process_audio()

            if btn_orig.collidepoint(event.pos):
                play(path_original, original)

            if btn_den.collidepoint(event.pos):
                play(path_denoised, denoised)

            if btn_rec.collidepoint(event.pos):
                play(path_reconstructed, reconstructed)

            if btn_stop.collidepoint(event.pos):
                stop()

            if slider_rect.collidepoint(event.pos):
                slider_dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False

        if event.type == pygame.MOUSEMOTION and slider_dragging:
            x = event.pos[0]
            rel = (x - slider_rect.x) / slider_rect.w
            rel = max(0, min(1, rel))

            seek_time = rel * duration
            play(current_path, current_signal, start_pos=seek_time)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()