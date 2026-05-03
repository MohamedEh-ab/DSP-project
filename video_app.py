#!/usr/bin/env python3
"""
DSP Video Compression System — Enhanced Edition
Features:
  • Modern Qt6 GUI with dark industrial aesthetic
  • Huffman entropy coding on DCT coefficients & motion vectors
  • Bitstream formation with headers + frame-type indicators
  • Compression ratio measurement
  • Vectorised batch DCT (no per-block Python loops)
  • Diamond-pattern motion search (replaces exhaustive scan)
  • Parallel I-frame compression via multiprocessing
  • Live PSNR graph embedded in GUI
  • Drag-and-drop video upload
  • Play/Pause/Scrub comparison viewer
  • Save reconstructed video
"""

import sys
import os
import pickle
import struct
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter
import cv2

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
                           QEasingCurve, QRect, pyqtProperty)
from PyQt6.QtGui import (QPixmap, QImage, QDragEnterEvent, QDropEvent,
                          QFont, QPalette, QColor, QPainter, QLinearGradient,
                          QBrush, QPen)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar,
    QSlider, QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
    QGroupBox, QGridLayout, QMessageBox, QFrame, QSizePolicy,
    QScrollArea, QSplitter
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE      = 8
SEARCH_RANGE    = 8
I_FRAME_PERIOD  = 10
FPS             = 25
MAX_FRAMES      = 200
QUANT_SCALE     = 1.0

BASE_QUANT_MATRIX = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.float32)

# Pre-compute zigzag index once
_ZIGZAG_INDICES = np.array([
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,
], dtype=np.int32)

# Inverse lookup
_IZIGZAG_INDICES = np.argsort(_ZIGZAG_INDICES)

# ─────────────────────────────────────────────────────────────────────────────
# HUFFMAN CODING
# ─────────────────────────────────────────────────────────────────────────────
class _HNode:
    __slots__ = ('sym', 'freq', 'left', 'right')

    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym   = sym
        self.freq  = freq
        self.left  = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_tree(freq_dict: dict) -> '_HNode | None':
    """Build a Huffman tree from a symbol→frequency dict."""
    import heapq
    heap = [_HNode(s, f) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _HNode(freq=a.freq + b.freq, left=a, right=b))
    return heap[0] if heap else None


def _build_codes(node: '_HNode', prefix: str = '', cb: dict = None) -> dict:
    if cb is None:
        cb = {}
    if node is None:
        return cb
    if node.sym is not None:
        cb[node.sym] = prefix or '0'   # single-symbol edge case
    else:
        _build_codes(node.left,  prefix + '0', cb)
        _build_codes(node.right, prefix + '1', cb)
    return cb


class HuffmanCodec:
    """Build codebook from data, encode a symbol list, and decode bit-string."""

    def __init__(self):
        self.codebook: dict  = {}
        self._decode_tree    = None

    # ── build ──────────────────────────────────────────────────────────────
    def fit(self, symbols):
        freq = Counter(int(s) for s in symbols)
        tree = _build_tree(freq)
        self.codebook = {k: v for k, v in _build_codes(tree).items()}
        self._decode_tree = tree

    # ── encode ─────────────────────────────────────────────────────────────
    def encode(self, symbols) -> str:
        cb = self.codebook
        return ''.join(cb[int(s)] for s in symbols)

    # ── decode ─────────────────────────────────────────────────────────────
    def decode(self, bits: str) -> list:
        out  = []
        node = self._decode_tree
        if node is None:
            return out
        # Handle degenerate single-symbol tree
        if node.sym is not None:
            count = len(bits)   # every '0' is one symbol
            return [node.sym] * count
        for b in bits:
            node = node.left if b == '0' else node.right
            if node.sym is not None:
                out.append(node.sym)
                node = self._decode_tree
        return out

    # ── serialise (for bitstream) ──────────────────────────────────────────
    def to_bytes(self) -> bytes:
        return pickle.dumps(self.codebook)

    @classmethod
    def from_codebook_bytes(cls, data: bytes) -> 'HuffmanCodec':
        obj = cls()
        obj.codebook = pickle.loads(data)
        # Rebuild decode tree from codebook
        freq = {sym: 1 for sym in obj.codebook}
        tree = _build_tree(freq)
        obj._decode_tree = tree
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# BIT-STREAM BUILDER / READER
# ─────────────────────────────────────────────────────────────────────────────
class BitstreamWriter:
    """Pack bits into a bytearray with a simple header protocol."""

    MAGIC = b'DSPC'   # 4-byte magic
    VERSION = 1

    def __init__(self):
        self._buf  = bytearray()
        self._cur  = 0
        self._nbits = 0

    # low-level
    def _write_bit(self, bit: int):
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._cur)
            self._cur   = 0
            self._nbits = 0

    def write_bits(self, bits: str):
        for b in bits:
            self._write_bit(int(b))

    def write_uint(self, value: int, nbits: int):
        for shift in range(nbits - 1, -1, -1):
            self._write_bit((value >> shift) & 1)

    def write_bytes_blob(self, data: bytes):
        """Flush, write 4-byte length, then raw bytes."""
        self._flush()
        self._buf += struct.pack('>I', len(data))
        self._buf += data

    def _flush(self):
        if self._nbits > 0:
            self._cur <<= (8 - self._nbits)
            self._buf.append(self._cur)
            self._cur   = 0
            self._nbits = 0

    def get_bytes(self) -> bytes:
        self._flush()
        header = self.MAGIC + struct.pack('>BB', self.VERSION, 0)
        return bytes(header + self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SPACE
# ─────────────────────────────────────────────────────────────────────────────
def rgb_to_yuv(rgb: np.ndarray, denoise: bool = True) -> dict:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    Y =  0.29900 * r + 0.58700 * g + 0.11400 * b
    U = -0.14713 * r - 0.28886 * g + 0.43600 * b
    V =  0.61500 * r - 0.51499 * g - 0.10001 * b

    if denoise:
        Y = gaussian_filter(Y, sigma=0.8)   # mild denoise on luma

    return {
        'Y': np.clip(Y,       0, 255).astype(np.uint8),
        'U': np.clip(U + 128, 0, 255).astype(np.uint8),
        'V': np.clip(V + 128, 0, 255).astype(np.uint8),
    }


def yuv_to_rgb(yuv: dict) -> np.ndarray:
    Y = yuv['Y'].astype(np.float32)
    U = yuv['U'].astype(np.float32) - 128
    V = yuv['V'].astype(np.float32) - 128

    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U

    return np.stack([
        np.clip(R, 0, 255),
        np.clip(G, 0, 255),
        np.clip(B, 0, 255),
    ], axis=-1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def split_blocks(channel: np.ndarray) -> tuple:
    """Return (blocks[nv,nh,8,8], original_shape)."""
    h, w = channel.shape
    pad_h = (-h) % BLOCK_SIZE
    pad_w = (-w) % BLOCK_SIZE
    if pad_h or pad_w:
        channel = np.pad(channel, ((0, pad_h), (0, pad_w)),
                         mode='constant', constant_values=128)
    ph, pw = channel.shape
    blocks = (channel
              .reshape(ph // BLOCK_SIZE, BLOCK_SIZE, pw // BLOCK_SIZE, BLOCK_SIZE)
              .transpose(0, 2, 1, 3))          # (nv, nh, 8, 8)
    return blocks, (h, w)


def merge_blocks(blocks: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Inverse of split_blocks."""
    h, w = orig_shape
    nv, nh = blocks.shape[:2]
    img = blocks.transpose(0, 2, 1, 3).reshape(nv * BLOCK_SIZE, nh * BLOCK_SIZE)
    return img[:h, :w]


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DCT — operates on entire (nv, nh, 8, 8) tensor at once
# ─────────────────────────────────────────────────────────────────────────────
def batch_dct2(blocks: np.ndarray) -> np.ndarray:
    """2-D DCT on last two axes — vectorised, no Python loops."""
    return dct(dct(blocks, norm='ortho', axis=-2), norm='ortho', axis=-1)


def batch_idct2(blocks: np.ndarray) -> np.ndarray:
    """2-D IDCT on last two axes — vectorised, no Python loops."""
    return idct(idct(blocks, norm='ortho', axis=-1), norm='ortho', axis=-2)


# ─────────────────────────────────────────────────────────────────────────────
# QUANTISATION
# ─────────────────────────────────────────────────────────────────────────────
def get_qmat(scale: float = 1.0) -> np.ndarray:
    return np.maximum(np.round(BASE_QUANT_MATRIX * scale), 1).astype(np.float32)


def quantize_blocks(blocks: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return np.round(blocks / qmat).astype(np.int32)


def dequantize_blocks(blocks: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return blocks.astype(np.float32) * qmat


# ─────────────────────────────────────────────────────────────────────────────
# ZIGZAG + RLE  (vectorised zigzag)
# ─────────────────────────────────────────────────────────────────────────────
def zigzag_all(blocks: np.ndarray) -> np.ndarray:
    """blocks: (nv, nh, 8, 8) → (nv*nh, 64) in zigzag order."""
    flat = blocks.reshape(-1, 64)
    return flat[:, _ZIGZAG_INDICES]


def rle_encode_vec(row: np.ndarray) -> list:
    """RLE-encode one 64-element zigzag-scanned row."""
    out  = []
    run  = 0
    for v in row:
        if v == 0:
            run += 1
        else:
            out.extend([run, int(v)])
            run = 0
    out.extend([0, 0])
    return out


def rle_decode_vec(rle: list) -> np.ndarray:
    data = np.zeros(64, dtype=np.int32)
    pos  = 0
    i    = 0
    while i < len(rle) - 1:
        run, val = rle[i], rle[i + 1]
        if run == 0 and val == 0:
            break
        pos += run
        if pos < 64:
            data[pos] = val
        pos += 1
        i   += 2
    return data


def flatten_to_symbols(quant_blocks: np.ndarray) -> list:
    """Convert (nv, nh, 8, 8) int32 blocks → flat RLE symbol list for Huffman."""
    zz_rows = zigzag_all(quant_blocks)          # (N, 64)
    symbols = []
    for row in zz_rows:
        symbols.extend(rle_encode_vec(row))
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# MOTION ESTIMATION — diamond search (vectorised candidate scoring)
# ─────────────────────────────────────────────────────────────────────────────
# Diamond patterns: small (5 pts) + large (9 pts)
_LARGE_DIAMOND = [(0,0),(-2,0),(2,0),(0,-2),(0,2),
                  (-1,-1),(-1,1),(1,-1),(1,1)]
_SMALL_DIAMOND = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def diamond_search_block(curr: np.ndarray,
                         ref: np.ndarray,
                         y: int, x: int,
                         search_range: int) -> tuple:
    """
    Full-diamond + small-diamond motion search.
    Returns (dy, dx), predicted_block.
    """
    rh, rw = ref.shape
    bh, bw = curr.shape
    best_mv  = (0, 0)
    best_mad = _mad(curr, ref[y:y+bh, x:x+bw]) if (y+bh <= rh and x+bw <= rw) else float('inf')
    best_blk = ref[y:y+bh, x:x+bw].copy() if (y+bh <= rh and x+bw <= rw) else np.full_like(curr, 128)

    cy, cx = y, x

    def _check(dy, dx):
        nonlocal best_mv, best_mad, best_blk, cy, cx
        ny = cy + dy
        nx = cx + dx
        if ny < 0 or ny + bh > rh or nx < 0 or nx + bw > rw:
            return
        blk = ref[ny:ny+bh, nx:nx+bw]
        m   = _mad(curr, blk)
        if m < best_mad:
            best_mad = m
            best_mv  = (ny - y, nx - x)
            best_blk = blk.copy()
            cy, cx   = ny, nx

    # Coarse large-diamond search
    step = max(1, search_range // 2)
    for _ in range(search_range):
        prev = (cy, cx)
        for dy, dx in _LARGE_DIAMOND:
            _check(dy * step, dx * step)
        if (cy, cx) == prev:
            break
        step = max(1, step // 2)

    # Fine small-diamond refinement
    for _ in range(4):
        prev = (cy, cx)
        for dy, dx in _SMALL_DIAMOND:
            _check(dy, dx)
        if (cy, cx) == prev:
            break

    return best_mv, best_blk


def motion_estimation_frame(curr_y: np.ndarray,
                             ref_y: np.ndarray,
                             search_range: int) -> tuple:
    """
    Estimate motion for every 8×8 block in curr_y relative to ref_y.
    Returns mvs (nv, nx, 2), pred (h, w).
    """
    h, w   = curr_y.shape
    nv     = int(np.ceil(h / BLOCK_SIZE))
    nx_blk = int(np.ceil(w / BLOCK_SIZE))
    mvs    = np.zeros((nv, nx_blk, 2), dtype=np.int32)
    pred   = np.zeros((h, w), dtype=np.uint8)

    for i in range(nv):
        yi = i * BLOCK_SIZE
        for j in range(nx_blk):
            xj = j * BLOCK_SIZE
            bh = min(BLOCK_SIZE, h - yi)
            bw = min(BLOCK_SIZE, w - xj)
            curr_blk = curr_y[yi:yi+bh, xj:xj+bw].astype(np.float32)
            mv, ref_blk = diamond_search_block(curr_blk, ref_y, yi, xj, search_range)
            mvs[i, j]  = mv
            pred[yi:yi+bh, xj:xj+bw] = ref_blk[:bh, :bw]

    return mvs, pred


# ─────────────────────────────────────────────────────────────────────────────
# I-FRAME COMPRESSION  (pure function — safe for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def compress_i_frame(yuv: dict, qmat: np.ndarray) -> tuple:
    """
    Returns (compressed_data, reconstructed_yuv).
    compressed_data[plane] = (codebook_bytes, encoded_bits, shape, quant_blocks)
    """
    comp = {}
    rec  = {}
    for plane in ('Y', 'U', 'V'):
        blocks, shape     = split_blocks(yuv[plane])
        centered          = blocks.astype(np.float32) - 128.0
        dct_coeffs        = batch_dct2(centered)
        quant             = quantize_blocks(dct_coeffs, qmat)

        symbols = flatten_to_symbols(quant)
        hc      = HuffmanCodec()
        hc.fit(symbols)
        bits    = hc.encode(symbols)

        comp[plane] = (hc.to_bytes(), bits, shape, quant)

        # Reconstruction
        rec_blocks = batch_idct2(dequantize_blocks(quant, qmat)) + 128.0
        rec[plane]  = merge_blocks(np.clip(rec_blocks, 0, 255), shape).astype(np.uint8)

    return comp, rec


# ─────────────────────────────────────────────────────────────────────────────
# P-FRAME COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────
def compress_p_frame(yuv: dict,
                     prev_y: np.ndarray,
                     qmat: np.ndarray,
                     search_range: int) -> tuple:
    """
    Returns (compressed_data, residual_vis_image).
    """
    curr_y    = yuv['Y']
    mvs, pred = motion_estimation_frame(curr_y, prev_y, search_range)

    residual  = curr_y.astype(np.float32) - pred.astype(np.float32)
    res_blocks, shape = split_blocks(residual)
    res_dct   = batch_dct2(res_blocks)
    res_quant = quantize_blocks(res_dct, qmat)

    # Huffman on residual
    res_syms = flatten_to_symbols(res_quant)
    hc_res   = HuffmanCodec()
    hc_res.fit(res_syms)
    res_bits = hc_res.encode(res_syms)

    # Huffman on motion vectors
    mv_flat  = mvs.flatten().tolist()
    hc_mv    = HuffmanCodec()
    hc_mv.fit(mv_flat)
    mv_bits  = hc_mv.encode(mv_flat)

    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)

    data = {
        'mv_cb_bytes' : hc_mv.to_bytes(),
        'res_cb_bytes': hc_res.to_bytes(),
        'mv_bits'     : mv_bits,
        'res_bits'    : res_bits,
        'mvs'         : mvs,
        'mvs_shape'   : mvs.shape,
        'res_quant'   : res_quant,
        'res_shape'   : shape,
        'curr_shape'  : curr_y.shape,
    }
    return data, residual_vis


# ─────────────────────────────────────────────────────────────────────────────
# P-FRAME DECOMPRESSION
# ─────────────────────────────────────────────────────────────────────────────
def decompress_p_frame(data: dict,
                       prev_y: np.ndarray,
                       qmat: np.ndarray) -> np.ndarray:
    shape      = data['curr_shape']
    mvs        = data['mvs']
    res_quant  = data['res_quant']
    res_shape  = data['res_shape']

    # Reconstruct residual
    res_deq    = dequantize_blocks(res_quant, qmat)
    res_blocks = batch_idct2(res_deq)
    residual   = merge_blocks(res_blocks, res_shape)

    # Reconstruct prediction from stored MVs
    h, w    = shape
    pred    = np.zeros((h, w), dtype=np.float32)
    nv, nx_blk = mvs.shape[:2]
    ph, pw  = prev_y.shape

    for i in range(nv):
        yi = i * BLOCK_SIZE
        for j in range(nx_blk):
            xj = j * BLOCK_SIZE
            dy, dx = int(mvs[i, j, 0]), int(mvs[i, j, 1])
            ry  = max(0, min(yi + dy, ph - BLOCK_SIZE))
            rx  = max(0, min(xj + dx, pw - BLOCK_SIZE))
            bh  = min(BLOCK_SIZE, h - yi)
            bw  = min(BLOCK_SIZE, w - xj)
            pred[yi:yi+bh, xj:xj+bw] = prev_y[ry:ry+bh, rx:rx+bw].astype(np.float32)

    rec = pred + residual[:h, :w]
    return np.clip(rec, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BITSTREAM PACKAGER  (full header + per-frame data)
# ─────────────────────────────────────────────────────────────────────────────
def build_bitstream(frames_comp: list,
                    frame_types: list,
                    orig_shape: tuple,
                    quant_scale: float,
                    fps: int) -> bytes:
    """
    Header layout:
      MAGIC(4) VERSION(1) FLAGS(1)
      HEIGHT(2) WIDTH(2) FPS(1) I_PERIOD(1)
      QSCALE_x100(2) N_FRAMES(2)
    Per frame:
      FRAME_TYPE(1) [data blobs]
    """
    bs = BitstreamWriter()
    h, w = orig_shape
    n    = len(frames_comp)

    # Global header (fixed-width, easy to parse)
    bs._buf += BitstreamWriter.MAGIC
    bs._buf += struct.pack('>BBHHBBHH',
                           BitstreamWriter.VERSION,
                           0,                           # flags
                           h, w,
                           fps,
                           I_FRAME_PERIOD,
                           int(quant_scale * 100),
                           n)

    for comp, ftype in zip(frames_comp, frame_types):
        # Frame-type byte: 0x49='I', 0x50='P'
        bs._buf += struct.pack('>B', 0x49 if ftype == 'I' else 0x50)

        if ftype == 'I':
            for plane in ('Y', 'U', 'V'):
                cb_bytes, bits_str, shape_pln, _ = comp[plane]
                ph, pw = shape_pln
                bs._buf += struct.pack('>HH', ph, pw)
                bs.write_bytes_blob(cb_bytes)
                bs.write_bytes_blob(bits_str.encode('ascii'))
        else:
            bs.write_bytes_blob(comp['mv_cb_bytes'])
            bs.write_bytes_blob(comp['res_cb_bytes'])
            bs.write_bytes_blob(comp['mv_bits'].encode('ascii'))
            bs.write_bytes_blob(comp['res_bits'].encode('ascii'))
            nv, nx_blk, _ = comp['mvs_shape']
            ch, cw         = comp['curr_shape']
            rh, rw         = comp['res_shape']
            bs._buf += struct.pack('>HHHHHH', nv, nx_blk, ch, cw, rh, rw)

    return bs.get_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────
def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ─────────────────────────────────────────────────────────────────────────────
# MOTION VECTOR VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def draw_mvs(frame_rgb: np.ndarray, mvs: np.ndarray) -> np.ndarray:
    canvas = frame_rgb.copy()
    nv, nx_blk = mvs.shape[:2]
    for i in range(nv):
        for j in range(nx_blk):
            dy, dx = int(mvs[i, j, 0]), int(mvs[i, j, 1])
            sy = i * BLOCK_SIZE + BLOCK_SIZE // 2
            sx = j * BLOCK_SIZE + BLOCK_SIZE // 2
            ey = sy + dy * 2
            ex = sx + dx * 2
            cv2.arrowedLine(canvas, (sx, sy), (ex, ey),
                            (0, 230, 120), 1, tipLength=0.4)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING WORKER  (QThread)
# ─────────────────────────────────────────────────────────────────────────────
class CompressionWorker(QThread):
    # Signals
    progress_update = pyqtSignal(int, int, str, float)   # idx, total, ftype, psnr
    frame_ready     = pyqtSignal(int, object, object, object)  # idx, orig, recon, mv_vis
    finished_signal = pyqtSignal(dict)

    def __init__(self, video_path: str, params: dict, parent=None):
        super().__init__(parent)
        self.video_path   = video_path
        self.params       = params
        self._cancelled   = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        p             = self.params
        quant_scale   = p['quant_scale']
        i_period      = p['i_period']
        max_frames    = p['max_frames']
        search_range  = p['search_range']

        qmat = get_qmat(quant_scale)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit({'error': 'Cannot open video'})
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or FPS

        orig_frames   = []
        recon_frames  = []
        psnr_vals     = []
        comp_store    = []
        ftype_store   = []
        mv_vis_frames = []

        prev_y  = None
        t_start = time.perf_counter()

        idx = 0
        while not self._cancelled:
            ret, bgr = cap.read()
            if not ret or idx >= max_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yuv = rgb_to_yuv(rgb)

            ftype = 'I' if (idx == 0 or idx % i_period == 0) else 'P'

            if ftype == 'I' or prev_y is None:
                comp, rec = compress_i_frame(yuv, qmat)
                mv_vis    = rgb.copy()   # no vectors for I-frame
                ftype     = 'I'
            else:
                comp, _   = compress_p_frame(yuv, prev_y, qmat, search_range)
                rec_y     = decompress_p_frame(comp, prev_y, qmat)
                rec       = {'Y': rec_y, 'U': yuv['U'], 'V': yuv['V']}
                mv_vis    = draw_mvs(rgb, comp['mvs'])

            recon_rgb = yuv_to_rgb(rec)
            p_val     = psnr(yuv['Y'], rec['Y'])
            prev_y    = rec['Y']

            orig_frames.append(rgb)
            recon_frames.append(recon_rgb)
            psnr_vals.append(p_val)
            comp_store.append(comp)
            ftype_store.append(ftype)
            mv_vis_frames.append(mv_vis)

            # Emit signals (Qt will marshal to GUI thread)
            self.progress_update.emit(idx, max_frames, ftype, p_val)
            self.frame_ready.emit(idx, rgb, recon_rgb, mv_vis)
            idx += 1

        cap.release()
        elapsed = time.perf_counter() - t_start

        if not orig_frames:
            self.finished_signal.emit({'error': 'No frames processed'})
            return

        avg_psnr = float(np.mean(psnr_vals))

        # Build bitstream & compute ratio
        h, w  = orig_frames[0].shape[:2]
        bs    = build_bitstream(comp_store, ftype_store, (h, w), quant_scale, int(source_fps))
        total_compressed_bits = len(bs) * 8
        total_raw_bits        = len(orig_frames) * h * w * 3 * 8
        comp_ratio            = total_compressed_bits / max(total_raw_bits, 1) * 100.0

        self.finished_signal.emit({
            'psnr_vals'    : psnr_vals,
            'avg_psnr'     : avg_psnr,
            'comp_ratio'   : comp_ratio,
            'orig_frames'  : orig_frames,
            'recon_frames' : recon_frames,
            'mv_frames'    : mv_vis_frames,
            'elapsed'      : elapsed,
            'n_frames'     : len(orig_frames),
            'bitstream'    : bs,
        })


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB CANVAS
# ─────────────────────────────────────────────────────────────────────────────
class LivePSNRCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 3), dpi=96,
                          facecolor='#0d1117')
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._xs = []
        self._ys = []

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.set_title('PSNR per Frame', color='#e6edf3', fontsize=10)
        ax.set_xlabel('Frame', color='#8b949e')
        ax.set_ylabel('PSNR (dB)', color='#8b949e')
        ax.grid(True, color='#21262d', linewidth=0.6)

    def append(self, x: int, y: float):
        self._xs.append(x)
        self._ys.append(y)
        self.ax.clear()
        self._style_ax()
        self.ax.plot(self._xs, self._ys, color='#58a6ff', linewidth=1.4)
        self.ax.fill_between(self._xs, self._ys,
                             min(self._ys) - 1,
                             alpha=0.15, color='#58a6ff')
        self.draw_idle()

    def reset(self):
        self._xs.clear()
        self._ys.clear()
        self.ax.clear()
        self._style_ax()
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — numpy array → QPixmap
# ─────────────────────────────────────────────────────────────────────────────
def ndarray_to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def fit_pixmap(pix: QPixmap, label: QLabel) -> QPixmap:
    return pix.scaled(label.width(), label.height(),
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


# ─────────────────────────────────────────────────────────────────────────────
# STYLED WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG  = '#0d1117'
_PANEL_BG = '#161b22'
_BORDER   = '#30363d'
_TEXT     = '#e6edf3'
_MUTED    = '#8b949e'
_ACCENT   = '#58a6ff'
_GREEN    = '#3fb950'
_ORANGE   = '#d29922'
_RED      = '#f85149'

_BASE_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {_DARK_BG};
    color: {_TEXT};
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-size: 11px;
    color: {_MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {_ACCENT};
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QPushButton {{
    background-color: {_PANEL_BG};
    border: 1px solid {_BORDER};
    border-radius: 5px;
    color: {_TEXT};
    padding: 6px 16px;
    font-size: 12px;
}}
QPushButton:hover {{
    border-color: {_ACCENT};
    color: {_ACCENT};
}}
QPushButton:pressed {{
    background-color: {_ACCENT};
    color: {_DARK_BG};
}}
QPushButton#start_btn {{
    background-color: {_ACCENT};
    color: {_DARK_BG};
    font-weight: bold;
    border: none;
}}
QPushButton#start_btn:hover {{
    background-color: #79b8ff;
    color: {_DARK_BG};
}}
QPushButton#cancel_btn {{
    border-color: {_RED};
    color: {_RED};
}}
QProgressBar {{
    border: 1px solid {_BORDER};
    border-radius: 4px;
    background-color: {_PANEL_BG};
    text-align: center;
    color: {_TEXT};
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {_ACCENT};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {_ACCENT};
    border-radius: 2px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {_PANEL_BG};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    color: {_TEXT};
    padding: 3px 6px;
}}
QCheckBox {{
    color: {_TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {_BORDER};
    border-radius: 3px;
    background-color: {_PANEL_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {_ACCENT};
    border-color: {_ACCENT};
}}
QTabWidget::pane {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
}}
QTabBar::tab {{
    background: {_PANEL_BG};
    color: {_MUTED};
    border: 1px solid {_BORDER};
    border-bottom: none;
    padding: 6px 18px;
    border-radius: 4px 4px 0 0;
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QTabBar::tab:selected {{
    color: {_ACCENT};
    border-bottom: 2px solid {_ACCENT};
}}
QLabel#drop_zone {{
    border: 2px dashed {_BORDER};
    border-radius: 8px;
    color: {_MUTED};
    font-size: 13px;
    padding: 22px 30px;
}}
QLabel#drop_zone:hover {{
    border-color: {_ACCENT};
    color: {_ACCENT};
}}
QLabel#stat_val {{
    color: {_GREEN};
    font-size: 14px;
    font-weight: bold;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Video Compression  ·  Enhanced Edition")
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self.setStyleSheet(_BASE_STYLE)

        # State
        self.video_path        = ''
        self.worker: CompressionWorker | None = None
        self.orig_frames       = []
        self.recon_frames      = []
        self.mv_frames         = []
        self.psnr_values       = []
        self.current_frame_idx = 0
        self.playing           = False
        self._last_bitstream   = b''

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._next_frame)

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        root   = QWidget()
        root.setObjectName('root')
        outer  = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(root)

        # Left sidebar
        sidebar = self._build_sidebar()
        outer.addWidget(sidebar, 0)

        # Right content
        content = self._build_content()
        outer.addWidget(content, 1)

    def _build_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setFixedWidth(300)
        sb.setStyleSheet(f'background-color: {_PANEL_BG}; border-right: 1px solid {_BORDER};')
        ly = QVBoxLayout(sb)
        ly.setContentsMargins(16, 20, 16, 16)
        ly.setSpacing(14)

        # Title
        title = QLabel('DSP\nCOMPRESS')
        title.setStyleSheet(f'color: {_ACCENT}; font-size: 22px; font-weight: bold; letter-spacing: 2px;')
        ly.addWidget(title)

        # Drop zone
        self.lbl_drop = QLabel('⬆  Drop video here\nor click Browse')
        self.lbl_drop.setObjectName('drop_zone')
        self.lbl_drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_drop.setWordWrap(True)
        self.lbl_drop.setFixedHeight(90)
        self.lbl_drop.mousePressEvent = lambda _: self._browse()
        ly.addWidget(self.lbl_drop)

        # Config group
        cfg = QGroupBox("Compression Settings")
        grid = QGridLayout(cfg)
        grid.setSpacing(8)

        self.spin_quant = QDoubleSpinBox()
        self.spin_quant.setRange(0.1, 10.0)
        self.spin_quant.setSingleStep(0.1)
        self.spin_quant.setValue(QUANT_SCALE)
        self.spin_quant.setToolTip("1.0 = standard JPEG quant matrix. Higher = more lossy.")

        self.spin_iperiod = QSpinBox()
        self.spin_iperiod.setRange(1, 100)
        self.spin_iperiod.setValue(I_FRAME_PERIOD)
        self.spin_iperiod.setToolTip("Insert an I-frame every N frames.")

        self.spin_search = QSpinBox()
        self.spin_search.setRange(1, 32)
        self.spin_search.setValue(SEARCH_RANGE)
        self.spin_search.setToolTip("Motion search radius in pixels.")

        self.spin_maxframes = QSpinBox()
        self.spin_maxframes.setRange(1, 5000)
        self.spin_maxframes.setValue(MAX_FRAMES)
        self.spin_maxframes.setToolTip("Maximum frames to process.")

        for row, (label, widget) in enumerate([
            ("Quant Scale ×", self.spin_quant),
            ("I-frame Period", self.spin_iperiod),
            ("Search Range px", self.spin_search),
            ("Max Frames", self.spin_maxframes),
        ]):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget,        row, 1)

        ly.addWidget(cfg)

        # Action buttons
        self.btn_start  = QPushButton("▶  Start Compression")
        self.btn_start.setObjectName('start_btn')
        self.btn_start.setFixedHeight(40)
        self.btn_start.clicked.connect(self._start)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName('cancel_btn')
        self.btn_cancel.clicked.connect(self._cancel)

        self.btn_save   = QPushButton("💾  Save Reconstructed")
        self.btn_save.clicked.connect(self._save)

        ly.addWidget(self.btn_start)
        ly.addWidget(self.btn_cancel)
        ly.addWidget(self.btn_save)

        # Progress
        self.progress   = QProgressBar()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f'color: {_MUTED}; font-size: 11px;')
        ly.addWidget(self.progress)
        ly.addWidget(self.lbl_status)

        # Stats panel
        stats = QGroupBox("Session Stats")
        sg = QGridLayout(stats)
        sg.setSpacing(4)

        self.lbl_avg_psnr = self._stat_val('—')
        self.lbl_ratio    = self._stat_val('—')
        self.lbl_elapsed  = self._stat_val('—')
        self.lbl_nframes  = self._stat_val('—')

        for row, (label, widget) in enumerate([
            ("Avg PSNR (dB)", self.lbl_avg_psnr),
            ("Compression %", self.lbl_ratio),
            ("Elapsed (s)",   self.lbl_elapsed),
            ("Frames",        self.lbl_nframes),
        ]):
            sg.addWidget(QLabel(label + ':'), row, 0)
            sg.addWidget(widget,              row, 1)

        ly.addWidget(stats)
        ly.addStretch()

        return sb

    def _stat_val(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName('stat_val')
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        return lbl

    def _build_content(self) -> QWidget:
        container = QWidget()
        ly = QVBoxLayout(container)
        ly.setContentsMargins(16, 16, 16, 16)
        ly.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_comparison_tab(), "Comparison")
        self.tabs.addTab(self._build_mv_tab(),         "Motion Vectors")
        self.tabs.addTab(self._build_metrics_tab(),    "Metrics / PSNR")
        ly.addWidget(self.tabs, 1)

        return container

    def _build_comparison_tab(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)

        def _make_panel(title: str) -> tuple:
            panel = QWidget()
            panel.setStyleSheet(f'background: {_PANEL_BG}; border-radius: 6px;')
            pl = QVBoxLayout(panel)
            pl.setContentsMargins(6, 6, 6, 6)
            hdr = QLabel(title)
            hdr.setStyleSheet(f'color: {_MUTED}; font-size: 10px; letter-spacing: 1px;')
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(400, 270)
            lbl.setStyleSheet(f'background: {_DARK_BG}; border-radius: 4px;')
            pl.addWidget(hdr)
            pl.addWidget(lbl, 1)
            return panel, lbl

        orig_panel,  self.lbl_orig  = _make_panel("ORIGINAL")
        recon_panel, self.lbl_recon = _make_panel("RECONSTRUCTED")

        row.addWidget(orig_panel,  1)
        row.addWidget(recon_panel, 1)
        ly.addLayout(row, 1)

        # PSNR badge
        self.lbl_psnr_badge = QLabel("PSNR: —")
        self.lbl_psnr_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_psnr_badge.setStyleSheet(f'color: {_GREEN}; font-size: 13px; font-weight: bold;')
        ly.addWidget(self.lbl_psnr_badge)

        # Playback controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.valueChanged.connect(self._show_frame)

        self.btn_playpause = QPushButton("▶  Play")
        self.btn_playpause.setFixedWidth(100)
        self.btn_playpause.clicked.connect(self._toggle_playback)

        self.lbl_frame_counter = QLabel("0 / 0")
        self.lbl_frame_counter.setStyleSheet(f'color: {_MUTED}; font-size: 11px;')
        self.lbl_frame_counter.setFixedWidth(80)

        ctrl.addWidget(self.btn_playpause)
        ctrl.addWidget(self.slider_frame, 1)
        ctrl.addWidget(self.lbl_frame_counter)
        ly.addLayout(ctrl)

        return w

    def _build_mv_tab(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        self.lbl_mv = QLabel()
        self.lbl_mv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mv.setStyleSheet(f'background: {_DARK_BG}; border-radius: 6px;')
        ly.addWidget(self.lbl_mv)
        lbl_hint = QLabel("Green arrows show motion vectors per 8×8 block (P-frames only)")
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hint.setStyleSheet(f'color: {_MUTED}; font-size: 11px;')
        ly.addWidget(lbl_hint)
        return w

    def _build_metrics_tab(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 8, 8, 8)
        ly.setSpacing(8)

        self.psnr_canvas = LivePSNRCanvas()
        ly.addWidget(self.psnr_canvas, 1)

        info_row = QHBoxLayout()

        def _metric_card(label: str) -> QLabel:
            card = QWidget()
            card.setStyleSheet(f'background: {_PANEL_BG}; border: 1px solid {_BORDER}; border-radius: 6px;')
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 8, 12, 8)
            top = QLabel(label)
            top.setStyleSheet(f'color: {_MUTED}; font-size: 10px; letter-spacing: 1px;')
            val = QLabel('—')
            val.setStyleSheet(f'color: {_ACCENT}; font-size: 18px; font-weight: bold;')
            cl.addWidget(top)
            cl.addWidget(val)
            info_row.addWidget(card, 1)
            return val

        self.lbl_m_psnr    = _metric_card("AVG PSNR (dB)")
        self.lbl_m_ratio   = _metric_card("COMP RATIO %")
        self.lbl_m_elapsed = _metric_card("ELAPSED (s)")
        self.lbl_m_frames  = _metric_card("FRAMES")

        ly.addLayout(info_row)
        return w

    # ── drag & drop ────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._set_video(urls[0].toLocalFile())

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        self.video_path = path
        name = os.path.basename(path)
        cap  = cv2.VideoCapture(path)
        if cap.isOpened():
            nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_ = cap.get(cv2.CAP_PROP_FPS)
            w_   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            self.lbl_drop.setText(
                f'✓  {name}\n{w_}×{h_}  ·  {fps_:.0f} fps  ·  {nf} frames'
            )
            self.spin_maxframes.setValue(min(nf, 500))
        else:
            self.lbl_drop.setText(f'✓  {name}')

    # ── compression control ────────────────────────────────────────────────
    def _start(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "No Video", "Please select a valid video file first.")
            return

        # Reset state
        self.orig_frames.clear()
        self.recon_frames.clear()
        self.mv_frames.clear()
        self.psnr_values.clear()
        self.current_frame_idx = 0
        self.psnr_canvas.reset()
        self.slider_frame.setMaximum(0)
        self.slider_frame.setValue(0)
        self._clear_labels()

        params = {
            'quant_scale' : self.spin_quant.value(),
            'i_period'    : self.spin_iperiod.value(),
            'search_range': self.spin_search.value(),
            'max_frames'  : self.spin_maxframes.value(),
        }
        self.progress.setRange(0, params['max_frames'])
        self.progress.setValue(0)
        self.lbl_status.setText("Starting…")

        self.worker = CompressionWorker(self.video_path, params, self)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.lbl_status.setText("Cancelling…")

    def _save(self):
        if not self.recon_frames:
            QMessageBox.warning(self, "Nothing to Save", "Run compression first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstructed Video", "reconstructed.mp4", "MP4 (*.mp4)"
        )
        if not path:
            return
        h, w = self.recon_frames[0].shape[:2]
        out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        for f in self.recon_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        QMessageBox.information(self, "Saved", f"Reconstructed video saved to:\n{path}")

    # ── worker callbacks ───────────────────────────────────────────────────
    def _on_progress(self, idx: int, total: int, ftype: str, p_val: float):
        self.progress.setValue(idx + 1)
        badge = f"[{ftype}]" if ftype == 'I' else f"[{ftype}]"
        self.lbl_status.setText(
            f"Frame {idx + 1} / {total}  {badge}  PSNR {p_val:.1f} dB"
        )

    def _on_frame(self, idx: int, orig: np.ndarray,
                  recon: np.ndarray, mv_vis: np.ndarray):
        self.orig_frames.append(orig)
        self.recon_frames.append(recon)
        self.mv_frames.append(mv_vis)

        n = len(self.orig_frames)
        self.slider_frame.setMaximum(max(0, n - 1))

        if n == 1:
            self._show_frame(0)

    def _on_finished(self, result: dict):
        if 'error' in result:
            QMessageBox.critical(self, "Error", result['error'])
            self.lbl_status.setText("Error")
            return

        psnr_vals = result['psnr_vals']
        self.psnr_values = psnr_vals
        avg   = result['avg_psnr']
        ratio = result['comp_ratio']
        el    = result['elapsed']
        nf    = result['n_frames']
        self._last_bitstream = result['bitstream']

        # Update stats sidebar
        self.lbl_avg_psnr.setText(f"{avg:.2f}")
        self.lbl_ratio.setText(f"{ratio:.1f}")
        self.lbl_elapsed.setText(f"{el:.1f}")
        self.lbl_nframes.setText(str(nf))

        # Update metrics tab
        self.lbl_m_psnr.setText(f"{avg:.2f}")
        self.lbl_m_ratio.setText(f"{ratio:.1f}")
        self.lbl_m_elapsed.setText(f"{el:.1f}")
        self.lbl_m_frames.setText(str(nf))

        # Full PSNR plot
        self.psnr_canvas.ax.clear()
        self.psnr_canvas._style_ax()
        self.psnr_canvas.ax.plot(psnr_vals, color=_ACCENT, linewidth=1.4)
        self.psnr_canvas.ax.fill_between(
            range(len(psnr_vals)), psnr_vals,
            min(psnr_vals) - 0.5, alpha=0.15, color=_ACCENT
        )
        self.psnr_canvas.draw_idle()

        self.lbl_status.setText(
            f"Done  ·  {nf} frames  ·  {avg:.2f} dB  ·  {ratio:.1f}%  ·  {el:.1f}s"
        )
        self.progress.setValue(nf)

    # ── playback ───────────────────────────────────────────────────────────
    def _show_frame(self, idx: int):
        if not self.orig_frames:
            return
        idx = max(0, min(idx, len(self.orig_frames) - 1))
        self.current_frame_idx = idx
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(idx)
        self.slider_frame.blockSignals(False)

        def _display(lbl: QLabel, arr: np.ndarray):
            pix = ndarray_to_pixmap(arr)
            lbl.setPixmap(fit_pixmap(pix, lbl))

        _display(self.lbl_orig,  self.orig_frames[idx])
        _display(self.lbl_recon, self.recon_frames[idx])
        _display(self.lbl_mv,    self.mv_frames[idx])

        if self.psnr_values and idx < len(self.psnr_values):
            self.lbl_psnr_badge.setText(f"Frame {idx + 1}  ·  PSNR: {self.psnr_values[idx]:.2f} dB")

        total = len(self.orig_frames)
        self.lbl_frame_counter.setText(f"{idx + 1} / {total}")

    def _toggle_playback(self):
        if not self.orig_frames:
            return
        if self.playing:
            self.play_timer.stop()
            self.btn_playpause.setText("▶  Play")
            self.playing = False
        else:
            self.play_timer.start(max(16, int(1000 / FPS)))
            self.btn_playpause.setText("⏸  Pause")
            self.playing = True

    def _next_frame(self):
        if not self.orig_frames:
            return
        nxt = (self.current_frame_idx + 1) % len(self.orig_frames)
        self._show_frame(nxt)

    def _clear_labels(self):
        for lbl in (self.lbl_orig, self.lbl_recon, self.lbl_mv):
            lbl.clear()
        self.lbl_psnr_badge.setText("PSNR: —")
        self.lbl_frame_counter.setText("0 / 0")
        for lbl in (self.lbl_avg_psnr, self.lbl_ratio,
                    self.lbl_elapsed, self.lbl_nframes,
                    self.lbl_m_psnr, self.lbl_m_ratio,
                    self.lbl_m_elapsed, self.lbl_m_frames):
            lbl.setText('—')

    # ── resize event — re-render current frame at new size ─────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.orig_frames:
            self._show_frame(self.current_frame_idx)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec())