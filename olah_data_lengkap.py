"""
=============================================================================
EEG + PPG BATCH PROCESSING PIPELINE
Dataset: OpenBCI 8-channel EEG + 1-channel Digital PPG
Format  : TabSep TXT, header 2 baris (% komentar), lalu data
Alur    : Load → Trim (65s awal → buang 65s, ambil 60s) →
          Zapline → MAICA → Bandpass per band →
          EEG Features (AbsPow, RelPow, Ratios, Asym, SASI) +
          PPG HRV Features → Label dari nama file (DASS) →
          Export ke CSV
=============================================================================
Struktur nama file: <Nama><RecID>_S<stres>_A<anxiety>_D<depresi>.txt
Contoh            : Eldi3_S6_A10_D1.txt
=============================================================================
"""
import matplotlib
# matplotlib.use('TkAgg')  # ← paksa backend interaktif
import matplotlib.pyplot as plt
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, welch, iirnotch, hilbert, find_peaks
from scipy.linalg import eigh
from scipy.ndimage import convolve1d
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import nolds          # untuk DFA
import antropy        # untuk MSE
warnings.filterwarnings("ignore")
from scipy.signal import hilbert as scipy_hilbert
from itertools import combinations
from sklearn.metrics import mutual_info_score
# ============================================================
# COMPATIBILITY PATCH NUMPY 2.x
# np.trapz lama diganti menjadi np.trapezoid di NumPy baru.
# Patch ini membuat kode lama tetap jalan tanpa mengganti semua trapz.
# ============================================================
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
FS          = 250           # Sampling rate (Hz)
SKIP_SEC    = 0.0         # Potong 1 menit 5 detik pertama
KEEP_SEC    = 120.0       # Potong 2 menit terakhir
SKIP_SAMP   = SKIP_SEC * FS
KEEP_SAMP   = KEEP_SEC * FS
overlap = 0.5
DATA_DIR   = r""
OUTPUT_CSV = r""
CH_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4']

# Band EEG standar (filter butterworth)
EEG_BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta' : (13.0, 30.0),
}
CCM_ENABLED  = True    # set False untuk skip saat debugging cepat
CCM_E        = 5       # embedding dimension (jurnal: 10, tapi mahal komputasinya)
CCM_TAU      = 1       # time delay (setelah downsample)
CCM_L        = 500     # max library length per pasang (batasi waktu komputasi)
CCM_DS       = 15      # downsample envelope (250Hz/15 ≈ 16.7Hz)
# ─────────────────────────────────────────────
# 1. UTILITAS FILTER
# ─────────────────────────────────────────────

def bandpass_filter(x, fs, low=1.0, high=40.0, order=4):
    """Bandpass butterworth, x shape: (n_ch, n_samp)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def bandpass_single(x, fs, low, high, order=4):
    """Bandpass butterworth, x shape: (n_ch, n_samp) — untuk 1 band"""
    nyq = fs / 2.0
    low  = max(low,  0.1)
    high = min(high, nyq - 0.1)
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def notch_filter(x, fs, freq=50.0, Q=50):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, x, axis=-1)

# ─────────────────────────────────────────────
# 2. ZAPLINE  (sama persis dengan eeg_processor_8.py)
# ─────────────────────────────────────────────

# def zapline_strong_auto_plus1(data, fs, fline=50, width=1.5):
#     """
#     Zapline: hapus noise harmonik fline dari data EEG.
#     data shape: (n_ch, n_time)
#     """
#     n_ch, n_time = data.shape
#     harmonics = [fline, fline * 2]
#     ref = data.copy()
#     for f in harmonics:
#         if f < fs / 2:
#             q = f / width
#             b, a = iirnotch(f, q, fs)
#             ref = filtfilt(b, a, ref, axis=-1)

#     noise_ref = data - ref
#     C1 = (noise_ref @ noise_ref.T) / n_time
#     C0 = (data @ data.T) / n_time
#     C0 += np.eye(n_ch) * (1e-6 * np.trace(C0) / n_ch)

#     vals, vecs = eigh(C1, C0)
#     idx = np.argsort(vals)[::-1]
#     vals, vecs = vals[idx], vecs[:, idx]

#     threshold = np.mean(vals) + np.std(vals)
#     n_auto   = np.sum(vals > threshold)
#     nremove  = min(n_auto, max(1, n_ch // 2))

#     W = vecs[:, :nremove]
#     S = W.T @ data
#     T = (data @ S.T) @ np.linalg.inv(S @ S.T + np.eye(nremove) * 1e-9)
#     return data - T @ S, nremove

def variance_burst_ratio(sig, fs, win_sec=0.5, step_frac=0.5):
    """
    Hitung rasio variansi lokal maksimum vs median.
    EMG bursty → rasio tinggi. EEG stasioner → rasio rendah.
    """
    win  = int(win_sec * fs)          # 0.5s × 250 = 125 samp
    step = max(1, int(win * step_frac))
    if len(sig) < win:
        return 1.0
    local_vars = np.array([
        np.var(sig[i : i + win])
        for i in range(0, len(sig) - win, step)
    ])
    median_var = np.median(local_vars)
    return float(np.max(local_vars) / (median_var + 1e-12))

# ─────────────────────────────────────────────
# 3. MAICA  (sama persis dengan eeg_processor_8.py — FIXED PARTIAL MAICA)
# ─────────────────────────────────────────────
def run_maica(data, fs, beats=None):  # <-- Tambahkan parameter beats
    """
    MAICA: Multi-Augmented ICA artifact rejection.
    data shape: (n_ch, n_time) → return shape sama (n_ch, n_time)
    """
    from scipy.signal import filtfilt
    from numpy import corrcoef

    n_ch, n_time = data.shape

    # ── 1. Tidak ada ICA_CUT — langsung pakai seluruh data ──────────────
    X_raw = data.T   # (n_time, n_ch)
    n_samples, n_ch_in = X_raw.shape

    # ── 2. Hitung L_bank adaptif dari channel pertama ────────────────────
    def zero_phase_ma(x, N):
        b = np.ones(N) / N
        return filtfilt(b, [1.0], x)

    def pearson_abs(a, b):
        c = corrcoef(a, b)
        return abs(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

    x1 = X_raw[:, 0]
    L_bank = []
    previous_input = None
    for N in range(2, n_samples // 4):
        new_input = zero_phase_ma(x1, N)
        if pearson_abs(new_input, x1) < 0.6:   # sudah terlalu halus → stop
            break
        if pearson_abs(new_input, x1) > 0.98:  # masih terlalu mirip asli → skip
            continue
        if previous_input is not None and pearson_abs(new_input, previous_input) > 0.98:
            continue                                                             # terlalu mirip window sebelumnya → skip
        previous_input = new_input
        L_bank.append(N)
        if len(L_bank) >= 2:                    # cukup 2 filter per paper
            break
    if not L_bank:
        L_bank = [2, 3]   # fallback minimal

    print(f"  [MAICA] L_bank: {L_bank}")

    # ── 3. Augmentasi dengan cek korelasi ────────────────────────────────
    X_aug_list = []
    orig_col_indices = []   # posisi kolom channel asli di X_aug

    for i in range(n_ch_in):
        orig = X_raw[:, i]
        orig_col_indices.append(len(X_aug_list))   # catat posisi SEBELUM ditambah
        X_aug_list.append(orig)

        for N in L_bank:
            virt = zero_phase_ma(orig, N)
            corr_ok = True
            for existing in X_aug_list:
                if abs(corrcoef(virt, existing)[0, 1]) > 0.98:
                    corr_ok = False
                    break
            if corr_ok:
                X_aug_list.append(virt)

    X_aug = np.column_stack(X_aug_list)
    print(f"  [MAICA] Augmented matrix: {X_aug.shape}, "
          f"orig_col_indices: {orig_col_indices}")

    # ── 4. Normalisasi ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_aug)

    # ── 5. FastICA — DISESUAIKAN DENGAN NOTEBOOK v3 ──────────────────────
    n_comp = min(16, X_aug.shape[1])
    ica = FastICA(n_components=n_comp, algorithm='parallel', 
                  whiten='unit-variance', whiten_solver='eigh', 
                  max_iter=2500, tol=0.0001, random_state=42) # <-- solver dikunci, tol diperketat
    S_ica = ica.fit_transform(X_norm)

    # ── TAMBAHAN SINKRONISASI: Bentuk Sinyal Kontinu Referensi BCG ────────
    pulse_smooth = None
    if beats is not None and len(beats) > 2:
        pulse_ref = np.zeros(n_samples)
        for bt in beats:
            if bt < n_samples:
                pulse_ref[bt] = 1.0
        window_len = int(fs * 0.3)
        pulse_smooth = np.convolve(pulse_ref, np.ones(window_len) / window_len, 'same')

    # ── 6. Rejection IC — Kurtosis Terkunci + Korelasi BCG ───────────────
    S_clean = S_ica.copy()
    n_rejected = 0
    KURT_THRESHOLD = 1.6  # <-- Diselaraskan dengan threshold eksak Notebook Anda

    ic_info = []  # simpan info tiap IC untuk visualisasi

    for i in range(S_ica.shape[1]):
        sig_ic = S_ica[:, i]
        k_val = kurtosis(sig_ic)
        # EMG burst detection
        vbr      = variance_burst_ratio(sig_ic, fs)
        is_emg   = vbr > 6   # threshold awal konservatif
        # Hitung korelasi dengan template BCG kontinu
        is_bcg = False
        corr_pulse = 0.0
        if pulse_smooth is not None:
            c_matrix = corrcoef(pulse_smooth, sig_ic)
            corr_pulse = abs(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0
            is_bcg = corr_pulse > 0.065 # <-- Threshold BCG identik

        rejected = k_val > KURT_THRESHOLD or is_bcg or is_emg

        # Jika lolos salah satu kriteria abnormalitas artifak
        if rejected:
            S_clean[:, i] = 0
            n_rejected += 1

        ic_info.append({
            'ic_idx'    : i,
            'signal'    : sig_ic.copy(),
            'kurtosis'  : k_val,
            'corr_bcg'  : corr_pulse,
            'is_bcg'    : is_bcg,
            'vbr'      : vbr,        # ← tambah
            'is_emg'   : is_emg, 
            'rejected'  : rejected,
        })
            
    print(f"  [MAICA] IC ditolak: {n_rejected}/{S_ica.shape[1]}")

    # ── 7. Rekonstruksi — ambil channel asli via orig_col_indices ─────────
    if n_rejected == 0:
        return data.copy(), 0, ic_info

    X_rec = scaler.inverse_transform(ica.inverse_transform(S_clean))
    clean_segment = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])

    return clean_segment, n_rejected, ic_info
# ─────────────────────────────────────────────
# 4. EKSTRAKSI BAND (Butterworth — standar)
# ─────────────────────────────────────────────

def extract_bands(clean_eeg, fs):
    band_signals = {}
    for band, (lo, hi) in EEG_BANDS.items():
        band_signals[band] = bandpass_single(clean_eeg, fs, lo, hi, order=5)
        print(f"  [BPF] Band {band} ({lo}-{hi} Hz) selesai")
    return band_signals

# ─────────────────────────────────────────────
# 5. POWER FEATURES
# ─────────────────────────────────────────────

def compute_band_power(sig_ch, fs, lo, hi):
    """Hitung power spektral di band [lo, hi] Hz untuk 1 channel (1-D array)."""
    f, Pxx = welch(sig_ch, fs=fs, nperseg=int(fs * 2))
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(Pxx[mask], f[mask])) if np.any(mask) else 1e-12

# ─────────────────────────────────────────────
# 5b. NONLINEAR EEG HELPER FUNCTIONS
# ─────────────────────────────────────────────

def compute_dfa_eeg(sig_1d):
    """DFA exponent untuk 1 channel EEG (1-D array)."""
    try:
        if len(sig_1d) < 50:
            return np.nan
        return float(nolds.dfa(sig_1d))
    except Exception:
        return np.nan

def compute_mse_eeg(sig_1d, max_scale=5):
    """
    Multiscale Entropy (MSE) rata-rata skala 1..max_scale.
    antropy.multiscale_entropy mengembalikan array per skala.
    """
    try:
        if len(sig_1d) < 200:
            return np.nan
        mse_vals = antropy.multiscale_entropy(sig_1d, scale=max_scale)
        return float(np.nanmean(mse_vals))
    except Exception:
        return np.nan

def compute_rqa_eeg(sig_1d, dim=3, tau=1, threshold=0.1):
    """
    Recurrence Quantification Analysis (RQA) — RR, DET, LAM.
    Implementasi manual ringan (tanpa library eksternal berat).
    """
    try:
        n = len(sig_1d)
        if n < 50:
            return np.nan, np.nan, np.nan

        # Bangun matriks rekurensi
        # Ambil subsample jika sinyal terlalu panjang (hemat memori)
        max_n = 500
        if n > max_n:
            idx = np.linspace(0, n - dim * tau - 1, max_n, dtype=int)
            x = sig_1d[idx]
        else:
            x = sig_1d

        # State-space reconstruction (delay embedding)
        N = len(x) - (dim - 1) * tau
        if N < 10:
            return np.nan, np.nan, np.nan
        X = np.array([x[i:i + dim * tau:tau] for i in range(N)])

        # Jarak Euclidean ternormalisasi
        dists = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        thr   = threshold * np.std(sig_1d)
        R     = (dists <= thr).astype(np.float32)
        np.fill_diagonal(R, 0)

        total_pts = N * N - N
        rr  = float(np.sum(R) / (total_pts + 1e-12))

        # Diagonal lines (determinism)
        det_count = 0
        total_diag = 0
        for d in range(2, N):
            diag = np.diag(R, d)
            runs = np.diff(np.concatenate([[0], diag, [0]]))
            starts = np.where(runs == 1)[0]
            ends   = np.where(runs == -1)[0]
            lengths = ends - starts
            total_diag += len(diag)
            det_count  += np.sum(lengths[lengths >= 2])
        det = float(det_count / (np.sum(R) / 2 + 1e-12))

        # Vertical lines (laminarity)
        lam_count = 0
        for col in range(N):
            col_arr = R[:, col]
            runs = np.diff(np.concatenate([[0], col_arr, [0]]))
            starts = np.where(runs == 1)[0]
            ends   = np.where(runs == -1)[0]
            lengths = ends - starts
            lam_count += np.sum(lengths[lengths >= 2])
        lam = float(lam_count / (np.sum(R) + 1e-12))

        return rr, det, lam
    except Exception:
        return np.nan, np.nan, np.nan
    
# ─────────────────────────────────────────────
# 5c. LZC + FUNCTIONAL CONNECTIVITY FEATURES
# ─────────────────────────────────────────────

def compute_lzc_eeg(sig_1d, max_len=5000):
    """
    Lempel-Ziv Complexity (LZC) untuk 1 sinyal EEG.
    - Sinyal dibinarisasi berdasarkan median.
    - Dinormalisasi dengan n/log2(n).
    - max_len dipakai agar komputasi tidak terlalu berat untuk segmen 2 menit.
    """
    try:
        x = np.asarray(sig_1d, dtype=float)

        if len(x) < 50 or np.nanstd(x) < 1e-12:
            return np.nan

        x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmedian(x), neginf=np.nanmedian(x))

        # Supaya LZC tidak terlalu mahal pada sinyal 2 menit
        if len(x) > max_len:
            idx = np.linspace(0, len(x) - 1, max_len).astype(int)
            x = x[idx]

        med = np.median(x)
        s = ''.join('1' if v > med else '0' for v in x)

        n = len(s)
        i, k, l = 0, 1, 1
        c = 1

        while True:
            if i + k > n or l + k > n:
                c += 1
                break

            if s[i:i+k] == s[l:l+k]:
                k += 1
            else:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    if l + 1 > n:
                        break
                    i = 0
                    k = 1

        norm = n / np.log2(n)
        return float(c / (norm + 1e-12))

    except Exception:
        return np.nan


def compute_pcc_pair(x, y):
    """Pearson correlation coefficient untuk dua sinyal."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            return np.nan

        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return np.nan


def compute_pli_pair(x, y):
    """
    Phase Lag Index.
    Input sebaiknya sudah bandpass per band.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            return np.nan

        phase_x = np.angle(hilbert(x))
        phase_y = np.angle(hilbert(y))

        phase_diff = phase_x - phase_y
        pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))

        return float(pli)

    except Exception:
        return np.nan


def compute_mi_pair(x, y, n_bins=16):
    """
    Mutual Information sederhana dengan diskretisasi histogram.
    Input sebaiknya sudah bandpass per band.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            return np.nan

        x = np.nan_to_num(x, nan=np.nanmedian(x))
        y = np.nan_to_num(y, nan=np.nanmedian(y))

        x_edges = np.histogram_bin_edges(x, bins=n_bins)
        y_edges = np.histogram_bin_edges(y, bins=n_bins)

        x_bin = np.digitize(x, x_edges)
        y_bin = np.digitize(y, y_edges)

        return float(mutual_info_score(x_bin, y_bin))

    except Exception:
        return np.nan


def extract_original_connectivity_features(band_signals, channels=None, prefix="ORI_CONN"):
    """
    Functional connectivity dari channel original.
    Fokus default: frontal channel Fp1, Fp2, F3, F4 untuk anxiety/GAD.
    """
    features = {}

    if channels is None:
        channels = ["Fp1", "Fp2", "F3", "F4"]

    ch_to_idx = {ch: i for i, ch in enumerate(CH_NAMES)}
    valid_channels = [ch for ch in channels if ch in ch_to_idx]

    for band, band_data in band_signals.items():
        for ch1, ch2 in combinations(valid_channels, 2):
            i = ch_to_idx[ch1]
            j = ch_to_idx[ch2]

            if i >= band_data.shape[0] or j >= band_data.shape[0]:
                continue

            x = band_data[i]
            y = band_data[j]
            pair = f"{ch1}-{ch2}"

            features[f"{prefix}_PCC_{band}_{pair}"] = compute_pcc_pair(x, y)
            features[f"{prefix}_PLI_{band}_{pair}"] = compute_pli_pair(x, y)
            features[f"{prefix}_MI_{band}_{pair}"]  = compute_mi_pair(x, y)

    return features


def extract_differential_connectivity_features(band_signals, source_channels=None, prefix="DIFF_CONN"):
    """
    Differential channel features:
    1. Buat sinyal selisih antar channel frontal.
    2. Hitung connectivity antar differential channels.
    
    Adaptasi dari paper GAD:
    differential channel = X_channel_i - X_channel_j
    lalu fitur connectivity diekstrak dari sinyal differential tersebut.
    """
    features = {}

    if source_channels is None:
        source_channels = ["Fp1", "Fp2", "F3", "F4"]

    ch_to_idx = {ch: i for i, ch in enumerate(CH_NAMES)}
    valid_channels = [ch for ch in source_channels if ch in ch_to_idx]

    for band, band_data in band_signals.items():

        diff_signals = {}

        # Buat differential channels dari channel frontal
        for ch1, ch2 in combinations(valid_channels, 2):
            i = ch_to_idx[ch1]
            j = ch_to_idx[ch2]

            if i >= band_data.shape[0] or j >= band_data.shape[0]:
                continue

            diff_name = f"{ch1}_minus_{ch2}"
            diff_sig = band_data[i] - band_data[j]
            diff_sig = diff_sig - np.nanmean(diff_sig)

            diff_signals[diff_name] = diff_sig

        # Connectivity antar differential channels
        diff_names = list(diff_signals.keys())

        for d1, d2 in combinations(diff_names, 2):
            x = diff_signals[d1]
            y = diff_signals[d2]

            pair = f"{d1}__{d2}"

            features[f"{prefix}_PCC_{band}_{pair}"] = compute_pcc_pair(x, y)
            features[f"{prefix}_PLI_{band}_{pair}"] = compute_pli_pair(x, y)
            features[f"{prefix}_MI_{band}_{pair}"]  = compute_mi_pair(x, y)

    return features

# ─────────────────────────────────────────────────────────────────
# TAMBAHAN BARU: EKSTRAKSI AMPLOP OSILASI EEG (HILBERT TRANSFORM)
# ─────────────────────────────────────────────────────────────────

def compute_band_envelope(signal_data, downsample=15):
    """
    Menghitung amplitudo instan (amplop) dari sinyal EEG menggunakan 
    Hilbert Transform, kemudian melakukan downsampling untuk menyesuaikan 
    skala waktu komputasi CCM sesuai dengan paper Zhang et al. (2025).
    """
    # 1. Hitung snyal analitik menggunakan Hilbert Transform
    analytic_signal = hilbert(signal_data)
    
    # 2. Ambil nilai absolutnya untuk mendapatkan amplop amplitudo
    amplitude_envelope = np.abs(analytic_signal)
    
    # 3. Lakukan downsampling (mengambil sampel setiap kelipatan 'downsample')
    # Jika FS asli 250 Hz dan downsample=15, fs baru menjadi ~16.67 Hz
    downsampled_envelope = amplitude_envelope[::downsample]
    
    return downsampled_envelope

import numpy as np
from scipy.signal import argrelmin

# ─────────────────────────────────────────────────────────────
# ESTIMASI τ OPTIMAL — via Average Mutual Information (AMI)
# ─────────────────────────────────────────────────────────────

def compute_ami(x, tau, n_bins=32):
    """
    Hitung Average Mutual Information antara x(t) dan x(t+tau).
    Gunakan histogram 2D sebagai estimasi distribusi joint.
    """
    x1 = x[:-tau]
    x2 = x[tau:]
    if not np.all(np.isfinite(x1)) or not np.all(np.isfinite(x2)):
        return 0.0
    
    # Normalisasi ke range [0, 1] untuk stabilitas bin
    x_min, x_max = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    eps = 1e-12
    x1_n = (x1 - x_min) / (x_max - x_min + eps)
    x2_n = (x2 - x_min) / (x_max - x_min + eps)
    
    # Hitung distribusi joint P(x1, x2)
    joint_hist, _, _ = np.histogram2d(x1_n, x2_n, bins=n_bins)
    joint_hist = joint_hist / (joint_hist.sum() + eps)
    
    # Distribusi marginal
    p_x1 = joint_hist.sum(axis=1, keepdims=True)
    p_x2 = joint_hist.sum(axis=0, keepdims=True)
    
    # Hitung MI = sum P(x,y) * log(P(x,y) / P(x)*P(y))
    denom = p_x1 * p_x2 + eps
    log_term = np.where(joint_hist > eps, np.log2(joint_hist / denom + eps), 0.0)
    ami = np.sum(joint_hist * log_term)
    return max(ami, 0.0)


def estimate_tau_ami(x, max_tau=20, n_bins=32):
    """
    Cari τ optimal = lag pada minimum pertama kurva AMI.
    Jika tidak ada minimum, ambil lag di mana AMI turun ke 1/e dari nilai awal.
    """
    ami_values = [compute_ami(x, tau, n_bins) for tau in range(1, max_tau + 1)]
    ami_arr = np.array(ami_values)
    
    # Cari minimum lokal pertama
    local_mins = argrelmin(ami_arr, order=1)[0]
    if len(local_mins) > 0:
        return int(local_mins[0]) + 1  # +1 karena index mulai dari tau=1
    
    # Fallback: turun ke 1/e dari nilai awal
    threshold = ami_arr[0] / np.e
    below = np.where(ami_arr < threshold)[0]
    if len(below) > 0:
        return int(below[0]) + 1
    
    # Fallback terakhir
    return max_tau // 2


# ─────────────────────────────────────────────────────────────
# ESTIMASI E OPTIMAL — via False Nearest Neighbors (FNN)
# ─────────────────────────────────────────────────────────────

def compute_fnn_ratio(x, E, tau, r_tol=15.0, a_tol=2.0):
    """
    Hitung rasio False Nearest Neighbors untuk embedding dimension E.
    
    r_tol: threshold jarak relatif (Kennel et al. gunakan 15)
    a_tol: threshold jarak absolut relatif terhadap std sinyal
    """
    n = len(x)
    n_vec = n - E * tau  # jumlah vektor embedding di dimensi E+1
    if n_vec <= E + 2:
        return 1.0  # tidak cukup data → anggap semua FNN
    
    # Bangun manifold di dimensi E
    def make_manifold_local(dim):
        nv = n - dim * tau
        M = np.zeros((nv, dim))
        for i in range(dim):
            M[:, i] = x[i * tau : i * tau + nv]
        return M
    
    M_E   = make_manifold_local(E)
    M_E1  = make_manifold_local(E + 1)
    
    n_common = min(M_E.shape[0], M_E1.shape[0])
    M_E  = M_E[:n_common]
    M_E1 = M_E1[:n_common]
    
    Ra = np.std(x)  # standar deviasi sinyal untuk normalisasi
    
    fnn_count = 0
    valid_count = 0
    
    for i in range(n_common):
        # Jarak di dimensi E
        diffs_E = M_E - M_E[i]
        dists_E = np.linalg.norm(diffs_E, axis=1)
        dists_E[i] = np.inf
        
        nn_idx = np.argmin(dists_E)
        r_nn   = dists_E[nn_idx]
        
        if r_nn < 1e-10:
            continue  # titik identik, skip
        
        # Jarak di dimensi E+1 untuk pasangan yang sama
        r_nn1 = np.linalg.norm(M_E1[i] - M_E1[nn_idx])
        
        # Kriteria FNN ganda (Kennel et al. 1992)
        crit1 = (abs(r_nn1 - r_nn) / r_nn) > r_tol
        crit2 = (r_nn1 / Ra) > a_tol
        
        if crit1 or crit2:
            fnn_count += 1
        valid_count += 1
    
    if valid_count == 0:
        return 1.0
    return fnn_count / valid_count


def estimate_E_fnn(x, tau, max_E=15, fnn_threshold=0.05):
    """
    Cari E minimal di mana rasio FNN turun di bawah threshold (default 5%).
    """
    for E in range(1, max_E + 1):
        fnn_ratio = compute_fnn_ratio(x, E, tau)
        if fnn_ratio < fnn_threshold:
            return E
    
    # Jika tidak konvergen, kembalikan max_E
    return max_E


# ─────────────────────────────────────────────────────────────
# WRAPPER: Estimasi Parameter CCM untuk Sepasang Sinyal
# ─────────────────────────────────────────────────────────────

def estimate_ccm_params(eeg_env, hrv_seq, max_tau=20, max_E=15, fnn_threshold=0.05):
    """
    Estimasi τ dan E optimal dari KEDUA sinyal, ambil nilai yang lebih konservatif.
    Sesuai pendekatan paper: AMI untuk τ, FNN untuk E.
    
    Returns:
        tau_opt (int), E_opt (int)
    """
    # Estimasi τ dari kedua sinyal, ambil nilai maksimum (lebih konservatif)
    tau_eeg = estimate_tau_ami(eeg_env, max_tau=max_tau)
    tau_hrv = estimate_tau_ami(hrv_seq, max_tau=max_tau)
    tau_opt = max(tau_eeg, tau_hrv)
    tau_opt = max(1, min(tau_opt, max_tau))  # clamp
    
    # Estimasi E dengan τ yang sudah dipilih
    E_eeg = estimate_E_fnn(eeg_env, tau=tau_opt, max_E=max_E, fnn_threshold=fnn_threshold)
    E_hrv = estimate_E_fnn(hrv_seq, tau=tau_opt, max_E=max_E, fnn_threshold=fnn_threshold)
    E_opt = max(E_eeg, E_hrv)  # ambil lebih besar untuk keamanan
    E_opt = max(2, min(E_opt, max_E))  # clamp
    
    return tau_opt, E_opt

def extract_eeg_features(clean_eeg, band_signals, ppg_peaks, fs):
    """
    Dari dict band signals → AbsPow, RelPow, Power Ratios, Asym, SASI.
    Mengembalikan dict fitur datar (key = nama fitur).
    """
    eps = 1e-12
    n_ch = next(iter(band_signals.values())).shape[0]
    bands = list(band_signals.keys())   # Delta, Theta, Alpha, Beta

    # ── Absolute Power ──────────────────────────────────────
    abs_pow = {band: [] for band in bands}
    for band in bands:
        lo, hi = EEG_BANDS[band]
        for ch in range(n_ch):
            p = compute_band_power(band_signals[band][ch], fs, lo, hi)
            abs_pow[band].append(p)

    # ── Relative Power ──────────────────────────────────────
    rel_pow = {band: [] for band in bands}
    for ch in range(n_ch):
        total = sum(abs_pow[b][ch] for b in bands) + eps
        for band in bands:
            rel_pow[band].append(abs_pow[band][ch] / total)

    # ── Kumpulkan ke dict ────────────────────────────────────
    features = {}

    for band in bands:
        for ch in range(n_ch):
            ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
            features[f"AbsPow_{band}_{ch_lbl}"]  = abs_pow[band][ch]
            features[f"RelPow_{band}_{ch_lbl}"]  = rel_pow[band][ch]

    # ── Power Ratios (per channel) ───────────────────────────
    for ch in range(n_ch):
        ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
        
        d = abs_pow['Delta'][ch]
        t = abs_pow['Theta'][ch]
        a = abs_pow['Alpha'][ch]
        b = abs_pow['Beta'][ch]
        features[f"Ratio_DeltaTheta_{ch_lbl}"] = d / (t + eps)
        features[f"Ratio_DeltaAlpha_{ch_lbl}"] = d / (a + eps)
        features[f"Ratio_DeltaBeta_{ch_lbl}"]  = d / (b + eps)
        features[f"Ratio_ThetaAlpha_{ch_lbl}"] = t / (a + eps)
        features[f"Ratio_ThetaBeta_{ch_lbl}"]  = t / (b + eps)
        features[f"Ratio_AlphaBeta_{ch_lbl}"]  = a / (b + eps)

    # ── Amplitude Asymmetry (pasangan elektrod) ──────────────
    # Pasangan: Fp1-Fp2, F3-F4, T3-T4, P3-P4  → index (0,1),(2,3),(4,5),(6,7)
    asym_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    pair_labels = [('Fp1', 'Fp2'), ('F3', 'F4'), ('T3', 'T4'), ('P3', 'P4')]
    for band in bands:
        for (L, R), (lbl_l, lbl_r) in zip(asym_pairs, pair_labels):
            if L < n_ch and R < n_ch:
                aL = abs_pow[band][L]
                aR = abs_pow[band][R]
                asym = (aL - aR) / (aL + aR + eps)
                features[f"Asym_{band}_{lbl_l}-{lbl_r}"] = asym

    # ── SASI (Stress-Alpha Suppression Index) ────────────────
    # SASI = (Beta - Alpha) / (Beta + Alpha)  per channel
    for ch in range(n_ch):
        ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
        b_val = abs_pow['Beta'][ch]
        a_val = abs_pow['Alpha'][ch]
        features[f"SASI_{ch_lbl}"] = (b_val - a_val) / (b_val + a_val + eps)
# ── EEG Nonlinear Features (per channel) ────────────────
# =========================================================================
    # PERBAIKAN: FITUR NON-LINEAR TUNGGAL (EEG ONLY) USING BROADBAND
    # =========================================================================
    for ch in range(n_ch):
        ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
        
        # PERBAIKAN KEBOCORAN: Gunakan clean_eeg broadband
        broadband_signal = clean_eeg[ch] 

        features[f"DFA_{ch_lbl}"]  = compute_dfa_eeg(broadband_signal)
        features[f"MSE_{ch_lbl}"]  = compute_mse_eeg(broadband_signal)
        
        rr_val, det_val, lam_val   = compute_rqa_eeg(broadband_signal)
        features[f"RQA_RR_{ch_lbl}"]  = rr_val
        features[f"RQA_DET_{ch_lbl}"] = det_val
        features[f"RQA_LAM_{ch_lbl}"] = lam_val

        # ============================================================
        # BARU — LZC untuk paper depression/MDD
        # LZC broadband per channel
        # ============================================================
        features[f"LZC_{ch_lbl}"] = compute_lzc_eeg(broadband_signal)

        # LZC per band per channel
        for band in bands:
            features[f"LZC_{band}_{ch_lbl}"] = compute_lzc_eeg(
                band_signals[band][ch]
            )

    # =========================================================================
    # BARU — ORI + DIFF FUNCTIONAL CONNECTIVITY untuk paper anxiety/GAD
    # Fokus frontal: Fp1, Fp2, F3, F4.
    # Fitur: PCC, PLI, MI.
    # =========================================================================
    ori_conn_features = extract_original_connectivity_features(
        band_signals=band_signals,
        channels=["Fp1", "Fp2", "F3", "F4"],
        prefix="ORI_CONN"
    )
    features.update(ori_conn_features)

    diff_conn_features = extract_differential_connectivity_features(
        band_signals=band_signals,
        source_channels=["Fp1", "Fp2", "F3", "F4"],
        prefix="DIFF_CONN"
    )
    features.update(diff_conn_features)
    
    # =========================================================================
    # IMPLEMENTASI UTAMA: FUSI NON-LINEAR CCM (EEG-HRV) SESUAI PAPER
    # =========================================================================
    if CCM_ENABLED and ppg_peaks is not None:
        # Hitung target sampling rate setelah downsample eeg (250 / 15 = 16.6667 Hz)
        target_hrv_fs = fs / CCM_DS 
        
        # Ekstrak deretan waktu HRV yang sinkron secara durasi (KEEP_SEC = 60s)
        hrv_seq = get_hrv_sequence_fixed(ppg_peaks, fs, target_fs=target_hrv_fs, total_seconds=KEEP_SEC)
        
        if hrv_seq is not None:
            # Hitung min_len sekali, konsisten untuk semua band×ch
            sample_env_len = len(compute_band_envelope(band_signals[bands[0]][0], downsample=CCM_DS))
            min_len = min(sample_env_len, len(hrv_seq))
            Y_hrv = hrv_seq[:min_len]

            for band in bands:
                for ch in range(n_ch):
                    
                    ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
                    
                    # 1. Amplop Sinyal Osilasi EEG per-band (Hilbert + Downsample)
                    eeg_env = compute_band_envelope(band_signals[band][ch], downsample=CCM_DS)
                    if abs(len(eeg_env) - len(hrv_seq)) > 2:
                        print(f"  [WARN] CCM skip {band}/{ch_lbl}: env={len(eeg_env)} vs hrv={len(hrv_seq)}")
                        continue
                    # 2. Sinkronisasi panjang array (menghindari error pembulatan float)
                    # min_len = min(len(eeg_env), len(hrv_seq))
                    X_eeg = eeg_env[:min_len]
                    # Y_hrv = hrv_seq[:min_len]
                    # ── BARU: Estimasi parameter adaptif per pasang sinyal ──
                    tau_opt, E_opt = estimate_ccm_params(X_eeg, Y_hrv)
                    # Simpan parameter yang digunakan (opsional, untuk debugging/analisis)
                    features[f"CCM_tau_{band}_{ch_lbl}"] = tau_opt
                    features[f"CCM_E_{band}_{ch_lbl}"]   = E_opt
                    
                    # 3. Hitung Interaksi Kausalitas Non-linier Dua Arah Terarah
                    ccm_h2b, ccm_b2h = compute_ccm_fixed(X_eeg, Y_hrv, E=E_opt, tau=tau_opt)
                    # 4. Simpan ke dalam dictionary fitur fusi
                    features[f"CCM_HeartToBrain_{band}_{ch_lbl}"] = ccm_h2b
                    features[f"CCM_BrainToHeart_{band}_{ch_lbl}"] = ccm_b2h
                    
    return features

# ─────────────────────────────────────────────
# 6. PPG / HRV FEATURES
# ─────────────────────────────────────────────

def detect_ppg_peaks(ppg_raw, fs):
    """
    Deteksi peak PPG digital/analog menggunakan:
    - threshold adaptif
    - midpoint pulse
    - validasi pulse width
    - refractory period
    """

    # =========================================================
    # THRESHOLD ADAPTIF
    # =========================================================
    unique_vals = np.unique(ppg_raw)

    if len(unique_vals) <= 10:
        threshold_ppg = (unique_vals.max() + unique_vals.min()) / 2
    else:
        threshold_ppg = np.median(ppg_raw)

    # =========================================================
    # BINARISASI
    # =========================================================
    ppg_binary = (ppg_raw > threshold_ppg).astype(int)
    diff_binary = np.diff(ppg_binary)

    rising_edges  = np.where(diff_binary > 0)[0] + 1
    falling_edges = np.where(diff_binary < 0)[0] + 1

    # =========================================================
    # MIDPOINT DETECTION
    # =========================================================
    raw_midpoints = []

    for r in rising_edges:

        f_candidates = falling_edges[falling_edges > r]

        if len(f_candidates) == 0:
            continue

        f = f_candidates[0]

        pulse_width = f - r

        # validasi 20–600 ms
        if pulse_width < int(0.020 * fs):
            continue

        if pulse_width > int(0.600 * fs):
            continue

        midpoint = (r + f) // 2
        raw_midpoints.append(midpoint)

    raw_midpoints = np.array(raw_midpoints)

    # =========================================================
    # REFRACTORY PERIOD / DEBOUNCE
    # =========================================================
    min_rr_samples = int(0.35 * fs)

    if len(raw_midpoints) == 0:
        return np.array([])

    ppg_peaks = [raw_midpoints[0]]

    for pk in raw_midpoints[1:]:

        if pk - ppg_peaks[-1] >= min_rr_samples:
            ppg_peaks.append(pk)

    return np.array(ppg_peaks)

# ─────────────────────────────────────────────────────────────────
# REVISI PERBAIKAN SINKRONISASI HRV & CCM
# ─────────────────────────────────────────────────────────────────

def get_hrv_sequence_fixed(ppg_peaks, fs, target_fs=16.6667, total_seconds=60.0):
    """
    Ekstrak urutan HRV seragam yang selaras dengan panjang amplop EEG.
    ppg_peaks: output berupa array indeks sampel dari detect_ppg_peaks
    """
    if len(ppg_peaks) < 5:
        return None
        
    # Hitung interval NN (RR) dalam milidetik
    rr_ms = np.diff(ppg_peaks / fs) * 1000.0
    t_rr = ppg_peaks[1:] / fs  # Waktu kemunculan detak (detik)
    
    if t_rr[-1] - t_rr[0] < 10.0:
        return None
        
    # Buat sumbu waktu yang presisi dimulai dari 0s hingga durasi segmen (60s)
    # Panjang array akan tepat sama dengan data EEG setelah downsample
    n_expected_samples = int(total_seconds * target_fs)
    t_uniform = np.linspace(0.0, total_seconds, n_expected_samples, endpoint=False)
    
    # Lakukan ekstrapolasi batas aman jika puncak pertama/terakhir terpotong trim
    from scipy.interpolate import interp1d
    f_interp = interp1d(t_rr - t_rr[0], rr_ms, kind='linear', fill_value="extrapolate")
    hrv_seq = f_interp(t_uniform)

    # Clamp nilai fisiologis RR (300–2000 ms) — cegah extrapolasi liar
    hrv_seq = np.clip(hrv_seq, 300, 2000)

    # Zero-mean standard sesuai paper
    hrv_seq -= np.mean(hrv_seq)
    return hrv_seq


def cross_map_fixed(M_lib, target, E, tau):
    """
    Fungsi Helper untuk mengestimasi target Y dari Manifold M_x
    M_lib: Manifold pengondisi (shape: n_vec, E)
    target: Sinyal target yang akan direkonstruksi (1-D array)
    """
    n_vec = M_lib.shape[0]
    offset = (E - 1) * tau
    
    # Target harus disesuaikan indeksnya dengan manifold (karena delay embedding memotong awal data)
    target_sliced = target[offset : offset + n_vec]
    estimated_target = np.zeros(n_vec)
    
    for t in range(n_vec):
        query = M_lib[t]
        
        # Hitung jarak Euclidean ke seluruh titik lain di manifold
        diffs = M_lib - query
        dists = np.linalg.norm(diffs, axis=-1)
        dists[t] = np.inf # Jangan hitung diri sendiri
        
        # Cari E + 1 tetangga terdekat (Nearest Neighbors)
        nn_indices = np.argsort(dists)[:(E + 1)]
        min_dists = dists[nn_indices]
        
        # Jika ada jarak 0, beri bobot epsilon aman
        if min_dists[0] == 0:
            weights = np.zeros(E + 1)
            weights[0] = 1.0
        else:
            # Bobot eksponensial Sugihara
            u = np.exp(-min_dists / min_dists[0])
            weights = u / np.sum(u)
            
        # Estimasi nilai target berdasarkan kombinasi linear tetangga
        estimated_target[t] = np.sum(weights * target_sliced[nn_indices])
        
    # Hitung korelasi Pearson antara target asli vs estimasi
    from numpy import corrcoef
    if np.std(estimated_target) == 0 or np.std(target_sliced) == 0:
        return np.nan
    c = corrcoef(target_sliced, estimated_target)
    return abs(c[0, 1]) if not np.isnan(c[0, 1]) else np.nan


def compute_ccm_fixed(X, Y, E=5, tau=1):
    """
    Menerapkan Teorema Sugihara secara tepat.
    X = Amplop EEG, Y = Deretan HRV (Panjang Harus Sama)
    """
    n = len(X)
    
    def make_manifold(x, E, tau):
        n_vec = len(x) - (E - 1) * tau
        if n_vec <= E + 1: return None
        M = np.zeros((n_vec, E))
        for i in range(E):
            M[:, i] = x[i * tau : i * tau + n_vec]
        return M

    Mx = make_manifold(X, E, tau)
    My = make_manifold(Y, E, tau)
    
    if Mx is None or My is None:
        return np.nan, np.nan
        
    # Sesuai Teorema: Manifold X memprediksi target Y -> Kekuatan X mempengaruhi Y
    # Maka Mx -> Y mengukur pengaruh Jantung (X) ke Otak (Y) jika X diset sebagai HRV.
    # Agar konsisten dengan istilah paper:
    # Jika input fungsi: X = EEG_Envelope, Y = HRV_Sequence:
    ccm_brain_to_heart = cross_map_fixed(Mx, Y, E, tau) # Mx merekonstruksi Y
    ccm_heart_to_brain = cross_map_fixed(My, X, E, tau) # My merekonstruksi X
    
    return ccm_heart_to_brain, ccm_brain_to_heart

# def extract_bhf_features(band_signals, ppg_digital, fs,
#                           E=5, tau=1, downsample=15, ccm_L=500):
#     """
#     Ekstrak fitur Brain-Heart Fusion (BHF) berbasis CCM.
    
#     Untuk setiap band × setiap channel:
#       - CCM_b2h_Band_Ch : brain → heart (band drive HRV?)
#       - CCM_h2b_Band_Ch : heart → brain (HRV drive band?)
    
#     Sesuai jurnal Zhang et al. (2025) tapi tanpa MEMD —
#     band langsung dari extract_bands_fir.
    
#     Returns: dict fitur datar.
#     """
#     features = {}
    
#     # Dapatkan HRV sebagai time series
#     hrv_seq = get_hrv_sequence_fixed(ppg_digital, fs)
#     if hrv_seq is None or len(hrv_seq) < 30:
#         # Fallback: isi NaN untuk semua fitur CCM
#         n_ch = next(iter(band_signals.values())).shape[0]
#         for band in band_signals:
#             for ch in range(n_ch):
#                 ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
#                 features[f"CCM_b2h_{band}_{ch_lbl}"] = np.nan
#                 features[f"CCM_h2b_{band}_{ch_lbl}"] = np.nan
#         return features
    
#     # Downsample HRV ke panjang yang sama dengan envelope
#     # (envelope sudah di-downsample saat compute_band_envelope)
#     n_ch = next(iter(band_signals.values())).shape[0]
    
#     for band, band_sig in band_signals.items():
#         for ch in range(n_ch):
#             ch_lbl = CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}"
            
#             # Amplop band channel ini
#             envelope = compute_band_envelope(band_sig[ch], downsample=downsample)
            
#             # Sesuaikan panjang HRV dengan envelope
#             # HRV sudah di 4Hz, envelope di 250/15 ≈ 16.7Hz — perlu diseragamkan
#             min_len  = min(len(envelope), len(hrv_seq))
#             env_cut  = envelope[:min_len]
#             hrv_cut  = hrv_seq[:min_len]
            
#             # Seragamkan dengan resample sederhana jika panjang berbeda jauh
#             if abs(len(envelope) - len(hrv_seq)) > 5:
#                 # Resample hrv_seq ke panjang envelope
#                 from scipy.signal import resample
#                 hrv_rs  = resample(hrv_seq, len(envelope))
#                 min_len = min(len(envelope), len(hrv_rs))
#                 env_cut = envelope[:min_len]
#                 hrv_cut = hrv_rs[:min_len]
            
#             # Batasi panjang untuk efisiensi
#             if len(env_cut) > ccm_L:
#                 env_cut = env_cut[:ccm_L]
#                 hrv_cut = hrv_cut[:ccm_L]
            
         
            
#             features[f"CCM_b2h_{band}_{ch_lbl}"] = ccm_b2h
#             features[f"CCM_h2b_{band}_{ch_lbl}"] = ccm_h2b
    
#     return features

def extract_ppg_features(ppg_digital, fs):
    """
    Hitung fitur HRV dari sinyal PPG digital (kotak).
    MeanNN, MeanHR, RMSSD, SDNN, pNN50, TotalPower, LF, HF, SD1, SD2, SD12.
    """
    eps = 1e-12
    beats = detect_ppg_peaks(ppg_digital, fs)

    if len(beats) < 3:
        return {k: np.nan for k in [
            'PPG_MeanNN', 'PPG_MeanHR', 'PPG_RMSSD', 'PPG_SDNN', 'PPG_pNN50',
            'PPG_TotalPower', 'PPG_LF', 'PPG_HF',
            'PPG_SD1', 'PPG_SD2', 'PPG_SD12']}

    rr_ms  = np.diff(beats / fs) * 1000.0  # interval NN dalam ms

    mean_nn = float(np.mean(rr_ms))
    mean_hr = float(60000.0 / (mean_nn + eps))
    rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    sdnn    = float(np.std(rr_ms, ddof=1))
    pnn50   = float(100.0 * np.sum(np.abs(np.diff(rr_ms)) > 50) / (len(rr_ms) - 1 + eps))

    # ── Frekuensi Domain HRV ─────────────────────────────────
    # Interpolasi RR menjadi sinyal seragam pada 4 Hz
    t_rr      = beats[1:] / fs
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / 4.0)
    if len(t_rr) > 1 and len(t_uniform) > 4:
        rr_interp = np.interp(t_uniform, t_rr, rr_ms)
        rr_interp -= np.mean(rr_interp)
        f_hrv, Pxx_hrv = welch(rr_interp, fs=4.0,
                                nperseg=min(len(rr_interp), 256))
        lf_mask  = (f_hrv >= 0.04) & (f_hrv <= 0.15)
        hf_mask  = (f_hrv >= 0.15) & (f_hrv <= 0.40)
        tp_mask  = (f_hrv >= 0.003) & (f_hrv <= 0.40)

        lf_pow   = float(np.trapezoid(Pxx_hrv[lf_mask],  f_hrv[lf_mask]))  if np.any(lf_mask) else np.nan
        hf_pow   = float(np.trapezoid(Pxx_hrv[hf_mask],  f_hrv[hf_mask]))  if np.any(hf_mask) else np.nan
        tp_pow   = float(np.trapezoid(Pxx_hrv[tp_mask],  f_hrv[tp_mask]))   if np.any(tp_mask) else np.nan
    else:
        lf_pow = hf_pow = tp_pow = np.nan

    # ── Poincaré Plot (SD1, SD2) ─────────────────────────────
    if len(rr_ms) >= 2:
        rr1 = rr_ms[:-1]
        rr2 = rr_ms[1:]
        sd1 = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
        sd2 = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
        sd12 = float(sd1 / (sd2 + eps))
    else:
        sd1 = sd2 = sd12 = np.nan
    # ── DFA α1, α2 (nonlinear HRV) ──────────────────────────
    try:
        if len(rr_ms) >= 16:
            # α1: fluktuasi pendek (4–16 beat)
            dfa_alpha1 = float(nolds.dfa(rr_ms, nvals=range(4, 17)))
            # α2: fluktuasi panjang (16–64 beat, jika cukup data)
            if len(rr_ms) >= 64:
                dfa_alpha2 = float(nolds.dfa(rr_ms, nvals=range(16, 65)))
            else:
                dfa_alpha2 = np.nan
        else:
            dfa_alpha1 = dfa_alpha2 = np.nan
    except Exception:
        dfa_alpha1 = dfa_alpha2 = np.nan
    return {
        'PPG_MeanNN'    : mean_nn,
        'PPG_MeanHR'    : mean_hr,
        'PPG_RMSSD'     : rmssd,
        'PPG_SDNN'      : sdnn,
        'PPG_pNN50'     : pnn50,
        'PPG_TotalPower': tp_pow,
        'PPG_LF'        : lf_pow,
        'PPG_HF'        : hf_pow,
        'PPG_SD1'       : sd1,
        'PPG_SD2'       : sd2,
        'PPG_SD12'      : sd12,
        'PPG_DFA_alpha1': dfa_alpha1,  # BARU
        'PPG_DFA_alpha2': dfa_alpha2,  # BARU
 
    }

# ─────────────────────────────────────────────
# 7. PARSER NAMA FILE → LABEL DASS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 7. PARSER NAMA FILE → LABEL DASS
# ─────────────────────────────────────────────
def parse_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]

    # ============================================================
    # FORMAT BARU VIRTUAL+FISIK:
    # nur_rec1_clean_14.59dB_nur1_S20_A17_D14_F_seg004_MAICA
    # ============================================================
    pattern_virtual = (
        r'^([A-Za-z]+)_rec(\d+)_clean_([-]?\d+\.?\d*)dB_'
        r'([A-Za-z]+)(\d+)_S(\d+)_A(\d+)_D(\d+)_([FM])'
        r'(?:_seg\d+)?(?:_MAICA)?$'
    )

    m = re.match(pattern_virtual, base)
    if m:
        return {
            'snr_db'    : float(m.group(3)),
            'subject'   : m.group(1),      # contoh: nur
            'rec_id'    : int(m.group(2)), # contoh: rec1 -> 1
            'stress'    : int(m.group(6)),
            'anxiety'   : int(m.group(7)),
            'depression': int(m.group(8)),
            'gender'    : m.group(9),
            'filename'  : os.path.basename(fname),
        }

    # ============================================================
    # FORMAT LAMA:
    # clean_8.21dB_Zainab3_S4_A6_D0_F
    # ============================================================
    pattern_old = r'^clean_([-]?\d+\.?\d*)dB_([A-Za-z]+)(\d+)_S(\d+)_A(\d+)_D(\d+)_([FM])'

    m = re.match(pattern_old, base)
    if m:
        return {
            'snr_db'    : float(m.group(1)),
            'subject'   : m.group(2),
            'rec_id'    : int(m.group(3)),
            'stress'    : int(m.group(4)),
            'anxiety'   : int(m.group(5)),
            'depression': int(m.group(6)),
            'gender'    : m.group(7),
            'filename'  : os.path.basename(fname),
        }

    else:
        print(f"[WARN] Nama file tidak cocok pola DASS: {os.path.basename(fname)}")
        return {
            'snr_db'    : np.nan,
            'subject'   : base,
            'rec_id'    : 0,
            'stress'    : np.nan,
            'anxiety'   : np.nan,
            'depression': np.nan,
            'gender'    : np.nan,
            'filename'  : os.path.basename(fname),
        }
# ─────────────────────────────────────────────
# 8. LOAD & TRIM FILE TXT
# ─────────────────────────────────────────────

# SESUDAH
def load_csv(filepath, fs=FS, seg_sec=120, overlap=0.5):
    df = pd.read_csv(filepath)

    # ============================================================
    # PRIORITAS 1: format EEG_ch1..EEG_ch8
    # ============================================================
    eeg_cols = [c for c in df.columns if c.startswith('EEG_')]

    # ============================================================
    # PRIORITAS 2: format channel name virtual+fisik
    # Sesuaikan urutan agar konsisten dengan pipeline 8 channel:
    # Fp1, Fp2, F3, F4, T3, T4, P3, P4
    # ============================================================
    if len(eeg_cols) == 0:
        preferred_cols = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4']

        # Kalau suatu saat file memakai T7/T8, tetap bisa jalan
        if not all(c in df.columns for c in preferred_cols):
            preferred_cols = ['Fp1', 'Fp2', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']

        missing = [c for c in preferred_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom EEG tidak lengkap di file {filepath}. "
                f"Missing: {missing}. Kolom tersedia: {list(df.columns)}"
            )

        eeg_cols = preferred_cols

    print(f"  [INFO] EEG cols dipakai: {eeg_cols}")

    ppg_col = 'PPG'

    raw_eeg = df[eeg_cols].values.T.astype(np.float64)

    ppg_digital = df[ppg_col].values.astype(np.float64) if ppg_col in df.columns \
                  else np.zeros(len(df))

    seg_samp  = int(seg_sec * fs)
    step_samp = int(seg_samp * (1 - overlap))
    n_total   = raw_eeg.shape[1]

    segments = []
    start = 0

    while start + seg_samp <= n_total:
        end = start + seg_samp
        segments.append((
            raw_eeg[:, start:end],
            ppg_digital[start:end]
        ))
        start += step_samp

    if not segments:
        raise ValueError(
            f"Durasi data ({n_total/fs:.1f}s) tidak mencukupi 1 segmen {seg_sec}s. "
            f"File: {filepath}"
        )

    return segments

# ─────────────────────────────────────────────
# 9. PIPELINE PER FILE
# ─────────────────────────────────────────────
from scipy.signal import firwin, kaiserord, lfilter

def design_fir_bandpass(lo, hi, fs, ripple_db=60, width_hz=0.5):
    """
    Desain FIR bandpass dengan Kaiser window.
    
    ripple_db : atenuasi stopband (dB) — 60 cukup untuk EEG
    width_hz  : lebar transition band (Hz) — 0.5 Hz cukup tajam
    """
    nyq = fs / 2.0
    width_norm = width_hz / nyq
    
    # Kaiser order & beta otomatis dari spesifikasi ripple
    numtaps, beta = kaiserord(ripple_db, width_norm)
    
    # Pastikan numtaps ganjil (syarat FIR zero-phase)
    if numtaps % 2 == 0:
        numtaps += 1
    
    # Batas minimal: sinyal harus > 3× numtaps
    # Untuk 60s × 250Hz = 15000 samp, numtaps ~251 → aman
    
    lo_norm = lo / nyq
    hi_norm = hi / nyq
    
    # Klip agar tidak menyentuh 0 atau 1 (Nyquist)
    lo_norm = max(lo_norm, width_norm * 1.5)
    hi_norm = min(hi_norm, 1.0 - width_norm * 1.5)
    
    taps = firwin(numtaps, [lo_norm, hi_norm],
                  window=('kaiser', beta),
                  pass_zero=False)
    return taps

def bandpass_fir(x, fs, lo, hi, ripple_db=60, width_hz=0.5):
    """
    FIR bandpass zero-phase via filtfilt.
    x shape: (n_ch, n_samp)
    """
    taps = design_fir_bandpass(lo, hi, fs, ripple_db, width_hz)
    
    # filtfilt butuh padlen < n_samp; cek aman
    padlen = 3 * (len(taps) - 1)
    if x.shape[1] <= padlen:
        # Fallback ke Butterworth jika sinyal terlalu pendek
        print(f"  [WARN] FIR padlen={padlen} > n_samp={x.shape[1]}, fallback Butterworth")
        return bandpass_single(x, fs, lo, hi)
    
    return filtfilt(taps, 1.0, x, axis=-1)

# def extract_bands_fir(clean_eeg, fs):
#     """
#     Ekstraksi band EEG dengan FIR Kaiser — lebih tajam dari Butterworth.
#     Return dict: {band_name: (n_ch, n_time)}
#     """
#     band_signals = {}
#     for band, (lo, hi) in EEG_BANDS.items():
#         band_signals[band] = bandpass_fir(clean_eeg, fs, lo, hi)
#         print(f"  [FIR] Band {band} ({lo}-{hi} Hz) selesai")
#     return band_signals

def plot_ica_components_popup(ic_info_list, fs, filename, seg_idx):
    """
    Popup grid 4×4 menampilkan semua IC hasil MAICA untuk satu segmen.
    - IC yang REJECTED: latar merah muda, label merah, garis merah
    - IC yang DITERIMA: latar putih, label hitam, garis biru gelap
    - Setiap subplot menampilkan: nomor IC, nilai kurtosis, korelasi BCG,
      alasan reject (KURT / BCG / keduanya) jika ditolak
    """
    n_ic   = len(ic_info_list)
    n_cols = 4
    n_rows = int(np.ceil(n_ic / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 2.8 * n_rows))
    fig.patch.set_facecolor('#f5f5f5')
    fig.suptitle(
        f"{filename}  |  Segmen {seg_idx}  —  Hasil 16 IC MAICA\n"
        f"(merah = ditolak, biru = diterima)",
        fontsize=12, fontweight='bold', y=1.01
    )

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    KURT_THRESHOLD = 1.6
    BCG_THRESHOLD  =0.1

    for i, ic in enumerate(ic_info_list):
        ax = axes_flat[i]
        sig    = ic['signal']
        kurt   = ic['kurtosis']
        corr   = ic['corr_bcg']
        rej    = ic['rejected']
        is_bcg = ic['is_bcg']

        t = np.arange(len(sig)) / fs

        # Warna berdasarkan status
        if rej:
            bg_color   = '#ffe5e5'
            line_color = '#cc0000'
            txt_color  = '#cc0000'
            title_weight = 'bold'
        else:
            bg_color   = '#ffffff'
            line_color = '#1a5276'
            txt_color  = '#1a5276'
            title_weight = 'normal'

        ax.set_facecolor(bg_color)
        ax.plot(t, sig, color=line_color, lw=0.6, alpha=0.85)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')

        # Tentukan alasan reject
        reasons = []
        if kurt > KURT_THRESHOLD:
            reasons.append('KURT')
        if is_bcg:
            reasons.append('BCG')
        rej_str = f"  ✗ {'+'.join(reasons)}" if reasons else ""

        # Judul IC
        ax.set_title(
            f"IC {i+1}{rej_str}",
            fontsize=9, fontweight=title_weight,
            color=txt_color, pad=3
        )

        # Anotasi kurtosis & korelasi BCG di dalam plot
        kurt_color = '#cc0000' if kurt > KURT_THRESHOLD else '#2c3e50'
        corr_color = '#cc0000' if is_bcg else '#2c3e50'

        ax.text(0.02, 0.96,
                f"Kurt: {kurt:.2f}",
                transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left',
                color=kurt_color,
                fontweight='bold' if kurt > KURT_THRESHOLD else 'normal')
        ax.text(0.02, 0.78,
                f"rBCG: {corr:.3f}",
                transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left',
                color=corr_color,
                fontweight='bold' if is_bcg else 'normal')

        # Garis threshold kurtosis (referensi visual)
        if abs(sig).max() > 0:
            ax.axhline(0, color='#aaaaaa', lw=0.4, ls='--')

        # Ambil dari ic_info
        vbr    = ic.get('vbr', 0.0)
        is_emg = ic.get('is_emg', False)

        VBR_THRESHOLD = 8.0

        # Warna & alasan reject — update bagian reasons:
        if ic.get('is_emg'):
            reasons.append('EMG')

        # Teks VBR di subplot:
        vbr_color = '#cc0000' if is_emg else '#2c3e50'
        ax.text(0.02, 0.065,
                f"VBR:  {vbr:.1f}",
                transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left',
                color=vbr_color,
                fontweight='bold' if is_emg else 'normal')


    # Sembunyikan axes kosong jika IC < 16
    for j in range(n_ic, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend ringkasan
    n_rej = sum(1 for ic in ic_info_list if ic['rejected'])
    n_acc = n_ic - n_rej
    legend_txt = (
        f"Total IC: {n_ic}   |   Diterima: {n_acc}   |   Ditolak: {n_rej}\n"
        f"Threshold → Kurt > {KURT_THRESHOLD}  |  rBCG > {BCG_THRESHOLD} |  VBR > {VBR_THRESHOLD}"
    )
    fig.text(0.5, -0.01, legend_txt,
             ha='center', va='top', fontsize=9,
             color='#555555',
             bbox=dict(boxstyle='round,pad=0.4', fc='#eeeeee', ec='#bbbbbb'))

    plt.tight_layout()
    plt.show()


def plot_segments_popup(segments_before, segments_after, fs, filename,
                        seg_indices=None, batch_size=2):
    """
    Popup perbandingan before vs after ICA, ditampilkan per BATCH_SIZE segmen.
    Before: merah transparan, After: biru solid.
    seg_indices: list nomor segmen asli (1-based) untuk judul popup.
    """
    n_seg = len(segments_after)
    n_ch  = segments_after[0].shape[0]
    if seg_indices is None:
        seg_indices = list(range(1, n_seg + 1))

    # Tampilkan per batch_size segmen
    for batch_start in range(0, n_seg, batch_size):
        batch_end  = min(batch_start + batch_size, n_seg)
        b_before   = segments_before[batch_start:batch_end]
        b_after    = segments_after[batch_start:batch_end]
        b_indices  = seg_indices[batch_start:batch_end]
        nb         = len(b_after)

        fig, axes = plt.subplots(n_ch, nb,
                                 figsize=(7 * nb, n_ch * 1.5),
                                 sharex='col', sharey='row')
        fig.suptitle(
            f"{filename} — Segmen {b_indices[0]}–{b_indices[-1]} "
            f"| Before (merah) vs After ICA (biru)",
            fontsize=11
        )

        if nb == 1: axes = axes[:, np.newaxis]
        if n_ch == 1: axes = axes[np.newaxis, :]

        for s in range(nb):
            t = np.arange(b_after[s].shape[1]) / fs
            axes[0, s].set_title(f"Segmen {b_indices[s]}", fontsize=10)

            for ch in range(n_ch):
                ax = axes[ch, s]
                sig_before = b_before[s][ch] - np.mean(b_before[s][ch])
                sig_after  = b_after[s][ch]  - np.mean(b_after[s][ch])
                ax.plot(t, sig_before, color='#e53935', lw=0.7, alpha=0.35)
                ax.plot(t, sig_after,  color='#1a73e8', lw=0.9, alpha=0.90)
                ax.grid(alpha=0.3)
                ax.set_yticks([])
                if s == 0:
                    ax.set_ylabel(CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}",
                                  fontsize=9, rotation=0, labelpad=30)
            axes[-1, s].set_xlabel("Waktu (s)")

        axes[0, -1].legend(
            handles=[
                plt.Line2D([0],[0], color='#e53935', alpha=0.35, lw=1.5, label='Before ICA'),
                plt.Line2D([0],[0], color='#1a73e8', alpha=0.90, lw=1.5, label='After ICA'),
            ],
            loc='upper right', fontsize=8, framealpha=0.7
        )
        plt.tight_layout()
        plt.show()
    
# def plot_segment_popup(eeg_clean, fs, seg_idx, filename):
#     """Tampilkan popup plot semua channel hasil preprocessing."""
#     n_ch, n_samp = eeg_clean.shape
#     t = np.arange(n_samp) / fs

#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(n_ch, 1, figsize=(14, n_ch * 1.5), sharex=True)
#     fig.suptitle(f"{filename} — Segmen {seg_idx} (post-MAICA)", fontsize=12)

#     if n_ch == 1:
#         axes = [axes]

#     for ch, ax in enumerate(axes):
#         ax.plot(t, eeg_clean[ch], linewidth=0.6, color='steelblue')
#         ax.set_ylabel(CH_NAMES[ch] if ch < len(CH_NAMES) else f"ch{ch}",
#                       fontsize=9, rotation=0, labelpad=30)
#         ax.grid(alpha=0.3)
#         ax.set_yticks([])

#     axes[-1].set_xlabel("Waktu (s)")
#     plt.tight_layout()
#     plt.show()  # ini yang memunculkan popup
def process_file(filepath, fs=FS):
    print(f"  → Memproses: {os.path.basename(filepath)}")
    
    label_info = parse_filename(filepath)
    
    try:
        # PERBAIKAN BUG 1: Ganti 'fpath' menjadi 'filepath' sesuai argumen fungsi
        # Masukkan konstanta baru: KEEP_SEC (120) dan overlap (0.5)
        segments = load_csv(filepath, fs=fs, seg_sec=KEEP_SEC, overlap=overlap)
    except Exception as e:
        print(f"     SKIP (load error): {e}")
        return None
        
    rows_file = []
    print(f"     {len(segments)} segmen ditemukan (Durasi {KEEP_SEC}s, Overlap {overlap*100}%)")
    
    MAX_REJECTED = 11  # segmen tidak valid jika IC ditolak > nilai ini

    rows            = []
    before_segments = []
    clean_segments  = []
    valid_seg_indices = []   # nomor segmen yang lolos validasi (1-based)

    for seg_idx, (raw_eeg, ppg_digital) in enumerate(segments, start=1):

        # Hitung beats dari PPG untuk BCG detection
        beats = detect_ppg_peaks(ppg_digital, fs)
        print(f"     Segmen {seg_idx} -> PPG peaks detected: {len(beats)}")

        # MAICA — kirim beats agar BCG aktif
        try:
            eeg_clean, n_rej, ic_info = run_maica(raw_eeg, fs, beats=beats)
        except Exception as e:
            print(f"     Segmen {seg_idx}: MAICA gagal ({e}), pakai raw")
            eeg_clean, n_rej, ic_info = raw_eeg, 0, []

        # Skip segmen jika terlalu banyak IC ditolak
        if n_rej > MAX_REJECTED:
            print(f"     Segmen {seg_idx}: SKIP — {n_rej} IC ditolak (> {MAX_REJECTED})")
            continue

        before_segments.append(raw_eeg)
        clean_segments.append(eeg_clean)
        valid_seg_indices.append(seg_idx)

        # Ekstraksi band menggunakan FIR
        band_signals = extract_bands(eeg_clean, fs)

        # ── AKTIFKAN FITUR GABUNGAN + NON-LINEAR CCM ─────────────────
        # eeg_clean (broadband) dikirim agar DFA/MSE/RQA terekstrak secara valid
        eeg_features = extract_eeg_features(eeg_clean, band_signals, beats, fs)
        ppg_features = extract_ppg_features(ppg_digital, fs)

        row = {}
        row.update(label_info)
        row['seg_id'] = seg_idx
        row['n_ic_rejected'] = n_rej
        row.update(eeg_features)
        row.update(ppg_features)
        
        rows.append(row)

    # PERBAIKAN BUG 2: Bungkus log print ke dalam kondisi pembacaan array yang tepat
    # Cetak pesan "SEMUA segmen di-skip" HANYA jika list 'rows' benar-benar kosong
    if not rows:
        print(f"     SEMUA segmen di-skip (semua IC > {MAX_REJECTED}) pada file ini.")

    return rows if rows else None


# ─────────────────────────────────────────────
# 10. MAIN LOOP BATCH
# ─────────────────────────────────────────────

def run_pipeline(data_dir=DATA_DIR, output_csv=OUTPUT_CSV, fs=FS):
    """
    Cari semua file CSV di data_dir, proses satu per satu,
    simpan hasil ke CSV.
    """

    # Format nama sekarang:
    # nur_rec1_clean_14.59dB_nur1_S20_A17_D14_F_seg004_MAICA.csv
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Kalau ternyata file ada di subfolder, fallback recursive
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))

    if not csv_files:
        print(f"[ERROR] Tidak ada file .csv ditemukan di: {data_dir}")
        return

    print(f"[INFO] Ditemukan {len(csv_files)} file CSV")
    print(f"[INFO] Setiap file: skip {SKIP_SEC}s, ambil {KEEP_SEC}s\n")

    results = []
    for i, fpath in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] {os.path.basename(fpath)}")
        rows = process_file(fpath, fs)
        if rows is not None:
            results.extend(rows)

    if not results:
        print("[ERROR] Tidak ada data berhasil diproses.")
        return

    df = pd.DataFrame(results)

    info_cols  = ['filename', 'subject', 'rec_id', 'gender']
    label_cols = ['stress', 'anxiety', 'depression']
    feat_cols  = [c for c in df.columns if c not in info_cols + label_cols]
    df = df[info_cols + feat_cols + label_cols]

    df.to_csv(output_csv, index=False)
    print(f"\n[SELESAI] {len(df)} baris disimpan → {output_csv}")
    print(f"[INFO]    Total fitur: {len(feat_cols)}")
    return df

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df_hasil = run_pipeline()

    if df_hasil is not None:
        print("\n=== Preview 5 Baris Pertama ===")
        # Tampilkan kolom info + label saja agar tidak memenuhi terminal
        preview_cols = ['filename', 'subject', 'rec_id', 'stress', 'anxiety', 'depression']
        print(df_hasil[preview_cols].head())
        print(f"\nShape hasil akhir: {df_hasil.shape}")
