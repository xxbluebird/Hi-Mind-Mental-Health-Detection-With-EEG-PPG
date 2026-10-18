"""
EEG Super-Resolution / Virtual Channel Reconstruction — Supercomputer-ready
Dataset : SEED converted .npy, SAVE_MODE=sr_8ch
Input   : Fp1, Fp2, P3, P4
Target  : F3, F4, T7, T8
Window  : 2 menit, 250 Hz, shape file (8, 30000)

Strategi:
- Model lama separate frontal/temporal + alpha per-channel + learnable input gate.
- Loss temporal dengan beta-specific objective ringan.
- FFT loss dioptimalkan: 1x rFFT untuk broad/beta/relative beta per forward loss.
- AMP mixed precision + TF32 untuk GPU besar.
- Checkpoint minimal: best validation loss dan best corr-score saja.
- Mendukung grid ablation via environment variable RUN_GRID=1.

Contoh single run:
python eeg_sr_supercomputer_grid.py

Contoh grid di 1 GPU/job:
CUDA_VISIBLE_DEVICES=0 CONFIG_ID=baseline_inputgate python eeg_sr_supercomputer_grid.py
CUDA_VISIBLE_DEVICES=1 CONFIG_ID=beta_001_000 python eeg_sr_supercomputer_grid.py
CUDA_VISIBLE_DEVICES=2 CONFIG_ID=beta_002_001 python eeg_sr_supercomputer_grid.py

Catatan: untuk supercomputer multi-GPU, biasanya lebih efisien menjalankan beberapa job 1 GPU
untuk beberapa konfigurasi daripada 1 job DataParallel besar, kecuali dataset/model sangat besar.
"""

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from scipy.integrate import simpson
import mne

try:
    from IPython.display import display
except Exception:
    display = print

# ============================================================
# 1. CONFIG
# ============================================================
SEED = int(os.getenv("SEED", "42"))

# Data root: boleh set manual lewat env DATA_ROOT, atau auto-cari di Kaggle.
DATA_ROOT = os.getenv("DATA_ROOT", "")
if DATA_ROOT == "":
    matches = list(Path("/kaggle/input").rglob("seed_sr_250hz_2min_bp1_40_npy"))
    print("Folder kandidat hasil konversi:")
    for m in matches:
        print(m)
    if len(matches) == 0:
        raise RuntimeError(
            "Folder seed_sr_250hz_2min_bp1_40_npy tidak ditemukan. "
            "Set DATA_ROOT manual jika menjalankan di supercomputer."
        )
    DATA_ROOT = str(matches[0])
print("DATA_ROOT digunakan:", DATA_ROOT)

# Sampling
ORIGINAL_SFREQ = 200
TARGET_SFREQ = 250
SFREQ = TARGET_SFREQ
SEGMENT_SECONDS = 120
SEGMENT_SAMPLES = SFREQ * SEGMENT_SECONDS

SR_CHANNELS = ["Fp1", "Fp2", "P3", "P4", "F3", "F4", "T7", "T8"]
INPUT_CH = ["Fp1", "Fp2", "P3", "P4"]
TARGET_CH = ["F3", "F4", "T7", "T8"]
INPUT_IDX = [0, 1, 2, 3]
TARGET_IDX = [4, 5, 6, 7]

TEST_RATIO = float(os.getenv("TEST_RATIO", "0.20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # A100 bisa coba 32/64; Kaggle T4 pakai 16
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4")) # Kaggle sering aman 0; supercomputer bisa 4/8
MAX_TRAIN_SEGMENTS = None
MAX_TEST_SEGMENTS = None
EPOCHS = int(os.getenv("EPOCHS", "100"))
SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    EPOCHS = 2

USE_AMP = os.getenv("USE_AMP", "1") == "1"
USE_TF32 = os.getenv("USE_TF32", "1") == "1"
USE_DATA_PARALLEL = os.getenv("USE_DATA_PARALLEL", "1") == "1"
SAVE_CORR_CHECKPOINT = os.getenv("SAVE_CORR_CHECKPOINT", "1") == "1"
SAVE_AUX_CHECKPOINTS = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = USE_TF32
    torch.backends.cudnn.allow_tf32 = USE_TF32
    # Untuk speed supercomputer. Kalau butuh deterministik penuh, set CUDNN_BENCHMARK=0.
    torch.backends.cudnn.benchmark = os.getenv("CUDNN_BENCHMARK", "1") == "1"

# Base hyperparameters
CHANNEL_DROPOUT_PROB = 0.0
FRONTAL_DROPOUT = 0.10
TEMPORAL_DROPOUT = 0.05
FRONTAL_ALPHA_INIT = 0.85
TEMPORAL_ALPHA_INIT = 0.70
FRONTAL_ALPHA_MIN = 0.60
FRONTAL_ALPHA_MAX = 0.97
TEMPORAL_ALPHA_MIN = 0.35
TEMPORAL_ALPHA_MAX = 0.97
FRONTAL_AMP_LOSS_WEIGHT = 0.08
TEMPORAL_AMP_LOSS_WEIGHT_T7 = 0.08
TEMPORAL_AMP_LOSS_WEIGHT_T8 = 0.10

USE_INPUT_GATE = True
FRONTAL_INPUT_GATE_INIT = [1.10, 1.10, 0.90, 0.90]
TEMPORAL_INPUT_GATE_INIT = [0.85, 0.85, 1.15, 1.15]

BETA_FMIN = 13.0
BETA_FMAX = 30.0
TOTAL_FMIN = 1.0
TOTAL_FMAX = 40.0

# ============================================================
# 2. EXPERIMENT GRID / PRESETS
# ============================================================
# Grid ringan: beta loss dibuat halus karena beta loss besar menurunkan corr.
PRESETS = {
    "baseline_inputgate": {
        "beta_shape_w": 0.00,
        "beta_rel_w": 0.00,
        "corr_w_t": 0.55,
        "broad_spec_w_t": 0.15,
        "name": "baseline_inputgate_no_beta",
    },
    "beta_001_000": {
        "beta_shape_w": 0.01,
        "beta_rel_w": 0.00,
        "corr_w_t": 0.58,
        "broad_spec_w_t": 0.12,
        "name": "beta_shape001_rel000",
    },
    "beta_002_000": {
        "beta_shape_w": 0.02,
        "beta_rel_w": 0.00,
        "corr_w_t": 0.58,
        "broad_spec_w_t": 0.12,
        "name": "beta_shape002_rel000",
    },
    "beta_001_001": {
        "beta_shape_w": 0.01,
        "beta_rel_w": 0.01,
        "corr_w_t": 0.60,
        "broad_spec_w_t": 0.10,
        "name": "beta_shape001_rel001",
    },
    "beta_002_001": {
        "beta_shape_w": 0.02,
        "beta_rel_w": 0.01,
        "corr_w_t": 0.60,
        "broad_spec_w_t": 0.10,
        "name": "beta_shape002_rel001",
    },
    "beta_002_002": {
        "beta_shape_w": 0.02,
        "beta_rel_w": 0.02,
        "corr_w_t": 0.60,
        "broad_spec_w_t": 0.10,
        "name": "beta_shape002_rel002",
    },
}

CONFIG_ID = os.getenv("CONFIG_ID", "beta_002_001")
if CONFIG_ID not in PRESETS:
    raise ValueError(f"CONFIG_ID={CONFIG_ID} tidak ada. Pilihan: {list(PRESETS.keys())}")
CFG = PRESETS[CONFIG_ID]

TEMPORAL_BETA_SHAPE_WEIGHT = float(os.getenv("BETA_SHAPE_W", str(CFG["beta_shape_w"])))
TEMPORAL_BETA_REL_POWER_WEIGHT = float(os.getenv("BETA_REL_W", str(CFG["beta_rel_w"])))
TEMPORAL_CORR_W = float(os.getenv("TEMPORAL_CORR_W", str(CFG["corr_w_t"])))
TEMPORAL_BROAD_SPEC_W = float(os.getenv("TEMPORAL_BROAD_SPEC_W", str(CFG["broad_spec_w_t"])))

# Menjaga total base loss tetap masuk akal.
TEMPORAL_MAE_W = 0.20
TEMPORAL_DIFF_W = 0.10

# ============================================================
# 2b. MODEL SWITCH
# ============================================================
# MODEL_ARCH="separate"  -> model lama: frontal head + temporal head
# MODEL_ARCH="maser"     -> MASER-lite: patch token + eSS2D-lite/SSM-inspired block
# MODEL_ARCH="hybrid"    -> Hybrid CNN + MASER fusion
MODEL_ARCH = os.getenv("MODEL_ARCH", "separate").lower().strip()
if MODEL_ARCH not in ["separate", "maser", "hybrid"]:
    raise ValueError("MODEL_ARCH harus 'separate', 'maser', atau 'hybrid'.")

# MASER-lite hyperparameters.
# Patch_len=250 cocok untuk data 250 Hz karena 1 patch = 1 detik, 120 patch untuk 2 menit.
MASER_PATCH_LEN = int(os.getenv("MASER_PATCH_LEN", "250"))
MASER_D_MODEL = int(os.getenv("MASER_D_MODEL", "64"))
MASER_DEPTH = int(os.getenv("MASER_DEPTH", "2"))
MASER_DROPOUT = float(os.getenv("MASER_DROPOUT", "0.08"))
MASER_FFN_MULT = int(os.getenv("MASER_FFN_MULT", "2"))
MASER_LR = float(os.getenv("MASER_LR", "2e-4"))
MASER_WEIGHT_DECAY = float(os.getenv("MASER_WEIGHT_DECAY", "1e-4"))
MASER_SMOOTH_W = float(os.getenv("MASER_SMOOTH_W", "0.001"))
MASER_LOSS_TEMPORAL_W = float(os.getenv("MASER_LOSS_TEMPORAL_W", "0.45"))

# ============================================================
# 2c. HYBRID CNN + MASER FUSION CONFIG
# ============================================================
# Hybrid memakai dua backbone pretrained:
# - CNN lama separate frontal/temporal
# - MASER-lite
# lalu menggabungkan prediksi keduanya dengan alpha per-channel dan opsional residual refiner.
HYBRID_CNN_CKPT = os.getenv(
    "HYBRID_CNN_CKPT",
    "/drive0-storage/Anna/EEG_SR_SUPERGRID/sr_beta_shape001_rel001_seed42/"
    "best_corr_sr_beta_shape001_rel001_seed42.pt",
)
HYBRID_MASER_CKPT = os.getenv(
    "HYBRID_MASER_CKPT",
    "/drive0-storage/Anna/EEG_SR_SUPERGRID/maser_beta002_rel002_patch250_d96_depth3_lowsmooth_seed42/"
    "best_corr_maser_beta002_rel002_patch250_d96_depth3_lowsmooth_seed42.pt",
)

# Pilihan:
# fixed_alpha      : tidak training, hanya pakai bobot alpha tetap.
# learned_alpha    : training alpha per-channel saja.
# residual_refine  : alpha + residual Conv1D refiner.
# alpha_residual   : sama dengan residual_refine, nama alternatif.
HYBRID_MODE = os.getenv("HYBRID_MODE", "learned_alpha").lower().strip()
if HYBRID_MODE not in ["fixed_alpha", "learned_alpha", "residual_refine", "alpha_residual"]:
    raise ValueError("HYBRID_MODE harus fixed_alpha, learned_alpha, residual_refine, atau alpha_residual")

HYBRID_FREEZE_BACKBONES = os.getenv("HYBRID_FREEZE_BACKBONES", "1") == "1"
HYBRID_INIT_OLD_WEIGHTS_STR = os.getenv("HYBRID_INIT_OLD_WEIGHTS", "0.80,0.75,0.85,0.75")
HYBRID_INIT_OLD_WEIGHTS = [float(x.strip()) for x in HYBRID_INIT_OLD_WEIGHTS_STR.split(",")]
if len(HYBRID_INIT_OLD_WEIGHTS) != len(TARGET_CH):
    raise ValueError("HYBRID_INIT_OLD_WEIGHTS harus berisi 4 angka: F3,F4,T7,T8")

HYBRID_FUSION_LR = float(os.getenv("HYBRID_FUSION_LR", "5e-3"))
HYBRID_WEIGHT_DECAY = float(os.getenv("HYBRID_WEIGHT_DECAY", "1e-4"))
HYBRID_RESIDUAL_HIDDEN = int(os.getenv("HYBRID_RESIDUAL_HIDDEN", "32"))
HYBRID_RESIDUAL_KERNEL = int(os.getenv("HYBRID_RESIDUAL_KERNEL", "15"))
HYBRID_RESIDUAL_SCALE_INIT = float(os.getenv("HYBRID_RESIDUAL_SCALE_INIT", "0.05"))
HYBRID_DROPOUT = float(os.getenv("HYBRID_DROPOUT", "0.05"))

# Loss hybrid. Alpha regularization menjaga bobot tidak lari terlalu jauh dari ensemble terbaik manual.
HYBRID_ALPHA_REG_W = float(os.getenv("HYBRID_ALPHA_REG_W", "0.001"))
HYBRID_TIME_W = float(os.getenv("HYBRID_TIME_W", "0.55"))
HYBRID_TEMPORAL_W = float(os.getenv("HYBRID_TEMPORAL_W", "0.30"))
HYBRID_AMP_W = float(os.getenv("HYBRID_AMP_W", "0.08"))
HYBRID_BETA_TOPO_W = float(os.getenv("HYBRID_BETA_TOPO_W", "0.00"))
HYBRID_BETA_POWER_W = float(os.getenv("HYBRID_BETA_POWER_W", "0.00"))

if "EXPERIMENT_NAME" in os.environ:
    EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
else:
    if MODEL_ARCH == "maser":
        EXPERIMENT_NAME = (
            f"maser_lite_{CFG['name']}_patch{MASER_PATCH_LEN}_"
            f"d{MASER_D_MODEL}_depth{MASER_DEPTH}_seed{SEED}"
        )
    elif MODEL_ARCH == "hybrid":
        EXPERIMENT_NAME = f"hybrid_{HYBRID_MODE}_{CFG['name']}_seed{SEED}"
    else:
        EXPERIMENT_NAME = f"sr_{CFG['name']}_seed{SEED}"

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/kaggle/working/EEG_SR_SUPERGRID")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, EXPERIMENT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_val_loss_{EXPERIMENT_NAME}.pt")
BEST_CORR_MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_corr_{EXPERIMENT_NAME}.pt")

FREQ_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
}

# ============================================================
# 3. UTILITIES
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_subject_id(folder_name: str) -> str:
    return folder_name.split("_")[0]


def discover_folders(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT tidak ditemukan: {root_dir}")
    folders = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not folders:
        raise RuntimeError(f"Tidak ada subfolder session di {root_dir}")
    return folders


def split_subject_folders(all_folders, test_ratio=0.2, seed=42):
    subjects = sorted(set(parse_subject_id(f) for f in all_folders))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = sorted(subjects[-n_test:])
    train_subjects = sorted(subjects[:-n_test])
    train_folders = [f for f in all_folders if parse_subject_id(f) in train_subjects]
    test_folders = [f for f in all_folders if parse_subject_id(f) in test_subjects]
    return train_subjects, test_subjects, train_folders, test_folders


def audit_training_input(data_root):
    data_root = Path(data_root)
    files = sorted(data_root.rglob("*.npy"))
    print("=" * 80)
    print("AUDIT INPUT TRAINING")
    print("=" * 80)
    print("DATA_ROOT:", data_root)
    print("Jumlah file .npy:", len(files))
    if not files:
        raise RuntimeError("Tidak ada file .npy di DATA_ROOT.")
    arr = np.load(files[0], mmap_mode="r")
    print("Contoh file:", files[0])
    print("Shape contoh:", arr.shape)
    print("Dtype contoh:", arr.dtype)
    expected_shape = (8, SEGMENT_SAMPLES)
    if arr.shape != expected_shape:
        raise RuntimeError(f"Shape tidak sesuai. Ditemukan {arr.shape}, seharusnya {expected_shape}.")
    summary_files = sorted(data_root.rglob("summary_seed_250hz_2min_segments.csv"))
    if summary_files:
        df_summary = pd.read_csv(summary_files[0])
        print("Summary CSV:", summary_files[0])
        print("Jumlah baris summary:", len(df_summary))
        display(df_summary.head())
    print("=" * 80)

# ============================================================
# 4. DATASET
# ============================================================
class SEEDSR8NpyDataset(Dataset):
    def __init__(self, folder_list, root_dir, input_idx, target_idx, segment_samples,
                 train=True, max_segments=None):
        self.root_dir = root_dir
        self.input_idx = input_idx
        self.target_idx = target_idx
        self.segment_samples = int(segment_samples)
        self.train = bool(train)
        self.all_segments = []
        for folder_name in folder_list:
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for dirpath, _, filenames in os.walk(folder_path):
                for fname in filenames:
                    if fname.endswith(".npy"):
                        self.all_segments.append(os.path.join(dirpath, fname))
        self.all_segments = sorted(self.all_segments)
        if self.train:
            random.shuffle(self.all_segments)
        if max_segments is not None:
            self.all_segments = self.all_segments[:int(max_segments)]
        if not self.all_segments:
            raise RuntimeError("Tidak ada file .npy yang ditemukan untuk dataset ini.")
        print(
            f"{'Train' if self.train else 'Test'} dataset:\n"
            f"  Jumlah segmen .npy : {len(self.all_segments)}\n"
            f"  File shape         : (8, {self.segment_samples})\n"
            f"  Input shape        : ({len(self.input_idx)}, {self.segment_samples})\n"
            f"  Target shape       : ({len(self.target_idx)}, {self.segment_samples})\n"
            f"  Input idx          : {self.input_idx}\n"
            f"  Target idx         : {self.target_idx}"
        )

    def __len__(self):
        return len(self.all_segments)

    def _load_and_validate(self, path):
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError(f"Data harus 2D, ditemukan {data.shape} pada {path}")
        if data.shape == (8, self.segment_samples):
            return data.astype(np.float32, copy=False)
        if data.shape == (self.segment_samples, 8):
            return data.T.astype(np.float32, copy=False)
        raise ValueError(f"Shape tidak sesuai pada {path}: {data.shape}")

    @staticmethod
    def _robust_scale_by_input(x_raw, y_raw):
        q75, q25 = np.percentile(x_raw, [75, 25])
        iqr = (q75 - q25) + 1e-6
        median = np.median(x_raw)
        x = (x_raw - median) / iqr
        y = (y_raw - median) / iqr
        x = np.tanh(x / 5.0)
        y = np.tanh(y / 5.0)
        return x.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, idx):
        data = self._load_and_validate(self.all_segments[idx])
        x_raw = data[self.input_idx, :]
        y_raw = data[self.target_idx, :]
        x, y = self._robust_scale_by_input(x_raw, y_raw)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ============================================================
# 5. MODEL
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _ = x.size()
        w = self.fc(x.mean(dim=-1)).view(b, c, 1)
        return x * w


class LearnableInputGate(nn.Module):
    def __init__(self, in_ch=4, init_weights=None):
        super().__init__()
        self.in_ch = int(in_ch)
        if init_weights is None:
            init_weights = [1.0] * self.in_ch
        init_weights = torch.tensor(init_weights, dtype=torch.float32)
        if init_weights.numel() != self.in_ch:
            raise ValueError(f"Jumlah init_weights harus {self.in_ch}")
        init_weights = init_weights * (self.in_ch / init_weights.sum())
        self.logits = nn.Parameter(torch.log(init_weights).view(1, self.in_ch, 1))
    def forward(self, x):
        gate = F.softmax(self.logits, dim=1) * self.in_ch
        return x * gate
    def get_gate(self):
        with torch.no_grad():
            gate = F.softmax(self.logits, dim=1) * self.in_ch
        return gate.detach().cpu().view(-1).numpy()


class EEGSingleHeadSRNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, dropout=0.10, alpha_init=0.85,
                 alpha_min=0.60, alpha_max=0.97,
                 use_input_gate=True, input_gate_init=None):
        super().__init__()
        self.use_input_gate = bool(use_input_gate)
        self.input_gate = LearnableInputGate(in_ch, input_gate_init) if self.use_input_gate else nn.Identity()
        self.input_proj = nn.Conv1d(in_ch, 32, kernel_size=1)
        self.encoder = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=15, padding=7, dilation=1),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=15, padding=14, dilation=2),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=15, padding=28, dilation=4),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=15, padding=56, dilation=8),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=15, padding=112, dilation=16),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=9, padding=128, dilation=32),
            nn.InstanceNorm1d(128, affine=True), nn.GELU(), SEBlock(128),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.InstanceNorm1d(64, affine=True), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.InstanceNorm1d(64, affine=True), nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.InstanceNorm1d(32, affine=True), nn.GELU(),
            nn.Conv1d(32, out_ch, kernel_size=1),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.full((1, out_ch, 1), float(alpha_init)))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.scale = nn.Parameter(torch.ones(1, out_ch, 1))

    def forward(self, x):
        x_gated = self.input_gate(x)
        feat = self.encoder(self.input_proj(x_gated))
        enc_out = self.head(feat)
        skip_out = self.skip(x_gated)
        a = self.alpha.clamp(self.alpha_min, self.alpha_max)
        return (a * enc_out + (1.0 - a) * skip_out) * self.scale


# ============================================================
# 5b. MASER-LITE MODEL
# ============================================================
# Implementasi ini mengikuti ide utama MASER:
# 1) patch embedding sinyal EEG,
# 2) low-resolution feature extractor,
# 3) high-resolution placeholder token untuk kanal target,
# 4) HR predictor berbasis eSS2D-lite / SSM-inspired scan,
# 5) output regression ke target F3, F4, T7, T8.
#
# Catatan penting:
# - Ini "MASER-lite" pure PyTorch agar tidak perlu dependency mamba-ssm.
# - SSM block di sini memakai gated recurrent scan (GRU) + depthwise temporal conv
#   sebagai pengganti S6/Mamba selective scan, tetapi tetap memproses urutan token
#   channel-time dalam 4 arah seperti konsep eSS2D.


class ChannelAttentionBlock(nn.Module):
    def __init__(self, n_channels, reduction=2):
        super().__init__()
        hidden = max(2, n_channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, N, D)
        w = x.mean(dim=(2, 3))          # (B, C)
        w = self.net(w).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ChannelTemporalEnhancement(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, N, D)
        # paper MASER menghitung bobot temporal dari rata-rata patch/channel.
        # Di sini bobot dihitung pada domain patch-token: (B, N, D).
        w = x.mean(dim=1)               # (B, N, D)
        w = self.conv(w.transpose(1, 2)).transpose(1, 2)  # (B, N, D)
        w = self.act(w).unsqueeze(1)    # (B, 1, N, D)
        return x * w


class SSMScanLite(nn.Module):
    def __init__(self, d_model, dropout=0.08):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=5, padding=2,
            groups=d_model,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        # seq: (B, L, D)
        u, gate = self.in_proj(seq).chunk(2, dim=-1)
        u = F.silu(u)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        y, _ = self.gru(u)
        y = y * torch.sigmoid(gate)
        y = self.out_proj(y)
        return self.dropout(y)


class ESS2DLiteBlock(nn.Module):
    def __init__(self, n_channels, d_model, dropout=0.08, ffn_mult=2):
        super().__init__()
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.cte = ChannelTemporalEnhancement(d_model)
        self.scan = SSMScanLite(d_model, dropout=dropout)
        self.cab = ChannelAttentionBlock(n_channels)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def _scan_row_major(self, x):
        # x: (B, C, N, D)
        b, c, n, d = x.shape
        seq = x.reshape(b, c * n, d)
        out = self.scan(seq).reshape(b, c, n, d)
        return out

    def _scan_row_reverse(self, x):
        b, c, n, d = x.shape
        seq = torch.flip(x.reshape(b, c * n, d), dims=[1])
        out = self.scan(seq)
        out = torch.flip(out, dims=[1]).reshape(b, c, n, d)
        return out

    def _scan_col_major(self, x):
        b, c, n, d = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b, n * c, d)
        out = self.scan(seq).reshape(b, n, c, d).permute(0, 2, 1, 3)
        return out

    def _scan_col_reverse(self, x):
        b, c, n, d = x.shape
        seq = torch.flip(x.permute(0, 2, 1, 3).reshape(b, n * c, d), dims=[1])
        out = self.scan(seq)
        out = torch.flip(out, dims=[1]).reshape(b, n, c, d).permute(0, 2, 1, 3)
        return out

    def forward(self, x):
        # x: (B, C, N, D)
        h = self.norm1(x)
        h = self.cte(h)

        y = (
            self._scan_row_major(h)
            + self._scan_row_reverse(h)
            + self._scan_col_major(h)
            + self._scan_col_reverse(h)
        ) / 4.0

        y = self.cab(y)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class MASERLiteNet(nn.Module):
    def __init__(
        self,
        in_ch=4,
        out_ch=4,
        total_ch=8,
        patch_len=250,
        d_model=64,
        depth=2,
        dropout=0.08,
        ffn_mult=2,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.total_ch = int(total_ch)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.lr_blocks = nn.ModuleList([
            ESS2DLiteBlock(self.in_ch, d_model, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(depth)
        ])

        # Token identitas kanal target agar F3/F4/T7/T8 tidak diperlakukan identik.
        self.target_channel_emb = nn.Parameter(torch.zeros(1, self.out_ch, 1, d_model))
        nn.init.normal_(self.target_channel_emb, mean=0.0, std=0.02)

        self.hr_blocks = nn.ModuleList([
            ESS2DLiteBlock(self.total_ch, d_model, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(depth)
        ])
        self.latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Linear(d_model, self.patch_len)

        # Skip regresi ringan dari input rata-rata patch ke semua target.
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.full((1, out_ch, 1), 0.85))
        self.scale = nn.Parameter(torch.ones(1, out_ch, 1))

    def _patchify(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        pad_len = (self.patch_len - (t % self.patch_len)) % self.patch_len
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="reflect")
        t_pad = x.shape[-1]
        n_patch = t_pad // self.patch_len
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # patches: (B, C, N, patch_len)
        return patches.contiguous(), t

    def forward(self, x):
        # x: (B, 4, 30000)
        patches, original_t = self._patchify(x)
        z_lr = self.patch_embed(patches)  # (B, 4, N, D)

        for blk in self.lr_blocks:
            z_lr = blk(z_lr)

        # Placeholder target token seperti MASER:
        # mean patch embedding + mean feature embedding + learned target identity.
        raw_emb = self.patch_embed(patches)
        x_mean = raw_emb.mean(dim=1, keepdim=True).repeat(1, self.out_ch, 1, 1)
        f_mean = z_lr.mean(dim=1, keepdim=True).repeat(1, self.out_ch, 1, 1)
        z_pred = x_mean + f_mean + self.target_channel_emb

        z_hr = torch.cat([z_lr, z_pred], dim=1)  # (B, 8, N, D)
        for blk in self.hr_blocks:
            z_hr = blk(z_hr)

        z_target = self.latent(z_hr[:, self.in_ch:, :, :])
        out_patch = self.decoder(z_target)       # (B, 4, N, patch_len)
        b, c, n, p = out_patch.shape
        out = out_patch.reshape(b, c, n * p)
        out = out[..., :original_t]

        skip_out = self.skip(x)
        a = self.alpha.clamp(0.45, 0.97)
        return (a * out + (1.0 - a) * skip_out) * self.scale

# ============================================================
# 6. LOSSES WITH FFT CACHE
# ============================================================
_MASK_CACHE = {}

def get_band_mask(n_times, sfreq, fmin, fmax, device):
    key = (int(n_times), float(sfreq), float(fmin), float(fmax), str(device))
    if key not in _MASK_CACHE:
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq).to(device)
        _MASK_CACHE[key] = (freqs >= fmin) & (freqs <= fmax)
    return _MASK_CACHE[key]


def pearson_corr_loss(pred, target):
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(dim=-1) * (t ** 2).sum(dim=-1) + 1e-8)
    corr = num / den
    return 1.0 - corr.mean()


def derivative_loss(pred, target):
    return F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def fft_mag_loss_from_fft(pred_fft, target_fft, mask):
    pred_mag = torch.log1p(torch.abs(pred_fft[..., mask]))
    target_mag = torch.log1p(torch.abs(target_fft[..., mask]))
    return F.l1_loss(pred_mag, target_mag)


def beta_relative_loss_from_fft(pred_fft, target_fft, beta_mask, total_mask):
    eps = 1e-8
    pred_power = torch.abs(pred_fft) ** 2
    target_power = torch.abs(target_fft) ** 2
    pred_beta = pred_power[..., beta_mask].mean(dim=-1)
    target_beta = target_power[..., beta_mask].mean(dim=-1)
    pred_total = pred_power[..., total_mask].mean(dim=-1)
    target_total = target_power[..., total_mask].mean(dim=-1)
    pred_ratio = pred_beta / (pred_total + eps)
    target_ratio = target_beta / (target_total + eps)
    return F.smooth_l1_loss(torch.log(pred_ratio + eps), torch.log(target_ratio + eps))


def spectral_loss(pred, target, sfreq=250, fmin=1.0, fmax=40.0):
    n_times = pred.shape[-1]
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    mask = get_band_mask(n_times, sfreq, fmin, fmax, pred.device)
    return fft_mag_loss_from_fft(pred_fft, target_fft, mask)


def refined_eeg_loss(pred, target, sfreq=250):
    mae = F.l1_loss(pred, target)
    corr = pearson_corr_loss(pred, target)
    diff = derivative_loss(pred, target)
    spec = spectral_loss(pred, target, sfreq=sfreq, fmin=1.0, fmax=40.0)
    return 0.35 * mae + 0.40 * corr + 0.10 * diff + 0.15 * spec


def temporal_focused_loss(pred, target, sfreq=250):
    mae = F.l1_loss(pred, target)
    corr = pearson_corr_loss(pred, target)
    diff = derivative_loss(pred, target)

    n_times = pred.shape[-1]
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)

    broad_mask = get_band_mask(n_times, sfreq, 4.0, 30.0, pred.device)
    beta_mask = get_band_mask(n_times, sfreq, BETA_FMIN, BETA_FMAX, pred.device)
    total_mask = get_band_mask(n_times, sfreq, TOTAL_FMIN, TOTAL_FMAX, pred.device)

    broad_spec = fft_mag_loss_from_fft(pred_fft, target_fft, broad_mask)
    beta_shape = fft_mag_loss_from_fft(pred_fft, target_fft, beta_mask)
    beta_rel = beta_relative_loss_from_fft(pred_fft, target_fft, beta_mask, total_mask)

    base_loss = (
        TEMPORAL_MAE_W * mae
        + TEMPORAL_CORR_W * corr
        + TEMPORAL_DIFF_W * diff
        + TEMPORAL_BROAD_SPEC_W * broad_spec
    )
    beta_loss = TEMPORAL_BETA_SHAPE_WEIGHT * beta_shape + TEMPORAL_BETA_REL_POWER_WEIGHT * beta_rel
    return base_loss + beta_loss

# ============================================================
# 7. OPTIMIZER / TRAIN / VALID
# ============================================================
def unwrap(model):
    return model.module if hasattr(model, "module") else model


def make_optimizer_single(model):
    m = unwrap(model)
    param_groups = [
        {"params": m.input_proj.parameters(), "lr": 1e-4},
        {"params": m.encoder.parameters(), "lr": 1e-4},
        {"params": m.head.parameters(), "lr": 4e-4},
        {"params": m.skip.parameters(), "lr": 4e-4},
        {"params": [m.alpha, m.scale], "lr": 1e-3},
    ]
    if isinstance(m.input_gate, LearnableInputGate):
        param_groups.append({"params": m.input_gate.parameters(), "lr": 5e-4})
    return optim.AdamW(param_groups, weight_decay=1e-4)


def make_optimizer_temporal(model):
    m = unwrap(model)
    param_groups = [
        {"params": m.input_proj.parameters(), "lr": 5e-5},
        {"params": m.encoder.parameters(), "lr": 5e-5},
        {"params": m.head.parameters(), "lr": 2e-4},
        {"params": m.skip.parameters(), "lr": 2e-4},
        {"params": [m.alpha, m.scale], "lr": 5e-4},
    ]
    if isinstance(m.input_gate, LearnableInputGate):
        param_groups.append({"params": m.input_gate.parameters(), "lr": 5e-4})
    return optim.AdamW(param_groups, weight_decay=3e-4)


def apply_channel_dropout(xb, prob=0.0):
    if prob <= 0 or random.random() >= prob:
        return xb
    mask = torch.ones(xb.shape[0], xb.shape[1], 1, device=xb.device)
    ch_to_drop = random.randint(0, xb.shape[1] - 1)
    mask[:, ch_to_drop, :] = 0.0
    return xb * mask


def curriculum_weights(epoch):
    # temporal lebih diprioritaskan di awal, tapi tidak ekstrem.
    if epoch < 40:
        return 0.30, 0.70
    if epoch < 80:
        return 0.40, 0.60
    return 0.50, 0.50


def train_one_epoch_separate(frontal_model, temporal_model, loader, optimizer_f, optimizer_t,
                             scaler, device, frontal_weight=0.5, temporal_weight=0.5):
    frontal_model.train(); temporal_model.train()
    total_loss = total_mse = total_loss_f = total_loss_t = 0.0
    use_amp = USE_AMP and device.type == "cuda"
    for xb, yb in tqdm(loader, desc="Train", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)
        xb = apply_channel_dropout(xb, prob=CHANNEL_DROPOUT_PROB)
        y_f = yb[:, :2, :]
        y_t = yb[:, 2:, :]

        optimizer_f.zero_grad(set_to_none=True)
        optimizer_t.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_f = frontal_model(xb)
            pred_t = temporal_model(xb)
            loss_f = refined_eeg_loss(pred_f, y_f, sfreq=SFREQ)
            loss_t = temporal_focused_loss(pred_t, y_t, sfreq=SFREQ)
            amp_loss_f = F.mse_loss(pred_f.std(dim=-1), y_f.std(dim=-1))
            amp_loss_t7 = F.mse_loss(pred_t[:, 0:1].std(dim=-1), y_t[:, 0:1].std(dim=-1))
            amp_loss_t8 = F.mse_loss(pred_t[:, 1:2].std(dim=-1), y_t[:, 1:2].std(dim=-1))
            loss_f_total = loss_f + FRONTAL_AMP_LOSS_WEIGHT * amp_loss_f
            loss_t_total = loss_t + TEMPORAL_AMP_LOSS_WEIGHT_T7 * amp_loss_t7 + TEMPORAL_AMP_LOSS_WEIGHT_T8 * amp_loss_t8
            loss_total = frontal_weight * loss_f_total + temporal_weight * loss_t_total

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer_f)
        scaler.unscale_(optimizer_t)
        torch.nn.utils.clip_grad_norm_(frontal_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), max_norm=1.0)
        scaler.step(optimizer_f)
        scaler.step(optimizer_t)
        scaler.update()

        with torch.no_grad():
            pred = torch.cat([pred_f, pred_t], dim=1)
            batch_mse = F.mse_loss(pred.float(), yb.float()).item()
        total_loss += loss_total.item()
        total_mse += batch_mse
        total_loss_f += loss_f_total.item()
        total_loss_t += loss_t_total.item()
    n = len(loader)
    return {"loss": total_loss/n, "mse": total_mse/n, "loss_f": total_loss_f/n, "loss_t": total_loss_t/n}


def batch_corr_torch(pred, target):
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(dim=-1) * (t ** 2).sum(dim=-1) + 1e-8)
    return num / den


@torch.no_grad()
def validate_one_epoch_separate(frontal_model, temporal_model, loader, device):
    frontal_model.eval(); temporal_model.eval()
    total_loss = total_mse = total_loss_f = total_loss_t = 0.0
    corr_sum = torch.zeros(4, device=device)
    corr_count = torch.zeros(4, device=device)
    std_ratio_sum = torch.zeros(4, device=device)
    std_ratio_count = torch.zeros(4, device=device)
    use_amp = USE_AMP and device.type == "cuda"
    for xb, yb in tqdm(loader, desc="Valid", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)
        y_f = yb[:, :2, :]
        y_t = yb[:, 2:, :]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_f = frontal_model(xb)
            pred_t = temporal_model(xb)
            loss_f = refined_eeg_loss(pred_f, y_f, sfreq=SFREQ)
            loss_t = temporal_focused_loss(pred_t, y_t, sfreq=SFREQ)
            amp_loss_f = F.mse_loss(pred_f.std(dim=-1), y_f.std(dim=-1))
            amp_loss_t7 = F.mse_loss(pred_t[:, 0:1].std(dim=-1), y_t[:, 0:1].std(dim=-1))
            amp_loss_t8 = F.mse_loss(pred_t[:, 1:2].std(dim=-1), y_t[:, 1:2].std(dim=-1))
            loss_f_total = loss_f + FRONTAL_AMP_LOSS_WEIGHT * amp_loss_f
            loss_t_total = loss_t + TEMPORAL_AMP_LOSS_WEIGHT_T7 * amp_loss_t7 + TEMPORAL_AMP_LOSS_WEIGHT_T8 * amp_loss_t8
            pred = torch.cat([pred_f, pred_t], dim=1)
            loss = 0.50 * loss_f_total + 0.50 * loss_t_total
            mse = F.mse_loss(pred.float(), yb.float())

        corr_batch = batch_corr_torch(pred.float(), yb.float())
        valid = torch.isfinite(corr_batch)
        corr_sum += torch.where(valid, corr_batch, torch.zeros_like(corr_batch)).sum(dim=0)
        corr_count += valid.sum(dim=0)
        std_ratio_batch = pred.float().std(dim=-1) / (yb.float().std(dim=-1) + 1e-8)
        valid_std = torch.isfinite(std_ratio_batch)
        std_ratio_sum += torch.where(valid_std, std_ratio_batch, torch.zeros_like(std_ratio_batch)).sum(dim=0)
        std_ratio_count += valid_std.sum(dim=0)
        total_loss += loss.item(); total_mse += mse.item(); total_loss_f += loss_f_total.item(); total_loss_t += loss_t_total.item()
    n = len(loader)
    corr_ch = (corr_sum / torch.clamp(corr_count, min=1)).detach().cpu().numpy()
    std_ratio_ch = (std_ratio_sum / torch.clamp(std_ratio_count, min=1)).detach().cpu().numpy()
    mean_corr = float(np.mean(corr_ch)); min_corr = float(np.min(corr_ch))
    return {
        "loss": total_loss/n, "mse": total_mse/n, "loss_f": total_loss_f/n, "loss_t": total_loss_t/n,
        "corr_ch": corr_ch,
        "corr_f3": float(corr_ch[0]), "corr_f4": float(corr_ch[1]),
        "corr_t7": float(corr_ch[2]), "corr_t8": float(corr_ch[3]),
        "std_ratio_f3": float(std_ratio_ch[0]), "std_ratio_f4": float(std_ratio_ch[1]),
        "std_ratio_t7": float(std_ratio_ch[2]), "std_ratio_t8": float(std_ratio_ch[3]),
        "mean_corr": mean_corr, "min_corr": min_corr,
        "corr_score": mean_corr + 0.25 * min_corr,
    }


@torch.no_grad()
def collect_predictions_separate(frontal_model, temporal_model, loader, device):
    frontal_model.eval(); temporal_model.eval()
    all_preds, all_trues, all_inputs = [], [], []
    for xb, yb in tqdm(loader, desc="Collect Pred", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        pred = torch.cat([frontal_model(xb), temporal_model(xb)], dim=1)
        all_preds.append(pred.detach().cpu().numpy())
        all_trues.append(yb.detach().cpu().numpy())
        all_inputs.append(xb.detach().cpu().numpy())
    return np.concatenate(all_inputs), np.concatenate(all_trues), np.concatenate(all_preds)



def make_optimizer_maser(model):
    m = unwrap(model)
    return optim.AdamW(
        [
            {"params": m.patch_embed.parameters(), "lr": MASER_LR},
            {"params": m.lr_blocks.parameters(), "lr": MASER_LR},
            {"params": [m.target_channel_emb], "lr": MASER_LR},
            {"params": m.hr_blocks.parameters(), "lr": MASER_LR},
            {"params": m.latent.parameters(), "lr": 2.0 * MASER_LR},
            {"params": m.decoder.parameters(), "lr": 2.0 * MASER_LR},
            {"params": m.skip.parameters(), "lr": 2.0 * MASER_LR},
            {"params": [m.alpha, m.scale], "lr": 5.0 * MASER_LR},
        ],
        weight_decay=MASER_WEIGHT_DECAY,
    )


def maser_reconstruction_loss(pred, target, sfreq=250):
    # Loss global untuk semua target
    loss_all = refined_eeg_loss(pred, target, sfreq=sfreq)

    # Loss temporal tetap memakai beta-specific objective untuk T7/T8.
    loss_temporal = temporal_focused_loss(pred[:, 2:, :], target[:, 2:, :], sfreq=sfreq)

    # Amplitude matching agar power tidak terlalu pucat.
    amp_loss = F.mse_loss(pred.std(dim=-1), target.std(dim=-1))

    # Smoothness constraint ala MASER: menghukum prediksi yang terlalu bergerigi.
    smooth_loss = torch.mean((pred[:, :, 1:] - pred[:, :, :-1]) ** 2)

    return (
        (1.0 - MASER_LOSS_TEMPORAL_W) * loss_all
        + MASER_LOSS_TEMPORAL_W * loss_temporal
        + 0.08 * amp_loss
        + MASER_SMOOTH_W * smooth_loss
    )


def train_one_epoch_maser(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = total_mse = 0.0
    use_amp = USE_AMP and device.type == "cuda"

    for xb, yb in tqdm(loader, desc="Train MASER", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)
        xb = apply_channel_dropout(xb, prob=CHANNEL_DROPOUT_PROB)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(xb)
            loss = maser_reconstruction_loss(pred, yb, sfreq=SFREQ)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_mse = F.mse_loss(pred.float(), yb.float()).item()

        total_loss += loss.item()
        total_mse += batch_mse

    n = len(loader)
    return {"loss": total_loss / n, "mse": total_mse / n}


@torch.no_grad()
def validate_one_epoch_maser(model, loader, device):
    model.eval()
    total_loss = total_mse = 0.0
    corr_sum = torch.zeros(4, device=device)
    corr_count = torch.zeros(4, device=device)
    std_ratio_sum = torch.zeros(4, device=device)
    std_ratio_count = torch.zeros(4, device=device)
    use_amp = USE_AMP and device.type == "cuda"

    for xb, yb in tqdm(loader, desc="Valid MASER", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(xb)
            loss = maser_reconstruction_loss(pred, yb, sfreq=SFREQ)
            mse = F.mse_loss(pred.float(), yb.float())

        corr_batch = batch_corr_torch(pred.float(), yb.float())
        valid = torch.isfinite(corr_batch)
        corr_sum += torch.where(valid, corr_batch, torch.zeros_like(corr_batch)).sum(dim=0)
        corr_count += valid.sum(dim=0)

        std_ratio_batch = pred.float().std(dim=-1) / (yb.float().std(dim=-1) + 1e-8)
        valid_std = torch.isfinite(std_ratio_batch)
        std_ratio_sum += torch.where(valid_std, std_ratio_batch, torch.zeros_like(std_ratio_batch)).sum(dim=0)
        std_ratio_count += valid_std.sum(dim=0)

        total_loss += loss.item()
        total_mse += mse.item()

    n = len(loader)
    corr_ch = (corr_sum / torch.clamp(corr_count, min=1)).detach().cpu().numpy()
    std_ratio_ch = (std_ratio_sum / torch.clamp(std_ratio_count, min=1)).detach().cpu().numpy()
    mean_corr = float(np.mean(corr_ch))
    min_corr = float(np.min(corr_ch))

    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "corr_ch": corr_ch,
        "corr_f3": float(corr_ch[0]),
        "corr_f4": float(corr_ch[1]),
        "corr_t7": float(corr_ch[2]),
        "corr_t8": float(corr_ch[3]),
        "std_ratio_f3": float(std_ratio_ch[0]),
        "std_ratio_f4": float(std_ratio_ch[1]),
        "std_ratio_t7": float(std_ratio_ch[2]),
        "std_ratio_t8": float(std_ratio_ch[3]),
        "mean_corr": mean_corr,
        "min_corr": min_corr,
        "corr_score": mean_corr + 0.25 * min_corr,
    }


@torch.no_grad()
def collect_predictions_maser(model, loader, device):
    model.eval()
    all_preds, all_trues, all_inputs = [], [], []
    for xb, yb in tqdm(loader, desc="Collect Pred MASER", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        pred = model(xb)
        all_preds.append(pred.detach().cpu().numpy())
        all_trues.append(yb.detach().cpu().numpy())
        all_inputs.append(xb.detach().cpu().numpy())
    return np.concatenate(all_inputs), np.concatenate(all_trues), np.concatenate(all_preds)


def plot_training_curves_maser(history):
    def plot_line(cols, title, fname, ylabel):
        plt.figure(figsize=(10, 4))
        for c in cols:
            if c in history.columns:
                plt.plot(history[c], label=c)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close()

    plot_line(["train_loss", "val_loss"], "MASER Loss Curve", "loss_curve.png", "Loss")
    plot_line(["train_mse", "val_mse"], "MASER MSE Curve", "mse_curve.png", "MSE")
    plot_line(
        ["val_corr_f3", "val_corr_f4", "val_corr_t7", "val_corr_t8"],
        "MASER Validation Corr per Channel",
        "corr_curve.png",
        "Corr",
    )
    plot_line(
        ["val_std_ratio_f3", "val_std_ratio_f4", "val_std_ratio_t7", "val_std_ratio_t8"],
        "MASER StdRatio per Channel",
        "stdratio_curve.png",
        "Pred std / True std",
    )



# ============================================================
# 7b. HYBRID CNN + MASER FUSION
# ============================================================
def _logit_from_prob(p, eps=1e-4):
    p = torch.as_tensor(p, dtype=torch.float32).clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def build_separate_models_from_ckpt(ckpt_path, device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HYBRID_CNN_CKPT tidak ditemukan: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    f_model = EEGSingleHeadSRNet(
        in_ch=len(INPUT_CH), out_ch=2, dropout=FRONTAL_DROPOUT,
        alpha_init=FRONTAL_ALPHA_INIT, alpha_min=FRONTAL_ALPHA_MIN, alpha_max=FRONTAL_ALPHA_MAX,
        use_input_gate=USE_INPUT_GATE, input_gate_init=FRONTAL_INPUT_GATE_INIT,
    ).to(device)
    t_model = EEGSingleHeadSRNet(
        in_ch=len(INPUT_CH), out_ch=2, dropout=TEMPORAL_DROPOUT,
        alpha_init=TEMPORAL_ALPHA_INIT, alpha_min=TEMPORAL_ALPHA_MIN, alpha_max=TEMPORAL_ALPHA_MAX,
        use_input_gate=USE_INPUT_GATE, input_gate_init=TEMPORAL_INPUT_GATE_INIT,
    ).to(device)
    f_model.load_state_dict(ckpt["frontal_model_state_dict"])
    t_model.load_state_dict(ckpt["temporal_model_state_dict"])
    return f_model, t_model, ckpt


def build_maser_model_from_ckpt(ckpt_path, device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HYBRID_MASER_CKPT tidak ditemukan: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    patch_len = int(ckpt.get("maser_patch_len", MASER_PATCH_LEN))
    d_model = int(ckpt.get("maser_d_model", MASER_D_MODEL))
    depth = int(ckpt.get("maser_depth", MASER_DEPTH))
    dropout = float(ckpt.get("maser_dropout", MASER_DROPOUT))
    ffn_mult = int(ckpt.get("maser_ffn_mult", MASER_FFN_MULT))
    model = MASERLiteNet(
        in_ch=len(INPUT_CH),
        out_ch=len(TARGET_CH),
        total_ch=len(INPUT_CH) + len(TARGET_CH),
        patch_len=patch_len,
        d_model=d_model,
        depth=depth,
        dropout=dropout,
        ffn_mult=ffn_mult,
    ).to(device)
    model.load_state_dict(ckpt["maser_model_state_dict"])
    return model, ckpt


class HybridResidualRefiner(nn.Module):
    def __init__(self, in_ch=12, out_ch=4, hidden=32, kernel_size=15, dropout=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel_size, padding=pad),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=pad),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, out_ch, kernel_size=1),
        )
        # Awali residual = 0, supaya model mulai dari ensemble manual yang sudah terbukti bagus.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class HybridCNNMASERFusionNet(nn.Module):
    def __init__(
        self,
        frontal_model,
        temporal_model,
        maser_model,
        mode="learned_alpha",
        init_old_weights=None,
        freeze_backbones=True,
        residual_hidden=32,
        residual_kernel=15,
        residual_scale_init=0.05,
        dropout=0.05,
    ):
        super().__init__()
        self.frontal_model = frontal_model
        self.temporal_model = temporal_model
        self.maser_model = maser_model
        self.mode = str(mode)
        self.freeze_backbones = bool(freeze_backbones)
        self.use_residual = self.mode in ["residual_refine", "alpha_residual"]

        if init_old_weights is None:
            init_old_weights = [0.80, 0.75, 0.85, 0.75]
        self.register_buffer("alpha_init", torch.tensor(init_old_weights, dtype=torch.float32).view(1, 4, 1))

        if self.mode == "fixed_alpha":
            self.register_buffer("alpha_fixed", torch.tensor(init_old_weights, dtype=torch.float32).view(1, 4, 1))
            self.alpha_logits = None
        else:
            self.alpha_logits = nn.Parameter(_logit_from_prob(init_old_weights).view(1, 4, 1))

        if self.use_residual:
            # Input residual: x_input(4) + cnn_pred(4) + maser_pred(4) = 12 channel.
            self.residual_refiner = HybridResidualRefiner(
                in_ch=len(INPUT_CH) + len(TARGET_CH) + len(TARGET_CH),
                out_ch=len(TARGET_CH),
                hidden=residual_hidden,
                kernel_size=residual_kernel,
                dropout=dropout,
            )
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.residual_refiner = None
            self.residual_scale = None

        if self.freeze_backbones:
            for m in [self.frontal_model, self.temporal_model, self.maser_model]:
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_alpha(self):
        if self.mode == "fixed_alpha":
            return self.alpha_fixed
        return torch.sigmoid(self.alpha_logits)

    def backbone_forward(self, x):
        if self.freeze_backbones:
            with torch.no_grad():
                y_cnn = torch.cat([self.frontal_model(x), self.temporal_model(x)], dim=1)
                y_maser = self.maser_model(x)
        else:
            y_cnn = torch.cat([self.frontal_model(x), self.temporal_model(x)], dim=1)
            y_maser = self.maser_model(x)
        return y_cnn, y_maser

    def forward(self, x, return_branches=False):
        y_cnn, y_maser = self.backbone_forward(x)
        alpha = self.get_alpha().to(dtype=y_cnn.dtype, device=y_cnn.device)
        y_base = alpha * y_cnn + (1.0 - alpha) * y_maser

        if self.use_residual:
            residual_in = torch.cat([x, y_cnn, y_maser], dim=1)
            residual = self.residual_refiner(residual_in)
            y_out = y_base + self.residual_scale * residual
        else:
            residual = None
            y_out = y_base

        if return_branches:
            return y_out, y_cnn, y_maser, residual
        return y_out



def beta_power_per_channel_torch(x, sfreq=250, fmin=13.0, fmax=30.0):
    n_times = x.shape[-1]
    fft = torch.fft.rfft(x.float(), dim=-1)
    mask = get_band_mask(n_times, sfreq, fmin, fmax, x.device)
    power = (torch.abs(fft[..., mask]) ** 2).mean(dim=-1)  # (B, C)
    return power


def differentiable_topomap_corr_loss(pred, target, sfreq=250):
    # Korelasi spasial bandpower beta antar 4 target channel.
    pred_p = beta_power_per_channel_torch(pred, sfreq, BETA_FMIN, BETA_FMAX).mean(dim=0)  # (C,)
    true_p = beta_power_per_channel_torch(target, sfreq, BETA_FMIN, BETA_FMAX).mean(dim=0)
    pred_c = pred_p - pred_p.mean()
    true_c = true_p - true_p.mean()
    corr = (pred_c * true_c).sum() / (torch.sqrt((pred_c ** 2).sum() * (true_c ** 2).sum()) + 1e-8)
    return 1.0 - corr


def beta_absolute_power_loss(pred, target, sfreq=250):
    eps = 1e-8
    pred_p = beta_power_per_channel_torch(pred, sfreq, BETA_FMIN, BETA_FMAX)
    true_p = beta_power_per_channel_torch(target, sfreq, BETA_FMIN, BETA_FMAX)
    return F.smooth_l1_loss(torch.log(pred_p + eps), torch.log(true_p + eps))


def hybrid_reconstruction_loss(model, pred, target, sfreq=250):
    loss_time = refined_eeg_loss(pred, target, sfreq=sfreq)
    loss_temp = temporal_focused_loss(pred[:, 2:, :], target[:, 2:, :], sfreq=sfreq)
    amp_loss = F.mse_loss(pred.std(dim=-1), target.std(dim=-1))

    total = HYBRID_TIME_W * loss_time + HYBRID_TEMPORAL_W * loss_temp + HYBRID_AMP_W * amp_loss

    if HYBRID_BETA_TOPO_W > 0:
        total = total + HYBRID_BETA_TOPO_W * differentiable_topomap_corr_loss(pred, target, sfreq=sfreq)
    if HYBRID_BETA_POWER_W > 0:
        total = total + HYBRID_BETA_POWER_W * beta_absolute_power_loss(pred, target, sfreq=sfreq)

    if getattr(model, "alpha_logits", None) is not None and HYBRID_ALPHA_REG_W > 0:
        alpha = model.get_alpha()
        total = total + HYBRID_ALPHA_REG_W * F.mse_loss(alpha, model.alpha_init.to(alpha.device))

    return total


def make_optimizer_hybrid(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    return optim.AdamW(params, lr=HYBRID_FUSION_LR, weight_decay=HYBRID_WEIGHT_DECAY)


@torch.no_grad()
def get_hybrid_alpha_np(model):
    m = unwrap(model)
    alpha = m.get_alpha().detach().cpu().view(-1).numpy()
    return alpha


def train_one_epoch_hybrid(model, loader, optimizer, scaler, device):
    model.train()
    if unwrap(model).freeze_backbones:
        unwrap(model).frontal_model.eval()
        unwrap(model).temporal_model.eval()
        unwrap(model).maser_model.eval()

    total_loss = total_mse = 0.0
    use_amp = USE_AMP and device.type == "cuda"

    for xb, yb in tqdm(loader, desc="Train HYBRID", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(xb)
            loss = hybrid_reconstruction_loss(unwrap(model), pred, yb, sfreq=SFREQ)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mse = F.mse_loss(pred.float(), yb.float()).item()
        total_loss += loss.item()
        total_mse += mse

    n = len(loader)
    return {"loss": total_loss / n, "mse": total_mse / n}


@torch.no_grad()
def validate_one_epoch_hybrid(model, loader, device):
    model.eval()
    total_loss = total_mse = 0.0
    corr_sum = torch.zeros(4, device=device)
    corr_count = torch.zeros(4, device=device)
    std_ratio_sum = torch.zeros(4, device=device)
    std_ratio_count = torch.zeros(4, device=device)
    use_amp = USE_AMP and device.type == "cuda"

    for xb, yb in tqdm(loader, desc="Valid HYBRID", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        yb = yb.contiguous().to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(xb)
            loss = hybrid_reconstruction_loss(unwrap(model), pred, yb, sfreq=SFREQ)
            mse = F.mse_loss(pred.float(), yb.float())

        corr_batch = batch_corr_torch(pred.float(), yb.float())
        valid = torch.isfinite(corr_batch)
        corr_sum += torch.where(valid, corr_batch, torch.zeros_like(corr_batch)).sum(dim=0)
        corr_count += valid.sum(dim=0)

        std_ratio_batch = pred.float().std(dim=-1) / (yb.float().std(dim=-1) + 1e-8)
        valid_std = torch.isfinite(std_ratio_batch)
        std_ratio_sum += torch.where(valid_std, std_ratio_batch, torch.zeros_like(std_ratio_batch)).sum(dim=0)
        std_ratio_count += valid_std.sum(dim=0)

        total_loss += loss.item()
        total_mse += mse.item()

    corr_ch = (corr_sum / torch.clamp(corr_count, min=1)).detach().cpu().numpy()
    std_ratio_ch = (std_ratio_sum / torch.clamp(std_ratio_count, min=1)).detach().cpu().numpy()
    mean_corr = float(np.mean(corr_ch))
    min_corr = float(np.min(corr_ch))
    n = len(loader)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "corr_ch": corr_ch,
        "corr_f3": float(corr_ch[0]),
        "corr_f4": float(corr_ch[1]),
        "corr_t7": float(corr_ch[2]),
        "corr_t8": float(corr_ch[3]),
        "std_ratio_f3": float(std_ratio_ch[0]),
        "std_ratio_f4": float(std_ratio_ch[1]),
        "std_ratio_t7": float(std_ratio_ch[2]),
        "std_ratio_t8": float(std_ratio_ch[3]),
        "mean_corr": mean_corr,
        "min_corr": min_corr,
        "corr_score": mean_corr + 0.25 * min_corr,
    }


@torch.no_grad()
def collect_predictions_hybrid(model, loader, device):
    model.eval()
    all_preds, all_trues, all_inputs = [], [], []
    all_cnn, all_maser = [], []
    for xb, yb in tqdm(loader, desc="Collect Pred HYBRID", leave=False, unit="batch"):
        xb = xb.contiguous().to(device, non_blocking=True)
        pred, y_cnn, y_maser, _ = model(xb, return_branches=True)
        all_preds.append(pred.detach().cpu().numpy())
        all_cnn.append(y_cnn.detach().cpu().numpy())
        all_maser.append(y_maser.detach().cpu().numpy())
        all_trues.append(yb.detach().cpu().numpy())
        all_inputs.append(xb.detach().cpu().numpy())
    return (
        np.concatenate(all_inputs),
        np.concatenate(all_trues),
        np.concatenate(all_preds),
        np.concatenate(all_cnn),
        np.concatenate(all_maser),
    )


def plot_training_curves_hybrid(history):
    def plot_line(cols, title, fname, ylabel):
        plt.figure(figsize=(10, 4))
        for c in cols:
            if c in history.columns:
                plt.plot(history[c], label=c)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close()

    plot_line(["train_loss", "val_loss"], "HYBRID Loss Curve", "loss_curve.png", "Loss")
    plot_line(["train_mse", "val_mse"], "HYBRID MSE Curve", "mse_curve.png", "MSE")
    plot_line(["val_corr_f3", "val_corr_f4", "val_corr_t7", "val_corr_t8"], "HYBRID Validation Corr", "corr_curve.png", "Corr")
    plot_line(["alpha_f3", "alpha_f4", "alpha_t7", "alpha_t8"], "HYBRID Alpha Old-Branch Weights", "alpha_history.png", "Old weight alpha")

# ============================================================
# 8. METRICS AND PLOTS
# ============================================================
def safe_corrcoef(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def evaluate_per_channel(y_true, y_pred, target_ch, save_name="metrics_per_channel_test.csv"):
    rows = []
    print("\n=== TIME-DOMAIN METRICS PER TARGET CHANNEL ===")
    for i, name in enumerate(target_ch):
        corrs = [safe_corrcoef(y_true[s, i, :], y_pred[s, i, :]) for s in range(len(y_true))]
        mse_ch = np.mean((y_true[:, i, :] - y_pred[:, i, :]) ** 2)
        mae_ch = np.mean(np.abs(y_true[:, i, :] - y_pred[:, i, :]))
        avg_corr = np.nanmean(corrs)
        print(f"{name:>3s} | MSE {mse_ch:.6f} | MAE {mae_ch:.6f} | Corr {avg_corr:.4f}")
        rows.append({"channel": name, "mse": mse_ch, "mae": mae_ch, "corr": avg_corr})
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, save_name)
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df


def compute_log_psd(data, sfreq, fmin=1.0, fmax=40.0, n_fft=512):
    n_times = data.shape[-1]
    n_fft = min(n_fft, n_times)
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=n_fft, n_overlap=n_fft//2, average="mean", verbose=False,
    )
    return np.log1p(psds), freqs


def evaluate_spectral_correlation(y_true, y_pred, target_ch, sfreq=250, fmin=1, fmax=40,
                                  save_name="metrics_spectral_corr_test.csv"):
    true_log_psd, freqs = compute_log_psd(y_true, sfreq, fmin, fmax)
    pred_log_psd, _ = compute_log_psd(y_pred, sfreq, fmin, fmax)
    rows = []
    for i, ch in enumerate(target_ch):
        corrs = [safe_corrcoef(true_log_psd[s, i, :], pred_log_psd[s, i, :]) for s in range(y_true.shape[0])]
        rows.append({
            "channel": ch,
            "psd_mse": np.mean((true_log_psd[:, i, :] - pred_log_psd[:, i, :]) ** 2),
            "psd_mae": np.mean(np.abs(true_log_psd[:, i, :] - pred_log_psd[:, i, :])),
            "spectral_corr_1_40hz": np.nanmean(corrs),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, save_name)
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df


def evaluate_spectral_correlation_by_band(y_true, y_pred, target_ch, sfreq=250,
                                          save_name="metrics_spectral_corr_by_band_test.csv"):
    true_log_psd, freqs = compute_log_psd(y_true, sfreq, 1, 40)
    pred_log_psd, _ = compute_log_psd(y_pred, sfreq, 1, 40)
    rows = []
    for band_name, (flo, fhi) in FREQ_BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        for i, ch in enumerate(target_ch):
            corrs = [safe_corrcoef(true_log_psd[s, i, mask], pred_log_psd[s, i, mask]) for s in range(y_true.shape[0])]
            rows.append({
                "band": band_name, "channel": ch,
                "spectral_corr": np.nanmean(corrs),
                "band_logpsd_mae": np.mean(np.abs(true_log_psd[:, i, mask] - pred_log_psd[:, i, mask])),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, save_name)
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df


def get_band_power(data, sfreq):
    n_times = data.shape[-1]
    n_fft = min(512, n_times)
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=40,
        n_fft=n_fft, n_overlap=n_fft//2, verbose=False,
    )
    band_powers = {}
    for band_name, (flo, fhi) in FREQ_BANDS.items():
        idx = np.logical_and(freqs >= flo, freqs <= fhi)
        power = simpson(y=psds[:, :, idx], x=freqs[idx], axis=-1)
        band_powers[band_name] = power.mean(axis=0)
    return band_powers


def evaluate_bandpower_topomap_similarity(y_true, y_pred, target_ch, sfreq=250,
                                          save_name="metrics_bandpower_topomap_similarity_test.csv"):
    true_power = get_band_power(y_true, sfreq)
    pred_power = get_band_power(y_pred, sfreq)
    rows = []
    for band_name in FREQ_BANDS:
        tv = true_power[band_name]
        pv = pred_power[band_name]
        rows.append({
            "band": band_name,
            "target_channels": "|".join(target_ch),
            "topomap_spatial_corr": safe_corrcoef(tv, pv),
            "bandpower_mse": np.mean((tv - pv) ** 2),
            "bandpower_mae": np.mean(np.abs(tv - pv)),
            "mean_pred_true_ratio": np.mean(pv / (tv + 1e-8)),
            "min_pred_true_ratio": np.min(pv / (tv + 1e-8)),
            "max_pred_true_ratio": np.max(pv / (tv + 1e-8)),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, save_name)
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df


def get_mne_interpolation(data_input_np, sfreq, input_names, target_names):
    full_names = input_names + target_names
    info = mne.create_info(ch_names=full_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)
    n_epochs, _, n_times = data_input_np.shape
    combined = np.zeros((n_epochs, len(full_names), n_times), dtype=np.float32)
    combined[:, :len(input_names), :] = data_input_np
    epochs_mne = mne.EpochsArray(combined, info, verbose=False)
    epochs_mne.info["bads"] = list(target_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs_mne.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    return epochs_mne.get_data(picks=target_names)


def plot_topomap_safe(values, info, axes, vmin, vmax):
    try:
        im, _ = mne.viz.plot_topomap(values, info, axes=axes, show=False, vlim=(vmin, vmax))
    except TypeError:
        im, _ = mne.viz.plot_topomap(values, info, axes=axes, show=False, vmin=vmin, vmax=vmax)
    return im


def plot_training_curves(history):
    def plot_line(cols, title, fname, ylabel):
        plt.figure(figsize=(10, 4))
        for c in cols:
            plt.plot(history[c], label=c)
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150); plt.close()
    plot_line(["train_loss", "val_loss"], "Loss Curve", "loss_curve.png", "Loss")
    plot_line(["train_mse", "val_mse"], "MSE Curve", "mse_curve.png", "MSE")
    plot_line(["val_corr_f3", "val_corr_f4", "val_corr_t7", "val_corr_t8"], "Validation Corr per Channel", "corr_curve.png", "Corr")
    plot_line(["val_std_ratio_f3", "val_std_ratio_f4", "val_std_ratio_t7", "val_std_ratio_t8"], "StdRatio per Channel", "stdratio_curve.png", "Pred std / True std")
    plot_line(["alpha_f3", "alpha_f4", "alpha_t7", "alpha_t8"], "Alpha History", "alpha_history.png", "Alpha")
    plot_line(["temporal_gate_fp1", "temporal_gate_fp2", "temporal_gate_p3", "temporal_gate_p4"], "Temporal Input Gate", "temporal_input_gate_history.png", "Gate")


def plot_signal_examples(y_true, y_pred, target_ch, sample_idx=0, sfreq=250, zoom_seconds=10, zoom_start_sec=30):
    sample_idx = min(sample_idx, len(y_true)-1)
    n_times = y_true.shape[-1]
    start = max(0, int(zoom_start_sec * sfreq))
    end = min(n_times, start + int(zoom_seconds * sfreq))
    t = np.arange(start, end) / sfreq
    plt.figure(figsize=(14, 10))
    for i, ch in enumerate(target_ch):
        plt.subplot(len(target_ch), 1, i+1)
        plt.plot(t, y_true[sample_idx, i, start:end], label=f"GT {ch}", alpha=0.9)
        plt.plot(t, y_pred[sample_idx, i, start:end], label=f"Pred {ch}", linestyle="--", alpha=0.9)
        plt.title(f"{ch} Reconstruction | sample={sample_idx} | Zoom {start/sfreq:.1f}-{end/sfreq:.1f}s")
        plt.ylabel("Amplitude"); plt.legend(); plt.grid(True, alpha=0.2)
    plt.xlabel("Time (s)"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"signal_reconstruction_zoom_{zoom_seconds}s_sample_{sample_idx}.png"), dpi=150)
    plt.close()


def plot_brain_maps(x_input, y_true, y_pred):
    print("\nMembuat MNE spline dan brain maps...")
    y_interp = get_mne_interpolation(x_input, SFREQ, INPUT_CH, TARGET_CH)
    np.save(os.path.join(OUTPUT_DIR, "y_interp_mne_test.npy"), y_interp)
    power_in = get_band_power(x_input, SFREQ)
    power_gt = get_band_power(y_true, SFREQ)
    power_pred = get_band_power(y_pred, SFREQ)
    power_interp = get_band_power(y_interp, SFREQ)
    full_ch_names = INPUT_CH + TARGET_CH
    info_8ch = mne.create_info(ch_names=full_ch_names, sfreq=SFREQ, ch_types="eeg")
    info_8ch.set_montage("standard_1020", match_case=False)
    for band_name in FREQ_BANDS:
        vals_gt = np.concatenate([power_in[band_name], power_gt[band_name]])
        vals_ai = np.concatenate([power_in[band_name], power_pred[band_name]])
        vals_spline = np.concatenate([power_in[band_name], power_interp[band_name]])
        vmax = max(vals_gt.max(), vals_ai.max(), vals_spline.max())
        vmin = min(vals_gt.min(), vals_ai.min(), vals_spline.min())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im = plot_topomap_safe(vals_gt, info_8ch, axes[0], vmin, vmax); axes[0].set_title("Ground Truth")
        plot_topomap_safe(vals_ai, info_8ch, axes[1], vmin, vmax); axes[1].set_title("AI Reconstruction")
        plot_topomap_safe(vals_spline, info_8ch, axes[2], vmin, vmax); axes[2].set_title("MNE Spline")
        cbar_ax = fig.add_axes([0.93, 0.20, 0.015, 0.60]); fig.colorbar(im, cax=cbar_ax)
        plt.suptitle(f"Brain Map: {band_name} Band", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"brainmap_{band_name.lower()}.png"), dpi=150)
        plt.close()
    return y_interp


def save_ai_vs_mne_summary(df_time_ai, df_time_mne, df_spec_ai, df_spec_mne, df_topo_ai, df_topo_mne):
    rows = [
        {"metric_group":"time", "metric":"mean_corr", "AI":df_time_ai["corr"].mean(), "MNE_spline":df_time_mne["corr"].mean()},
        {"metric_group":"time", "metric":"mean_mse", "AI":df_time_ai["mse"].mean(), "MNE_spline":df_time_mne["mse"].mean()},
        {"metric_group":"spectral", "metric":"mean_spectral_corr_1_40hz", "AI":df_spec_ai["spectral_corr_1_40hz"].mean(), "MNE_spline":df_spec_mne["spectral_corr_1_40hz"].mean()},
        {"metric_group":"spectral", "metric":"mean_psd_mae", "AI":df_spec_ai["psd_mae"].mean(), "MNE_spline":df_spec_mne["psd_mae"].mean()},
        {"metric_group":"topomap", "metric":"mean_topomap_spatial_corr", "AI":df_topo_ai["topomap_spatial_corr"].mean(), "MNE_spline":df_topo_mne["topomap_spatial_corr"].mean()},
        {"metric_group":"topomap", "metric":"mean_bandpower_mae", "AI":df_topo_ai["bandpower_mae"].mean(), "MNE_spline":df_topo_mne["bandpower_mae"].mean()},
    ]
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "summary_ai_vs_mne_time_spectral.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df

# ============================================================
# 9. MAIN
# ============================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    print("="*80)
    print("EEG SUPER-RESOLUTION — SUPERCOMPUTER GRID")
    print("="*80)
    print("CONFIG_ID       :", CONFIG_ID)
    print("Experiment      :", EXPERIMENT_NAME)
    print("Output dir      :", OUTPUT_DIR)
    print("Device          :", device)
    print("GPU count       :", torch.cuda.device_count())
    print("Batch size      :", BATCH_SIZE)
    print("Num workers     :", NUM_WORKERS)
    print("AMP / TF32      :", USE_AMP, USE_TF32)
    print("Beta weights    :", TEMPORAL_BETA_SHAPE_WEIGHT, TEMPORAL_BETA_REL_POWER_WEIGHT)
    print("Temporal loss w : mae", TEMPORAL_MAE_W, "corr", TEMPORAL_CORR_W, "diff", TEMPORAL_DIFF_W, "broad", TEMPORAL_BROAD_SPEC_W)
    print("="*80)

    audit_training_input(DATA_ROOT)
    all_folders = discover_folders(DATA_ROOT)
    train_subjects, test_subjects, train_folders, test_folders = split_subject_folders(all_folders, TEST_RATIO, SEED)
    print("Train subjects:", train_subjects)
    print("Test subjects :", test_subjects)
    print("Train folders :", len(train_folders), "| Test folders:", len(test_folders))

    train_dataset = SEEDSR8NpyDataset(train_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
    test_dataset = SEEDSR8NpyDataset(test_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin_memory, drop_last=True, persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_memory, drop_last=False, persistent_workers=(NUM_WORKERS > 0))
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader))

    frontal_model = EEGSingleHeadSRNet(
        in_ch=len(INPUT_CH), out_ch=2, dropout=FRONTAL_DROPOUT,
        alpha_init=FRONTAL_ALPHA_INIT, alpha_min=FRONTAL_ALPHA_MIN, alpha_max=FRONTAL_ALPHA_MAX,
        use_input_gate=USE_INPUT_GATE, input_gate_init=FRONTAL_INPUT_GATE_INIT,
    ).to(device)
    temporal_model = EEGSingleHeadSRNet(
        in_ch=len(INPUT_CH), out_ch=2, dropout=TEMPORAL_DROPOUT,
        alpha_init=TEMPORAL_ALPHA_INIT, alpha_min=TEMPORAL_ALPHA_MIN, alpha_max=TEMPORAL_ALPHA_MAX,
        use_input_gate=USE_INPUT_GATE, input_gate_init=TEMPORAL_INPUT_GATE_INIT,
    ).to(device)

    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU via DataParallel.")
        frontal_model = nn.DataParallel(frontal_model)
        temporal_model = nn.DataParallel(temporal_model)

    optimizer_f = make_optimizer_single(frontal_model)
    optimizer_t = make_optimizer_temporal(temporal_model)
    scheduler_f = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_f, T_0=20, T_mult=1, eta_min=1e-5)
    scheduler_t = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_t, T_0=20, T_mult=1, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_val_loss = float("inf")
    best_corr_score = -np.inf
    best_epoch = -1
    best_corr_epoch = -1
    history_rows = []

    print("\n" + "="*80)
    print(f"MEMULAI TRAINING {EPOCHS} EPOCH")
    print("="*80)
    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        f_weight, t_weight = curriculum_weights(epoch)
        train_metrics = train_one_epoch_separate(frontal_model, temporal_model, train_loader,
                                                 optimizer_f, optimizer_t, scaler, device,
                                                 frontal_weight=f_weight, temporal_weight=t_weight)
        val_metrics = validate_one_epoch_separate(frontal_model, temporal_model, test_loader, device)
        scheduler_f.step(); scheduler_t.step()

        af = unwrap(frontal_model); at = unwrap(temporal_model)
        alpha_f_values = af.alpha.detach().cpu().view(-1).numpy()
        alpha_t_values = at.alpha.detach().cpu().view(-1).numpy()
        gate_f = af.input_gate.get_gate() if USE_INPUT_GATE else np.ones(4)
        gate_t = at.input_gate.get_gate() if USE_INPUT_GATE else np.ones(4)

        row = {
            "epoch": epoch+1,
            "frontal_weight": f_weight, "temporal_weight": t_weight,
            "lr_f": optimizer_f.param_groups[0]["lr"], "lr_t": optimizer_t.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
            "train_mse": train_metrics["mse"], "val_mse": val_metrics["mse"],
            "train_loss_f": train_metrics["loss_f"], "train_loss_t": train_metrics["loss_t"],
            "val_loss_f": val_metrics["loss_f"], "val_loss_t": val_metrics["loss_t"],
            "val_corr_f3": val_metrics["corr_f3"], "val_corr_f4": val_metrics["corr_f4"],
            "val_corr_t7": val_metrics["corr_t7"], "val_corr_t8": val_metrics["corr_t8"],
            "val_mean_corr": val_metrics["mean_corr"], "val_min_corr": val_metrics["min_corr"],
            "val_corr_score": val_metrics["corr_score"],
            "val_std_ratio_f3": val_metrics["std_ratio_f3"], "val_std_ratio_f4": val_metrics["std_ratio_f4"],
            "val_std_ratio_t7": val_metrics["std_ratio_t7"], "val_std_ratio_t8": val_metrics["std_ratio_t8"],
            "alpha_f3": float(alpha_f_values[0]), "alpha_f4": float(alpha_f_values[1]),
            "alpha_t7": float(alpha_t_values[0]), "alpha_t8": float(alpha_t_values[1]),
            "alpha_fmean": float(np.mean(alpha_f_values)), "alpha_tmean": float(np.mean(alpha_t_values)),
            "frontal_gate_fp1": float(gate_f[0]), "frontal_gate_fp2": float(gate_f[1]),
            "frontal_gate_p3": float(gate_f[2]), "frontal_gate_p4": float(gate_f[3]),
            "temporal_gate_fp1": float(gate_t[0]), "temporal_gate_fp2": float(gate_t[1]),
            "temporal_gate_p3": float(gate_t[2]), "temporal_gate_p4": float(gate_t[3]),
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"W F/T {f_weight:.2f}/{t_weight:.2f} | "
            f"LR_F {row['lr_f']:.6f} | LR_T {row['lr_t']:.6f} | "
            f"Train Loss {row['train_loss']:.4f} | Val Loss {row['val_loss']:.4f} | "
            f"Train MSE {row['train_mse']:.6f} | Val MSE {row['val_mse']:.6f} | "
            f"Corr F3 {row['val_corr_f3']:.4f} | F4 {row['val_corr_f4']:.4f} | "
            f"T7 {row['val_corr_t7']:.4f} | T8 {row['val_corr_t8']:.4f} | "
            f"MeanCorr {row['val_mean_corr']:.4f} | MinCorr {row['val_min_corr']:.4f} | "
            f"StdRatio T7 {row['val_std_ratio_t7']:.3f} | T8 {row['val_std_ratio_t8']:.3f} | "
            f"GateT [{gate_t[0]:.2f},{gate_t[1]:.2f},{gate_t[2]:.2f},{gate_t[3]:.2f}] | "
            f"A_T7 {row['alpha_t7']:.4f} | A_T8 {row['alpha_t8']:.4f}"
        )

        common_meta = {
            "epoch": epoch+1,
            "config_id": CONFIG_ID,
            "experiment_name": EXPERIMENT_NAME,
            "model_type": "separate_singlehead_inputgate_temporal_beta_loss",
            "frontal_model_state_dict": unwrap(frontal_model).state_dict(),
            "temporal_model_state_dict": unwrap(temporal_model).state_dict(),
            "optimizer_f_state_dict": optimizer_f.state_dict(),
            "optimizer_t_state_dict": optimizer_t.state_dict(),
            "input_ch": INPUT_CH, "target_ch": TARGET_CH,
            "input_idx": INPUT_IDX, "target_idx": TARGET_IDX,
            "sfreq": SFREQ, "segment_seconds": SEGMENT_SECONDS, "segment_samples": SEGMENT_SAMPLES,
            "train_subjects": train_subjects, "test_subjects": test_subjects,
            "use_input_gate": USE_INPUT_GATE,
            "frontal_input_gate_init": FRONTAL_INPUT_GATE_INIT,
            "temporal_input_gate_init": TEMPORAL_INPUT_GATE_INIT,
            "final_frontal_input_gate": gate_f.tolist(),
            "final_temporal_input_gate": gate_t.tolist(),
            "temporal_beta_shape_weight": TEMPORAL_BETA_SHAPE_WEIGHT,
            "temporal_beta_rel_power_weight": TEMPORAL_BETA_REL_POWER_WEIGHT,
            "temporal_corr_w": TEMPORAL_CORR_W,
            "temporal_broad_spec_w": TEMPORAL_BROAD_SPEC_W,
            "beta_band": [BETA_FMIN, BETA_FMAX],
            "total_band_for_beta_ratio": [TOTAL_FMIN, TOTAL_FMAX],
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch+1
            save_obj = dict(common_meta)
            save_obj.update({"best_val_loss": best_val_loss, "val_metrics": val_metrics})
            torch.save(save_obj, BEST_MODEL_PATH)
            print("  -> Best val-loss model disimpan:", BEST_MODEL_PATH)

        if val_metrics["corr_score"] > best_corr_score:
            best_corr_score = val_metrics["corr_score"]
            best_corr_epoch = epoch+1
            if SAVE_CORR_CHECKPOINT:
                save_obj = dict(common_meta)
                save_obj.update({
                    "best_corr_score": best_corr_score,
                    "best_mean_corr": val_metrics["mean_corr"],
                    "best_min_corr": val_metrics["min_corr"],
                    "corr_f3": val_metrics["corr_f3"], "corr_f4": val_metrics["corr_f4"],
                    "corr_t7": val_metrics["corr_t7"], "corr_t8": val_metrics["corr_t8"],
                    "val_metrics": val_metrics,
                })
                torch.save(save_obj, BEST_CORR_MODEL_PATH)
                print("  -> Best corr-score model disimpan:", BEST_CORR_MODEL_PATH)

        # Early stop/prune opsional: aktifkan dengan PRUNE_BAD_RUN=1.
        if os.getenv("PRUNE_BAD_RUN", "0") == "1" and epoch+1 >= 60:
            recent_mean = np.mean([r["val_mean_corr"] for r in history_rows[-5:]])
            recent_t8 = np.mean([r["val_corr_t8"] for r in history_rows[-5:]])
            if recent_mean < 0.710 and recent_t8 < 0.545:
                print("PRUNE: run dihentikan karena performa tidak menjanjikan.")
                break

    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print("Training history tersimpan:", history_path)
    plot_training_curves(history_df)

    # ========================================================
    # Final evaluation: evaluasi best val-loss dan best corr
    # ========================================================
    eval_targets = [("best_val_loss", BEST_MODEL_PATH)]
    if SAVE_CORR_CHECKPOINT and os.path.exists(BEST_CORR_MODEL_PATH):
        eval_targets.append(("best_corr", BEST_CORR_MODEL_PATH))

    for tag, ckpt_path in eval_targets:
        print("\n" + "="*80)
        print("FINAL EVALUATION:", tag)
        print("Checkpoint:", ckpt_path)
        print("="*80)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unwrap(frontal_model).load_state_dict(ckpt["frontal_model_state_dict"])
        unwrap(temporal_model).load_state_dict(ckpt["temporal_model_state_dict"])
        x_input, y_true, y_pred = collect_predictions_separate(frontal_model, temporal_model, test_loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"x_input_test_{tag}.npy"), x_input)
        np.save(os.path.join(OUTPUT_DIR, f"y_true_test_{tag}.npy"), y_true)
        np.save(os.path.join(OUTPUT_DIR, f"y_pred_test_{tag}.npy"), y_pred)

        df_time = evaluate_per_channel(y_true, y_pred, TARGET_CH, save_name=f"metrics_per_channel_{tag}.csv")
        df_spec = evaluate_spectral_correlation(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_{tag}.csv")
        df_spec_band = evaluate_spectral_correlation_by_band(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_by_band_{tag}.csv")
        df_topo = evaluate_bandpower_topomap_similarity(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_bandpower_topomap_similarity_{tag}.csv")

        # Plot hanya untuk checkpoint pertama agar tidak terlalu banyak file, kecuali PLOT_ALL_CKPT=1.
        if tag == "best_val_loss" or os.getenv("PLOT_ALL_CKPT", "0") == "1":
            plot_signal_examples(y_true, y_pred, TARGET_CH, sample_idx=0, sfreq=SFREQ, zoom_seconds=10, zoom_start_sec=30)
            y_interp = plot_brain_maps(x_input, y_true, y_pred)
            df_mne_time = evaluate_per_channel(y_true, y_interp, TARGET_CH, save_name=f"metrics_per_channel_mne_spline_{tag}.csv")
            df_mne_spec = evaluate_spectral_correlation(y_true, y_interp, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_mne_spline_{tag}.csv")
            df_mne_topo = evaluate_bandpower_topomap_similarity(y_true, y_interp, TARGET_CH, SFREQ, save_name=f"metrics_bandpower_topomap_similarity_mne_spline_{tag}.csv")
            df_summary = save_ai_vs_mne_summary(df_time, df_mne_time, df_spec, df_mne_spec, df_topo, df_mne_topo)
            display(df_summary)

    print("\nSELESAI. Output dir:", OUTPUT_DIR)



# ============================================================
# 10. MAIN MASER-LITE
# ============================================================
def main_maser():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    print("=" * 80)
    print("EEG SUPER-RESOLUTION — MASER-LITE")
    print("=" * 80)
    print("CONFIG_ID       :", CONFIG_ID)
    print("MODEL_ARCH      :", MODEL_ARCH)
    print("Experiment      :", EXPERIMENT_NAME)
    print("Output dir      :", OUTPUT_DIR)
    print("Device          :", device)
    print("GPU count       :", torch.cuda.device_count())
    print("Batch size      :", BATCH_SIZE)
    print("Num workers     :", NUM_WORKERS)
    print("AMP / TF32      :", USE_AMP, USE_TF32)
    print("MASER patch len :", MASER_PATCH_LEN)
    print("MASER d/depth   :", MASER_D_MODEL, MASER_DEPTH)
    print("MASER lr/wd     :", MASER_LR, MASER_WEIGHT_DECAY)
    print("Smooth weight   :", MASER_SMOOTH_W)
    print("Temporal loss w :", MASER_LOSS_TEMPORAL_W)
    print("Beta weights    :", TEMPORAL_BETA_SHAPE_WEIGHT, TEMPORAL_BETA_REL_POWER_WEIGHT)
    print("=" * 80)

    if SEGMENT_SAMPLES % MASER_PATCH_LEN != 0:
        print(
            f"[INFO] SEGMENT_SAMPLES={SEGMENT_SAMPLES} tidak habis dibagi "
            f"MASER_PATCH_LEN={MASER_PATCH_LEN}. Model akan padding reflect lalu crop output."
        )

    audit_training_input(DATA_ROOT)
    all_folders = discover_folders(DATA_ROOT)
    train_subjects, test_subjects, train_folders, test_folders = split_subject_folders(all_folders, TEST_RATIO, SEED)

    print("Train subjects:", train_subjects)
    print("Test subjects :", test_subjects)
    print("Train folders :", len(train_folders), "| Test folders:", len(test_folders))

    train_dataset = SEEDSR8NpyDataset(
        train_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES,
        True, MAX_TRAIN_SEGMENTS
    )
    test_dataset = SEEDSR8NpyDataset(
        test_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES,
        False, MAX_TEST_SEGMENTS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader))

    model = MASERLiteNet(
        in_ch=len(INPUT_CH),
        out_ch=len(TARGET_CH),
        total_ch=len(INPUT_CH) + len(TARGET_CH),
        patch_len=MASER_PATCH_LEN,
        d_model=MASER_D_MODEL,
        depth=MASER_DEPTH,
        dropout=MASER_DROPOUT,
        ffn_mult=MASER_FFN_MULT,
    ).to(device)

    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU via DataParallel.")
        model = nn.DataParallel(model)

    optimizer = make_optimizer_maser(model)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_val_loss = float("inf")
    best_corr_score = -np.inf
    best_epoch = -1
    best_corr_epoch = -1
    history_rows = []

    print("\n" + "=" * 80)
    print(f"MEMULAI TRAINING MASER-LITE {EPOCHS} EPOCH")
    print("=" * 80)

    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train_metrics = train_one_epoch_maser(model, train_loader, optimizer, scaler, device)
        val_metrics = validate_one_epoch_maser(model, test_loader, device)
        scheduler.step()

        m = unwrap(model)
        alpha_values = m.alpha.detach().cpu().view(-1).numpy()

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "val_corr_f3": val_metrics["corr_f3"],
            "val_corr_f4": val_metrics["corr_f4"],
            "val_corr_t7": val_metrics["corr_t7"],
            "val_corr_t8": val_metrics["corr_t8"],
            "val_mean_corr": val_metrics["mean_corr"],
            "val_min_corr": val_metrics["min_corr"],
            "val_corr_score": val_metrics["corr_score"],
            "val_std_ratio_f3": val_metrics["std_ratio_f3"],
            "val_std_ratio_f4": val_metrics["std_ratio_f4"],
            "val_std_ratio_t7": val_metrics["std_ratio_t7"],
            "val_std_ratio_t8": val_metrics["std_ratio_t8"],
            "alpha_f3": float(alpha_values[0]),
            "alpha_f4": float(alpha_values[1]),
            "alpha_t7": float(alpha_values[2]),
            "alpha_t8": float(alpha_values[3]),
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"LR {row['lr']:.6f} | "
            f"Train Loss {row['train_loss']:.4f} | Val Loss {row['val_loss']:.4f} | "
            f"Train MSE {row['train_mse']:.6f} | Val MSE {row['val_mse']:.6f} | "
            f"Corr F3 {row['val_corr_f3']:.4f} | F4 {row['val_corr_f4']:.4f} | "
            f"T7 {row['val_corr_t7']:.4f} | T8 {row['val_corr_t8']:.4f} | "
            f"MeanCorr {row['val_mean_corr']:.4f} | MinCorr {row['val_min_corr']:.4f} | "
            f"StdRatio T7 {row['val_std_ratio_t7']:.3f} | T8 {row['val_std_ratio_t8']:.3f} | "
            f"A_T7 {row['alpha_t7']:.4f} | A_T8 {row['alpha_t8']:.4f}"
        )

        common_meta = {
            "epoch": epoch + 1,
            "config_id": CONFIG_ID,
            "experiment_name": EXPERIMENT_NAME,
            "model_arch": "maser_lite",
            "model_type": "maser_lite_patch_token_ess2d_lite",
            "maser_model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_ch": INPUT_CH,
            "target_ch": TARGET_CH,
            "input_idx": INPUT_IDX,
            "target_idx": TARGET_IDX,
            "sfreq": SFREQ,
            "segment_seconds": SEGMENT_SECONDS,
            "segment_samples": SEGMENT_SAMPLES,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "maser_patch_len": MASER_PATCH_LEN,
            "maser_d_model": MASER_D_MODEL,
            "maser_depth": MASER_DEPTH,
            "maser_dropout": MASER_DROPOUT,
            "maser_ffn_mult": MASER_FFN_MULT,
            "maser_smooth_w": MASER_SMOOTH_W,
            "maser_loss_temporal_w": MASER_LOSS_TEMPORAL_W,
            "temporal_beta_shape_weight": TEMPORAL_BETA_SHAPE_WEIGHT,
            "temporal_beta_rel_power_weight": TEMPORAL_BETA_REL_POWER_WEIGHT,
            "temporal_corr_w": TEMPORAL_CORR_W,
            "temporal_broad_spec_w": TEMPORAL_BROAD_SPEC_W,
            "beta_band": [BETA_FMIN, BETA_FMAX],
            "total_band_for_beta_ratio": [TOTAL_FMIN, TOTAL_FMAX],
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            save_obj = dict(common_meta)
            save_obj.update({"best_val_loss": best_val_loss, "val_metrics": val_metrics})
            torch.save(save_obj, BEST_MODEL_PATH)
            print("  -> Best val-loss MASER disimpan:", BEST_MODEL_PATH)

        if val_metrics["corr_score"] > best_corr_score:
            best_corr_score = val_metrics["corr_score"]
            best_corr_epoch = epoch + 1
            if SAVE_CORR_CHECKPOINT:
                save_obj = dict(common_meta)
                save_obj.update({
                    "best_corr_score": best_corr_score,
                    "best_mean_corr": val_metrics["mean_corr"],
                    "best_min_corr": val_metrics["min_corr"],
                    "corr_f3": val_metrics["corr_f3"],
                    "corr_f4": val_metrics["corr_f4"],
                    "corr_t7": val_metrics["corr_t7"],
                    "corr_t8": val_metrics["corr_t8"],
                    "val_metrics": val_metrics,
                })
                torch.save(save_obj, BEST_CORR_MODEL_PATH)
                print("  -> Best corr-score MASER disimpan:", BEST_CORR_MODEL_PATH)

        if os.getenv("PRUNE_BAD_RUN", "0") == "1" and epoch + 1 >= 40:
            recent_mean = np.mean([r["val_mean_corr"] for r in history_rows[-5:]])
            if recent_mean < 0.60:
                print("PRUNE: MASER run dihentikan karena performa tidak menjanjikan.")
                break

    print(f"Best val-loss epoch: {best_epoch} | Best corr epoch: {best_corr_epoch}")

    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print("Training history tersimpan:", history_path)
    plot_training_curves_maser(history_df)

    eval_targets = [("best_val_loss", BEST_MODEL_PATH)]
    if SAVE_CORR_CHECKPOINT and os.path.exists(BEST_CORR_MODEL_PATH):
        eval_targets.append(("best_corr", BEST_CORR_MODEL_PATH))

    for tag, ckpt_path in eval_targets:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION MASER:", tag)
        print("Checkpoint:", ckpt_path)
        print("=" * 80)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ckpt["maser_model_state_dict"])

        x_input, y_true, y_pred = collect_predictions_maser(model, test_loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"x_input_test_{tag}.npy"), x_input)
        np.save(os.path.join(OUTPUT_DIR, f"y_true_test_{tag}.npy"), y_true)
        np.save(os.path.join(OUTPUT_DIR, f"y_pred_test_{tag}.npy"), y_pred)

        df_time = evaluate_per_channel(
            y_true, y_pred, TARGET_CH,
            save_name=f"metrics_per_channel_{tag}.csv"
        )
        df_spec = evaluate_spectral_correlation(
            y_true, y_pred, TARGET_CH, SFREQ,
            save_name=f"metrics_spectral_corr_{tag}.csv"
        )
        df_spec_band = evaluate_spectral_correlation_by_band(
            y_true, y_pred, TARGET_CH, SFREQ,
            save_name=f"metrics_spectral_corr_by_band_{tag}.csv"
        )
        df_topo = evaluate_bandpower_topomap_similarity(
            y_true, y_pred, TARGET_CH, SFREQ,
            save_name=f"metrics_bandpower_topomap_similarity_{tag}.csv"
        )

        if tag == "best_val_loss" or os.getenv("PLOT_ALL_CKPT", "0") == "1":
            plot_signal_examples(
                y_true, y_pred, TARGET_CH,
                sample_idx=0, sfreq=SFREQ,
                zoom_seconds=10, zoom_start_sec=30,
            )
            y_interp = plot_brain_maps(x_input, y_true, y_pred)
            df_mne_time = evaluate_per_channel(
                y_true, y_interp, TARGET_CH,
                save_name=f"metrics_per_channel_mne_spline_{tag}.csv"
            )
            df_mne_spec = evaluate_spectral_correlation(
                y_true, y_interp, TARGET_CH, SFREQ,
                save_name=f"metrics_spectral_corr_mne_spline_{tag}.csv"
            )
            df_mne_topo = evaluate_bandpower_topomap_similarity(
                y_true, y_interp, TARGET_CH, SFREQ,
                save_name=f"metrics_bandpower_topomap_similarity_mne_spline_{tag}.csv"
            )
            df_summary = save_ai_vs_mne_summary(
                df_time, df_mne_time, df_spec, df_mne_spec, df_topo, df_mne_topo
            )
            display(df_summary)

    print("\nSELESAI MASER-LITE. Output dir:", OUTPUT_DIR)




def main_hybrid():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    print("=" * 80)
    print("EEG SUPER-RESOLUTION — HYBRID CNN + MASER FUSION")
    print("=" * 80)
    print("CONFIG_ID        :", CONFIG_ID)
    print("Experiment       :", EXPERIMENT_NAME)
    print("Output dir       :", OUTPUT_DIR)
    print("Device           :", device)
    print("GPU count        :", torch.cuda.device_count())
    print("Batch size       :", BATCH_SIZE)
    print("Hybrid mode      :", HYBRID_MODE)
    print("Freeze backbones :", HYBRID_FREEZE_BACKBONES)
    print("Init old weights :", HYBRID_INIT_OLD_WEIGHTS)
    print("CNN ckpt         :", HYBRID_CNN_CKPT)
    print("MASER ckpt       :", HYBRID_MASER_CKPT)
    print("Loss w           : time", HYBRID_TIME_W, "temp", HYBRID_TEMPORAL_W, "amp", HYBRID_AMP_W,
          "beta_topo", HYBRID_BETA_TOPO_W, "beta_power", HYBRID_BETA_POWER_W)
    print("=" * 80)

    audit_training_input(DATA_ROOT)
    all_folders = discover_folders(DATA_ROOT)
    train_subjects, test_subjects, train_folders, test_folders = split_subject_folders(all_folders, TEST_RATIO, SEED)
    print("Train subjects:", train_subjects)
    print("Test subjects :", test_subjects)
    print("Train folders :", len(train_folders), "| Test folders:", len(test_folders))

    train_dataset = SEEDSR8NpyDataset(train_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
    test_dataset = SEEDSR8NpyDataset(test_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin_memory, drop_last=True, persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_memory, drop_last=False, persistent_workers=(NUM_WORKERS > 0))
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader))

    frontal_model, temporal_model, cnn_ckpt = build_separate_models_from_ckpt(HYBRID_CNN_CKPT, device)
    maser_model, maser_ckpt = build_maser_model_from_ckpt(HYBRID_MASER_CKPT, device)

    model = HybridCNNMASERFusionNet(
        frontal_model=frontal_model,
        temporal_model=temporal_model,
        maser_model=maser_model,
        mode=HYBRID_MODE,
        init_old_weights=HYBRID_INIT_OLD_WEIGHTS,
        freeze_backbones=HYBRID_FREEZE_BACKBONES,
        residual_hidden=HYBRID_RESIDUAL_HIDDEN,
        residual_kernel=HYBRID_RESIDUAL_KERNEL,
        residual_scale_init=HYBRID_RESIDUAL_SCALE_INIT,
        dropout=HYBRID_DROPOUT,
    ).to(device)

    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU via DataParallel.")
        model = nn.DataParallel(model)

    optimizer = make_optimizer_hybrid(model)
    scheduler = None
    if optimizer is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_val_loss = float("inf")
    best_corr_score = -np.inf
    best_epoch = -1
    best_corr_epoch = -1
    history_rows = []

    if optimizer is None or HYBRID_MODE == "fixed_alpha":
        print("\nHYBRID_MODE fixed_alpha: skip training, langsung evaluasi bobot tetap.")
        val_metrics = validate_one_epoch_hybrid(model, test_loader, device)
        alpha = get_hybrid_alpha_np(model)
        history_rows.append({
            "epoch": 0, "lr": 0.0,
            "train_loss": np.nan, "val_loss": val_metrics["loss"],
            "train_mse": np.nan, "val_mse": val_metrics["mse"],
            "val_corr_f3": val_metrics["corr_f3"], "val_corr_f4": val_metrics["corr_f4"],
            "val_corr_t7": val_metrics["corr_t7"], "val_corr_t8": val_metrics["corr_t8"],
            "val_mean_corr": val_metrics["mean_corr"], "val_min_corr": val_metrics["min_corr"],
            "val_corr_score": val_metrics["corr_score"],
            "alpha_f3": alpha[0], "alpha_f4": alpha[1], "alpha_t7": alpha[2], "alpha_t8": alpha[3],
        })
        save_obj = {
            "epoch": 0,
            "config_id": CONFIG_ID,
            "experiment_name": EXPERIMENT_NAME,
            "model_arch": "hybrid",
            "hybrid_mode": HYBRID_MODE,
            "hybrid_model_state_dict": unwrap(model).state_dict(),
            "input_ch": INPUT_CH, "target_ch": TARGET_CH,
            "sfreq": SFREQ, "segment_seconds": SEGMENT_SECONDS,
            "train_subjects": train_subjects, "test_subjects": test_subjects,
            "hybrid_init_old_weights": HYBRID_INIT_OLD_WEIGHTS,
            "hybrid_final_old_weights": alpha.tolist(),
            "val_metrics": val_metrics,
        }
        torch.save(save_obj, BEST_MODEL_PATH)
        torch.save(save_obj, BEST_CORR_MODEL_PATH)
        best_epoch = 0
        best_corr_epoch = 0
    else:
        print("\n" + "=" * 80)
        print(f"MEMULAI TRAINING HYBRID {EPOCHS} EPOCH")
        print("=" * 80)

        for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
            train_metrics = train_one_epoch_hybrid(model, train_loader, optimizer, scaler, device)
            val_metrics = validate_one_epoch_hybrid(model, test_loader, device)
            if scheduler is not None:
                scheduler.step()

            alpha = get_hybrid_alpha_np(model)
            lr0 = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
            residual_scale = np.nan
            if hasattr(unwrap(model), "residual_scale") and unwrap(model).residual_scale is not None:
                residual_scale = float(unwrap(model).residual_scale.detach().cpu().item())

            row = {
                "epoch": epoch + 1,
                "lr": lr0,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "val_corr_f3": val_metrics["corr_f3"],
                "val_corr_f4": val_metrics["corr_f4"],
                "val_corr_t7": val_metrics["corr_t7"],
                "val_corr_t8": val_metrics["corr_t8"],
                "val_mean_corr": val_metrics["mean_corr"],
                "val_min_corr": val_metrics["min_corr"],
                "val_corr_score": val_metrics["corr_score"],
                "val_std_ratio_f3": val_metrics["std_ratio_f3"],
                "val_std_ratio_f4": val_metrics["std_ratio_f4"],
                "val_std_ratio_t7": val_metrics["std_ratio_t7"],
                "val_std_ratio_t8": val_metrics["std_ratio_t8"],
                "alpha_f3": float(alpha[0]),
                "alpha_f4": float(alpha[1]),
                "alpha_t7": float(alpha[2]),
                "alpha_t8": float(alpha[3]),
                "residual_scale": residual_scale,
            }
            history_rows.append(row)

            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | "
                f"LR {row['lr']:.6f} | "
                f"Train Loss {row['train_loss']:.4f} | Val Loss {row['val_loss']:.4f} | "
                f"Train MSE {row['train_mse']:.6f} | Val MSE {row['val_mse']:.6f} | "
                f"Corr F3 {row['val_corr_f3']:.4f} | F4 {row['val_corr_f4']:.4f} | "
                f"T7 {row['val_corr_t7']:.4f} | T8 {row['val_corr_t8']:.4f} | "
                f"MeanCorr {row['val_mean_corr']:.4f} | MinCorr {row['val_min_corr']:.4f} | "
                f"Alpha [{alpha[0]:.3f},{alpha[1]:.3f},{alpha[2]:.3f},{alpha[3]:.3f}] | "
                f"ResScale {residual_scale:.4f}"
            )

            common_meta = {
                "epoch": epoch + 1,
                "config_id": CONFIG_ID,
                "experiment_name": EXPERIMENT_NAME,
                "model_arch": "hybrid",
                "hybrid_mode": HYBRID_MODE,
                "hybrid_model_state_dict": unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "input_ch": INPUT_CH, "target_ch": TARGET_CH,
                "input_idx": INPUT_IDX, "target_idx": TARGET_IDX,
                "sfreq": SFREQ, "segment_seconds": SEGMENT_SECONDS, "segment_samples": SEGMENT_SAMPLES,
                "train_subjects": train_subjects, "test_subjects": test_subjects,
                "cnn_ckpt": HYBRID_CNN_CKPT,
                "maser_ckpt": HYBRID_MASER_CKPT,
                "hybrid_init_old_weights": HYBRID_INIT_OLD_WEIGHTS,
                "hybrid_final_old_weights": alpha.tolist(),
                "hybrid_freeze_backbones": HYBRID_FREEZE_BACKBONES,
                "hybrid_alpha_reg_w": HYBRID_ALPHA_REG_W,
                "hybrid_beta_topo_w": HYBRID_BETA_TOPO_W,
                "hybrid_beta_power_w": HYBRID_BETA_POWER_W,
            }

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                save_obj = dict(common_meta)
                save_obj.update({"best_val_loss": best_val_loss, "val_metrics": val_metrics})
                torch.save(save_obj, BEST_MODEL_PATH)
                print("  -> Best val-loss HYBRID disimpan:", BEST_MODEL_PATH)

            if val_metrics["corr_score"] > best_corr_score:
                best_corr_score = val_metrics["corr_score"]
                best_corr_epoch = epoch + 1
                if SAVE_CORR_CHECKPOINT:
                    save_obj = dict(common_meta)
                    save_obj.update({
                        "best_corr_score": best_corr_score,
                        "best_mean_corr": val_metrics["mean_corr"],
                        "best_min_corr": val_metrics["min_corr"],
                        "corr_f3": val_metrics["corr_f3"],
                        "corr_f4": val_metrics["corr_f4"],
                        "corr_t7": val_metrics["corr_t7"],
                        "corr_t8": val_metrics["corr_t8"],
                        "val_metrics": val_metrics,
                    })
                    torch.save(save_obj, BEST_CORR_MODEL_PATH)
                    print("  -> Best corr-score HYBRID disimpan:", BEST_CORR_MODEL_PATH)

            if os.getenv("PRUNE_BAD_RUN", "0") == "1" and epoch + 1 >= 20:
                recent_mean = np.mean([r["val_mean_corr"] for r in history_rows[-5:]])
                if recent_mean < 0.710:
                    print("PRUNE: HYBRID run dihentikan karena performa tidak menjanjikan.")
                    break

    print(f"Best val-loss epoch: {best_epoch} | Best corr epoch: {best_corr_epoch}")

    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print("Training history tersimpan:", history_path)
    plot_training_curves_hybrid(history_df)

    eval_targets = [("best_val_loss", BEST_MODEL_PATH)]
    if SAVE_CORR_CHECKPOINT and os.path.exists(BEST_CORR_MODEL_PATH):
        eval_targets.append(("best_corr", BEST_CORR_MODEL_PATH))

    for tag, ckpt_path in eval_targets:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION HYBRID:", tag)
        print("Checkpoint:", ckpt_path)
        print("=" * 80)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ckpt["hybrid_model_state_dict"])

        x_input, y_true, y_pred, y_cnn, y_maser = collect_predictions_hybrid(model, test_loader, device)
        np.save(os.path.join(OUTPUT_DIR, f"x_input_test_{tag}.npy"), x_input)
        np.save(os.path.join(OUTPUT_DIR, f"y_true_test_{tag}.npy"), y_true)
        np.save(os.path.join(OUTPUT_DIR, f"y_pred_test_{tag}.npy"), y_pred)
        np.save(os.path.join(OUTPUT_DIR, f"y_cnn_test_{tag}.npy"), y_cnn)
        np.save(os.path.join(OUTPUT_DIR, f"y_maser_test_{tag}.npy"), y_maser)

        df_time = evaluate_per_channel(y_true, y_pred, TARGET_CH, save_name=f"metrics_per_channel_{tag}.csv")
        df_spec = evaluate_spectral_correlation(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_{tag}.csv")
        df_spec_band = evaluate_spectral_correlation_by_band(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_by_band_{tag}.csv")
        df_topo = evaluate_bandpower_topomap_similarity(y_true, y_pred, TARGET_CH, SFREQ, save_name=f"metrics_bandpower_topomap_similarity_{tag}.csv")

        if tag == "best_val_loss" or os.getenv("PLOT_ALL_CKPT", "0") == "1":
            plot_signal_examples(y_true, y_pred, TARGET_CH, sample_idx=0, sfreq=SFREQ, zoom_seconds=10, zoom_start_sec=30)
            y_interp = plot_brain_maps(x_input, y_true, y_pred)
            df_mne_time = evaluate_per_channel(y_true, y_interp, TARGET_CH, save_name=f"metrics_per_channel_mne_spline_{tag}.csv")
            df_mne_spec = evaluate_spectral_correlation(y_true, y_interp, TARGET_CH, SFREQ, save_name=f"metrics_spectral_corr_mne_spline_{tag}.csv")
            df_mne_topo = evaluate_bandpower_topomap_similarity(y_true, y_interp, TARGET_CH, SFREQ, save_name=f"metrics_bandpower_topomap_similarity_mne_spline_{tag}.csv")
            df_summary = save_ai_vs_mne_summary(df_time, df_mne_time, df_spec, df_mne_spec, df_topo, df_mne_topo)
            display(df_summary)

    print("\nSELESAI HYBRID. Output dir:", OUTPUT_DIR)

if __name__ == "__main__":
    if MODEL_ARCH == "maser":
        main_maser()
    elif MODEL_ARCH == "hybrid":
        main_hybrid()
    else:
        main()
