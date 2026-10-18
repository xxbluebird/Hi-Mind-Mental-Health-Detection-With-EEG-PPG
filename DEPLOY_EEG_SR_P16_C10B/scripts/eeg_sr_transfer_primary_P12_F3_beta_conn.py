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

# ===================== C1 PATCH HELPERS =====================
def _as_float_list_C1(x, fallback=(0.5, 0.5, 0.5, 0.5)):
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if hasattr(x, "tolist"):
            x = x.tolist()
        vals = [float(v) for v in list(x)]
        return vals
    except Exception:
        return list(fallback)

def _parse_float_list_env_C1(name, default=(0.5, 0.5, 0.5, 0.5)):
    import os as _os_C1
    raw = _os_C1.environ.get(name, "").strip()
    if not raw:
        return list(default)
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != len(default):
        raise ValueError(f"{name} harus berisi {len(default)} angka, dapat {len(vals)}: {raw}")
    return vals
# =================== END C1 PATCH HELPERS ===================



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

# Format dataset:
# - seed_npy     : SEED converted .npy folder seperti eksperimen sebelumnya
# - primary_csv  : dataset primer CSV MAICA, kolom: time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG
# - auto         : otomatis pakai primary_csv jika ditemukan file .csv, selain itu seed_npy
DATA_FORMAT = os.getenv("DATA_FORMAT", "auto").lower().strip()

# Sampling
ORIGINAL_SFREQ = 200
TARGET_SFREQ = 250
SFREQ = TARGET_SFREQ
SEGMENT_SECONDS = int(os.getenv("SEGMENT_SECONDS", "120"))
SEGMENT_SAMPLES = SFREQ * SEGMENT_SECONDS

# Setup baru:
# Input fisik  : Fp1, Fp2, T7/T8
# Target virtual: F3, F4, P3, P4
# Pada CSV primer, T7/T8 biasanya bernama T3/T4.
SR_CHANNELS = ["Fp1", "Fp2", "T7", "T8", "F3", "F4", "P3", "P4"]
INPUT_CH = ["Fp1", "Fp2", "T7", "T8"]
TARGET_CH = ["F3", "F4", "P3", "P4"]
INPUT_IDX = [0, 1, 2, 3]
TARGET_IDX = [4, 5, 6, 7]

TEST_RATIO = float(os.getenv("TEST_RATIO", "0.20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # A100 bisa coba 32/64; Kaggle T4 pakai 16
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4")) # Kaggle sering aman 0; supercomputer bisa 4/8
MAX_TRAIN_SEGMENTS = None
MAX_TEST_SEGMENTS = None
EPOCHS = int(os.getenv("EPOCHS", "100"))
SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"
ADAPT_ALL_PRIMARY = os.getenv("ADAPT_ALL_PRIMARY", "0") == "1"
HOLDOUT_SUBJECTS = [x.strip() for x in __import__("os").environ.get("C1_TEST_SUBJECT", "Anna").split(",") if x.strip()]  # C1_FORCE_SINGLE_TEST_SUBJECT
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
# Batas maksimum kontribusi residual. Penting agar residual tidak over-correct waveform.
HYBRID_RESIDUAL_SCALE_MAX = float(os.getenv("HYBRID_RESIDUAL_SCALE_MAX", "0.03"))
HYBRID_DROPOUT = float(os.getenv("HYBRID_DROPOUT", "0.05"))

# Loss hybrid. Alpha regularization menjaga bobot tidak lari terlalu jauh dari ensemble terbaik manual.
HYBRID_ALPHA_REG_W = float(os.getenv("HYBRID_ALPHA_REG_W", "0.001"))
HYBRID_TIME_W = float(os.getenv("HYBRID_TIME_W", "0.55"))
HYBRID_TEMPORAL_W = float(os.getenv("HYBRID_TEMPORAL_W", "0.30"))
HYBRID_AMP_W = float(os.getenv("HYBRID_AMP_W", "0.08"))
HYBRID_BETA_TOPO_W = float(os.getenv("HYBRID_BETA_TOPO_W", "0.00"))
HYBRID_BETA_POWER_W = float(os.getenv("HYBRID_BETA_POWER_W", "0.00"))

# ============================================================
# P12: F3 beta-connectivity focused loss
# Fokus: memperbaiki F3 dan hubungan beta Fp1-F3 untuk fitur anxiety.
# TARGET_CH order diasumsikan: F3,F4,P3,P4 atau F3,F4,T7,T8.
# F3 index = 0, Fp1 input index = 0.
# ============================================================
HYBRID_RESUME_CKPT = os.getenv("HYBRID_RESUME_CKPT", "").strip()

P12_F3_TIME_W = float(os.getenv("P12_F3_TIME_W", "0.00"))
P12_F3_BETA_W = float(os.getenv("P12_F3_BETA_W", "0.00"))
P12_FP1F3_BETA_CORR_W = float(os.getenv("P12_FP1F3_BETA_CORR_W", "0.00"))
P12_BETA_FMIN = float(os.getenv("P12_BETA_FMIN", "13.0"))
P12_BETA_FMAX = float(os.getenv("P12_BETA_FMAX", "30.0"))

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




def parse_subject_id_from_csv(path):
    # Nama subjek = kata pertama sebelum underscore.
    # Contoh: Anna_rec1_clean_...csv -> Anna
    name = Path(path).stem
    return name.split("_")[0]


def discover_csv_files(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT tidak ditemukan: {root_dir}")
    files = sorted(root.rglob("*.csv"))
    # Abaikan file ringkasan/metrics kalau tidak sengaja ada di dalam folder.
    files = [p for p in files if not p.name.lower().startswith(("summary", "metrics", "training_history"))]
    if not files:
        raise RuntimeError(f"Tidak ada file .csv di {root_dir}")
    return files


def split_subject_files(all_files, test_ratio=0.2, seed=42):
    subjects = sorted(set(parse_subject_id_from_csv(p) for p in all_files))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = sorted(subjects[-n_test:])
    train_subjects = sorted(subjects[:-n_test])
    train_files = [p for p in all_files if parse_subject_id_from_csv(p) in train_subjects]
    test_files = [p for p in all_files if parse_subject_id_from_csv(p) in test_subjects]
    return train_subjects, test_subjects, train_files, test_files


def resolve_data_format(data_root):
    fmt = DATA_FORMAT
    if fmt == "auto":
        root = Path(data_root)
        csv_files = list(root.rglob("*.csv")) if root.exists() else []
        npy_files = list(root.rglob("*.npy")) if root.exists() else []
        fmt = "primary_csv" if len(csv_files) > 0 else "seed_npy"
        print(f"DATA_FORMAT auto -> {fmt} | csv={len(csv_files)} | npy={len(npy_files)}")
    if fmt not in ["seed_npy", "primary_csv"]:
        raise ValueError("DATA_FORMAT harus auto, seed_npy, atau primary_csv")
    return fmt


def audit_primary_csv_input(data_root):
    files = discover_csv_files(data_root)
    print("=" * 80)
    print("AUDIT INPUT PRIMARY CSV")
    print("=" * 80)
    print("DATA_ROOT:", data_root)
    print("Jumlah file .csv:", len(files))
    print("Contoh file:", files[0])
    df_head = pd.read_csv(files[0], nrows=5)
    print("Kolom contoh:", list(df_head.columns))
    required = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
    missing = [c for c in required if c not in df_head.columns]
    if missing:
        raise RuntimeError(f"Kolom wajib tidak ada di CSV: {missing}")
    # Hitung jumlah baris contoh secara hemat memori.
    n_rows = sum(1 for _ in open(files[0], "r")) - 1
    print("Jumlah baris contoh:", n_rows)
    print("Expected segment samples:", SEGMENT_SAMPLES)
    subjects = sorted(set(parse_subject_id_from_csv(p) for p in files))
    print("Jumlah subjek:", len(subjects))
    print("Subjek contoh:", subjects[:10])
    print("=" * 80)
    return files

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



class PrimaryCSVSRDataset(Dataset):
    """Dataset primer MAICA CSV untuk transfer learning EEG-SR.

    CSV diharapkan memiliki kolom:
      time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG

    Mapping untuk kompatibilitas model SEED:
      input  = Fp1,Fp2,P3,P4
      target = F3,F4,T3,T4, tetapi diberi label internal F3,F4,T7,T8
    """
    INPUT_FILE_COLS = ["Fp1", "Fp2", "T3", "T4"]
    TARGET_FILE_COLS = ["F3", "F4", "P3", "P4"]

    def __init__(self, file_list, segment_samples, train=True, max_segments=None):
        self.files = [Path(p) for p in file_list]
        self.segment_samples = int(segment_samples)
        self.train = bool(train)
        self.files = sorted(self.files)
        if self.train:
            random.shuffle(self.files)
        if max_segments is not None:
            self.files = self.files[:int(max_segments)]
        if not self.files:
            raise RuntimeError("Tidak ada file CSV untuk PrimaryCSVSRDataset.")
        print(
            f"{'Train' if self.train else 'Test'} PRIMARY CSV dataset:\n"
            f"  Jumlah file CSV : {len(self.files)}\n"
            f"  Input columns   : {self.INPUT_FILE_COLS}\n"
            f"  Target columns  : {self.TARGET_FILE_COLS} -> labels {TARGET_CH}\n"
            f"  Segment samples : {self.segment_samples}"
        )

    def __len__(self):
        return len(self.files)

    def _fix_length(self, arr):
        # arr: (C,T)
        t = arr.shape[-1]
        if t == self.segment_samples:
            return arr
        if t > self.segment_samples:
            # CSV ini sudah berupa segment; center-crop agar deterministik.
            start = (t - self.segment_samples) // 2
            return arr[:, start:start + self.segment_samples]
        # Jika terlalu pendek, pad agar model tetap bisa jalan.
        pad = self.segment_samples - t
        if t <= 1:
            return np.pad(arr, ((0, 0), (0, pad)), mode="edge")
        return np.pad(arr, ((0, 0), (0, pad)), mode="reflect")

    def _load_csv(self, path):
        usecols = self.INPUT_FILE_COLS + self.TARGET_FILE_COLS
        try:
            df = pd.read_csv(path, usecols=usecols)
        except ValueError as e:
            df0 = pd.read_csv(path, nrows=1)
            raise ValueError(f"Gagal membaca {path}. Kolom tersedia: {list(df0.columns)}. Error: {e}")
        df = df.replace([np.inf, -np.inf], np.nan)
        # Interpolasi ringan jika ada NaN kecil; lalu fill sisa dengan 0.
        df = df.interpolate(limit_direction="both").fillna(0.0)
        x_raw = df[self.INPUT_FILE_COLS].to_numpy(dtype=np.float32).T
        y_raw = df[self.TARGET_FILE_COLS].to_numpy(dtype=np.float32).T
        x_raw = self._fix_length(x_raw)
        y_raw = self._fix_length(y_raw)
        return x_raw, y_raw

    @staticmethod
    def _robust_scale_by_input(x_raw, y_raw):
        # Sama seperti SEED: statistik normalisasi hanya dari input channel.
        q75, q25 = np.percentile(x_raw, [75, 25])
        iqr = (q75 - q25) + 1e-6
        median = np.median(x_raw)
        x = (x_raw - median) / iqr
        y = (y_raw - median) / iqr
        x = np.tanh(x / 5.0)
        y = np.tanh(y / 5.0)
        return x.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, idx):
        x_raw, y_raw = self._load_csv(self.files[idx])
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
        residual_scale_max=0.03,
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
            self.residual_scale_max = float(residual_scale_max)
            init = float(residual_scale_init)
            # Parameterisasi bounded: scale = max * tanh(raw).
            # Ini mencegah residual membesar terlalu jauh seperti pada E0/D0.
            ratio = max(min(init / max(self.residual_scale_max, 1e-8), 0.98), -0.98)
            raw_init = float(np.arctanh(ratio))
            self.residual_scale_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.residual_refiner = None
            self.residual_scale_max = None
            self.residual_scale_raw = None

        if self.freeze_backbones:
            for m in [self.frontal_model, self.temporal_model, self.maser_model]:
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_alpha(self):
        if self.mode == "fixed_alpha":
            return self.alpha_fixed
        return torch.sigmoid(self.alpha_logits)

    def get_residual_scale(self):
        if self.residual_scale_raw is None:
            return None
        return self.residual_scale_max * torch.tanh(self.residual_scale_raw)

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
            residual_scale = self.get_residual_scale().to(dtype=y_base.dtype, device=y_base.device)
            y_out = y_base + residual_scale * residual
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



def p12_fft_bandpass_torch(x, sfreq=250, fmin=13.0, fmax=30.0):
    """
    Differentiable FFT bandpass.
    x shape: (B,T) atau (B,C,T)
    """
    x_float = x.float()
    n_times = x_float.shape[-1]
    fft = torch.fft.rfft(x_float, dim=-1)

    freqs = torch.fft.rfftfreq(
        n_times,
        d=1.0 / float(sfreq),
        device=x_float.device
    )

    mask = ((freqs >= float(fmin)) & (freqs <= float(fmax))).to(fft.dtype)
    mask_shape = [1] * (fft.ndim - 1) + [-1]
    fft = fft * mask.view(*mask_shape)

    y = torch.fft.irfft(fft, n=n_times, dim=-1)
    return y.to(dtype=x.dtype)


def p12_corr_1d(a, b):
    """
    Korelasi per batch.
    a,b shape: (B,T)
    return shape: (B,)
    """
    a = a.float()
    b = b.float()

    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)

    num = (a * b).sum(dim=-1)
    den = torch.sqrt((a ** 2).sum(dim=-1) * (b ** 2).sum(dim=-1)) + 1e-8
    return num / den


def p12_f3_time_loss(pred, target):
    """
    Loss khusus F3 time-domain.
    F3 index target = 0.
    """
    p = pred[:, 0, :]
    t = target[:, 0, :]

    l1 = F.smooth_l1_loss(p, t)
    corr_loss = 1.0 - p12_corr_1d(p, t).mean()

    return 0.70 * l1 + 0.30 * corr_loss


def p12_f3_beta_loss(pred, target, sfreq=250):
    """
    Loss khusus F3 pada beta band 13-30 Hz.
    """
    p = pred[:, 0, :]
    t = target[:, 0, :]

    p_beta = p12_fft_bandpass_torch(p, sfreq, P12_BETA_FMIN, P12_BETA_FMAX)
    t_beta = p12_fft_bandpass_torch(t, sfreq, P12_BETA_FMIN, P12_BETA_FMAX)

    l1 = F.smooth_l1_loss(p_beta, t_beta)
    corr_loss = 1.0 - p12_corr_1d(p_beta, t_beta).mean()

    return 0.50 * l1 + 0.50 * corr_loss


def p12_fp1_f3_beta_connectivity_loss(pred, target, x_input, sfreq=250):
    """
    Proxy konektivitas beta Fp1-F3.
    Tujuan: korelasi beta Fp1-F3 hasil prediksi mendekati korelasi beta Fp1-F3 ground truth.
    Ini bukan MI langsung, tetapi proxy yang differentiable dan relevan untuk fitur MI beta.
    """
    if x_input is None:
        return pred.new_tensor(0.0)

    # Input channel order: Fp1,Fp2,T7,T8 atau Fp1,Fp2,....
    fp1 = x_input[:, 0, :]
    f3_pred = pred[:, 0, :]
    f3_true = target[:, 0, :]

    fp1_beta = p12_fft_bandpass_torch(fp1, sfreq, P12_BETA_FMIN, P12_BETA_FMAX)
    pred_beta = p12_fft_bandpass_torch(f3_pred, sfreq, P12_BETA_FMIN, P12_BETA_FMAX)
    true_beta = p12_fft_bandpass_torch(f3_true, sfreq, P12_BETA_FMIN, P12_BETA_FMAX)

    corr_pred = p12_corr_1d(fp1_beta, pred_beta)
    corr_true = p12_corr_1d(fp1_beta, true_beta)

    return F.smooth_l1_loss(corr_pred, corr_true)


def hybrid_reconstruction_loss(model, pred, target, sfreq=250, x_input=None):
    loss_time = refined_eeg_loss(pred, target, sfreq=sfreq)
    loss_temp = temporal_focused_loss(pred[:, 2:, :], target[:, 2:, :], sfreq=sfreq)
    amp_loss = F.mse_loss(pred.std(dim=-1), target.std(dim=-1))

    total = HYBRID_TIME_W * loss_time + HYBRID_TEMPORAL_W * loss_temp + HYBRID_AMP_W * amp_loss

    if HYBRID_BETA_TOPO_W > 0:
        total = total + HYBRID_BETA_TOPO_W * differentiable_topomap_corr_loss(pred, target, sfreq=sfreq)
    if HYBRID_BETA_POWER_W > 0:
        total = total + HYBRID_BETA_POWER_W * beta_absolute_power_loss(pred, target, sfreq=sfreq)

    # P12 tambahan: F3-focused objective
    if P12_F3_TIME_W > 0:
        total = total + P12_F3_TIME_W * p12_f3_time_loss(pred, target)

    if P12_F3_BETA_W > 0:
        total = total + P12_F3_BETA_W * p12_f3_beta_loss(pred, target, sfreq=sfreq)

    if P12_FP1F3_BETA_CORR_W > 0:
        total = total + P12_FP1F3_BETA_CORR_W * p12_fp1_f3_beta_connectivity_loss(
            pred, target, x_input, sfreq=sfreq
        )

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




# ===================== TRUE C7D HF BANDPASS LOSS PATCH =====================
import os as _os_C7D
import torch as _torch_C7D
import torch.nn.functional as _F_C7D

def _c7d_parse_float_list(txt, default):
    try:
        vals = [float(x.strip()) for x in str(txt).split(",") if x.strip()]
        return vals if len(vals) > 0 else default
    except Exception:
        return default

def _c7d_fft_bandpass(x, fs=250.0, f_low=13.0, f_high=40.0):
    """
    Differentiable FFT bandpass.
    x shape expected: [B, C, T].
    """
    if x is None:
        return x
    if x.ndim != 3:
        return x

    x = x.float()
    n = x.shape[-1]
    X = _torch_C7D.fft.rfft(x, dim=-1)
    freqs = _torch_C7D.fft.rfftfreq(n, d=1.0 / float(fs)).to(device=x.device)

    mask = ((freqs >= float(f_low)) & (freqs <= float(f_high))).to(dtype=X.dtype, device=x.device)
    while mask.ndim < X.ndim:
        mask = mask.view(*([1] * (X.ndim - 1)), -1)

    Y = X * mask
    y = _torch_C7D.fft.irfft(Y, n=n, dim=-1)
    return y

def _c7d_find_tensor(local_vars, names):
    for name in names:
        if name in local_vars:
            v = local_vars[name]
            if _torch_C7D.is_tensor(v) and v.ndim == 3:
                return v
    return None

def _c7d_locals_hf_loss(local_vars):
    """
    Tambahan loss eksplisit untuk HF band 13–40 Hz.
    Aktif hanya jika C7D_TRUE_HF_LOSS=1.

    Return sudah termasuk bobot C7D_HF_W, C7D_HF_DERIV_W, C7D_HF_STD_W.
    """
    if _os_C7D.environ.get("C7D_TRUE_HF_LOSS", "0") != "1":
        # cari device dari loss kalau ada
        loss = local_vars.get("loss", None)
        if _torch_C7D.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C7D.tensor(0.0)

    pred = _c7d_find_tensor(local_vars, ["pred", "y_pred", "yp", "output", "outputs", "out"])
    true = _c7d_find_tensor(local_vars, ["yb", "y_true", "yt", "target", "targets", "y"])

    if pred is None or true is None:
        loss = local_vars.get("loss", None)
        if _torch_C7D.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C7D.tensor(0.0)

    if pred.shape != true.shape:
        return pred.new_tensor(0.0)

    fs = float(_os_C7D.environ.get("C7D_HF_FS", _os_C7D.environ.get("SFREQ", "250")))
    f_low = float(_os_C7D.environ.get("C7D_HF_LOW", "13"))
    f_high = float(_os_C7D.environ.get("C7D_HF_HIGH", "40"))

    hf_w = float(_os_C7D.environ.get("C7D_HF_W", "1.0"))
    deriv_w = float(_os_C7D.environ.get("C7D_HF_DERIV_W", "0.0"))
    std_w = float(_os_C7D.environ.get("C7D_HF_STD_W", "0.0"))

    ch_weights = _c7d_parse_float_list(
        _os_C7D.environ.get("C7D_CHANNEL_WEIGHTS", "1,1,2.5,2.5"),
        [1.0, 1.0, 2.5, 2.5]
    )

    C = pred.shape[1]
    if len(ch_weights) < C:
        ch_weights = ch_weights + [ch_weights[-1]] * (C - len(ch_weights))
    ch_weights = ch_weights[:C]

    w = _torch_C7D.tensor(ch_weights, dtype=pred.dtype, device=pred.device).view(1, C, 1)

    pred_hf = _c7d_fft_bandpass(pred, fs=fs, f_low=f_low, f_high=f_high)
    true_hf = _c7d_fft_bandpass(true, fs=fs, f_low=f_low, f_high=f_high)

    # HF waveform loss
    loss_hf = ((pred_hf - true_hf).abs() * w).mean()

    total = hf_w * loss_hf

    # Derivative HF loss: memaksa perubahan cepat/high-frequency lebih sinkron
    if deriv_w > 0:
        dp = pred_hf[..., 1:] - pred_hf[..., :-1]
        dt = true_hf[..., 1:] - true_hf[..., :-1]
        loss_deriv = ((dp - dt).abs() * w).mean()
        total = total + deriv_w * loss_deriv

    # STD ratio loss: memaksa energi/amplitudo HF tidak terlalu kecil
    if std_w > 0:
        eps = 1e-6
        pred_std = pred_hf.std(dim=-1)
        true_std = true_hf.std(dim=-1).clamp_min(eps)
        ratio = pred_std / true_std
        ww = w.squeeze(-1)
        loss_std = ((ratio - 1.0).abs() * ww).mean()
        total = total + std_w * loss_std

    return total
# =================== END TRUE C7D HF BANDPASS LOSS PATCH ===================



# ===================== TRUE C9 FRONTAL BETA SPECTRAL LOSS PATCH =====================
import os as _os_C9
import torch as _torch_C9

def _c9_find_tensor(local_vars, names):
    for name in names:
        if name in local_vars:
            v = local_vars[name]
            if _torch_C9.is_tensor(v) and v.ndim == 3:
                return v
    return None

def _c9_parse_idx(txt, default):
    try:
        vals = [int(x.strip()) for x in str(txt).split(",") if x.strip()]
        return vals if len(vals) else default
    except Exception:
        return default

def _c9_frontal_beta_spectral_loss(local_vars):
    """
    C9 loss:
    Fokus menaikkan spectral similarity pada frontal beta.
    Default channel index:
      0 = F3
      1 = F4
    Band:
      beta 13–30 Hz
    """
    if _os_C9.environ.get("C9_FRONTAL_BETA_LOSS", "0") != "1":
        loss = local_vars.get("loss", None)
        if _torch_C9.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C9.tensor(0.0)

    pred = _c9_find_tensor(local_vars, ["pred", "y_pred", "yp", "output", "outputs", "out"])
    true = _c9_find_tensor(local_vars, ["yb", "y_true", "yt", "target", "targets", "y"])

    if pred is None or true is None:
        loss = local_vars.get("loss", None)
        if _torch_C9.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C9.tensor(0.0)

    if pred.shape != true.shape:
        return pred.new_tensor(0.0)

    fs = float(_os_C9.environ.get("C9_FS", _os_C9.environ.get("SFREQ", "250")))
    f_low = float(_os_C9.environ.get("C9_BETA_LOW", "13"))
    f_high = float(_os_C9.environ.get("C9_BETA_HIGH", "30"))
    ch_idx = _c9_parse_idx(_os_C9.environ.get("C9_CHANNEL_IDX", "0,1"), [0, 1])

    spec_w = float(_os_C9.environ.get("C9_SPEC_CORR_W", "1.0"))
    mae_w = float(_os_C9.environ.get("C9_LOGPSD_MAE_W", "0.2"))
    std_w = float(_os_C9.environ.get("C9_BETA_STD_W", "0.1"))

    eps = 1e-6

    pred = pred.float()
    true = true.float()

    # Select frontal channels
    ch_idx = [i for i in ch_idx if i >= 0 and i < pred.shape[1]]
    if not ch_idx:
        return pred.new_tensor(0.0)

    pred = pred[:, ch_idx, :]
    true = true[:, ch_idx, :]

    n = pred.shape[-1]
    freqs = _torch_C9.fft.rfftfreq(n, d=1.0 / fs).to(pred.device)
    mask = (freqs >= f_low) & (freqs <= f_high)

    if mask.sum() < 3:
        return pred.new_tensor(0.0)

    P = _torch_C9.fft.rfft(pred, dim=-1)
    T = _torch_C9.fft.rfft(true, dim=-1)

    # log power spectrum beta
    P_log = _torch_C9.log((_torch_C9.abs(P[..., mask]) ** 2) + eps)
    T_log = _torch_C9.log((_torch_C9.abs(T[..., mask]) ** 2) + eps)

    # spectral correlation across frequency bins
    Pc = P_log - P_log.mean(dim=-1, keepdim=True)
    Tc = T_log - T_log.mean(dim=-1, keepdim=True)

    corr = (Pc * Tc).sum(dim=-1) / (
        _torch_C9.sqrt((Pc ** 2).sum(dim=-1) + eps) *
        _torch_C9.sqrt((Tc ** 2).sum(dim=-1) + eps)
    )

    loss_corr = 1.0 - corr.mean()
    loss_mae = (P_log - T_log).abs().mean()

    total = spec_w * loss_corr + mae_w * loss_mae

    # beta std/power ratio supaya beta frontal tidak underpowered
    if std_w > 0:
        pred_beta = _torch_C9.fft.irfft(P * mask.to(P.dtype).view(1, 1, -1), n=n, dim=-1)
        true_beta = _torch_C9.fft.irfft(T * mask.to(T.dtype).view(1, 1, -1), n=n, dim=-1)

        pred_std = pred_beta.std(dim=-1)
        true_std = true_beta.std(dim=-1).clamp_min(eps)

        ratio = pred_std / true_std
        loss_std = (ratio - 1.0).abs().mean()
        total = total + std_w * loss_std

    return total
# =================== END TRUE C9 FRONTAL BETA SPECTRAL LOSS PATCH ===================



# ===================== TRUE C10 LOCAL TIME-DOMAIN LOSS PATCH =====================
import os as _os_C10
import torch as _torch_C10

def _c10_find_tensor(local_vars, names):
    for name in names:
        if name in local_vars:
            v = local_vars[name]
            if _torch_C10.is_tensor(v) and v.ndim == 3:
                return v
    return None

def _c10_parse_float_list(txt, default):
    try:
        vals = [float(x.strip()) for x in str(txt).split(",") if x.strip()]
        return vals if len(vals) else default
    except Exception:
        return default

def _c10_local_time_domain_loss(local_vars):
    """
    C10 loss:
    Fokus menaikkan korelasi time-domain lokal.
    Loss dihitung pada random short crop dari sinyal 120 detik.
    Cocok untuk EEG karena dinamika lokal lebih informatif daripada global 120 s.
    """
    if _os_C10.environ.get("C10_LOCAL_TIME_LOSS", "0") != "1":
        loss = local_vars.get("loss", None)
        if _torch_C10.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C10.tensor(0.0)

    pred = _c10_find_tensor(local_vars, ["pred", "y_pred", "yp", "output", "outputs", "out"])
    true = _c10_find_tensor(local_vars, ["yb", "y_true", "yt", "target", "targets", "y"])

    if pred is None or true is None:
        loss = local_vars.get("loss", None)
        if _torch_C10.is_tensor(loss):
            return loss.new_tensor(0.0)
        return _torch_C10.tensor(0.0)

    if pred.shape != true.shape:
        return pred.new_tensor(0.0)

    pred = pred.float()
    true = true.float()

    B, C, T = pred.shape

    fs = float(_os_C10.environ.get("C10_FS", _os_C10.environ.get("SFREQ", "250")))
    crop_sec = float(_os_C10.environ.get("C10_CROP_SEC", "12"))
    n_crops = int(_os_C10.environ.get("C10_N_CROPS", "3"))

    crop_len = int(crop_sec * fs)
    crop_len = max(256, min(crop_len, T))

    main_w = float(_os_C10.environ.get("C10_TIME_W", "1.0"))
    corr_w = float(_os_C10.environ.get("C10_CORR_W", "1.0"))
    mae_w = float(_os_C10.environ.get("C10_ZMAE_W", "0.15"))
    deriv_w = float(_os_C10.environ.get("C10_DERIV_CORR_W", "0.25"))
    amp_w = float(_os_C10.environ.get("C10_AMP_W", "0.10"))

    ch_w = _c10_parse_float_list(_os_C10.environ.get("C10_CHANNEL_WEIGHTS", "1,1,1,1"), [1.0] * C)
    if len(ch_w) < C:
        ch_w = ch_w + [1.0] * (C - len(ch_w))
    ch_w = _torch_C10.tensor(ch_w[:C], device=pred.device, dtype=pred.dtype).view(1, C)
    ch_w = ch_w / ch_w.mean().clamp_min(1e-6)

    eps = 1e-6

    total = pred.new_tensor(0.0)

    for _ in range(max(1, n_crops)):
        if T > crop_len:
            start = int(_torch_C10.randint(0, T - crop_len + 1, (1,), device=pred.device).item())
            p = pred[:, :, start:start+crop_len]
            t = true[:, :, start:start+crop_len]
        else:
            p = pred
            t = true

        # zero-mean
        p0 = p - p.mean(dim=-1, keepdim=True)
        t0 = t - t.mean(dim=-1, keepdim=True)

        # local Pearson corr
        corr = (p0 * t0).sum(dim=-1) / (
            _torch_C10.sqrt((p0 ** 2).sum(dim=-1) + eps) *
            _torch_C10.sqrt((t0 ** 2).sum(dim=-1) + eps)
        )
        loss_corr = ((1.0 - corr) * ch_w).mean()

        # z-normalized MAE, supaya fokus bentuk gelombang bukan skala saja
        pz = p0 / p0.std(dim=-1, keepdim=True).clamp_min(eps)
        tz = t0 / t0.std(dim=-1, keepdim=True).clamp_min(eps)
        loss_zmae = ((pz - tz).abs().mean(dim=-1) * ch_w).mean()

        # derivative correlation, supaya phase/dinamika lokal lebih ikut
        dp = p[:, :, 1:] - p[:, :, :-1]
        dt = t[:, :, 1:] - t[:, :, :-1]

        dp0 = dp - dp.mean(dim=-1, keepdim=True)
        dt0 = dt - dt.mean(dim=-1, keepdim=True)

        dcorr = (dp0 * dt0).sum(dim=-1) / (
            _torch_C10.sqrt((dp0 ** 2).sum(dim=-1) + eps) *
            _torch_C10.sqrt((dt0 ** 2).sum(dim=-1) + eps)
        )
        loss_deriv = ((1.0 - dcorr) * ch_w).mean()

        # amplitude std ratio
        std_p = p.std(dim=-1)
        std_t = t.std(dim=-1).clamp_min(eps)
        loss_amp = (((std_p / std_t) - 1.0).abs() * ch_w).mean()

        total = total + (
            corr_w * loss_corr
            + mae_w * loss_zmae
            + deriv_w * loss_deriv
            + amp_w * loss_amp
        )

    total = total / max(1, n_crops)
    return main_w * total
# =================== END TRUE C10 LOCAL TIME-DOMAIN LOSS PATCH ===================

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
            loss = hybrid_reconstruction_loss(unwrap(model), pred, yb, sfreq=SFREQ, x_input=xb)

        # C7D_HF_LOSS_INSERT_TRAIN
        loss = loss + _c7d_locals_hf_loss(locals())
        # C9_FRONTAL_BETA_INSERT_TRAIN
        loss = loss + _c9_frontal_beta_spectral_loss(locals())
        # C10_LOCAL_TIME_INSERT_TRAIN
        loss = loss + _c10_local_time_domain_loss(locals())
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
            loss = hybrid_reconstruction_loss(unwrap(model), pred, yb, sfreq=SFREQ, x_input=xb)
            mse = F.mse_loss(pred.float(), yb.float())

        corr_batch = batch_corr_torch(pred.float(), yb.float())
        valid = torch.isfinite(corr_batch)
        corr_sum += torch.where(valid, corr_batch, torch.zeros_like(corr_batch)).sum(dim=0)
        corr_count += valid.sum(dim=0)

        std_ratio_batch = pred.float().std(dim=-1) / (yb.float().std(dim=-1) + 1e-8)
        valid_std = torch.isfinite(std_ratio_batch)
        std_ratio_sum += torch.where(valid_std, std_ratio_batch, torch.zeros_like(std_ratio_batch)).sum(dim=0)
        std_ratio_count += valid_std.sum(dim=0)

        # C7D_HF_LOSS_INSERT_EVAL
        loss = loss + _c7d_locals_hf_loss(locals())
        # C9_FRONTAL_BETA_INSERT_EVAL
        loss = loss + _c9_frontal_beta_spectral_loss(locals())
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
    # C15_EXPORT_SEGMENT_CSV_CALL_INSIDE_TIME_METRICS
    try:
        _c15_export_segments_from_eval(locals())
    except Exception as e:
        print('[C15_EXPORT] ERROR inside time metrics:', repr(e))
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
    # C15_EXPORT_SEGMENT_CSV_CALL
    _c15_export_segments_from_eval(locals())
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

    # C1_PATCH_OUTROOT_BEGIN
    import os as _os_C1_out
    from pathlib import Path as _Path_C1_out
    if _os_C1_out.environ.get("OUT_ROOT", "").strip():
        OUTPUT_DIR = _Path_C1_out(_os_C1_out.environ["OUT_ROOT"]) / _Path_C1_out(str(OUTPUT_DIR)).name
    # C1_PATCH_OUTROOT_END
    print("Output dir      :", OUTPUT_DIR)
    print("Device          :", device)
    print("GPU count       :", torch.cuda.device_count())
    print("Batch size      :", BATCH_SIZE)
    print("Num workers     :", NUM_WORKERS)
    print("AMP / TF32      :", USE_AMP, USE_TF32)
    print("Beta weights    :", TEMPORAL_BETA_SHAPE_WEIGHT, TEMPORAL_BETA_REL_POWER_WEIGHT)
    print("Temporal loss w : mae", TEMPORAL_MAE_W, "corr", TEMPORAL_CORR_W, "diff", TEMPORAL_DIFF_W, "broad", TEMPORAL_BROAD_SPEC_W)
    print("="*80)

    data_format = resolve_data_format(DATA_ROOT)
    print("Data format      :", data_format)

    if data_format == "primary_csv":
        all_files = audit_primary_csv_input(DATA_ROOT)
        if HOLDOUT_SUBJECTS:
            all_subjects = sorted(set(parse_subject_id_from_csv(p) for p in all_files))
            missing = [s for s in HOLDOUT_SUBJECTS if s not in all_subjects]
            if missing:
                raise RuntimeError(f"HOLDOUT_SUBJECTS tidak ditemukan: {missing}. Subjek tersedia: {all_subjects}")

            test_subjects = [s for s in all_subjects if s in HOLDOUT_SUBJECTS]
            train_subjects = [s for s in all_subjects if s not in HOLDOUT_SUBJECTS]

            train_files = [p for p in all_files if parse_subject_id_from_csv(p) in train_subjects]
            test_files = [p for p in all_files if parse_subject_id_from_csv(p) in test_subjects]

            print(f"[HOLDOUT_SUBJECTS] Validasi subjek: {test_subjects}")
        elif ADAPT_ALL_PRIMARY:
            all_subjects = sorted(set(parse_subject_id_from_csv(p) for p in all_files))
            train_subjects = all_subjects
            test_subjects = all_subjects
            train_files = list(all_files)
            test_files = list(all_files)
            print("[ADAPT_ALL_PRIMARY=1] Semua data primer dipakai untuk adaptasi.")
            print("[INFO] Test internal hanya untuk monitoring training, bukan hasil final laporan.")
        else:
            train_subjects, test_subjects, train_files, test_files = split_subject_files(all_files, TEST_RATIO, SEED)


        # C1_PATCH_FORCE_LOSO_BEGIN

        import os as _os_C1_loso

        _c1_test_subject = _os_C1_loso.environ.get("C1_TEST_SUBJECT", "").strip()

        if _c1_test_subject:

            _all_subjects_c1 = sorted(set(train_subjects) | set(test_subjects))

            if _c1_test_subject not in _all_subjects_c1:

                raise ValueError(f"C1_TEST_SUBJECT={_c1_test_subject} tidak ada di subject list: {_all_subjects_c1}")

            test_subjects = [_c1_test_subject]

            train_subjects = [ss for ss in _all_subjects_c1 if ss != _c1_test_subject]

        # C1_PATCH_FORCE_LOSO_END

        print("Train subjects:", train_subjects)
        print("Test subjects :", test_subjects)
        print("Train files   :", len(train_files), "| Test files:", len(test_files))

        # Simpan split agar eksperimen transfer learning reproducible.
        pd.DataFrame({"file": [str(p) for p in train_files], "subject": [parse_subject_id_from_csv(p) for p in train_files]}).to_csv(
            os.path.join(OUTPUT_DIR, "train_files_primary.csv"), index=False
        )
        pd.DataFrame({"file": [str(p) for p in test_files], "subject": [parse_subject_id_from_csv(p) for p in test_files]}).to_csv(
            os.path.join(OUTPUT_DIR, "test_files_primary.csv"), index=False
        )

        train_dataset = PrimaryCSVSRDataset(train_files, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
        test_dataset = PrimaryCSVSRDataset(test_files, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)
    else:
        audit_training_input(DATA_ROOT)
        all_folders = discover_folders(DATA_ROOT)
        train_subjects, test_subjects, train_folders, test_folders = split_subject_folders(all_folders, TEST_RATIO, SEED)
        print("Train subjects:", train_subjects)
        print("Test subjects :", test_subjects)
        print("Train folders :", len(train_folders), "| Test folders:", len(test_folders))
        train_dataset = SEEDSR8NpyDataset(train_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
        test_dataset = SEEDSR8NpyDataset(test_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)

    train_drop_last = len(train_dataset) >= BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin_memory, drop_last=train_drop_last, persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_memory, drop_last=False, persistent_workers=(NUM_WORKERS > 0))
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader), "| drop_last:", train_drop_last)

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
        # C2_FORCE_ROW_GAIN_BIAS_HISTORY
        row = _c2_update_row_with_affine(row, locals())
        # C3_FORCE_ROW_BANDCAL_HISTORY
        row = _c3_update_row_with_bandcal(row, locals())
        # C4_FORCE_ROW_BANDMIX_HISTORY
        row = _c4_update_row_with_bandmix(row, locals())
        # C5_FORCE_ROW_TEMPRES_HISTORY
        row = _c5_update_row_with_tempres(row, locals())
        # C6_FORCE_ROW_HISTORY
        row = _c6_update_row(row, locals())
        # C17_FORCE_ROW_HISTORY
        row = _c17_update_row(row, locals())
        # C8_FORCE_ROW_HISTORY
        row = _c8_update_row(row, locals())
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
    # C1_FORCE_OPTIMIZER_LR_FINAL
    import os as _os_C1_lr_final
    _c1_lr_raw = _os_C1_lr_final.environ.get('LR', '').strip()
    if _c1_lr_raw:
        _c1_lr_val = float(_c1_lr_raw)
        for _pg in optimizer.param_groups:
            _pg['lr'] = _c1_lr_val
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
        # C2_FORCE_ROW_GAIN_BIAS_HISTORY
        row = _c2_update_row_with_affine(row, locals())
        # C3_FORCE_ROW_BANDCAL_HISTORY
        row = _c3_update_row_with_bandcal(row, locals())
        # C4_FORCE_ROW_BANDMIX_HISTORY
        row = _c4_update_row_with_bandmix(row, locals())
        # C5_FORCE_ROW_TEMPRES_HISTORY
        row = _c5_update_row_with_tempres(row, locals())
        # C6_FORCE_ROW_HISTORY
        row = _c6_update_row(row, locals())
        # C17_FORCE_ROW_HISTORY
        row = _c17_update_row(row, locals())
        # C8_FORCE_ROW_HISTORY
        row = _c8_update_row(row, locals())
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






# ===================== TRUE C2 AFFINE PATCH =====================
import torch as _torch_C2
import torch.nn as _nn_C2

class _C2AffineHybridWrapper(_nn_C2.Module):
    """
    Wrapper calibration-only:
      y_cal = gain * y_fusion + bias

    Backbone tetap frozen. Parameter trainable hanya:
      - base alpha, jika HYBRID_MODE=learned_alpha
      - c2_gain_raw
      - c2_bias_raw
    """
    def __init__(
        self,
        base_model,
        n_ch=4,
        gain_init=None,
        bias_init=None,
        gain_min=0.5,
        gain_max=2.0,
        bias_max=0.5,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_ch = int(n_ch)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.bias_max = float(bias_max)

        if gain_init is None:
            gain_init = [1.0] * self.n_ch
        if bias_init is None:
            bias_init = [0.0] * self.n_ch

        gain_init = _torch_C2.tensor(gain_init, dtype=_torch_C2.float32)
        bias_init = _torch_C2.tensor(bias_init, dtype=_torch_C2.float32)

        # inverse sigmoid untuk constrained gain
        eps = 1e-6
        z = (gain_init - self.gain_min) / max(self.gain_max - self.gain_min, eps)
        z = z.clamp(eps, 1.0 - eps)
        gain_raw_init = _torch_C2.log(z / (1.0 - z))

        # inverse tanh untuk constrained bias
        b = (bias_init / max(self.bias_max, eps)).clamp(-0.99, 0.99)
        bias_raw_init = 0.5 * _torch_C2.log((1.0 + b) / (1.0 - b))

        self.c2_gain_raw = _nn_C2.Parameter(gain_raw_init.clone())
        self.c2_bias_raw = _nn_C2.Parameter(bias_raw_init.clone())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            base = modules.get("base_model", None)
            if base is not None:
                return getattr(base, name)
            raise

    @property
    def c2_gain(self):
        return self.gain_min + (self.gain_max - self.gain_min) * _torch_C2.sigmoid(self.c2_gain_raw)

    @property
    def c2_bias(self):
        return self.bias_max * _torch_C2.tanh(self.c2_bias_raw)

    def _affine_tensor(self, y):
        if not _torch_C2.is_tensor(y):
            return y
        if y.ndim < 3:
            return y
        if y.shape[1] != self.n_ch:
            return y

        shape = [1, self.n_ch] + [1] * (y.ndim - 2)
        gain = self.c2_gain.to(device=y.device, dtype=y.dtype).reshape(*shape)
        bias = self.c2_bias.to(device=y.device, dtype=y.dtype).reshape(*shape)
        return gain * y + bias

    def _apply_affine_to_output(self, out):
        if _torch_C2.is_tensor(out):
            return self._affine_tensor(out)

        if isinstance(out, tuple):
            out_list = list(out)
            for i, item in enumerate(out_list):
                if _torch_C2.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out_list[i] = self._affine_tensor(item)
                    break
            return tuple(out_list)

        if isinstance(out, list):
            out_list = list(out)
            for i, item in enumerate(out_list):
                if _torch_C2.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out_list[i] = self._affine_tensor(item)
                    break
            return out_list

        if isinstance(out, dict):
            out = dict(out)
            for key in ["pred", "y_pred", "output", "out"]:
                if key in out and _torch_C2.is_tensor(out[key]) and out[key].ndim >= 3 and out[key].shape[1] == self.n_ch:
                    out[key] = self._affine_tensor(out[key])
                    break
            return out

        return out

    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        return self._apply_affine_to_output(out)

    def get_c2_affine_state(self):
        g = self.c2_gain.detach().cpu().float().numpy().tolist()
        b = self.c2_bias.detach().cpu().float().numpy().tolist()
        return {
            "gain_f3": g[0],
            "gain_f4": g[1],
            "gain_p3": g[2],
            "gain_p4": g[3],
            "bias_f3": b[0],
            "bias_f4": b[1],
            "bias_p3": b[2],
            "bias_p4": b[3],
            # alias lama agar summary yang cari t7/t8 tetap kebaca
            "gain_t7": g[2],
            "gain_t8": g[3],
            "bias_t7": b[2],
            "bias_t8": b[3],
        }

def _parse_c2_float_list_env(name, default):
    import os as _os_C2
    raw = _os_C2.environ.get(name, "").strip()
    if not raw:
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != len(default):
        raise ValueError(f"{name} harus berisi {len(default)} angka, dapat {len(vals)}: {raw}")
    return vals

def _maybe_wrap_c2_affine(model, target_ch, device):
    import os as _os_C2

    use_c2 = (
        _os_C2.environ.get("CALIBRATE_AFFINE", "0") == "1"
        or _os_C2.environ.get("CALIBRATE_GAIN_BIAS", "0") == "1"
    )

    if not use_c2:
        return model

    if isinstance(model, _C2AffineHybridWrapper):
        return model

    n_ch = len(target_ch)
    gain_init = _parse_c2_float_list_env("GAIN_INIT", [1.0] * n_ch)
    bias_init = _parse_c2_float_list_env("BIAS_INIT", [0.0] * n_ch)

    gain_min = float(_os_C2.environ.get("GAIN_MIN", "0.5"))
    gain_max = float(_os_C2.environ.get("GAIN_MAX", "2.0"))
    bias_max = float(_os_C2.environ.get("BIAS_MAX", "0.5"))

    wrapped = _C2AffineHybridWrapper(
        model,
        n_ch=n_ch,
        gain_init=gain_init,
        bias_init=bias_init,
        gain_min=gain_min,
        gain_max=gain_max,
        bias_max=bias_max,
    ).to(device)

    print("[TRUE_C2_AFFINE] Aktif: y_cal = gain * y_fusion + bias")
    print("[TRUE_C2_AFFINE] gain_init:", gain_init)
    print("[TRUE_C2_AFFINE] bias_init:", bias_init)
    print("[TRUE_C2_AFFINE] gain_min/max:", gain_min, gain_max)
    print("[TRUE_C2_AFFINE] bias_max:", bias_max)

    return wrapped

def _c2_update_row_with_affine(row, local_vars):
    model = None
    for name in ["model", "hybrid_model", "net"]:
        if name in local_vars:
            cand = local_vars[name]
            if hasattr(cand, "get_c2_affine_state"):
                model = cand
                break

    if model is None:
        return row

    try:
        row.update(model.get_c2_affine_state())
    except Exception:
        pass
    return row
# =================== END TRUE C2 AFFINE PATCH ===================



# ===================== TRUE C3 BAND CALIBRATION PATCH =====================
import torch as _torch_C3
import torch.nn as _nn_C3

class _C3BandCalHybridWrapper(_nn_C3.Module):
    """
    Calibration-only band-specific output correction.

    Formula:
      y_cal = y + sum_b (gain[c,b] - 1) * bandpass_b(y)

    Backbone tetap frozen. Parameter trainable:
      - alpha dari base hybrid, jika learned_alpha aktif
      - c3_band_gain_raw
    """
    def __init__(
        self,
        base_model,
        n_ch=4,
        sfreq=250,
        bands=None,
        gain_init=None,
        gain_min=0.2,
        gain_max=5.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_ch = int(n_ch)
        self.sfreq = float(sfreq)
        self.bands = bands or [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
        self.band_names = ["delta", "theta", "alpha", "beta"]
        self.n_bands = len(self.bands)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

        if gain_init is None:
            gain_init = _torch_C3.ones(self.n_ch, self.n_bands)
        else:
            gain_init = _torch_C3.tensor(gain_init, dtype=_torch_C3.float32)
            if gain_init.numel() == self.n_bands:
                gain_init = gain_init.reshape(1, self.n_bands).repeat(self.n_ch, 1)
            elif gain_init.numel() == self.n_ch * self.n_bands:
                gain_init = gain_init.reshape(self.n_ch, self.n_bands)
            else:
                raise ValueError(
                    f"C3_BAND_GAIN_INIT harus 4 angka global-band atau 16 angka channel-band, dapat {gain_init.numel()}"
                )

        eps = 1e-6
        z = (gain_init - self.gain_min) / max(self.gain_max - self.gain_min, eps)
        z = z.clamp(eps, 1.0 - eps)
        raw = _torch_C3.log(z / (1.0 - z))
        self.c3_band_gain_raw = _nn_C3.Parameter(raw.clone())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            base = modules.get("base_model", None)
            if base is not None:
                return getattr(base, name)
            raise

    @property
    def c3_band_gain(self):
        return self.gain_min + (self.gain_max - self.gain_min) * _torch_C3.sigmoid(self.c3_band_gain_raw)

    def _band_component(self, y, f_lo, f_hi):
        # y: [B, C, T]
        T = y.shape[-1]
        Y = _torch_C3.fft.rfft(y.float(), dim=-1)
        freqs = _torch_C3.fft.rfftfreq(T, d=1.0 / self.sfreq).to(device=y.device)

        mask = ((freqs >= float(f_lo)) & (freqs < float(f_hi))).to(dtype=Y.dtype, device=y.device)
        while mask.ndim < Y.ndim:
            mask = mask.unsqueeze(0)

        comp = _torch_C3.fft.irfft(Y * mask, n=T, dim=-1)
        return comp.to(dtype=y.dtype)

    def _apply_bandcal_tensor(self, y):
        if not _torch_C3.is_tensor(y):
            return y
        if y.ndim < 3:
            return y
        if y.shape[1] != self.n_ch:
            return y

        y0 = y
        y_cal = y.float()

        gains = self.c3_band_gain.to(device=y.device, dtype=y.float().dtype)

        for bi, (f_lo, f_hi) in enumerate(self.bands):
            comp = self._band_component(y0, f_lo, f_hi).float()
            g = gains[:, bi].reshape(1, self.n_ch, 1)
            y_cal = y_cal + (g - 1.0) * comp

        return y_cal.to(dtype=y.dtype)

    def _apply_to_output(self, out):
        if _torch_C3.is_tensor(out):
            return self._apply_bandcal_tensor(out)

        if isinstance(out, tuple):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C3.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_bandcal_tensor(item)
                    break
            return tuple(out)

        if isinstance(out, list):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C3.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_bandcal_tensor(item)
                    break
            return out

        if isinstance(out, dict):
            out = dict(out)
            for key in ["pred", "y_pred", "output", "out"]:
                if key in out and _torch_C3.is_tensor(out[key]) and out[key].ndim >= 3 and out[key].shape[1] == self.n_ch:
                    out[key] = self._apply_bandcal_tensor(out[key])
                    break
            return out

        return out

    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        return self._apply_to_output(out)

    def get_c3_bandcal_state(self):
        g = self.c3_band_gain.detach().cpu().float().numpy()
        row = {}

        ch_names = ["f3", "f4", "p3", "p4"]
        for ci, ch in enumerate(ch_names):
            for bi, band in enumerate(self.band_names):
                row[f"c3_gain_{ch}_{band}"] = float(g[ci, bi])

        # alias lama agar target index 2/3 tetap mudah dibaca
        row["c3_gain_t7_delta"] = row["c3_gain_p3_delta"]
        row["c3_gain_t7_theta"] = row["c3_gain_p3_theta"]
        row["c3_gain_t7_alpha"] = row["c3_gain_p3_alpha"]
        row["c3_gain_t7_beta"]  = row["c3_gain_p3_beta"]

        row["c3_gain_t8_delta"] = row["c3_gain_p4_delta"]
        row["c3_gain_t8_theta"] = row["c3_gain_p4_theta"]
        row["c3_gain_t8_alpha"] = row["c3_gain_p4_alpha"]
        row["c3_gain_t8_beta"]  = row["c3_gain_p4_beta"]

        return row

def _parse_c3_band_gain_env(default=(1.25, 1.25, 1.60, 3.00)):
    import os as _os_C3
    raw = _os_C3.environ.get("C3_BAND_GAIN_INIT", "").strip()
    if not raw:
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) not in [4, 16]:
        raise ValueError(f"C3_BAND_GAIN_INIT harus 4 atau 16 angka, dapat {len(vals)}: {raw}")
    return vals

def _maybe_wrap_c3_bandcal(model, target_ch, device):
    import os as _os_C3

    use_c3 = _os_C3.environ.get("CALIBRATE_BAND_GAIN", "0") == "1"

    if not use_c3:
        return model

    if isinstance(model, _C3BandCalHybridWrapper):
        return model

    n_ch = len(target_ch)
    sfreq = float(_os_C3.environ.get("SFREQ", "250"))

    gain_min = float(_os_C3.environ.get("C3_BAND_GAIN_MIN", "0.2"))
    gain_max = float(_os_C3.environ.get("C3_BAND_GAIN_MAX", "5.0"))
    gain_init = _parse_c3_band_gain_env()

    wrapped = _C3BandCalHybridWrapper(
        model,
        n_ch=n_ch,
        sfreq=sfreq,
        gain_init=gain_init,
        gain_min=gain_min,
        gain_max=gain_max,
    ).to(device)

    print("[TRUE_C3_BANDCAL] Aktif: y_cal = y + sum_b (gain[c,b]-1)*bandpass_b(y)")
    print("[TRUE_C3_BANDCAL] bands: delta(1-4), theta(4-8), alpha(8-13), beta(13-30)")
    print("[TRUE_C3_BANDCAL] gain_init:", gain_init)
    print("[TRUE_C3_BANDCAL] gain_min/max:", gain_min, gain_max)

    return wrapped

def _c3_update_row_with_bandcal(row, local_vars):
    model = None
    for name in ["model", "hybrid_model", "net"]:
        if name in local_vars:
            cand = local_vars[name]
            if hasattr(cand, "get_c3_bandcal_state"):
                model = cand
                break

    if model is None:
        return row

    try:
        row.update(model.get_c3_bandcal_state())
    except Exception:
        pass
    return row
# =================== END TRUE C3 BAND CALIBRATION PATCH ===================



# ===================== TRUE C4 BAND-SPATIAL MIXING PATCH =====================
import torch as _torch_C4
import torch.nn as _nn_C4

class _C4BandSpatialMixWrapper(_nn_C4.Module):
    """
    Calibration-only band-specific spatial mixing.

    Formula:
      y_cal = y + sum_b ( W_b @ bandpass_b(y) - bandpass_b(y) )

    Jika W_b = I, output tidak berubah.
    Jika W_b diagonal > 1, mirip C3 band gain.
    Jika W_b punya off-diagonal, C4 dapat mengoreksi distribusi spasial antar-channel.

    Backbone CNN/MASER tetap frozen.
    Parameter trainable:
      - alpha dari base hybrid, jika learned_alpha aktif
      - c4_mix_raw: matriks W_b untuk tiap band
    """
    def __init__(
        self,
        base_model,
        n_ch=4,
        sfreq=250,
        bands=None,
        diag_init=None,
        mix_min=-2.0,
        mix_max=5.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_ch = int(n_ch)
        self.sfreq = float(sfreq)
        self.bands = bands or [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
        self.band_names = ["delta", "theta", "alpha", "beta"]
        self.n_bands = len(self.bands)
        self.mix_min = float(mix_min)
        self.mix_max = float(mix_max)

        if diag_init is None:
            diag_init = [1.25, 1.25, 1.60, 3.00]

        vals = _torch_C4.tensor(diag_init, dtype=_torch_C4.float32)

        W_init = _torch_C4.zeros(self.n_bands, self.n_ch, self.n_ch, dtype=_torch_C4.float32)

        # 4 angka: diagonal global per band
        if vals.numel() == self.n_bands:
            for bi in range(self.n_bands):
                W_init[bi] = _torch_C4.eye(self.n_ch) * vals[bi]

        # 16 angka: diagonal per channel-band, urutan channel x band
        elif vals.numel() == self.n_ch * self.n_bands:
            vals2 = vals.reshape(self.n_ch, self.n_bands)
            for bi in range(self.n_bands):
                for ci in range(self.n_ch):
                    W_init[bi, ci, ci] = vals2[ci, bi]

        # 64 angka: full matrix, urutan band x out x in
        elif vals.numel() == self.n_bands * self.n_ch * self.n_ch:
            W_init = vals.reshape(self.n_bands, self.n_ch, self.n_ch)

        else:
            raise ValueError(
                f"C4_MIX_INIT harus 4, 16, atau 64 angka. Dapat {vals.numel()}"
            )

        eps = 1e-6
        z = (W_init - self.mix_min) / max(self.mix_max - self.mix_min, eps)
        z = z.clamp(eps, 1.0 - eps)
        raw = _torch_C4.log(z / (1.0 - z))
        self.c4_mix_raw = _nn_C4.Parameter(raw.clone())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            base = modules.get("base_model", None)
            if base is not None:
                return getattr(base, name)
            raise

    @property
    def c4_mix_matrix(self):
        return self.mix_min + (self.mix_max - self.mix_min) * _torch_C4.sigmoid(self.c4_mix_raw)

    def _band_component(self, y, f_lo, f_hi):
        # y: [B, C, T]
        T = y.shape[-1]
        Y = _torch_C4.fft.rfft(y.float(), dim=-1)
        freqs = _torch_C4.fft.rfftfreq(T, d=1.0 / self.sfreq).to(device=y.device)

        mask = ((freqs >= float(f_lo)) & (freqs < float(f_hi))).to(dtype=Y.dtype, device=y.device)
        while mask.ndim < Y.ndim:
            mask = mask.unsqueeze(0)

        comp = _torch_C4.fft.irfft(Y * mask, n=T, dim=-1)
        return comp.to(dtype=y.dtype)

    def _apply_bandmix_tensor(self, y):
        if not _torch_C4.is_tensor(y):
            return y
        if y.ndim < 3:
            return y
        if y.shape[1] != self.n_ch:
            return y

        y0 = y
        y_cal = y.float()
        W = self.c4_mix_matrix.to(device=y.device, dtype=y.float().dtype)

        for bi, (f_lo, f_hi) in enumerate(self.bands):
            comp = self._band_component(y0, f_lo, f_hi).float()  # [B,C,T]
            Wb = W[bi]  # [C_out, C_in]

            mixed = _torch_C4.einsum("oi,bit->bot", Wb, comp)

            # jika Wb = I, mixed - comp = 0
            y_cal = y_cal + (mixed - comp)

        return y_cal.to(dtype=y.dtype)

    def _apply_to_output(self, out):
        if _torch_C4.is_tensor(out):
            return self._apply_bandmix_tensor(out)

        if isinstance(out, tuple):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C4.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_bandmix_tensor(item)
                    break
            return tuple(out)

        if isinstance(out, list):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C4.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_bandmix_tensor(item)
                    break
            return out

        if isinstance(out, dict):
            out = dict(out)
            for key in ["pred", "y_pred", "output", "out"]:
                if key in out and _torch_C4.is_tensor(out[key]) and out[key].ndim >= 3 and out[key].shape[1] == self.n_ch:
                    out[key] = self._apply_bandmix_tensor(out[key])
                    break
            return out

        return out

    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        return self._apply_to_output(out)

    def get_c4_bandmix_state(self):
        W = self.c4_mix_matrix.detach().cpu().float().numpy()
        row = {}
        ch_names = ["f3", "f4", "p3", "p4"]

        for bi, band in enumerate(self.band_names):
            for oi, out_ch in enumerate(ch_names):
                for ii, in_ch in enumerate(ch_names):
                    row[f"c4_gain_{out_ch}_from_{in_ch}_{band}"] = float(W[bi, oi, ii])

        # ringkasan off-diagonal per band
        for bi, band in enumerate(self.band_names):
            mat = W[bi]
            diag_mean = float(mat.diagonal().mean())
            off = mat.copy()
            for i in range(self.n_ch):
                off[i, i] = 0.0
            row[f"c4_diag_mean_{band}"] = diag_mean
            row[f"c4_offdiag_abs_mean_{band}"] = float(abs(off).mean())

        return row

def _parse_c4_mix_init_env(default=(1.25, 1.25, 1.60, 3.00)):
    import os as _os_C4
    raw = _os_C4.environ.get("C4_MIX_INIT", "").strip()
    if not raw:
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) not in [4, 16, 64]:
        raise ValueError(f"C4_MIX_INIT harus 4, 16, atau 64 angka. Dapat {len(vals)}: {raw}")
    return vals

def _maybe_wrap_c4_bandmix(model, target_ch, device):
    import os as _os_C4

    use_c4 = _os_C4.environ.get("CALIBRATE_BAND_MIX", "0") == "1"

    if not use_c4:
        return model

    if isinstance(model, _C4BandSpatialMixWrapper):
        return model

    n_ch = len(target_ch)
    sfreq = float(_os_C4.environ.get("SFREQ", "250"))
    mix_min = float(_os_C4.environ.get("C4_MIX_MIN", "-2.0"))
    mix_max = float(_os_C4.environ.get("C4_MIX_MAX", "5.0"))
    diag_init = _parse_c4_mix_init_env()

    wrapped = _C4BandSpatialMixWrapper(
        model,
        n_ch=n_ch,
        sfreq=sfreq,
        diag_init=diag_init,
        mix_min=mix_min,
        mix_max=mix_max,
    ).to(device)

    print("[TRUE_C4_BANDMIX] Aktif: y_cal = y + sum_b(W_b @ bandpass_b(y) - bandpass_b(y))")
    print("[TRUE_C4_BANDMIX] bands: delta(1-4), theta(4-8), alpha(8-13), beta(13-30)")
    print("[TRUE_C4_BANDMIX] init diagonal/global:", diag_init)
    print("[TRUE_C4_BANDMIX] mix_min/max:", mix_min, mix_max)
    print("[TRUE_C4_BANDMIX] W_b shape: [band, out_channel, in_channel]")

    return wrapped

def _c4_update_row_with_bandmix(row, local_vars):
    model = None
    for name in ["model", "hybrid_model", "net"]:
        if name in local_vars:
            cand = local_vars[name]
            if hasattr(cand, "get_c4_bandmix_state"):
                model = cand
                break

    if model is None:
        return row

    try:
        row.update(model.get_c4_bandmix_state())
    except Exception:
        pass
    return row
# =================== END TRUE C4 BAND-SPATIAL MIXING PATCH ===================



# ===================== TRUE C5 TEMPORAL RESIDUAL PATCH =====================
import torch as _torch_C5
import torch.nn as _nn_C5

class _C5TemporalResidualWrapper(_nn_C5.Module):
    """
    Calibration-only temporal residual refinement.

    Formula:
      y_c5 = y_base + scale * R_phi(y_base)

    Tujuan:
      - mengurangi oversmoothing
      - menaikkan dynamic amplitude / local waveform detail
      - tetap menjaga C4 band-spatial calibration

    Backbone CNN/MASER tetap frozen.
    Parameter trainable:
      - alpha dari base hybrid
      - C4 bandmix matrix jika aktif
      - C5 temporal residual conv kecil
    """
    def __init__(
        self,
        base_model,
        n_ch=4,
        hidden=16,
        kernel=31,
        dropout=0.05,
        scale_init=0.10,
        scale_max=0.35,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_ch = int(n_ch)
        self.hidden = int(hidden)
        self.kernel = int(kernel)
        if self.kernel % 2 == 0:
            self.kernel += 1

        self.scale_max = float(scale_max)

        pad = self.kernel // 2

        self.res_net = _nn_C5.Sequential(
            _nn_C5.Conv1d(self.n_ch, self.hidden, kernel_size=self.kernel, padding=pad),
            _nn_C5.GELU(),
            _nn_C5.Dropout(float(dropout)),
            _nn_C5.Conv1d(self.hidden, self.hidden, kernel_size=self.kernel, padding=pad),
            _nn_C5.GELU(),
            _nn_C5.Dropout(float(dropout)),
            _nn_C5.Conv1d(self.hidden, self.n_ch, kernel_size=1),
        )

        # Zero-init final layer agar awal C5 sama persis dengan C4c.
        last = self.res_net[-1]
        _nn_C5.init.zeros_(last.weight)
        _nn_C5.init.zeros_(last.bias)

        # scale dibatasi 0..scale_max
        eps = 1e-6
        z = max(min(float(scale_init) / max(self.scale_max, eps), 1.0 - eps), eps)
        raw = _torch_C5.log(_torch_C5.tensor(z / (1.0 - z), dtype=_torch_C5.float32))
        self.c5_scale_raw = _nn_C5.Parameter(raw.clone())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            base = modules.get("base_model", None)
            if base is not None:
                return getattr(base, name)
            raise

    @property
    def c5_scale(self):
        return self.scale_max * _torch_C5.sigmoid(self.c5_scale_raw)

    def _apply_tempres_tensor(self, y):
        if not _torch_C5.is_tensor(y):
            return y
        if y.ndim < 3:
            return y
        if y.shape[1] != self.n_ch:
            return y

        dtype = y.dtype
        yy = y.float()

        resid = self.res_net(yy)
        out = yy + self.c5_scale * resid

        return out.to(dtype=dtype)

    def _apply_to_output(self, out):
        if _torch_C5.is_tensor(out):
            return self._apply_tempres_tensor(out)

        if isinstance(out, tuple):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C5.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_tempres_tensor(item)
                    break
            return tuple(out)

        if isinstance(out, list):
            out = list(out)
            for i, item in enumerate(out):
                if _torch_C5.is_tensor(item) and item.ndim >= 3 and item.shape[1] == self.n_ch:
                    out[i] = self._apply_tempres_tensor(item)
                    break
            return out

        if isinstance(out, dict):
            out = dict(out)
            for key in ["pred", "y_pred", "output", "out"]:
                if key in out and _torch_C5.is_tensor(out[key]) and out[key].ndim >= 3 and out[key].shape[1] == self.n_ch:
                    out[key] = self._apply_tempres_tensor(out[key])
                    break
            return out

        return out

    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        return self._apply_to_output(out)

    def get_c5_tempres_state(self):
        row = {}
        row["c5_scale"] = float(self.c5_scale.detach().cpu())
        row["c5_kernel"] = int(self.kernel)
        row["c5_hidden"] = int(self.hidden)

        with _torch_C5.no_grad():
            total_norm = 0.0
            n = 0
            for param in self.res_net.parameters():
                total_norm += float(param.detach().float().norm().cpu())
                n += 1
            row["c5_resnet_param_norm_sum"] = total_norm
            row["c5_resnet_n_params_tensors"] = n

        return row

def _maybe_wrap_c5_tempres(model, target_ch, device):
    import os as _os_C5

    use_c5 = _os_C5.environ.get("CALIBRATE_TEMP_RESIDUAL", "0") == "1"

    if not use_c5:
        return model

    if isinstance(model, _C5TemporalResidualWrapper):
        return model

    n_ch = len(target_ch)
    hidden = int(_os_C5.environ.get("C5_HIDDEN", "16"))
    kernel = int(_os_C5.environ.get("C5_KERNEL", "31"))
    dropout = float(_os_C5.environ.get("C5_DROPOUT", "0.05"))
    scale_init = float(_os_C5.environ.get("C5_SCALE_INIT", "0.10"))
    scale_max = float(_os_C5.environ.get("C5_SCALE_MAX", "0.35"))

    wrapped = _C5TemporalResidualWrapper(
        model,
        n_ch=n_ch,
        hidden=hidden,
        kernel=kernel,
        dropout=dropout,
        scale_init=scale_init,
        scale_max=scale_max,
    ).to(device)

    print("[TRUE_C5_TEMPRES] Aktif: y_c5 = y_base + scale * R_phi(y_base)")
    print("[TRUE_C5_TEMPRES] hidden:", hidden, "kernel:", kernel, "dropout:", dropout)
    print("[TRUE_C5_TEMPRES] scale_init/max:", scale_init, scale_max)

    return wrapped

def _c5_update_row_with_tempres(row, local_vars):
    model = None
    for name in ["model", "hybrid_model", "net"]:
        if name in local_vars:
            cand = local_vars[name]
            if hasattr(cand, "get_c5_tempres_state"):
                model = cand
                break

    if model is None:
        return row

    try:
        row.update(model.get_c5_tempres_state())
    except Exception:
        pass
    return row
# =================== END TRUE C5 TEMPORAL RESIDUAL PATCH ===================



# ===================== TRUE C6 PARTIAL HEAD UNFREEZE PATCH =====================
import os as _os_C6
import torch as _torch_C6

def _maybe_load_full_calibration_ckpt(model, device):
    """
    Load full calibration checkpoint after C4/C5 wrappers are constructed.
    Useful for starting C6 from C5b best_val_loss/best_corr.
    """
    path = _os_C6.environ.get("INIT_FULL_CKPT", "").strip()
    if not path:
        return model

    if not _os_C6.path.exists(path):
        print("[C6_FULL_CKPT] WARNING: checkpoint tidak ditemukan:", path)
        return model

    try:
        ckpt = _torch_C6.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = _torch_C6.load(path, map_location=device)

    if isinstance(ckpt, dict):
        # Prioritas khusus untuk checkpoint hybrid script ini
        if "hybrid_model_state_dict" in ckpt:
            sd = ckpt["hybrid_model_state_dict"]
            print("[C6_FULL_CKPT] memakai key: hybrid_model_state_dict")
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            print("[C6_FULL_CKPT] memakai key: model_state_dict")
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print("[C6_FULL_CKPT] memakai key: state_dict")
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
            print("[C6_FULL_CKPT] memakai key: model")
        else:
            sd = ckpt
            print("[C6_FULL_CKPT] WARNING: tidak ada key state dict eksplisit, pakai ckpt langsung.")
    else:
        print("[C6_FULL_CKPT] WARNING: format checkpoint tidak dikenali, skip.")
        return model

    clean = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        clean[kk] = v

    incompatible = model.load_state_dict(clean, strict=False)

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    print("[C6_FULL_CKPT] Loaded:", path)
    print("[C6_FULL_CKPT] missing keys:", len(missing), "| unexpected keys:", len(unexpected))
    if missing[:10]:
        print("[C6_FULL_CKPT] missing contoh:", missing[:10])
    if unexpected[:10]:
        print("[C6_FULL_CKPT] unexpected contoh:", unexpected[:10])

    return model


def _maybe_apply_c6_partial_unfreeze(model):
    """
    Partial unfreeze untuk output/head/projection layers saja.
    Tidak membuka seluruh backbone.
    """
    use_c6 = _os_C6.environ.get("C6_PARTIAL_UNFREEZE", "0") == "1"
    if not use_c6:
        return model

    raw_kw = _os_C6.environ.get(
        "C6_UNFREEZE_KEYWORDS",
        "frontal_head,temporal_head,frontal_skip,temporal_skip,output_head,out_head,head,out_proj,output_proj,proj_out,to_out,readout,final,decoder"
    )

    keywords = [x.strip().lower() for x in raw_kw.split(",") if x.strip()]

    matched = []
    total_params = 0

    for name, param in model.named_parameters():
        lname = name.lower()
        if any(kw in lname for kw in keywords):
            param.requires_grad = True
            matched.append((name, param.numel()))
            total_params += param.numel()

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_total = sum(p.numel() for p in model.parameters())

    print("[TRUE_C6_PARTIAL_UNFREEZE] Aktif: output/head partial unfreeze")
    print("[TRUE_C6_PARTIAL_UNFREEZE] keywords:", keywords)
    print("[TRUE_C6_PARTIAL_UNFREEZE] matched tensors:", len(matched))
    print("[TRUE_C6_PARTIAL_UNFREEZE] matched param count:", total_params)
    print("[TRUE_C6_PARTIAL_UNFREEZE] total trainable params:", trainable_total, "/", all_total)

    print("[TRUE_C6_PARTIAL_UNFREEZE] first matched names:")
    for name, n in matched[:80]:
        print("  -", name, "|", n)

    if len(matched) == 0:
        print("[TRUE_C6_PARTIAL_UNFREEZE] WARNING: tidak ada parameter head yang match. Perlu cek nama layer.")

    return model


def _c6_update_row(row, local_vars):
    try:
        model = local_vars.get("model", None)
        if model is None:
            return row

        row["c6_partial_unfreeze"] = int(_os_C6.environ.get("C6_PARTIAL_UNFREEZE", "0") == "1")
        row["c6_trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        row["c6_total_params"] = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass
    return row
# =================== END TRUE C6 PARTIAL HEAD UNFREEZE PATCH ===================



# ===================== TRUE C17 OUTPUT AMPLITUDE SCALE PATCH =====================
import os as _os_C17
import math as _math_C17
import torch as _torch_C17
import torch.nn as _nn_C17

class _C17OutputScaleWrapper(_nn_C17.Module):
    """
    Learnable per-channel output gain:
        y_out[c] = scale[c] * y_base[c]

    Tujuan: memperbaiki under-amplitude / terlalu smooth pada waveform prediksi.
    """
    def __init__(self, base_model, n_ch, init_scales=None, min_scale=0.20, max_scale=3.00):
        super().__init__()
        self.base_model = base_model
        self.n_ch = int(n_ch)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        if init_scales is None:
            init_scales = [1.0] * self.n_ch
        if len(init_scales) != self.n_ch:
            init_scales = [float(init_scales[0])] * self.n_ch

        init_scales = [min(max(float(x), self.min_scale + 1e-4), self.max_scale - 1e-4) for x in init_scales]

        # sigmoid parameterization: scale = min + (max-min)*sigmoid(raw)
        raws = []
        for sc in init_scales:
            z = (sc - self.min_scale) / (self.max_scale - self.min_scale)
            z = min(max(z, 1e-5), 1 - 1e-5)
            raws.append(_math_C17.log(z / (1 - z)))

        self.c17_scale_raw = _nn_C17.Parameter(_torch_C17.tensor(raws, dtype=_torch_C17.float32))

    def _find_base_attr(self, attr):
        """
        Cari atribut pada base_model secara rekursif.
        Diperlukan karena wrapper C7B berada di luar C5/C4/hybrid model.
        """
        m = self.base_model
        for _ in range(12):
            try:
                return getattr(m, attr)
            except AttributeError:
                if hasattr(m, "base_model"):
                    m = m.base_model
                else:
                    break
        raise AttributeError(f"{self.__class__.__name__} and nested base_model have no attribute {attr}")

    @property
    def freeze_backbones(self):
        try:
            return self._find_base_attr("freeze_backbones")
        except AttributeError:
            return False

    @freeze_backbones.setter
    def freeze_backbones(self, value):
        m = self.base_model
        for _ in range(12):
            if hasattr(m, "freeze_backbones"):
                m.freeze_backbones = value
                return
            if hasattr(m, "base_model"):
                m = m.base_model
            else:
                return

    @property
    def frontal_model(self):
        return self._find_base_attr("frontal_model")

    @property
    def temporal_model(self):
        return self._find_base_attr("temporal_model")

    @property
    def maser_model(self):
        return self._find_base_attr("maser_model")

    @property
    def alpha_logits(self):
        return self._find_base_attr("alpha_logits")

    @property
    def alpha_init(self):
        return self._find_base_attr("alpha_init")

    def get_alpha(self):
        return self._find_base_attr("get_alpha")()

    def get_scale(self):
        return self.min_scale + (self.max_scale - self.min_scale) * _torch_C17.sigmoid(self.c17_scale_raw)

    def forward(self, x, *args, **kwargs):
        out = self.base_model(x, *args, **kwargs)

        # Normal forward: base_model return tensor [B,C,T]
        if not isinstance(out, tuple):
            y = out
            sc = self.get_scale().to(device=y.device, dtype=y.dtype).view(1, -1, 1)
            return y * sc

        # Branch forward: base_model return tuple, biasanya:
        # pred, y_cnn, y_maser, alpha = model(x, return_branches=True)
        y = out[0]
        sc = self.get_scale().to(device=y.device, dtype=y.dtype).view(1, -1, 1)
        y_scaled = y * sc

        return (y_scaled,) + tuple(out[1:])


def _maybe_wrap_c17_outscale(model, n_ch, device):
    use = _os_C17.environ.get("C17_OUTPUT_SCALE", "0") == "1"
    if not use:
        return model

    # TARGET_CH kadang berupa list nama channel, misalnya ["F3","F4","P3","P4"].
    # Jadi jumlah channel harus dihitung aman.
    try:
        if isinstance(n_ch, (list, tuple)):
            nch = len(n_ch)
        else:
            nch = int(n_ch)
    except Exception:
        nch = len(n_ch)

    raw = _os_C17.environ.get("C17_SCALE_INIT", "1.0")
    init_scales = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(init_scales) == 1:
        init_scales = init_scales * nch

    min_scale = float(_os_C17.environ.get("C17_SCALE_MIN", "0.20"))
    max_scale = float(_os_C17.environ.get("C17_SCALE_MAX", "3.00"))

    model = _C17OutputScaleWrapper(
        model,
        n_ch=nch,
        init_scales=init_scales,
        min_scale=min_scale,
        max_scale=max_scale,
    ).to(device)

    print("[TRUE_C7B_OUTSCALE] Aktif: y_out = scale_channel * y_base")
    print("[TRUE_C7B_OUTSCALE] init_scales:", init_scales)
    print("[TRUE_C7B_OUTSCALE] min/max:", min_scale, max_scale)
    print("[TRUE_C7B_OUTSCALE] current scale:", model.get_scale().detach().cpu().numpy().tolist())

    return model


def _c17_update_row(row, local_vars):
    try:
        model = local_vars.get("model", None)
        if model is None:
            return row

        m = model
        found = None
        # model bisa wrapper C17 paling luar, atau nested.
        for _ in range(8):
            if hasattr(m, "get_scale") and hasattr(m, "c17_scale_raw"):
                found = m
                break
            if hasattr(m, "base_model"):
                m = m.base_model
            else:
                break

        if found is not None:
            scales = found.get_scale().detach().cpu().numpy().tolist()
            for i, v in enumerate(scales):
                row[f"c7b_scale_ch{i}"] = float(v)
            row["c7b_output_scale"] = 1
        else:
            row["c7b_output_scale"] = 0
    except Exception:
        pass
    return row
# =================== END TRUE C17 OUTPUT AMPLITUDE SCALE PATCH ===================



# ===================== TRUE C8 HF RESIDUAL BRANCH PATCH =====================
import os as _os_C8
import math as _math_C8
import torch as _torch_C8
import torch.nn as _nn_C8

def _c8_fft_bandpass(x, fs=250.0, f_low=13.0, f_high=40.0):
    """
    Differentiable FFT bandpass.
    x: [B, C, T]
    """
    if x is None or x.ndim != 3:
        return x

    x = x.float()
    n = x.shape[-1]
    X = _torch_C8.fft.rfft(x, dim=-1)
    freqs = _torch_C8.fft.rfftfreq(n, d=1.0 / float(fs)).to(device=x.device)

    mask = ((freqs >= float(f_low)) & (freqs <= float(f_high))).to(dtype=X.dtype, device=x.device)
    while mask.ndim < X.ndim:
        mask = mask.view(*([1] * (X.ndim - 1)), -1)

    Y = X * mask
    y = _torch_C8.fft.irfft(Y, n=n, dim=-1)
    return y


class _C8HFResidualWrapper(_nn_C8.Module):
    """
    C8 HF residual branch:
        y_final = y_base + scale * Bandpass_13_40Hz(R([x_input, y_base]))

    Tujuan:
    - menjaga base model C6/C7B tetap stabil
    - menambahkan detail osilasi cepat
    - mengurangi output terlalu smooth
    """
    def __init__(
        self,
        base_model,
        input_ch=4,
        target_ch=4,
        hidden=32,
        kernel=15,
        dropout=0.0,
        scale_init=0.05,
        scale_max=0.40,
        fs=250.0,
        f_low=13.0,
        f_high=40.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.input_ch = int(input_ch)
        self.target_ch = int(target_ch)
        self.hidden = int(hidden)
        self.kernel = int(kernel)
        self.dropout = float(dropout)
        self.scale_max = float(scale_max)
        self.fs = float(fs)
        self.f_low = float(f_low)
        self.f_high = float(f_high)

        k = self.kernel
        h = self.hidden
        in_ch = self.input_ch + self.target_ch

        self.c8_branch = _nn_C8.Sequential(
            _nn_C8.Conv1d(in_ch, h, kernel_size=k, padding=k//2),
            _nn_C8.GELU(),
            _nn_C8.Dropout(self.dropout),
            _nn_C8.Conv1d(h, h, kernel_size=k, padding=(k//2)*2, dilation=2),
            _nn_C8.GELU(),
            _nn_C8.Dropout(self.dropout),
            _nn_C8.Conv1d(h, h, kernel_size=k, padding=(k//2)*4, dilation=4),
            _nn_C8.GELU(),
            _nn_C8.Conv1d(h, self.target_ch, kernel_size=1),
        )

        # zero init final conv supaya awalnya identik dengan base model
        final = self.c8_branch[-1]
        _nn_C8.init.zeros_(final.weight)
        if final.bias is not None:
            _nn_C8.init.zeros_(final.bias)

        # scale = scale_max * sigmoid(raw)
        init = max(1e-5, min(float(scale_init) / float(scale_max), 1 - 1e-5))
        raw = _math_C8.log(init / (1 - init))
        self.c8_scale_raw = _nn_C8.Parameter(_torch_C8.tensor(float(raw), dtype=_torch_C8.float32))

    def _find_base_attr(self, attr):
        m = self.base_model
        for _ in range(16):
            try:
                return getattr(m, attr)
            except AttributeError:
                if hasattr(m, "base_model"):
                    m = m.base_model
                else:
                    break
        raise AttributeError(f"{self.__class__.__name__} and nested base_model have no attribute {attr}")

    @property
    def freeze_backbones(self):
        try:
            return self._find_base_attr("freeze_backbones")
        except AttributeError:
            return False

    @freeze_backbones.setter
    def freeze_backbones(self, value):
        m = self.base_model
        for _ in range(16):
            if hasattr(m, "freeze_backbones"):
                m.freeze_backbones = value
                return
            if hasattr(m, "base_model"):
                m = m.base_model
            else:
                return

    @property
    def frontal_model(self):
        return self._find_base_attr("frontal_model")

    @property
    def temporal_model(self):
        return self._find_base_attr("temporal_model")

    @property
    def maser_model(self):
        return self._find_base_attr("maser_model")

    @property
    def alpha_logits(self):
        return self._find_base_attr("alpha_logits")

    @property
    def alpha_init(self):
        return self._find_base_attr("alpha_init")

    def get_alpha(self):
        return self._find_base_attr("get_alpha")()

    def get_c8_scale(self):
        return self.scale_max * _torch_C8.sigmoid(self.c8_scale_raw)

    def _apply_c8(self, x, y_base):
        # x: [B,4,T], y_base: [B,4,T]
        xin = x.float()
        yb = y_base.float()

        if xin.shape[-1] != yb.shape[-1]:
            T = min(xin.shape[-1], yb.shape[-1])
            xin = xin[..., :T]
            yb = yb[..., :T]

        z = _torch_C8.cat([xin, yb], dim=1)
        residual = self.c8_branch(z)
        residual_hf = _c8_fft_bandpass(
            residual,
            fs=self.fs,
            f_low=self.f_low,
            f_high=self.f_high,
        )

        scale = self.get_c8_scale().to(device=yb.device, dtype=yb.dtype)
        return yb + scale * residual_hf.to(dtype=yb.dtype)

    def forward(self, x, *args, **kwargs):
        out = self.base_model(x, *args, **kwargs)

        if not isinstance(out, tuple):
            y_base = out
            y_final = self._apply_c8(x, y_base)
            return y_final

        y_base = out[0]
        y_final = self._apply_c8(x, y_base)
        return (y_final,) + tuple(out[1:])


def _maybe_wrap_c8_hf_residual(model, input_ch, target_ch, device):
    if _os_C8.environ.get("C8_HF_RESIDUAL", "0") != "1":
        return model

    hidden = int(_os_C8.environ.get("C8_HIDDEN", "32"))
    kernel = int(_os_C8.environ.get("C8_KERNEL", "15"))
    dropout = float(_os_C8.environ.get("C8_DROPOUT", "0.0"))
    scale_init = float(_os_C8.environ.get("C8_SCALE_INIT", "0.05"))
    scale_max = float(_os_C8.environ.get("C8_SCALE_MAX", "0.40"))
    fs = float(_os_C8.environ.get("C8_FS", _os_C8.environ.get("SFREQ", "250")))
    f_low = float(_os_C8.environ.get("C8_HF_LOW", "13"))
    f_high = float(_os_C8.environ.get("C8_HF_HIGH", "40"))

    try:
        inch = len(input_ch) if isinstance(input_ch, (list, tuple)) else int(input_ch)
    except Exception:
        inch = 4

    try:
        tch = len(target_ch) if isinstance(target_ch, (list, tuple)) else int(target_ch)
    except Exception:
        tch = 4

    model = _C8HFResidualWrapper(
        model,
        input_ch=inch,
        target_ch=tch,
        hidden=hidden,
        kernel=kernel,
        dropout=dropout,
        scale_init=scale_init,
        scale_max=scale_max,
        fs=fs,
        f_low=f_low,
        f_high=f_high,
    ).to(device)

    n_train = sum(p.numel() for p in model.c8_branch.parameters() if p.requires_grad) + model.c8_scale_raw.numel()

    print("[TRUE_C8_HF_RESIDUAL] Aktif: y_final = y_base + scale * HF_residual([x, y_base])")
    print("[TRUE_C8_HF_RESIDUAL] input_ch/target_ch:", inch, tch)
    print("[TRUE_C8_HF_RESIDUAL] hidden/kernel/dropout:", hidden, kernel, dropout)
    print("[TRUE_C8_HF_RESIDUAL] band:", f_low, "-", f_high, "Hz")
    print("[TRUE_C8_HF_RESIDUAL] scale_init/max:", scale_init, scale_max)
    print("[TRUE_C8_HF_RESIDUAL] current scale:", float(model.get_c8_scale().detach().cpu()))
    print("[TRUE_C8_HF_RESIDUAL] trainable C8 params:", n_train)

    return model


def _c8_update_row(row, local_vars):
    try:
        model = local_vars.get("model", None)
        if model is None:
            return row

        m = model
        found = None
        for _ in range(12):
            if hasattr(m, "get_c8_scale") and hasattr(m, "c8_branch"):
                found = m
                break
            if hasattr(m, "base_model"):
                m = m.base_model
            else:
                break

        if found is not None:
            row["c8_hf_residual"] = 1
            row["c8_scale"] = float(found.get_c8_scale().detach().cpu())
        else:
            row["c8_hf_residual"] = 0
    except Exception:
        pass
    return row
# =================== END TRUE C8 HF RESIDUAL BRANCH PATCH ===================



# ===================== TRUE C8B LOAD C7B AFTER C7B WRAP PATCH =====================
import os as _os_C8B
import torch as _torch_C8B

def _maybe_load_c8b_c7b_ckpt_after_wrap(model, device):
    """
    Load checkpoint C7B setelah C7B output-scale wrapper aktif.
    Ini diperlukan karena checkpoint C7B punya parameter c17_scale_raw dan struktur base_model.*
    """
    path = _os_C8B.environ.get("C8B_INIT_C7B_CKPT", "").strip()
    if not path:
        return model

    if not _os_C8B.path.exists(path):
        raise FileNotFoundError(f"[C8B_LOAD_C7B] checkpoint tidak ditemukan: {path}")

    ckpt = _torch_C8B.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "hybrid_model_state_dict" in ckpt:
            sd = ckpt["hybrid_model_state_dict"]
            key = "hybrid_model_state_dict"
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            key = "model_state_dict"
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            key = "state_dict"
        else:
            sd = ckpt
            key = "raw_dict"
    else:
        sd = ckpt
        key = "raw_object"

    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("[TRUE_C8B_LOAD_C7B] memakai key:", key)
    print("[TRUE_C8B_LOAD_C7B] Loaded:", path)
    print("[TRUE_C8B_LOAD_C7B] missing keys:", len(missing), "| unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("[TRUE_C8B_LOAD_C7B] first missing:", missing[:20])
    if len(unexpected) > 0:
        print("[TRUE_C8B_LOAD_C7B] first unexpected:", unexpected[:20])

    return model
# =================== END TRUE C8B LOAD C7B AFTER C7B WRAP PATCH ===================



# ===================== TRUE C8C STABLE CURVE FREEZE POLICY PATCH =====================
import os as _os_C8C

def _maybe_apply_c8c_freeze_policy(model):
    """
    C8C stable-curve policy:
    Freeze seluruh base model, lalu train hanya:
      - C8 HF residual branch
      - C7B/C17 output scale
    Tujuan: loss curve lebih stabil dan validation corr tidak turun karena base model drift.
    """
    if _os_C8C.environ.get("C8C_FREEZE_POLICY", "0") != "1":
        return model

    train_c7b_scale = _os_C8C.environ.get("C8C_TRAIN_C7B_SCALE", "1") == "1"
    train_alpha = _os_C8C.environ.get("C8C_TRAIN_ALPHA", "0") == "1"

    for _, p in model.named_parameters():
        p.requires_grad = False

    train_keywords = ["c8_branch", "c8_scale_raw"]

    if train_c7b_scale:
        train_keywords += ["c17_scale_raw"]

    if train_alpha:
        train_keywords += ["alpha_logits"]

    matched = []
    n_train = 0

    for name, p in model.named_parameters():
        if any(k in name for k in train_keywords):
            p.requires_grad = True
            matched.append(name)
            n_train += p.numel()

    total = sum(p.numel() for p in model.parameters())

    print("[TRUE_C8C_FREEZE_POLICY] Aktif")
    print("[TRUE_C8C_FREEZE_POLICY] train keywords:", train_keywords)
    print("[TRUE_C8C_FREEZE_POLICY] trainable params:", n_train, "/", total)
    print("[TRUE_C8C_FREEZE_POLICY] matched tensors:", len(matched))
    print("[TRUE_C8C_FREEZE_POLICY] first matched names:")
    for n in matched[:30]:
        print("  -", n)

    return model
# =================== END TRUE C8C STABLE CURVE FREEZE POLICY PATCH ===================



# ===================== TRUE C9 FINAL CKPT LOADER PATCH =====================
import os as _os_C9FINAL
import torch as _torch_C9FINAL

def _maybe_load_c9_final_ckpt_after_c8_wrap(model, device):
    """
    Load exact final C9/C8 checkpoint after all wrappers are already built:
    C4 -> C5 -> C7B output-scale -> C8 HF residual.
    """
    path = _os_C9FINAL.environ.get("C9_FINAL_CKPT", "").strip()
    if not path:
        return model

    if not _os_C9FINAL.path.exists(path):
        raise FileNotFoundError(f"[TRUE_C9_FINAL_CKPT] File tidak ditemukan: {path}")

    ckpt = _torch_C9FINAL.load(path, map_location=device, weights_only=False)

    key_used = None
    if isinstance(ckpt, dict):
        for k in ["hybrid_model_state_dict", "model_state_dict", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                key_used = k
                break
        else:
            sd = ckpt
            key_used = "raw_dict"
    else:
        raise RuntimeError("[TRUE_C9_FINAL_CKPT] Format checkpoint tidak dikenali.")

    clean = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        clean[kk] = v

    incompatible = model.load_state_dict(clean, strict=False)

    print("[TRUE_C9_FINAL_CKPT] Loaded:", path)
    print("[TRUE_C9_FINAL_CKPT] key used:", key_used)
    print("[TRUE_C9_FINAL_CKPT] missing keys:", len(incompatible.missing_keys),
          "| unexpected keys:", len(incompatible.unexpected_keys))

    if len(incompatible.missing_keys) > 0:
        print("[TRUE_C9_FINAL_CKPT] first missing:", incompatible.missing_keys[:20])
    if len(incompatible.unexpected_keys) > 0:
        print("[TRUE_C9_FINAL_CKPT] first unexpected:", incompatible.unexpected_keys[:20])

    return model
# =================== END TRUE C9 FINAL CKPT LOADER PATCH ===================



# ===================== TRUE C13 DELTA POST-GAIN WRAPPER PATCH =====================
import os as _os_C13
import torch as _torch_C13
import torch.nn as _nn_C13

def _c13_delta_bandpass_fft(y, fs=250.0, low=1.0, high=4.0):
    """
    FFT ideal bandpass untuk delta component.
    y shape: [B, C, T]
    """
    y = y.float()
    B, C, T = y.shape

    Y = _torch_C13.fft.rfft(y, dim=-1)
    freqs = _torch_C13.fft.rfftfreq(T, d=1.0 / fs).to(y.device)

    mask = (freqs >= low) & (freqs < high)
    Yf = _torch_C13.zeros_like(Y)
    Yf[..., mask] = Y[..., mask]

    yd = _torch_C13.fft.irfft(Yf, n=T, dim=-1)
    return yd

class _C13DeltaPostGainWrapper(_nn_C13.Module):
    """
    Post-calibration:
    y_final = y + (gain - 1) * delta_bandpass(y)

    Jadi hanya komponen delta 1-4 Hz yang diperbesar,
    komponen theta/alpha/beta relatif tetap.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    @property
    def freeze_backbones(self):
        return getattr(self.base_model, "freeze_backbones", None)

    @freeze_backbones.setter
    def freeze_backbones(self, value):
        if hasattr(self.base_model, "freeze_backbones"):
            self.base_model.freeze_backbones = value

    @property
    def frontal_model(self):
        return getattr(self.base_model, "frontal_model", None)

    @property
    def temporal_model(self):
        return getattr(self.base_model, "temporal_model", None)

    @property
    def maser_model(self):
        return getattr(self.base_model, "maser_model", None)

    @property
    def alpha_logits(self):
        return getattr(self.base_model, "alpha_logits", None)

    @property
    def alpha_init(self):
        return getattr(self.base_model, "alpha_init", None)

    def get_alpha(self):
        if hasattr(self.base_model, "get_alpha"):
            return self.base_model.get_alpha()
        return None

    def _apply_gain(self, y):
        fs = float(_os_C13.environ.get("C13_FS", _os_C13.environ.get("SFREQ", "250")))
        low = float(_os_C13.environ.get("C13_DELTA_LOW", "1.0"))
        high = float(_os_C13.environ.get("C13_DELTA_HIGH", "4.0"))
        gain = float(_os_C13.environ.get("C13_DELTA_GAIN", "2.22"))

        yd = _c13_delta_bandpass_fft(y, fs=fs, low=low, high=high)
        return y + (gain - 1.0) * yd

    def forward(self, x, *args, **kwargs):
        out = self.base_model(x, *args, **kwargs)

        if isinstance(out, tuple):
            y = out[0]
            y2 = self._apply_gain(y)
            return (y2,) + out[1:]

        return self._apply_gain(out)

def _maybe_wrap_c13_delta_postgain(model):
    if _os_C13.environ.get("C13_DELTA_POSTGAIN", "0") != "1":
        return model

    gain = float(_os_C13.environ.get("C13_DELTA_GAIN", "2.22"))
    low = float(_os_C13.environ.get("C13_DELTA_LOW", "1.0"))
    high = float(_os_C13.environ.get("C13_DELTA_HIGH", "4.0"))

    print(f"[TRUE_C13_DELTA_POSTGAIN] enabled | band={low}-{high} Hz | gain={gain}")
    return _C13DeltaPostGainWrapper(model)
# =================== END TRUE C13 DELTA POST-GAIN WRAPPER PATCH ===================



# ===================== TRUE C14 MULTIBAND POST-GAIN WRAPPER PATCH =====================
import os as _os_C14
import torch as _torch_C14
import torch.nn as _nn_C14

def _c14_bandpass_fft(y, fs=250.0, low=1.0, high=4.0):
    y = y.float()
    B, C, T = y.shape

    Y = _torch_C14.fft.rfft(y, dim=-1)
    freqs = _torch_C14.fft.rfftfreq(T, d=1.0 / fs).to(y.device)

    mask = (freqs >= low) & (freqs < high)
    Yf = _torch_C14.zeros_like(Y)
    Yf[..., mask] = Y[..., mask]

    return _torch_C14.fft.irfft(Yf, n=T, dim=-1)

class _C14MultibandPostGainWrapper(_nn_C14.Module):
    """
    Post-calibration multiband:
    y_final = y
              + (delta_gain - 1) * delta(y)
              + (theta_gain - 1) * theta(y)
              + (alpha_gain - 1) * alpha(y)
              + (beta_gain  - 1) * beta(y)

    Default:
    delta 1-4 Hz
    theta 4-8 Hz
    alpha 8-13 Hz
    beta  13-30 Hz
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    @property
    def freeze_backbones(self):
        return getattr(self.base_model, "freeze_backbones", None)

    @freeze_backbones.setter
    def freeze_backbones(self, value):
        if hasattr(self.base_model, "freeze_backbones"):
            self.base_model.freeze_backbones = value

    @property
    def frontal_model(self):
        return getattr(self.base_model, "frontal_model", None)

    @property
    def temporal_model(self):
        return getattr(self.base_model, "temporal_model", None)

    @property
    def maser_model(self):
        return getattr(self.base_model, "maser_model", None)

    @property
    def alpha_logits(self):
        return getattr(self.base_model, "alpha_logits", None)

    @property
    def alpha_init(self):
        return getattr(self.base_model, "alpha_init", None)

    def get_alpha(self):
        if hasattr(self.base_model, "get_alpha"):
            return self.base_model.get_alpha()
        return None

    def _apply_gain(self, y):
        fs = float(_os_C14.environ.get("C14_FS", _os_C14.environ.get("SFREQ", "250")))

        delta_gain = float(_os_C14.environ.get("C14_DELTA_GAIN", "1.0"))
        theta_gain = float(_os_C14.environ.get("C14_THETA_GAIN", "1.0"))
        alpha_gain = float(_os_C14.environ.get("C14_ALPHA_GAIN", "1.0"))
        beta_gain  = float(_os_C14.environ.get("C14_BETA_GAIN",  "1.0"))

        delta_low = float(_os_C14.environ.get("C14_DELTA_LOW", "1.0"))
        delta_high = float(_os_C14.environ.get("C14_DELTA_HIGH", "4.0"))

        theta_low = float(_os_C14.environ.get("C14_THETA_LOW", "4.0"))
        theta_high = float(_os_C14.environ.get("C14_THETA_HIGH", "8.0"))

        alpha_low = float(_os_C14.environ.get("C14_ALPHA_LOW", "8.0"))
        alpha_high = float(_os_C14.environ.get("C14_ALPHA_HIGH", "13.0"))

        beta_low = float(_os_C14.environ.get("C14_BETA_LOW", "13.0"))
        beta_high = float(_os_C14.environ.get("C14_BETA_HIGH", "30.0"))

        y2 = y

        if abs(delta_gain - 1.0) > 1e-8:
            yd = _c14_bandpass_fft(y, fs=fs, low=delta_low, high=delta_high)
            y2 = y2 + (delta_gain - 1.0) * yd

        if abs(theta_gain - 1.0) > 1e-8:
            yt = _c14_bandpass_fft(y, fs=fs, low=theta_low, high=theta_high)
            y2 = y2 + (theta_gain - 1.0) * yt

        if abs(alpha_gain - 1.0) > 1e-8:
            ya = _c14_bandpass_fft(y, fs=fs, low=alpha_low, high=alpha_high)
            y2 = y2 + (alpha_gain - 1.0) * ya

        if abs(beta_gain - 1.0) > 1e-8:
            yb = _c14_bandpass_fft(y, fs=fs, low=beta_low, high=beta_high)
            y2 = y2 + (beta_gain - 1.0) * yb

        return y2

    def forward(self, x, *args, **kwargs):
        out = self.base_model(x, *args, **kwargs)

        if isinstance(out, tuple):
            y = out[0]
            y2 = self._apply_gain(y)
            return (y2,) + out[1:]

        return self._apply_gain(out)

def _maybe_wrap_c14_multiband_postgain(model):
    if _os_C14.environ.get("C14_MULTIBAND_POSTGAIN", "0") != "1":
        return model

    print(
        "[TRUE_C14_MULTIBAND_POSTGAIN] enabled | "
        f"delta={_os_C14.environ.get('C14_DELTA_GAIN','1.0')} | "
        f"theta={_os_C14.environ.get('C14_THETA_GAIN','1.0')} | "
        f"alpha={_os_C14.environ.get('C14_ALPHA_GAIN','1.0')} | "
        f"beta={_os_C14.environ.get('C14_BETA_GAIN','1.0')}"
    )

    return _C14MultibandPostGainWrapper(model)
# =================== END TRUE C14 MULTIBAND POST-GAIN WRAPPER PATCH ===================



# ===================== TRUE C15 EXPORT SEGMENT CSV PATCH =====================
import os as _os_C15
import re as _re_C15
import glob as _glob_C15
import numpy as _np_C15
import pandas as _pd_C15
import torch as _torch_C15

def _c15_to_numpy(x):
    if _torch_C15.is_tensor(x):
        x = x.detach().cpu().numpy()
    return _np_C15.asarray(x)

def _c15_norm_arr(a):
    a = _c15_to_numpy(a)
    if a.ndim == 3:
        # ideal: N,C,T
        if a.shape[1] == 4:
            return a
        if a.shape[2] == 4:
            return _np_C15.transpose(a, (0, 2, 1))
    if a.ndim == 2:
        if a.shape[0] == 4:
            return a[None, :, :]
        if a.shape[1] == 4:
            return a.T[None, :, :]
    return None

def _c15_score_name(name, mode):
    n = str(name).lower()
    score = 0

    if mode == "pred":
        for tok in ["pred", "y_pred", "yp", "output", "outputs", "recon", "ai"]:
            if tok in n:
                score += 5
        for bad in ["mne", "spline", "true", "target", "yb", "input", "xb"]:
            if bad in n:
                score -= 6

    elif mode == "true":
        for tok in ["true", "target", "targets", "y_true", "yt", "yb"]:
            if tok in n:
                score += 5
        for bad in ["pred", "output", "recon", "ai", "mne", "spline", "input", "xb"]:
            if bad in n:
                score -= 6

    elif mode == "input":
        for tok in ["input", "inputs", "x", "xb"]:
            if tok == n or tok in n:
                score += 5
        for bad in ["pred", "true", "target", "output", "mne"]:
            if bad in n:
                score -= 5

    return score

def _c15_find_array(local_vars, mode):
    candidates = []
    for k, v in local_vars.items():
        try:
            if _torch_C15.is_tensor(v) or isinstance(v, _np_C15.ndarray):
                arr = _c15_norm_arr(v)
                if arr is not None and arr.ndim == 3 and arr.shape[1] == 4:
                    candidates.append((_c15_score_name(k, mode), k, arr))
        except Exception:
            pass

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    if candidates and candidates[0][0] > 0:
        print(f"[C15_EXPORT] selected {mode}: {candidates[0][1]} shape={candidates[0][2].shape}")
        return candidates[0][2]

    print(f"[C15_EXPORT] failed find {mode}. candidates:", [(c[0], c[1], c[2].shape) for c in candidates[:10]])
    return None

def _c15_get_dataset_paths(local_vars):
    # 1) explicit list/path variables
    for k, v in local_vars.items():
        lk = str(k).lower()
        if any(tok in lk for tok in ["file", "path", "csv"]):
            if isinstance(v, (list, tuple)) and len(v) > 0:
                vals = [str(x) for x in v]
                if any(x.lower().endswith(".csv") for x in vals):
                    return vals

    # 2) dataloader dataset attributes
    for k, v in local_vars.items():
        if "loader" in str(k).lower():
            ds = getattr(v, "dataset", None)
            if ds is None:
                continue
            for attr in ["files", "file_paths", "paths", "csv_files", "items", "samples", "records"]:
                if hasattr(ds, attr):
                    obj = getattr(ds, attr)
                    try:
                        vals = []
                        for item in obj:
                            if isinstance(item, str):
                                vals.append(item)
                            elif isinstance(item, (list, tuple)) and len(item) > 0:
                                vals.append(str(item[0]))
                            elif isinstance(item, dict):
                                for kk in ["file", "path", "csv", "csv_path", "filepath"]:
                                    if kk in item:
                                        vals.append(str(item[kk]))
                                        break
                        if vals and any(x.lower().endswith(".csv") for x in vals):
                            return vals
                    except Exception:
                        pass

    return None

def _c15_subject_from_file(path):
    base = _os_C15.path.basename(str(path))
    m = _re_C15.match(r"^([A-Za-z]+)_rec", base)
    if m:
        return m.group(1)
    return _os_C15.path.splitext(base)[0].split("_")[0]

def _c15_corr(a, b):
    a = _np_C15.asarray(a, dtype=float)
    b = _np_C15.asarray(b, dtype=float)
    m = _np_C15.isfinite(a) & _np_C15.isfinite(b)
    if m.sum() < 3:
        return _np_C15.nan
    a = a[m]
    b = b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return _np_C15.nan
    return float(_np_C15.corrcoef(a, b)[0, 1])

def _c15_export_segments_from_eval(local_vars):
    if _os_C15.environ.get("C15_EXPORT_SEGMENT_CSV", "0") != "1":
        return

    # supaya tidak export berkali-kali dalam satu process
    global _C15_ALREADY_EXPORTED
    try:
        if _C15_ALREADY_EXPORTED:
            return
    except NameError:
        _C15_ALREADY_EXPORTED = False

    pred = _c15_find_array(local_vars, "pred")
    true = _c15_find_array(local_vars, "true")

    if pred is None or true is None:
        print("[C15_EXPORT] pred/true tidak ditemukan. Export dibatalkan.")
        return

    n = min(pred.shape[0], true.shape[0])
    pred = pred[:n]
    true = true[:n]

    fs = float(_os_C15.environ.get("C15_FS", _os_C15.environ.get("SFREQ", "250")))
    gt_root = _os_C15.environ.get("C15_GT_ROOT", "/drive0-storage/Anna/new_signal_clean")
    out_root = _os_C15.environ.get("C15_OUT_DIR", "")

    if not out_root:
        # fallback cari dari env output
        base_out = (
            _os_C15.environ.get("OUT_ROOT")
            or _os_C15.environ.get("OUTPUT_DIR")
            or _os_C15.environ.get("OUT_DIR")
            or "."
        )
        out_root = _os_C15.path.join(base_out, "hybrid_learned_alpha_beta_shape002_rel001_seed42", "exported_segment_csv")

    _os_C15.makedirs(out_root, exist_ok=True)

    # simpan arrays untuk plotting cepat
    _np_C15.save(_os_C15.path.join(out_root, "_y_pred.npy"), pred)
    _np_C15.save(_os_C15.path.join(out_root, "_y_true.npy"), true)

    paths = _c15_get_dataset_paths(local_vars)

    gt_files = sorted(_glob_C15.glob(_os_C15.path.join(gt_root, "*.csv")))
    # buang dummy train kalau ada
    gt_files = [f for f in gt_files if not _os_C15.path.basename(f).startswith("ZZZTRAIN")]

    if paths is None or len(paths) < n:
        print("[C15_EXPORT] dataset paths tidak ditemukan/cukup. Pakai sorted GT_ROOT fallback.")
        paths = gt_files

    paths = [p for p in paths if str(p).lower().endswith(".csv")]
    paths = [p for p in paths if not _os_C15.path.basename(str(p)).startswith("ZZZTRAIN")]

    if len(paths) < n:
        print(f"[C15_EXPORT] WARNING paths={len(paths)} < n={n}. File akan dinamai segment index.")

    rows = []
    target_ch = ["F3", "F4", "P3", "P4"]

    for i in range(n):
        src_path = paths[i] if i < len(paths) else None
        T = pred.shape[2]

        if src_path is not None and _os_C15.path.exists(str(src_path)):
            df0 = _pd_C15.read_csv(src_path)
            base = _os_C15.path.splitext(_os_C15.path.basename(str(src_path)))[0]
            subj = _c15_subject_from_file(src_path)
        else:
            df0 = None
            base = f"segment_{i:04d}"
            subj = "UNKNOWN"

        # ambil panjang minimum
        if df0 is not None:
            L = min(len(df0), T)
        else:
            L = T

        out = _pd_C15.DataFrame()
        if df0 is not None and "time_s" in df0.columns:
            out["time_s"] = _pd_C15.to_numeric(df0["time_s"].iloc[:L], errors="coerce").values
        else:
            out["time_s"] = _np_C15.arange(L) / fs

        # physical channels dari CSV asli
        for ch in ["Fp1", "Fp2"]:
            if df0 is not None and ch in df0.columns:
                out[ch] = _pd_C15.to_numeric(df0[ch].iloc[:L], errors="coerce").values
            else:
                out[ch] = _np_C15.nan

        # virtual target dari deep learning prediction
        out["F3"] = pred[i, 0, :L]
        out["F4"] = pred[i, 1, :L]

        for ch in ["T3", "T4"]:
            if df0 is not None and ch in df0.columns:
                out[ch] = _pd_C15.to_numeric(df0[ch].iloc[:L], errors="coerce").values
            else:
                out[ch] = _np_C15.nan

        out["P3"] = pred[i, 2, :L]
        out["P4"] = pred[i, 3, :L]

        if df0 is not None and "PPG" in df0.columns:
            out["PPG"] = _pd_C15.to_numeric(df0["PPG"].iloc[:L], errors="coerce").values
        else:
            out["PPG"] = 0

        out = out[["time_s", "Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4", "PPG"]]

        save_path = _os_C15.path.join(out_root, f"{base}_SR.csv")
        out.to_csv(save_path, index=False)

        corr_map = {}
        for ci, ch in enumerate(target_ch):
            corr_map[ch] = _c15_corr(true[i, ci, :L], pred[i, ci, :L])

        rows.append({
            "idx": i,
            "subject": subj,
            "source_file": str(src_path) if src_path is not None else "",
            "output_csv": save_path,
            "n_samples": L,
            "mean_corr": _np_C15.nanmean(list(corr_map.values())),
            "corr_F3": corr_map["F3"],
            "corr_F4": corr_map["F4"],
            "corr_P3": corr_map["P3"],
            "corr_P4": corr_map["P4"],
        })

    man = _pd_C15.DataFrame(rows)
    man.to_csv(_os_C15.path.join(out_root, "_export_manifest.csv"), index=False)

    print(f"[C15_EXPORT] DONE. Exported {len(rows)} CSV files to:")
    print(out_root)
    print(f"[C15_EXPORT] Manifest: {_os_C15.path.join(out_root, '_export_manifest.csv')}")

    _C15_ALREADY_EXPORTED = True
# =================== END TRUE C15 EXPORT SEGMENT CSV PATCH ===================

def main_hybrid():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    print("=" * 80)
    print("EEG SUPER-RESOLUTION — TRANSFER / HYBRID CNN + MASER FUSION")
    print("=" * 80)
    print("CONFIG_ID        :", CONFIG_ID)
    print("Experiment       :", EXPERIMENT_NAME)
    print("Output dir       :", OUTPUT_DIR)
    print("Device           :", device)
    print("GPU count        :", torch.cuda.device_count())
    print("Batch size       :", BATCH_SIZE)
    print("Hybrid mode      :", HYBRID_MODE)
    print("Freeze backbones :", HYBRID_FREEZE_BACKBONES)

    # C1_PATCH_INIT_ALPHA_BEGIN
    HYBRID_INIT_OLD_WEIGHTS = _parse_float_list_env_C1("INIT_ALPHA", [0.5, 0.5, 0.5, 0.5])  # C1_PATCH_FIXED
    # C1_PATCH_INIT_ALPHA_END
    print("Init old weights :", HYBRID_INIT_OLD_WEIGHTS)
    print("CNN ckpt         :", HYBRID_CNN_CKPT)
    print("MASER ckpt       :", HYBRID_MASER_CKPT)
    print("Loss w           : time", HYBRID_TIME_W, "temp", HYBRID_TEMPORAL_W, "amp", HYBRID_AMP_W,
          "beta_topo", HYBRID_BETA_TOPO_W, "beta_power", HYBRID_BETA_POWER_W)
    print("=" * 80)

    data_format = resolve_data_format(DATA_ROOT)
    print("Data format      :", data_format)

    if data_format == "primary_csv":
        all_files = audit_primary_csv_input(DATA_ROOT)
        if HOLDOUT_SUBJECTS:
            all_subjects = sorted(set(parse_subject_id_from_csv(p) for p in all_files))
            missing = [s for s in HOLDOUT_SUBJECTS if s not in all_subjects]
            if missing:
                raise RuntimeError(f"HOLDOUT_SUBJECTS tidak ditemukan: {missing}. Subjek tersedia: {all_subjects}")

            test_subjects = [s for s in all_subjects if s in HOLDOUT_SUBJECTS]
            train_subjects = [s for s in all_subjects if s not in HOLDOUT_SUBJECTS]

            train_files = [p for p in all_files if parse_subject_id_from_csv(p) in train_subjects]
            test_files = [p for p in all_files if parse_subject_id_from_csv(p) in test_subjects]

            print(f"[HOLDOUT_SUBJECTS] Validasi subjek: {test_subjects}")
        elif ADAPT_ALL_PRIMARY:
            all_subjects = sorted(set(parse_subject_id_from_csv(p) for p in all_files))
            train_subjects = all_subjects
            test_subjects = all_subjects
            train_files = list(all_files)
            test_files = list(all_files)
            print("[ADAPT_ALL_PRIMARY=1] Semua data primer dipakai untuk adaptasi.")
            print("[INFO] Test internal hanya untuk monitoring training, bukan hasil final laporan.")
        else:
            train_subjects, test_subjects, train_files, test_files = split_subject_files(all_files, TEST_RATIO, SEED)

        print("Train subjects:", train_subjects)
        print("Test subjects :", test_subjects)
        print("Train files   :", len(train_files), "| Test files:", len(test_files))

        # Simpan split agar eksperimen transfer learning reproducible.
        pd.DataFrame({"file": [str(p) for p in train_files], "subject": [parse_subject_id_from_csv(p) for p in train_files]}).to_csv(
            os.path.join(OUTPUT_DIR, "train_files_primary.csv"), index=False
        )
        pd.DataFrame({"file": [str(p) for p in test_files], "subject": [parse_subject_id_from_csv(p) for p in test_files]}).to_csv(
            os.path.join(OUTPUT_DIR, "test_files_primary.csv"), index=False
        )

        train_dataset = PrimaryCSVSRDataset(train_files, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
        test_dataset = PrimaryCSVSRDataset(test_files, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)
    else:
        audit_training_input(DATA_ROOT)
        all_folders = discover_folders(DATA_ROOT)
        train_subjects, test_subjects, train_folders, test_folders = split_subject_folders(all_folders, TEST_RATIO, SEED)
        print("Train subjects:", train_subjects)
        print("Test subjects :", test_subjects)
        print("Train folders :", len(train_folders), "| Test folders:", len(test_folders))
        train_dataset = SEEDSR8NpyDataset(train_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, True, MAX_TRAIN_SEGMENTS)
        test_dataset = SEEDSR8NpyDataset(test_folders, DATA_ROOT, INPUT_IDX, TARGET_IDX, SEGMENT_SAMPLES, False, MAX_TEST_SEGMENTS)

    train_drop_last = len(train_dataset) >= BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin_memory, drop_last=train_drop_last, persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_memory, drop_last=False, persistent_workers=(NUM_WORKERS > 0))
    print("Train batches:", len(train_loader), "| Test batches:", len(test_loader), "| drop_last:", train_drop_last)

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
        residual_scale_max=HYBRID_RESIDUAL_SCALE_MAX,
        dropout=HYBRID_DROPOUT,
    ).to(device)

    # ========================================================
    # Optional resume dari checkpoint hybrid P10/P11/P12
    # ========================================================
    if HYBRID_RESUME_CKPT:
        print("[RESUME] Loading hybrid checkpoint:", HYBRID_RESUME_CKPT)
        ckpt_resume = torch.load(HYBRID_RESUME_CKPT, map_location=device, weights_only=False)

        state = None
        for key in ["model_state_dict", "hybrid_model_state_dict", "state_dict"]:
            if isinstance(ckpt_resume, dict) and key in ckpt_resume:
                state = ckpt_resume[key]
                print("[RESUME] Using key:", key)
                break

        if state is None:
            state = ckpt_resume

        # Handle checkpoint dari DataParallel
        if isinstance(state, dict) and any(str(k).startswith("module.") for k in state.keys()):
            state = {str(k).replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[RESUME] Missing keys   :", len(missing))
        print("[RESUME] Unexpected keys:", len(unexpected))
        if len(missing) > 0:
            print("[RESUME] Contoh missing:", missing[:10])
        if len(unexpected) > 0:
            print("[RESUME] Contoh unexpected:", unexpected[:10])

    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU via DataParallel.")
        model = nn.DataParallel(model)

    # C2_FORCE_WRAP_BEFORE_HYBRID_OPTIMIZER
    model = _maybe_wrap_c2_affine(model, TARGET_CH, device)
    # C3_FORCE_WRAP_BEFORE_HYBRID_OPTIMIZER
    model = _maybe_wrap_c3_bandcal(model, TARGET_CH, device)
    # C4_FORCE_WRAP_BEFORE_HYBRID_OPTIMIZER
    model = _maybe_wrap_c4_bandmix(model, TARGET_CH, device)
    # C5_FORCE_WRAP_BEFORE_HYBRID_OPTIMIZER
    model = _maybe_wrap_c5_tempres(model, TARGET_CH, device)
    # C6_FORCE_HEADUNFREEZE_BEFORE_HYBRID_OPTIMIZER
    model = _maybe_load_full_calibration_ckpt(model, device)
    # C17_FORCE_OUTSCALE_AFTER_FULL_CKPT
    model = _maybe_wrap_c17_outscale(model, TARGET_CH, device)
    # C8B_FORCE_LOAD_C7B_AFTER_C7B_WRAP
    model = _maybe_load_c8b_c7b_ckpt_after_wrap(model, device)
    model = _maybe_apply_c6_partial_unfreeze(model)
    # C8_FORCE_WRAP_AFTER_PARTIAL_UNFREEZE
    model = _maybe_wrap_c8_hf_residual(model, INPUT_CH, TARGET_CH, device)
    # C9_FINAL_CKPT_AFTER_C8_WRAP
    model = _maybe_load_c9_final_ckpt_after_c8_wrap(model, device)
    # C13_DELTA_POSTGAIN_AFTER_FINAL_CKPT
    model = _maybe_wrap_c13_delta_postgain(model)
    # C14_MULTIBAND_POSTGAIN_AFTER_C13
    model = _maybe_wrap_c14_multiband_postgain(model)
    # C8C_FORCE_FREEZE_AFTER_C8_WRAP
    model = _maybe_apply_c8c_freeze_policy(model)
    optimizer = make_optimizer_hybrid(model)
    # C1_FORCE_OPTIMIZER_LR_FINAL
    import os as _os_C1_lr_final
    _c1_lr_raw = _os_C1_lr_final.environ.get('LR', '').strip()
    if _c1_lr_raw:
        _c1_lr_val = float(_c1_lr_raw)
        for _pg in optimizer.param_groups:
            _pg['lr'] = _c1_lr_val
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
            if hasattr(unwrap(model), "get_residual_scale"):
                rs = unwrap(model).get_residual_scale()
                if rs is not None:
                    residual_scale = float(rs.detach().cpu().item())

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
            # C2_ROW_UPDATE_FINAL
            row = _c2_update_row_with_affine(row, locals())
            # C2_FORCE_ROW_GAIN_BIAS_HISTORY
            row = _c2_update_row_with_affine(row, locals())
            # C3_FORCE_ROW_BANDCAL_HISTORY
            row = _c3_update_row_with_bandcal(row, locals())
            # C4_FORCE_ROW_BANDMIX_HISTORY
            row = _c4_update_row_with_bandmix(row, locals())
            # C5_FORCE_ROW_TEMPRES_HISTORY
            row = _c5_update_row_with_tempres(row, locals())
            # C6_FORCE_ROW_HISTORY
            row = _c6_update_row(row, locals())
            # C17_FORCE_ROW_HISTORY
            row = _c17_update_row(row, locals())
            # C8_FORCE_ROW_HISTORY
            row = _c8_update_row(row, locals())
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
