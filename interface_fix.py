"""
Deployment GUI/Core EEG 4ch + PPG -> EEG Super-Resolution P16 -> fitur EEG+PPG ->
manual NumPy inference boundary cuML SVM untuk stress/anxiety/depression.

Alur yang diasumsikan untuk fungsi utama:
1) EEG 4 channel fisik sudah melewati preprocessing GUI lama: bandpass, Zapline, ICA/MAICA.
2) Channel fisik masuk SR P16 dalam urutan Fp1, Fp2, T7, T8.
   Jika input memakai T3/T4, otomatis dialias ke T7/T8 untuk model SR.
3) Output SR: F3, F4, P3, P4, lalu digabung menjadi:
   Fp1, Fp2, F3, F4, T7, T8, P3, P4.
4) Untuk ekstraksi fitur lama, T7/T8 direname ke T3/T4 agar fitur T3/T4 tidak hilang.
5) Fitur masuk ke boundary SVM manual. Tidak ada retraining sklearn/cuml.

Catatan integrasi dengan GUI lama:
- Panggil DeploymentPipeline.run_clean_4ch_array() setelah results['ica']['data'] tersedia.
- Untuk sistem lama 8 channel, ambil 4 channel fisik awal sesuai konfigurasi fisik: Fp1,Fp2,T3,T4.
- File olah_data_lengkap.py digunakan sebagai sumber fungsi ekstraksi fitur.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import pickle
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tkinter import filedialog
from datetime import datetime
import tempfile
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.pdfbase.pdfmetrics import stringWidth
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.platypus import KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image
import tempfile
import os
import numpy as np
from datetime import datetime
# ============================================================
# COMPATIBILITY PATCH NUMPY 2.x
# np.trapz lama diganti menjadi np.trapezoid di NumPy baru.
# Patch ini membuat kode lama tetap jalan tanpa mengganti semua trapz.
# ============================================================
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

def _call_process_eeg_data_compat(processor_module, filename, **kwargs):
    """
    Memanggil eeg_processor_8.process_eeg_data secara kompatibel.
    Beberapa versi eeg_processor_8 lama belum punya parameter seperti
    skip_zapline, k_thresh, bcg_thresh, atau vbr_thresh. Helper ini
    otomatis hanya mengirim keyword yang memang ada di signature fungsi.
    """
    func = processor_module.process_eeg_data

    try:
        sig = inspect.signature(func)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_kwargs:
            accepted_kwargs = kwargs
            dropped = []
        else:
            accepted_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            dropped = [k for k in kwargs if k not in sig.parameters]

        if dropped:
            print(f"[COMPAT] process_eeg_data versi ini tidak menerima {dropped}; parameter tersebut dilewati.")

        return func(filename, **accepted_kwargs)

    except TypeError as e:
        # Fallback terakhir untuk processor yang sangat lama.
        msg = str(e)
        print(f"[COMPAT] Fallback process_eeg_data karena TypeError: {msg}")
        try:
            return func(filename, fs=kwargs.get("fs", 250), skip_ica=kwargs.get("skip_ica", False))
        except TypeError:
            return func(filename, fs=kwargs.get("fs", 250))

try:
    import torch
except Exception:  # pragma: no cover - agar file tetap bisa dibaca meski torch belum terpasang
    torch = None

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover
    resample_poly = None


# =============================================================================
# KONSTANTA DEPLOYMENT
# =============================================================================
FS_EXPECTED = 250
SEGMENT_SECONDS = 120
SEGMENT_SAMPLES = FS_EXPECTED * SEGMENT_SECONDS

SR_INPUT_CHANNELS = ["Fp1", "Fp2", "T7", "T8"]
SR_TARGET_CHANNELS = ["F3", "F4", "P3", "P4"]
SR_OUTPUT_CHANNELS_T7T8 = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]
FEATURE_CHANNELS_T3T4 = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]

REQUIRED_FEATURES_BY_LABEL = {
    "stress": [
        "SASI_Fp1",
        "SASI_F3",
        "ORI_CONN_MI_Alpha_Fp1-Fp2",
        "AbsPow_Beta_Fp1",
        "LZC_Alpha_P3",
        "Ratio_AlphaBeta_Fp1",
        "SASI_P3",
    ],
    "anxiety": [
        "PPG_DFA_alpha2",
        "AbsPow_Beta_Fp1",
        "ORI_CONN_MI_Beta_Fp1-F3",
        "AbsPow_Beta_T4",
        "ORI_CONN_PCC_Delta_Fp1-Fp2",
    ],
    "depression": [
        "LZC_Alpha_T3",
        "LZC_Alpha_Fp1",
        "LZC_Alpha_T4",
        "LZC_T3",
        "CCM_HeartToBrain_Alpha_T3",
    ],
}
MODEL_SUFFIX = ""
DEFAULT_MODEL_PATHS = {
    "stress": Path(f"DEPLOY_cuml_boundary_stress_8CH_ALL_PPG{MODEL_SUFFIX}/cuml_svm_stress_rbf_cpu_boundary_UPDATED_THRESHOLD.pkl"),
    "anxiety": Path(f"DEPLOY_cuml_boundary_anxiety_8CH_ALL_PPG{MODEL_SUFFIX}/cuml_svm_anxiety_rbf_cpu_boundary_UPDATED_THRESHOLD.pkl"),
    "depression": Path(f"DEPLOY_cuml_boundary_depression_8CH_ALL_PPG{MODEL_SUFFIX}/cuml_svm_depression_rbf_cpu_boundary_UPDATED_THRESHOLD.pkl"),
}

PDF_BG_PATH = None

PDF_COLORS = {
    "bg": "#F6F0FB",
    "panel": "#EEE5F8",
    "panel2": "#E5D7F5",
    "accent": "#8E6CCF",
    "accent_dark": "#6F4DB8",
    "accent_soft": "#B89AE6",
    "text": "#2D2242",
    "muted": "#6E6485",
    "white": "#FFFFFF",
    "line": "#D7C8ED",
    "risk_low": "#63C7B2",
    "risk_mid": "#F2B84B",
    "risk_high": "#E66B8A",
}

# =============================================================================
# UTILITAS UMUM
# =============================================================================
class DeploymentError(RuntimeError):
    """Error yang sengaja dibuat jelas untuk ditampilkan ke GUI."""


def _norm_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _first_existing_path(paths: Sequence[Union[str, Path]]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def safe_sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    exp_z = np.exp(z_arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    if np.isscalar(z):
        return float(out.reshape(-1)[0])
    return out


def ensure_no_nan_inf(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise DeploymentError(f"{name} mengandung NaN/inf sebanyak {bad}. Periksa preprocessing/input.")


def import_module_from_path(module_name: str, path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File modul tidak ditemukan: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa membuat import spec untuk: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _deploy_rl_image_keep_aspect(self, img_path, max_w, max_h):
    """
    Membuat RLImage dengan aspect ratio tetap.
    Jadi brain map tidak gepeng / termampatkan.
    """
    img = Image.open(img_path)
    w, h = img.size

    if w <= 0 or h <= 0:
        return RLImage(img_path, width=max_w, height=max_h)

    scale = min(max_w / w, max_h / h)
    new_w = w * scale
    new_h = h * scale

    return RLImage(img_path, width=new_w, height=new_h)

def robust_scale_by_input(x_raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalisasi persis sesuai infer_P16_SR8_to_82_segments.py."""
    q75, q25 = np.percentile(x_raw, [75, 25])
    iqr = float(q75 - q25) + 1e-6
    median = float(np.median(x_raw))
    x = (x_raw - median) / iqr
    x = np.tanh(x / 5.0)
    return x.astype(np.float32), median, iqr


def inverse_tanh_scale(y_norm: np.ndarray, median: float, iqr: float) -> np.ndarray:
    """Inverse normalisasi persis sesuai infer_P16_SR8_to_82_segments.py."""
    y = np.clip(y_norm, -0.999999, 0.999999)
    y = np.arctanh(y) * 5.0 * iqr + median
    return y.astype(np.float32)

def _deploy_get_subject_name_from_file(self):
    """
    Ambil nama dari nama file.
    Contoh:
    clean_17.46dB_nur2_S20_A17_D14_F.csv -> nur2
    nur_rec1_clean_14.59dB_nur1_S20_A17_D14_F_seg004_MAICA.csv -> nur1
    """
    fname = ""

    for attr in ["last_filename", "loaded_file_path", "current_file_path"]:
        if hasattr(self, attr):
            val = getattr(self, attr)
            if val:
                fname = str(val)
                break

    if not fname and hasattr(self, "file_label"):
        try:
            fname = self.file_label.cget("text")
        except Exception:
            fname = ""

    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]

    # Ambil token sebelum _Sxx_Axx_Dxx
    m = re.search(r"([A-Za-z]+[0-9]*)_S\d+_A\d+_D\d+", stem)
    if m:
        return m.group(1)

    # Fallback untuk clean_17.46dB_nur2_S20...
    parts = stem.split("_")
    for p in parts:
        if re.match(r"^[A-Za-z]+[0-9]*$", p) and not p.lower().startswith("clean"):
            return p

    return stem if stem else "Unknown"
def _deploy_collect_report_summary(self):
    """
    Ambil data report dari deploy_outputs.
    Output:
    - subject name
    - session datetime
    - status/risk per label
    - fitur yang dipakai masing-masing model
    """
    if not hasattr(self, "deploy_outputs") or not self.deploy_outputs:
        raise RuntimeError("Belum ada hasil SR+ML. Klik Run SR + ML terlebih dahulu.")

    out = self.deploy_outputs[0]

    subject_name = self._get_subject_name_from_file()
    session_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_rows = []
    feature_rows = []

    for label in ["stress", "anxiety", "depression"]:
        if label not in out.predictions:
            continue

        pred = out.predictions[label]
        status = pred.get("status", "Unknown")
        risk = float(pred.get("risk_score_model", np.nan))

        model_rows.append({
            "label": label.capitalize(),
            "status": status,
            "risk": risk,
        })

        # Ambil feature_order dari model artifact agar fitur yang dilaporkan
        # benar-benar fitur yang masuk model.
        feature_order = []
        try:
            feature_order = list(self.deploy_pipeline.models[label].feature_order)
        except Exception:
            feature_order = []

        for feat in feature_order:
            val = out.features.get(feat, np.nan)
            feature_rows.append({
                "label": label.capitalize(),
                "feature": feat,
                "value": val
            })

    return {
        "subject_name": subject_name,
        "session_datetime": session_dt,
        "model_rows": model_rows,
        "feature_rows": feature_rows,
    }
    
def _deploy_collect_report_summary(self):
    """
    Ambil data report dari deploy_outputs.
    Output:
    - subject name
    - session datetime
    - status/risk per label
    - fitur yang dipakai masing-masing model
    """
    if not hasattr(self, "deploy_outputs") or not self.deploy_outputs:
        raise RuntimeError("Belum ada hasil SR+ML. Klik Run SR + ML terlebih dahulu.")

    out = self.deploy_outputs[0]

    subject_name = self._get_subject_name_from_file()
    session_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_rows = []
    feature_rows = []

    for label in ["stress", "anxiety", "depression"]:
        if label not in out.predictions:
            continue

        pred = out.predictions[label]
        status = pred.get("status", "Unknown")
        risk = float(pred.get("risk_score_model", np.nan))

        model_rows.append({
            "label": label.capitalize(),
            "status": status,
            "risk": risk,
        })

        # Ambil feature_order dari model artifact agar fitur yang dilaporkan
        # benar-benar fitur yang masuk model.
        feature_order = []
        try:
            feature_order = list(self.deploy_pipeline.models[label].feature_order)
        except Exception:
            feature_order = []

        for feat in feature_order:
            val = out.features.get(feat, np.nan)
            feature_rows.append({
                "label": label.capitalize(),
                "feature": feat,
                "value": val
            })

    return {
        "subject_name": subject_name,
        "session_datetime": session_dt,
        "model_rows": model_rows,
        "feature_rows": feature_rows,
    }
    
def _deploy_build_report_tab(self):
    _deploy_ensure_vars(self)

    if not hasattr(self, "report_tab") or self.report_tab is None:
        self.report_tab = ttk.Frame(self.notebook)

    for w in self.report_tab.winfo_children():
        w.destroy()

    container = ttk.Frame(self.report_tab)
    container.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    title = ttk.Label(
        container,
        text="EEG Report",
        font=("Segoe UI", 18, "bold")
    )
    title.pack(anchor="w", pady=(0, 8))

    info_frame = ttk.LabelFrame(container, text="Session Information", padding=8)
    info_frame.pack(fill=tk.X, pady=(0, 8))

    try:
        summary = self._collect_report_summary()
        subject = summary["subject_name"]
        session_dt = summary["session_datetime"]
    except Exception:
        subject = self._get_subject_name_from_file()
        session_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = None

    ttk.Label(info_frame, text=f"Name: {subject}", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=12)
    ttk.Label(info_frame, text=f"Session: {session_dt}", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=12)

    btn_frame = ttk.Frame(container)
    btn_frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Button(
        btn_frame,
        text="Refresh Report",
        command=self._build_report_tab
    ).pack(side=tk.LEFT, padx=4)

    ttk.Button(
        btn_frame,
        text="Save Report as PDF",
        command=self._save_report_pdf_dialog
    ).pack(side=tk.LEFT, padx=4)

    # =========================
    # Model Summary
    # =========================
    result_frame = ttk.LabelFrame(container, text="Model Results", padding=8)
    result_frame.pack(fill=tk.X, pady=(0, 8))

    tree = ttk.Treeview(
        result_frame,
        columns=("label", "status", "risk"),
        show="headings",
        height=4
    )

    tree.heading("label", text="Model")
    tree.heading("status", text="Status")
    tree.heading("risk", text="Skor Risiko Model")

    tree.column("label", width=140, anchor="center")
    tree.column("status", width=160, anchor="center")
    tree.column("risk", width=160, anchor="center")

    tree.pack(fill=tk.X)

    if summary is not None:
        for row in summary["model_rows"]:
            tree.insert(
                "",
                "end",
                values=(
                    row["label"],
                    row["status"],
                    f"{row['risk']:.1f}%"
                )
            )
    else:
        tree.insert("", "end", values=("Stress", "-", "-"))
        tree.insert("", "end", values=("Anxiety", "-", "-"))
        tree.insert("", "end", values=("Depression", "-", "-"))

    # =========================
    # Feature Values
    # =========================
    feat_frame = ttk.LabelFrame(container, text="Feature Values Used by Each Model", padding=8)
    feat_frame.pack(fill=tk.BOTH, expand=True)

    feat_tree = ttk.Treeview(
        feat_frame,
        columns=("label", "feature", "value"),
        show="headings",
        height=12
    )

    feat_tree.heading("label", text="Model")
    feat_tree.heading("feature", text="Feature")
    feat_tree.heading("value", text="Value")

    feat_tree.column("label", width=120, anchor="center")
    feat_tree.column("feature", width=360, anchor="w")
    feat_tree.column("value", width=160, anchor="center")

    feat_scroll = ttk.Scrollbar(feat_frame, orient="vertical", command=feat_tree.yview)
    feat_tree.configure(yscrollcommand=feat_scroll.set)

    feat_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    feat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    if summary is not None:
        for row in summary["feature_rows"]:
            val = row["value"]
            try:
                val_str = f"{float(val):.6g}"
            except Exception:
                val_str = str(val)

            feat_tree.insert(
                "",
                "end",
                values=(
                    row["label"],
                    row["feature"],
                    val_str
                )
            )

    disclaimer = ttk.Label(
        container,
        text="Disclaimer: Hasil ini merupakan estimasi model berbasis sinyal EEG/PPG dan bukan diagnosis klinis.",
        foreground=self.colors.get("text_dim", "gray")
    )
    disclaimer.pack(fill=tk.X, pady=(8, 0))

def _deploy_pdf_page_bg(self, canvas, doc):
    """
    Background PDF bernuansa wallpaper lavender.
    """
    page_w, page_h = doc.pagesize

    # 1) background dasar
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(PDF_COLORS["bg"]))
    canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    # 2) kalau file wallpaper ada, pakai sebagai header besar di atas
    if os.path.exists(PDF_BG_PATH):
        try:
            canvas.drawImage(
                PDF_BG_PATH,
                0,
                page_h - 4.0 * cm,
                width=page_w,
                height=4.0 * cm,
                preserveAspectRatio=True,
                mask='auto'
            )
        except Exception as e:
            print(f"[PDF BG] Gagal load wallpaper background: {e}")

    # 3) overlay transparan tidak mudah di reportlab,
    # jadi tambahkan panel putih untuk area konten
    canvas.setFillColor(colors.white)
    canvas.rect(
        0.8 * cm,
        1.0 * cm,
        page_w - 1.6 * cm,
        page_h - 5.0 * cm,
        fill=1,
        stroke=0
    )

    canvas.restoreState()
    
def pad_2d_time(arr: np.ndarray, target_len: int, mode: str = "reflect") -> Tuple[np.ndarray, bool]:
    """Pad array shape (C,T) pada axis waktu. EEG: reflect jika memungkinkan, edge jika T<=1."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"arr harus 2D (C,T), diterima {arr.shape}")
    t = arr.shape[1]
    if t == target_len:
        return arr, False
    if t > target_len:
        return arr[:, :target_len], False
    pad = target_len - t
    if t <= 1 or mode == "edge":
        return np.pad(arr, ((0, 0), (0, pad)), mode="edge"), True
    return np.pad(arr, ((0, 0), (0, pad)), mode="reflect"), True


def pad_1d_time(arr: np.ndarray, target_len: int, mode: str = "edge") -> Tuple[np.ndarray, bool]:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    t = len(arr)
    if t == target_len:
        return arr, False
    if t > target_len:
        return arr[:target_len], False
    pad = target_len - t
    if t <= 1 or mode == "edge":
        return np.pad(arr, (0, pad), mode="edge"), True
    return np.pad(arr, (0, pad), mode="reflect"), True


# =============================================================================
# INPUT STANDARDIZATION & SEGMENTATION
# =============================================================================
def standardize_4ch_dataframe(df: pd.DataFrame, ppg_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ambil Fp1,Fp2,T7/T3,T8/T4 dari DataFrame dan kembalikan urutan internal SR:
    Fp1, Fp2, T7, T8, PPG(optional).
    """
    if df is None or df.empty:
        raise DeploymentError("DataFrame input kosong.")

    col_lookup = {_norm_name(c): c for c in df.columns}

    def pick(candidates: Sequence[str], logical_name: str) -> str:
        for cand in candidates:
            key = _norm_name(cand)
            if key in col_lookup:
                return col_lookup[key]
        raise DeploymentError(
            f"Channel wajib '{logical_name}' tidak ditemukan. Kolom tersedia: {list(df.columns)}"
        )

    fp1 = pick(["Fp1", "EEG_ch1", "EXG Channel 0"], "Fp1")
    fp2 = pick(["Fp2", "EEG_ch2", "EXG Channel 1"], "Fp2")
    t7 = pick(["T7", "T3", "EEG_ch3", "EXG Channel 2"], "T7/T3")
    t8 = pick(["T8", "T4", "EEG_ch4", "EXG Channel 3"], "T8/T4")

    out = pd.DataFrame({
        "Fp1": pd.to_numeric(df[fp1], errors="coerce"),
        "Fp2": pd.to_numeric(df[fp2], errors="coerce"),
        "T7": pd.to_numeric(df[t7], errors="coerce"),
        "T8": pd.to_numeric(df[t8], errors="coerce"),
    })

    if ppg_col is not None and ppg_col in df.columns:
        out["PPG"] = pd.to_numeric(df[ppg_col], errors="coerce")
    else:
        # Deteksi otomatis PPG/Digital
        ppg_candidates = ["PPG", "Digital", "Digital Channel 0 (D11)", "D11", "Marker Channel"]
        for cand in ppg_candidates:
            if _norm_name(cand) in col_lookup:
                out["PPG"] = pd.to_numeric(df[col_lookup[_norm_name(cand)]], errors="coerce")
                break

    if out[["Fp1", "Fp2", "T7", "T8"]].isna().any().any():
        n_bad = int(out[["Fp1", "Fp2", "T7", "T8"]].isna().sum().sum())
        raise DeploymentError(f"EEG input 4 channel memiliki NaN setelah konversi numerik: {n_bad} nilai.")

    if "PPG" in out.columns and out["PPG"].isna().any():
        n_bad = int(out["PPG"].isna().sum())
        print(f"[WARN] PPG mengandung {n_bad} NaN; akan diinterpolasi lalu fill 0.")
        out["PPG"] = out["PPG"].interpolate(limit_direction="both").fillna(0.0)

    return out

def _deploy_make_risk_bar(self, label, risk_value, width=10*cm, height=1.0*cm):
    """
    Flowable bar untuk skor risiko.
    """
    risk_value = float(np.clip(risk_value, 0, 100))

    if risk_value < 40:
        bar_color = colors.HexColor(PDF_COLORS["risk_low"])
    elif risk_value < 70:
        bar_color = colors.HexColor(PDF_COLORS["risk_mid"])
    else:
        bar_color = colors.HexColor(PDF_COLORS["risk_high"])

    d = Drawing(width, 1.4 * cm)

    # label
    d.add(String(
        0,
        1.0 * cm,
        f"{label}",
        fontName="Helvetica-Bold",
        fontSize=10,
        fillColor=colors.HexColor(PDF_COLORS["text"])
    ))

    # background bar
    d.add(Rect(
        0,
        0.3 * cm,
        width,
        0.35 * cm,
        fillColor=colors.HexColor(PDF_COLORS["panel2"]),
        strokeColor=colors.HexColor(PDF_COLORS["line"])
    ))

    # filled bar
    d.add(Rect(
        0,
        0.3 * cm,
        width * (risk_value / 100.0),
        0.35 * cm,
        fillColor=bar_color,
        strokeColor=bar_color
    ))

    # value text
    d.add(String(
        width + 0.15 * cm,
        0.32 * cm,
        f"{risk_value:.1f}%",
        fontName="Helvetica-Bold",
        fontSize=9,
        fillColor=colors.HexColor(PDF_COLORS["text"])
    ))

    return d

def maybe_resample_dataframe(
    df: pd.DataFrame,
    fs: int,
    target_fs: int = FS_EXPECTED,
    allow_resample: bool = False,
) -> pd.DataFrame:
    """Tidak resample diam-diam. Jika fs bukan 250 dan allow_resample=False, raise error."""
    if int(fs) == int(target_fs):
        return df
    if not allow_resample:
        raise DeploymentError(
            f"Sampling rate input = {fs} Hz, tetapi deployment wajib {target_fs} Hz. "
            "Aktifkan allow_resample=True jika memang ingin resampling eksplisit."
        )
    if resample_poly is None:
        raise DeploymentError("scipy.signal.resample_poly tidak tersedia untuk resampling.")

    from math import gcd

    g = gcd(int(target_fs), int(fs))
    up = int(target_fs) // g
    down = int(fs) // g
    out = {}
    for col in df.columns:
        y = resample_poly(df[col].to_numpy(dtype=np.float64), up, down)
        out[col] = y
    min_len = min(len(v) for v in out.values())
    out = {k: v[:min_len] for k, v in out.items()}
    print(f"[INFO] Resampling eksplisit {fs} Hz -> {target_fs} Hz, panjang baru {min_len} sampel.")
    return pd.DataFrame(out)


@dataclass
class SegmentPack:
    segment_index: int
    start_sample: int
    end_sample: int
    eeg4: np.ndarray  # shape (4,T), Fp1,Fp2,T7,T8
    ppg: Optional[np.ndarray]
    padded: bool = False


def segment_4ch_dataframe(
    df4: pd.DataFrame,
    fs: int = FS_EXPECTED,
    segment_samples: int = SEGMENT_SAMPLES,
    pad_short: bool = False,
    overlap: float = 0.0,
) -> List[SegmentPack]:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap harus di rentang [0,1).")
    eeg = df4[SR_INPUT_CHANNELS].to_numpy(dtype=np.float32).T
    ppg = df4["PPG"].to_numpy(dtype=np.float32) if "PPG" in df4.columns else None
    ensure_no_nan_inf(eeg, "EEG 4ch sebelum segmentasi")

    n_total = eeg.shape[1]
    if n_total < segment_samples and not pad_short:
        raise DeploymentError(
            f"Durasi data hanya {n_total/fs:.2f}s ({n_total} sampel), "
            f"kurang dari 120s/30000 sampel. Aktifkan pad_short=True jika ingin padding eksplisit."
        )

    step = max(1, int(segment_samples * (1.0 - overlap)))
    packs: List[SegmentPack] = []

    if n_total < segment_samples and pad_short:
        eeg_pad, padded = pad_2d_time(eeg, segment_samples, mode="reflect")
        ppg_pad = None
        if ppg is not None:
            ppg_pad, _ = pad_1d_time(ppg, segment_samples, mode="edge")
        packs.append(SegmentPack(1, 0, n_total, eeg_pad, ppg_pad, padded=padded))
        return packs

    seg_idx = 1
    start = 0
    while start + segment_samples <= n_total:
        end = start + segment_samples
        packs.append(
            SegmentPack(
                segment_index=seg_idx,
                start_sample=start,
                end_sample=end,
                eeg4=eeg[:, start:end],
                ppg=ppg[start:end] if ppg is not None else None,
                padded=False,
            )
        )
        seg_idx += 1
        start += step

    if not packs:
        raise DeploymentError("Tidak ada segmen valid yang terbentuk.")
    return packs


# =============================================================================
# EEG SUPER-RESOLUTION P16
# =============================================================================
@dataclass
class SRSegmentResult:
    segment_index: int
    sr8_df: pd.DataFrame
    audit: Dict[str, Any]


class EEGSRP16Inference:
    def __init__(
        self,
        ckpt_path: Union[str, Path] = "DEPLOY_EEG_SR_P16/models/p16_sr_model_deploy.pt",
        arch_path: Union[str, Path] = "DEPLOY_EEG_SR_P16/scripts/eeg_sr_hybrid_fusion.py",
        config_path: Optional[Union[str, Path]] = "DEPLOY_EEG_SR_P16/config/p16_sr_deploy_config.json",
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise DeploymentError("PyTorch belum tersedia. Install torch CPU untuk Raspberry Pi.")
        self.ckpt_path = Path(ckpt_path)
        self.arch_path = Path(arch_path)
        self.config_path = Path(config_path) if config_path is not None else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = self._load_json_config(self.config_path)
        self.model = self._load_model()

    @staticmethod
    def _load_json_config(path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_import_arch(self) -> Any:
        # File training punya konfigurasi top-level; env ini mencegah pencarian Kaggle saat import.
        os.environ.setdefault("DATA_ROOT", str(self.arch_path.parent))
        os.environ.setdefault("OUTPUT_ROOT", str(self.arch_path.parent / "_deploy_import_cache"))
        os.environ.setdefault("MODEL_ARCH", "hybrid")
        os.environ.setdefault("CONFIG_ID", "baseline_inputgate")
        os.environ.setdefault("SMOKE_TEST", "1")
        return import_module_from_path("eeg_sr_hybrid_fusion_deploy", self.arch_path)

    def _load_model(self) -> Any:
        """
        Load model SR P16.

        Versi ini mendukung dua format checkpoint:
        1) HybridCNNMASERFusionNet polos:
           alpha_logits, frontal_model.*, temporal_model.*, maser_model.*
        2) C10B wrapper bertingkat:
           base_model.c8_scale_raw
           base_model.base_model.c17_scale_raw
           base_model.base_model.base_model.c5_scale_raw
           base_model.base_model.base_model.base_model.c4_mix_raw
           base_model.base_model.base_model.base_model.base_model.*

        Catatan:
        - File eeg_sr_hybrid_fusion.py tetap dipakai untuk class base model:
          EEGSingleHeadSRNet, MASERLiteNet, HybridCNNMASERFusionNet.
        - Wrapper C10B dibuat di sini agar GUI tetap bisa memuat checkpoint C10B
          tanpa mengganti seluruh file arsitektur training.
        """
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint SR tidak ditemukan: {self.ckpt_path}")
        if not self.arch_path.exists():
            raise FileNotFoundError(f"File arsitektur SR tidak ditemukan: {self.arch_path}")

        srmod = self._safe_import_arch()
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)

        state_key = "hybrid_model_state_dict"
        if state_key not in ckpt:
            raise DeploymentError(
                f"Checkpoint SR tidak memiliki key '{state_key}'. Key tersedia: {list(ckpt.keys())}"
            )

        sd = ckpt[state_key]
        if not isinstance(sd, dict):
            raise DeploymentError(f"Isi checkpoint['{state_key}'] bukan state_dict/dict.")

        sd_keys = list(sd.keys())
        is_c10b_wrapped = any(k.startswith("base_model.base_model.base_model.base_model.base_model.") for k in sd_keys)

        # ============================================================
        # Helper inferensi shape dari state_dict agar cocok dengan model baru.
        # Ini penting untuk C10B karena MASER bisa d_model/depth berbeda dari default.
        # ============================================================
        def _shape(key, default=None):
            if key in sd:
                return tuple(sd[key].shape)
            return default

        def _infer_base_prefix():
            if is_c10b_wrapped:
                return "base_model.base_model.base_model.base_model.base_model."
            return ""

        base_prefix = _infer_base_prefix()

        def _infer_maser_params():
            # MASERLiteNet.patch_embed adalah Linear, weight shape = (d_model, patch_len)
            w_key = base_prefix + "maser_model.patch_embed.weight"
            patch_len = int(ckpt.get("maser_patch_len", self.config.get("maser_patch_len", 250)))
            d_model = int(ckpt.get("maser_d_model", self.config.get("maser_d_model", 64)))
            depth = int(ckpt.get("maser_depth", self.config.get("maser_depth", 2)))
            ffn_mult = int(ckpt.get("maser_ffn_mult", self.config.get("maser_ffn_mult", 2)))

            if w_key in sd:
                w_shape = tuple(sd[w_key].shape)
                if len(w_shape) == 2:
                    d_model = int(w_shape[0])
                    patch_len = int(w_shape[1])

            # Depth dari jumlah lr_blocks.N
            lr_block_ids = []
            prefix = base_prefix + "maser_model.lr_blocks."
            for k in sd_keys:
                if k.startswith(prefix):
                    parts = k[len(prefix):].split(".")
                    if parts and parts[0].isdigit():
                        lr_block_ids.append(int(parts[0]))
            if lr_block_ids:
                depth = max(lr_block_ids) + 1

            # ffn_mult dari shape ffn.0.weight = (ffn_mult*d_model, d_model)
            ffn_key = base_prefix + "maser_model.lr_blocks.0.ffn.0.weight"
            if ffn_key in sd:
                ffn_shape = tuple(sd[ffn_key].shape)
                if len(ffn_shape) == 2 and d_model > 0:
                    ffn_mult = max(1, int(round(ffn_shape[0] / d_model)))

            return patch_len, d_model, depth, ffn_mult

        maser_patch_len, maser_d_model, maser_depth, maser_ffn_mult = _infer_maser_params()

        # ============================================================
        # Bangun base HybridCNNMASERFusionNet
        # ============================================================
        frontal_model = srmod.EEGSingleHeadSRNet(
            in_ch=4,
            out_ch=2,
            dropout=float(getattr(srmod, "FRONTAL_DROPOUT", 0.10)),
            alpha_init=float(getattr(srmod, "FRONTAL_ALPHA_INIT", 0.85)),
            alpha_min=float(getattr(srmod, "FRONTAL_ALPHA_MIN", 0.60)),
            alpha_max=float(getattr(srmod, "FRONTAL_ALPHA_MAX", 0.97)),
            use_input_gate=bool(getattr(srmod, "USE_INPUT_GATE", True)),
            input_gate_init=getattr(srmod, "FRONTAL_INPUT_GATE_INIT", [1.10, 1.10, 0.90, 0.90]),
        ).to(self.device)

        temporal_model = srmod.EEGSingleHeadSRNet(
            in_ch=4,
            out_ch=2,
            dropout=float(getattr(srmod, "TEMPORAL_DROPOUT", 0.05)),
            alpha_init=float(getattr(srmod, "TEMPORAL_ALPHA_INIT", 0.70)),
            alpha_min=float(getattr(srmod, "TEMPORAL_ALPHA_MIN", 0.35)),
            alpha_max=float(getattr(srmod, "TEMPORAL_ALPHA_MAX", 0.97)),
            use_input_gate=bool(getattr(srmod, "USE_INPUT_GATE", True)),
            input_gate_init=getattr(srmod, "TEMPORAL_INPUT_GATE_INIT", [0.85, 0.85, 1.15, 1.15]),
        ).to(self.device)

        maser_model = srmod.MASERLiteNet(
            in_ch=4,
            out_ch=4,
            total_ch=8,
            patch_len=maser_patch_len,
            d_model=maser_d_model,
            depth=maser_depth,
            dropout=float(ckpt.get("maser_dropout", self.config.get("maser_dropout", 0.08))),
            ffn_mult=maser_ffn_mult,
        ).to(self.device)

        base_model = srmod.HybridCNNMASERFusionNet(
            frontal_model=frontal_model,
            temporal_model=temporal_model,
            maser_model=maser_model,
            mode=ckpt.get("hybrid_mode", self.config.get("hybrid_mode", "learned_alpha")),
            init_old_weights=ckpt.get(
                "hybrid_init_old_weights",
                self.config.get("hybrid_init_old_weights", [0.7202764, 0.8898062, 0.4613816, 0.4711465]),
            ),
            freeze_backbones=bool(ckpt.get("hybrid_freeze_backbones", self.config.get("hybrid_freeze_backbones", True))),
            residual_hidden=int(ckpt.get("hybrid_residual_hidden", self.config.get("hybrid_residual_hidden", 32))),
            residual_kernel=int(ckpt.get("hybrid_residual_kernel", self.config.get("hybrid_residual_kernel", 15))),
            residual_scale_init=float(ckpt.get("hybrid_residual_scale_init", self.config.get("hybrid_residual_scale_init", 0.05))),
            dropout=float(ckpt.get("hybrid_dropout", self.config.get("hybrid_dropout", 0.05))),
        ).to(self.device)

        # ============================================================
        # C10B wrapper definitions.
        # Nama atribut dibuat supaya key state_dict cocok:
        # base_model.c8_scale_raw, base_model.base_model.c17_scale_raw, dst.
        # ============================================================
        import torch.nn as nn
        import torch.nn.functional as F

        def _make_raw_param(full_key: str, default_shape=(1, 4, 1)):
            sh = _shape(full_key, default_shape)
            return nn.Parameter(torch.zeros(sh, dtype=torch.float32))

        def _as_4ch_broadcast(raw, apply_sigmoid: bool = True, scale_max: Optional[float] = None):
            """
            Paksa parameter checkpoint menjadi bentuk broadcast aman untuk sinyal (B,4,T).

            Masalah sebelumnya:
            - c4_mix_raw bisa tersimpan sebagai (1,1,4)
            - PyTorch menganggap dimensi terakhir = waktu, sehingga bentrok dengan T=30000
            - Bentuk aman untuk parameter per-channel adalah (1,4,1)

            Fungsi ini sengaja agresif:
            - scalar      -> (1,1,1)
            - 4 elemen    -> (1,4,1)
            - >4 elemen   -> ambil 4 elemen pertama sebagai gain channel
            """
            val = raw
            if apply_sigmoid:
                val = torch.sigmoid(val)
            if scale_max is not None:
                val = val * float(scale_max)

            flat = val.reshape(-1)

            if flat.numel() == 1:
                return flat.reshape(1, 1, 1)

            if flat.numel() >= 4:
                return flat[:4].reshape(1, 4, 1)

            # Fallback aman kalau elemennya 2/3: pakai rata-rata sebagai scalar gain.
            return flat.mean().reshape(1, 1, 1)

        def _scale_from_raw(raw, scale_max: float):
            return _as_4ch_broadcast(raw, apply_sigmoid=True, scale_max=scale_max)

        def _as_channel_gain(raw):
            return _as_4ch_broadcast(raw, apply_sigmoid=True, scale_max=None)

        def _conv_from_sd(full_weight_key: str):
            if full_weight_key not in sd:
                raise DeploymentError(f"Key conv tidak ditemukan pada checkpoint: {full_weight_key}")
            w_shape = tuple(sd[full_weight_key].shape)
            if len(w_shape) != 3:
                raise DeploymentError(f"Shape conv tidak valid untuk {full_weight_key}: {w_shape}")
            out_ch, in_ch, kernel = map(int, w_shape)
            bias_key = full_weight_key.replace(".weight", ".bias")
            use_bias = bias_key in sd
            return nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=use_bias)

        class _C4BandMixWrapper(nn.Module):
            """
            Wrapper C4.
            State_dict wajib punya c4_mix_raw. Forward dibuat konservatif:
            output base tetap dominan, ditambah campuran residual berbasis smoothing ringan.
            """
            def __init__(self, base_model, full_prefix: str):
                super().__init__()
                self.base_model = base_model
                self.c4_mix_raw = _make_raw_param(full_prefix + "c4_mix_raw")

            def forward(self, x):
                y = self.base_model(x)
                # Residual band-mix ringan: high-frequency residual dari output sendiri.
                # Tidak mengubah shape, tetap (B,4,T).
                mix = _as_channel_gain(self.c4_mix_raw)
                y_smooth = F.avg_pool1d(y, kernel_size=9, stride=1, padding=4)

                # Debug ringan satu kali; hapus boleh setelah berhasil.
                if not hasattr(self, "_printed_c4_shape"):
                    print(
                        f"[SR DEBUG C4] y={tuple(y.shape)} | "
                        f"c4_mix_raw={tuple(self.c4_mix_raw.shape)} | "
                        f"mix={tuple(mix.shape)}"
                    )
                    self._printed_c4_shape = True

                # y shape = (B,4,T), mix shape wajib (1,4,1)
                y_mix = y + mix * 0.10 * (y - y_smooth)
                return y_mix

        class _C5TemporalResidualWrapper(nn.Module):
            def __init__(self, base_model, full_prefix: str, scale_max: float = 0.35):
                super().__init__()
                self.base_model = base_model
                self.c5_scale_raw = _make_raw_param(full_prefix + "c5_scale_raw", default_shape=())
                self.scale_max = float(scale_max)

                self.res_net = nn.Sequential(
                    _conv_from_sd(full_prefix + "res_net.0.weight"),
                    nn.GELU(),
                    nn.Dropout(float(self_config_get("architecture.C5.dropout", 0.05))),
                    _conv_from_sd(full_prefix + "res_net.3.weight"),
                    nn.GELU(),
                    nn.Dropout(float(self_config_get("architecture.C5.dropout", 0.05))),
                    _conv_from_sd(full_prefix + "res_net.6.weight"),
                )

            def forward(self, x):
                y = self.base_model(x)
                scale = _scale_from_raw(self.c5_scale_raw, self.scale_max)
                return y + scale * self.res_net(y)

        class _C17OutputScaleWrapper(nn.Module):
            def __init__(self, base_model, full_prefix: str, scale_max: float = 0.50):
                super().__init__()
                self.base_model = base_model
                self.c17_scale_raw = _make_raw_param(full_prefix + "c17_scale_raw")
                self.scale_max = float(scale_max)

            def forward(self, x):
                y = self.base_model(x)
                scale = _scale_from_raw(self.c17_scale_raw, self.scale_max)
                return y * (1.0 + scale)

        class _C8HFResidualWrapper(nn.Module):
            def __init__(self, base_model, full_prefix: str, scale_max: float = 0.40):
                super().__init__()
                self.base_model = base_model
                self.c8_scale_raw = _make_raw_param(full_prefix + "c8_scale_raw", default_shape=())
                self.scale_max = float(scale_max)

                self.c8_branch = nn.Sequential(
                    _conv_from_sd(full_prefix + "c8_branch.0.weight"),
                    nn.GELU(),
                    nn.Dropout(float(self_config_get("architecture.C8.dropout", 0.0))),
                    _conv_from_sd(full_prefix + "c8_branch.3.weight"),
                    nn.GELU(),
                    nn.Dropout(float(self_config_get("architecture.C8.dropout", 0.0))),
                    _conv_from_sd(full_prefix + "c8_branch.6.weight"),
                    nn.GELU(),
                    _conv_from_sd(full_prefix + "c8_branch.8.weight"),
                )

            def forward(self, x):
                y = self.base_model(x)
                scale = _scale_from_raw(self.c8_scale_raw, self.scale_max)

                # C8 branch checkpoint punya weight [64, 8, 11],
                # jadi inputnya harus 8 channel.
                # Gabungkan input fisik 4ch + output SR 4ch.
                branch_in = torch.cat([x, y], dim=1)  # shape: (B, 8, T)

                return y + scale * self.c8_branch(branch_in)

        class _C14MultibandPostGainWrapper(nn.Module):
            """
            Wrapper C14 tidak punya parameter trainable, jadi di state_dict hanya muncul
            prefix base_model.*. Gain diambil dari p16_sr_deploy_config.json.
            """
            def __init__(self, base_model, fs: float = 250.0, gains: Optional[Dict[str, float]] = None):
                super().__init__()
                self.base_model = base_model
                self.fs = float(fs)
                gains = gains or {}
                self.delta_gain = float(gains.get("delta_gain", 1.0))
                self.theta_gain = float(gains.get("theta_gain", 1.0))
                self.alpha_gain = float(gains.get("alpha_gain", 1.0))
                self.beta_gain = float(gains.get("beta_gain", 1.0))

            def _apply_fft_gain(self, y):
                # y: (B,4,T)
                try:
                    t = y.shape[-1]
                    freq = torch.fft.rfftfreq(t, d=1.0 / self.fs).to(y.device)
                    gain = torch.ones_like(freq, dtype=y.dtype, device=y.device)
                    gain = torch.where((freq >= 1.0) & (freq < 4.0), gain * self.delta_gain, gain)
                    gain = torch.where((freq >= 4.0) & (freq < 8.0), gain * self.theta_gain, gain)
                    gain = torch.where((freq >= 8.0) & (freq < 13.0), gain * self.alpha_gain, gain)
                    gain = torch.where((freq >= 13.0) & (freq <= 30.0), gain * self.beta_gain, gain)
                    Y = torch.fft.rfft(y, dim=-1)
                    return torch.fft.irfft(Y * gain.view(1, 1, -1), n=t, dim=-1)
                except Exception:
                    # Fallback aman kalau FFT bermasalah di environment tertentu.
                    return y

            def forward(self, x):
                y = self.base_model(x)
                return self._apply_fft_gain(y)

        def self_config_get(path: str, default=None):
            cur = self.config
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur

        # ============================================================
        # Susun model sesuai checkpoint.
        # Jika checkpoint C10B wrapped, susunan module sengaja dibuat:
        # C14(base_model=C8(base_model=C17(base_model=C5(base_model=C4(base_model=Hybrid)))))
        # sehingga key state_dict cocok 1:1.
        # ============================================================
        if is_c10b_wrapped:
            print("[SR] Detected C10B wrapped checkpoint. Building wrapper stack C4 -> C5 -> C17 -> C8 -> C14.")

            arch_cfg = self.config.get("architecture", {}) if isinstance(self.config, dict) else {}
            c5_cfg = arch_cfg.get("C5", {}) if isinstance(arch_cfg, dict) else {}
            c8_cfg = arch_cfg.get("C8", {}) if isinstance(arch_cfg, dict) else {}
            c14_cfg = arch_cfg.get("C14", {}) if isinstance(arch_cfg, dict) else {}

            model = _C4BandMixWrapper(
                base_model,
                full_prefix="base_model.base_model.base_model.base_model.",
            ).to(self.device)
            model = _C5TemporalResidualWrapper(
                model,
                full_prefix="base_model.base_model.base_model.",
                scale_max=float(c5_cfg.get("scale_max", 0.35)),
            ).to(self.device)
            model = _C17OutputScaleWrapper(
                model,
                full_prefix="base_model.base_model.",
                scale_max=float(arch_cfg.get("C17", {}).get("scale_max", 0.50)) if isinstance(arch_cfg.get("C17", {}), dict) else 0.50,
            ).to(self.device)
            model = _C8HFResidualWrapper(
                model,
                full_prefix="base_model.",
                scale_max=float(c8_cfg.get("scale_max", 0.40)),
            ).to(self.device)
            model = _C14MultibandPostGainWrapper(
                model,
                fs=float(self.config.get("sampling_rate_hz", FS_EXPECTED)) if isinstance(self.config, dict) else FS_EXPECTED,
                gains=c14_cfg,
            ).to(self.device)
        else:
            model = base_model

        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            # Tambahkan pesan yang lebih informatif untuk debugging deployment.
            example_keys = "\n".join(sd_keys[:30])
            raise RuntimeError(
                "Gagal load state_dict SR. Ini biasanya berarti arsitektur GUI belum sama dengan checkpoint.\n"
                f"is_c10b_wrapped={is_c10b_wrapped}\n"
                f"maser_patch_len={maser_patch_len}, maser_d_model={maser_d_model}, "
                f"maser_depth={maser_depth}, maser_ffn_mult={maser_ffn_mult}\n"
                f"Contoh key checkpoint:\n{example_keys}\n\n"
                f"Detail PyTorch:\n{e}"
            )

        model.eval()
        print(f"[SR] Loaded P16 model on {self.device}: {self.ckpt_path}")
        print(
            f"[SR] MASER params: patch_len={maser_patch_len}, "
            f"d_model={maser_d_model}, depth={maser_depth}, ffn_mult={maser_ffn_mult}"
        )
        if is_c10b_wrapped:
            print("[SR] C10B wrapper stack active.")
        if hasattr(model, "get_alpha"):
            with torch.no_grad():
                alpha = model.get_alpha().detach().cpu().view(-1).numpy()
            print(f"[SR] Hybrid alpha: {alpha}")
        elif is_c10b_wrapped:
            try:
                # Ambil alpha dari base Hybrid terdalam untuk audit.
                inner = model.base_model.base_model.base_model.base_model.base_model
                if hasattr(inner, "get_alpha"):
                    with torch.no_grad():
                        alpha = inner.get_alpha().detach().cpu().view(-1).numpy()
                    print(f"[SR] Hybrid alpha inner: {alpha}")
            except Exception:
                pass
        return model

    def infer_segment(self, eeg4: np.ndarray, ppg: Optional[np.ndarray] = None, alias_for_features: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        eeg4 = np.asarray(eeg4, dtype=np.float32)
        if eeg4.shape != (4, SEGMENT_SAMPLES):
            raise DeploymentError(f"Input SR wajib shape (4,{SEGMENT_SAMPLES}), diterima {eeg4.shape}.")
        ensure_no_nan_inf(eeg4, "EEG 4ch input SR")

        x_norm, median, iqr = robust_scale_by_input(eeg4)
        ensure_no_nan_inf(x_norm, "EEG 4ch normalized SR")

        x_tensor = torch.from_numpy(x_norm[None, :, :]).to(self.device)
        with torch.no_grad():
            y = self.model(x_tensor)
            if isinstance(y, (tuple, list)):
                y = y[0]
        if tuple(y.shape) != (1, 4, SEGMENT_SAMPLES):
            raise DeploymentError(f"Output model SR harus (1,4,{SEGMENT_SAMPLES}), diterima {tuple(y.shape)}.")

        y_norm = y.detach().cpu().numpy()[0]
        y_raw = inverse_tanh_scale(y_norm, median, iqr)
        ensure_no_nan_inf(y_raw, "Virtual EEG hasil inverse SR")

        # Output model: F3,F4,P3,P4. Gabung: Fp1,Fp2,F3,F4,T7,T8,P3,P4.
        sr8 = np.vstack([
            eeg4[0],
            eeg4[1],
            y_raw[0],
            y_raw[1],
            eeg4[2],
            eeg4[3],
            y_raw[2],
            y_raw[3],
        ])

        out_cols = FEATURE_CHANNELS_T3T4 if alias_for_features else SR_OUTPUT_CHANNELS_T7T8
        df = pd.DataFrame(sr8.T, columns=out_cols)
        if ppg is not None:
            ppg_arr = np.asarray(ppg, dtype=np.float32).reshape(-1)
            if len(ppg_arr) != SEGMENT_SAMPLES:
                raise DeploymentError(f"PPG harus panjang {SEGMENT_SAMPLES}, diterima {len(ppg_arr)}")
            df["PPG"] = ppg_arr

        audit = {
            "median": median,
            "iqr": iqr,
            "x_norm_min": float(np.min(x_norm)),
            "x_norm_max": float(np.max(x_norm)),
            "y_norm_min": float(np.min(y_norm)),
            "y_norm_max": float(np.max(y_norm)),
            "y_raw_min": float(np.min(y_raw)),
            "y_raw_max": float(np.max(y_raw)),
            "sr_output_columns": list(df.columns),
        }
        # Validasi kasar agar user tidak tidak sengaja meneruskan output tanh-normalized.
        if np.nanmax(np.abs(y_raw)) <= 1.5 and iqr > 2.0:
            audit["warning"] = (
                "Virtual channel tampak sangat kecil walaupun IQR input besar. "
                "Cek apakah inverse_tanh_scale sudah benar dan satuan input konsisten."
            )
        return df, audit

    def infer_segments(self, segments: Sequence[SegmentPack], alias_for_features: bool = False) -> List[SRSegmentResult]:
        results: List[SRSegmentResult] = []
        for pack in segments:
            df8, audit = self.infer_segment(pack.eeg4, pack.ppg, alias_for_features=alias_for_features)
            audit.update({
                "segment_index": pack.segment_index,
                "start_sample": pack.start_sample,
                "end_sample": pack.end_sample,
                "padded": pack.padded,
            })
            results.append(SRSegmentResult(pack.segment_index, df8, audit))
        return results


# =============================================================================
# FEATURE EXTRACTION WRAPPER — menggunakan olah_data_lengkap.py
# =============================================================================
class FeatureExtractorOlahData:
    def __init__(self, olah_data_path: Union[str, Path] = "olah_data_lengkap.py", fs: int = FS_EXPECTED) -> None:
        self.olah_data_path = Path(olah_data_path)
        if not self.olah_data_path.exists():
            raise FileNotFoundError(f"File olah_data_lengkap.py tidak ditemukan: {self.olah_data_path}")
        self.fs = int(fs)
        self.mod = import_module_from_path("olah_data_lengkap_deploy", self.olah_data_path)
        # Paksa konstanta channel sesuai deployment fitur lama.
        self.mod.FS = self.fs
        self.mod.KEEP_SEC = float(SEGMENT_SECONDS)
        self.mod.KEEP_SAMP = int(SEGMENT_SAMPLES)
        self.mod.CH_NAMES = list(FEATURE_CHANNELS_T3T4)

    @staticmethod
    def _to_feature_channel_df(sr8_df: pd.DataFrame) -> pd.DataFrame:
        col_lookup = {_norm_name(c): c for c in sr8_df.columns}
        rename_map = {}
        if _norm_name("T7") in col_lookup and _norm_name("T3") not in col_lookup:
            rename_map[col_lookup[_norm_name("T7")]] = "T3"
        if _norm_name("T8") in col_lookup and _norm_name("T4") not in col_lookup:
            rename_map[col_lookup[_norm_name("T8")]] = "T4"
        df = sr8_df.rename(columns=rename_map).copy()

        col_lookup2 = {_norm_name(c): c for c in df.columns}
        missing = [c for c in FEATURE_CHANNELS_T3T4 if _norm_name(c) not in col_lookup2]
        if missing:
            raise DeploymentError(f"Kolom 8ch untuk ekstraksi fitur tidak lengkap. Missing: {missing}. Kolom: {list(df.columns)}")
        ordered = [col_lookup2[_norm_name(c)] for c in FEATURE_CHANNELS_T3T4]
        out = df[ordered].copy()
        out.columns = FEATURE_CHANNELS_T3T4
        if _norm_name("PPG") in col_lookup2:
            out["PPG"] = df[col_lookup2[_norm_name("PPG")]].to_numpy()
        else:
            out["PPG"] = np.zeros(len(out), dtype=np.float32)
        return out

    def extract_features(self, sr8_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        df = self._to_feature_channel_df(sr8_df)
        if len(df) != SEGMENT_SAMPLES:
            raise DeploymentError(f"Ekstraksi fitur wajib 120s/30000 sampel, diterima {len(df)} sampel.")

        clean_eeg = df[FEATURE_CHANNELS_T3T4].to_numpy(dtype=np.float64).T
        ppg = df["PPG"].to_numpy(dtype=np.float64)
        ensure_no_nan_inf(clean_eeg, "EEG 8ch sebelum ekstraksi fitur")

        # PPG boleh digital/analog, deteksi peak mengikuti olah_data_lengkap.py.
        beats = self.mod.detect_ppg_peaks(ppg, self.fs)
        band_signals = self.mod.extract_bands(clean_eeg, self.fs)
        eeg_features = self.mod.extract_eeg_features(clean_eeg, band_signals, beats, self.fs)
        ppg_features = self.mod.extract_ppg_features(ppg, self.fs)
        features: Dict[str, float] = {}
        features.update(eeg_features)
        features.update(ppg_features)

        missing_by_label = {
            label: [f for f in feats if f not in features]
            for label, feats in REQUIRED_FEATURES_BY_LABEL.items()
        }
        audit = {
            "n_features": len(features),
            "n_ppg_peaks": int(len(beats)),
            "missing_required_features": missing_by_label,
            "nan_features": [k for k, v in features.items() if not np.isfinite(v)],
        }
        return features, audit


# =============================================================================
# MANUAL NumPy cuML SVM BOUNDARY INFERENCE
# =============================================================================
class CumLSVMBoundaryModel:
    def __init__(self, artifact_path: Union[str, Path], risk_temperature: Optional[float] = None) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact SVM tidak ditemukan: {self.artifact_path}")
        with open(self.artifact_path, "rb") as f:
            self.artifact = pickle.load(f)
        self.label = str(self.artifact.get("label", self.artifact_path.stem)).lower()
        self.feature_order = list(self.artifact.get("feature_order", []))
        if not self.feature_order:
            raise DeploymentError(f"Artifact {self.artifact_path} tidak memiliki feature_order.")
        self.risk_temperature = risk_temperature

    def _imputer_statistics(self) -> np.ndarray:
        imp = self.artifact.get("imputer", {})
        stats = imp.get("statistics", None) if isinstance(imp, dict) else None
        if stats is None:
            stats = self.artifact.get("imputer_statistics", None)
        if stats is None:
            return np.zeros(len(self.feature_order), dtype=np.float64)
        if isinstance(stats, Mapping):
            return np.array([float(stats.get(f, 0.0)) for f in self.feature_order], dtype=np.float64)
        arr = np.asarray(stats, dtype=np.float64).reshape(-1)
        if len(arr) != len(self.feature_order):
            raise DeploymentError(
                f"Panjang imputer statistics ({len(arr)}) tidak sama dengan feature_order ({len(self.feature_order)})."
            )
        return arr

    def _feature_vector(self, features: Mapping[str, Any]) -> Tuple[np.ndarray, List[str], List[str]]:
        missing = [f for f in self.feature_order if f not in features]
        if missing:
            raise DeploymentError(
                f"Fitur untuk model {self.label} tidak lengkap. Missing: {missing}. "
                "Cek mapping T7/T8 -> T3/T4 dan fungsi ekstraksi fitur."
            )
        x = np.array([features[f] for f in self.feature_order], dtype=np.float64)
        nan_mask = ~np.isfinite(x)
        nan_features = [self.feature_order[i] for i in np.where(nan_mask)[0]]
        if nan_mask.any():
            stats = self._imputer_statistics()
            x[nan_mask] = stats[nan_mask]
        return x.reshape(1, -1), missing, nan_features

    @staticmethod
    def _get_from_scaler_obj(scaler: Any, names: Sequence[str]) -> Optional[np.ndarray]:
        for name in names:
            if isinstance(scaler, Mapping) and name in scaler:
                return np.asarray(scaler[name], dtype=np.float64)
            if hasattr(scaler, name):
                return np.asarray(getattr(scaler, name), dtype=np.float64)
        return None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        scaler = self.artifact.get("scaler", None)
        if scaler is None:
            return x

        scaler_type = None
        if isinstance(scaler, Mapping):
            scaler_type = scaler.get("type") or scaler.get("method") or scaler.get("scaler_type") or scaler.get("name")
        if scaler_type is None:
            cls_name = scaler.__class__.__name__.lower()
            if "robust" in cls_name:
                scaler_type = "robust"
            elif "standard" in cls_name:
                scaler_type = "standard"
            elif "minmax" in cls_name:
                scaler_type = "minmax"
            else:
                scaler_type = str(self.artifact.get("scaler_type", "none"))
        scaler_type = str(scaler_type).lower()

        if scaler_type in ["none", "null", "identity"]:
            return x

        if "robust" in scaler_type:
            center = self._get_from_scaler_obj(scaler, ["center_", "center", "median", "median_"])
            scale = self._get_from_scaler_obj(scaler, ["scale_", "scale", "iqr", "iqr_"])
            if center is None or scale is None:
                raise DeploymentError("Scaler robust tidak memiliki center/scale.")
            return (x - center.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)

        if "standard" in scaler_type:
            mean = self._get_from_scaler_obj(scaler, ["mean_", "mean"])
            scale = self._get_from_scaler_obj(scaler, ["scale_", "scale", "std", "std_"])
            if mean is None or scale is None:
                raise DeploymentError("Scaler standard tidak memiliki mean/scale.")
            return (x - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)

        if "minmax" in scaler_type or "min_max" in scaler_type:
            scale = self._get_from_scaler_obj(scaler, ["scale_", "scale"])
            minv = self._get_from_scaler_obj(scaler, ["min_", "min"])
            if scale is None or minv is None:
                raise DeploymentError("Scaler minmax tidak memiliki scale/min.")
            return x * scale.reshape(1, -1) + minv.reshape(1, -1)

        raise DeploymentError(f"Jenis scaler tidak dikenali: {scaler_type}")

    def _kernel_params(self, x_scaled: np.ndarray) -> Tuple[str, float, int, float]:
        cfg = self.artifact.get("config", {}) or {}
        kernel = cfg.get("kernel") or self.artifact.get("kernel") or self.artifact.get("kernel_type") or "rbf"
        gamma = cfg.get("gamma") or self.artifact.get("gamma") or self.artifact.get("kernel_gamma")
        degree = cfg.get("degree") or self.artifact.get("degree") or self.artifact.get("kernel_degree") or 3
        coef0 = cfg.get("coef0") or self.artifact.get("coef0") or self.artifact.get("kernel_coef0") or 0.0

        kernel = str(kernel).lower().strip()

        # Gamma: cuML artifact kadang menyimpan string auto/scale/none.
        if gamma is None:
            gamma = 1.0 / x_scaled.shape[1]
        if isinstance(gamma, str):
            g = gamma.lower().strip()
            if g in ["none", "null", "nan", ""]:
                gamma = 1.0 / x_scaled.shape[1]
            elif g == "auto":
                gamma = 1.0 / x_scaled.shape[1]
            elif g == "scale":
                # Fallback agar tidak crash. Idealnya gamma numerik diexport dari Kaggle.
                gamma = 1.0 / (x_scaled.shape[1] * float(np.var(x_scaled) + 1e-12))
                print("[WARN] gamma='scale' dihitung dari input inference. Lebih aman export gamma numerik dari Kaggle.")
            else:
                gamma = float(g)

        # Degree tidak dipakai untuk RBF/linear, tetapi artifact bisa berisi 'none'.
        if isinstance(degree, str):
            d = degree.lower().strip()
            if d in ["none", "null", "nan", ""]:
                degree = 3
            else:
                degree = int(float(d))
        else:
            degree = int(degree)

        # Coef0 tidak dipakai untuk RBF/linear, tetapi artifact bisa berisi 'none'.
        if isinstance(coef0, str):
            c = coef0.lower().strip()
            if c in ["none", "null", "nan", ""]:
                coef0 = 0.0
            else:
                coef0 = float(c)
        else:
            coef0 = float(coef0)

        return kernel, float(gamma), int(degree), float(coef0)

    @staticmethod
    def _compute_kernel(x: np.ndarray, sv: np.ndarray, kernel: str, gamma: float, degree: int, coef0: float) -> np.ndarray:
        if kernel == "linear":
            return x @ sv.T
        if kernel == "rbf":
            x_norm = np.sum(x * x, axis=1, keepdims=True)
            sv_norm = np.sum(sv * sv, axis=1, keepdims=True).T
            dist2 = np.maximum(x_norm + sv_norm - 2.0 * (x @ sv.T), 0.0)
            return np.exp(-gamma * dist2)
        if kernel in ["poly", "polynomial"]:
            return (gamma * (x @ sv.T) + coef0) ** degree
        if kernel == "sigmoid":
            return np.tanh(gamma * (x @ sv.T) + coef0)
        raise DeploymentError(f"Kernel SVM tidak didukung: {kernel}")

    def decision_score(self, features: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
        x_raw, _, nan_features = self._feature_vector(features)
        x_scaled = self._scale(x_raw)
        ensure_no_nan_inf(x_scaled, f"X scaled untuk {self.label}")

        sv = np.asarray(self.artifact.get("support_vectors"), dtype=np.float64)
        dual = np.asarray(self.artifact.get("dual_coef"), dtype=np.float64).reshape(-1)
        intercept = np.asarray(self.artifact.get("intercept", 0.0), dtype=np.float64).reshape(-1)[0]
        if sv.ndim != 2:
            raise DeploymentError(f"support_vectors harus 2D, diterima shape {sv.shape}")
        if len(dual) != sv.shape[0]:
            raise DeploymentError(f"dual_coef length {len(dual)} tidak cocok dengan n_support_vectors {sv.shape[0]}")
        if sv.shape[1] != x_scaled.shape[1]:
            raise DeploymentError(f"Dimensi SV {sv.shape[1]} tidak cocok dengan fitur {x_scaled.shape[1]}")

        kernel, gamma, degree, coef0 = self._kernel_params(x_scaled)
        k = self._compute_kernel(x_scaled, sv, kernel, gamma, degree, coef0)
        decision_arr = np.asarray(k @ dual + intercept, dtype=np.float64).reshape(-1)

        if decision_arr.size != 1:
            raise DeploymentError(
                f"Output decision SVM tidak scalar. "
                f"shape k={k.shape}, shape dual={dual.shape}, "
                f"shape decision={decision_arr.shape}"
            )

        decision = float(decision_arr[0])
        decision *= float(self.artifact.get("decision_sign", 1.0))


        debug = {
            "kernel": kernel,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "intercept": float(intercept),
            "nan_features_imputed": nan_features,
            "n_features": len(self.feature_order),
        }
        return decision, debug

    def _raw_probability(self, decision: float) -> Optional[float]:
        A = self.artifact.get("platt_A", None)
        B = self.artifact.get("platt_B", None)
        if A is None or B is None:
            return None
        return float(safe_sigmoid(float(A) * decision + float(B)))

    def _threshold_info(self) -> Tuple[str, str, float]:
        mode = str(self.artifact.get("deployment_threshold_mode", "decision_score")).lower()
        direction = str(self.artifact.get("deployment_threshold_direction", ">=")).strip()
        threshold = self.artifact.get("deployment_decision_threshold", None)
        if threshold is None:
            threshold = self.artifact.get("segment_threshold", None)
        if threshold is None:
            threshold = 0.5 if mode == "probability" else 0.0
        return mode, direction, float(threshold)

    def _risk_temperature_auto(self, threshold: float) -> float:
        """
        Temperature untuk UI risk score berbasis margin decision-threshold.
        Ini bukan probabilitas klinis. Bila artifact menyediakan risk_temperature, nilai itu dipakai.
        Bila tidak, dipilih otomatis agar margin kecil pada boundary poly tidak selalu tampil 50%.
        """
        for key in ["risk_temperature", "deployment_risk_temperature", "margin_temperature"]:
            val = self.artifact.get(key, None)
            if val is not None:
                try:
                    return max(float(val), 1e-12)
                except Exception:
                    pass
        th = abs(float(threshold))
        if th < 1e-5:
            return 1e-7
        if th < 1.0:
            return 0.10
        return 0.50

    def predict(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        decision, debug = self.decision_score(features)
        prob = self._raw_probability(decision)
        mode, direction, threshold = self._threshold_info()

        # Margin utama UI HARUS berbasis decision score terhadap threshold, bukan raw probability.
        if direction == "<=":
            decision_pred = int(decision <= threshold)
            margin = threshold - decision
        elif direction == ">=":
            decision_pred = int(decision >= threshold)
            margin = decision - threshold
        else:
            raise DeploymentError(f"deployment_threshold_direction tidak valid: {direction}")

        # Keputusan akhir tetap mengikuti mode artifact. Namun risk score tetap memakai margin decision.
        if mode == "decision_score":
            pred = decision_pred
        elif mode in ["probability", "prob", "raw_probability"]:
            if prob is None:
                raise DeploymentError(f"Mode threshold probability, tetapi platt_A/platt_B tidak tersedia untuk {self.label}.")
            pred = int(prob >= threshold)
            print("[WARN] Mode artifact probability. Status memakai probability threshold, tetapi risk UI tetap memakai margin decision-threshold.")
        else:
            raise DeploymentError(f"deployment_threshold_mode tidak valid: {mode}")

        temp = self.risk_temperature
        if temp is None:
            temp = self._risk_temperature_auto(threshold)
        temp = max(float(temp), 1e-12)
        risk_score = float(safe_sigmoid(margin / temp) * 100.0)

        return {
            "label": self.label,
            "prediction": pred,
            "status": "Tidak normal" if pred == 1 else "Normal",
            "decision_score": float(decision),
            "prob_positive_raw": prob,
            "deployment_threshold_mode": mode,
            "deployment_threshold_direction": direction,
            "deployment_threshold": float(threshold),
            "margin_to_threshold": float(margin),
            "risk_temperature": float(temp),
            "risk_score_model": risk_score,
            "debug": debug,
        }


# =============================================================================
# FULL PIPELINE
# =============================================================================
@dataclass
class PipelineSegmentOutput:
    segment_index: int
    start_sample: int
    end_sample: int
    predictions: Dict[str, Dict[str, Any]]
    feature_audit: Dict[str, Any]
    sr_audit: Dict[str, Any]
    sr8_df: pd.DataFrame = field(repr=False)
    features: Dict[str, float] = field(repr=False)


class DeploymentPipeline:
    def __init__(
        self,
        sr_ckpt_path: Union[str, Path] = "DEPLOY_EEG_SR_P16/models/p16_sr_model_deploy.pt",
        sr_arch_path: Union[str, Path] = "DEPLOY_EEG_SR_P16/scripts/eeg_sr_hybrid_fusion.py",
        sr_config_path: Optional[Union[str, Path]] = "DEPLOY_EEG_SR_P16/config/p16_sr_deploy_config.json",
        olah_data_path: Union[str, Path] = "olah_data_lengkap.py",
        svm_artifact_paths: Optional[Mapping[str, Union[str, Path]]] = None,
        risk_temperatures: Optional[Mapping[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.sr = EEGSRP16Inference(sr_ckpt_path, sr_arch_path, sr_config_path, device=device)
        self.feature_extractor = FeatureExtractorOlahData(olah_data_path, fs=FS_EXPECTED)
        paths = dict(DEFAULT_MODEL_PATHS)
        if svm_artifact_paths:
            paths.update({k: Path(v) for k, v in svm_artifact_paths.items()})
        self.models = {
            label: CumLSVMBoundaryModel(path, risk_temperature=(risk_temperatures or {}).get(label))
            for label, path in paths.items()
        }

    def run_clean_4ch_dataframe(
        self,
        df: pd.DataFrame,
        fs: int = FS_EXPECTED,
        allow_resample: bool = False,
        pad_short: bool = False,
        overlap: float = 0.0,
        output_alias_for_features: bool = True,
    ) -> List[PipelineSegmentOutput]:
        df4 = standardize_4ch_dataframe(df)
        df4 = maybe_resample_dataframe(df4, fs, FS_EXPECTED, allow_resample=allow_resample)
        segments = segment_4ch_dataframe(df4, fs=FS_EXPECTED, pad_short=pad_short, overlap=overlap)
        sr_results = self.sr.infer_segments(segments, alias_for_features=output_alias_for_features)

        outputs: List[PipelineSegmentOutput] = []
        for pack, sr_res in zip(segments, sr_results):
            features, feat_audit = self.feature_extractor.extract_features(sr_res.sr8_df)
            predictions = {label: model.predict(features) for label, model in self.models.items()}
            outputs.append(
                PipelineSegmentOutput(
                    segment_index=pack.segment_index,
                    start_sample=pack.start_sample,
                    end_sample=pack.end_sample,
                    predictions=predictions,
                    feature_audit=feat_audit,
                    sr_audit=sr_res.audit,
                    sr8_df=sr_res.sr8_df,
                    features=features,
                )
            )
        return outputs

    def run_clean_4ch_array(
        self,
        eeg4: np.ndarray,
        channel_names: Sequence[str],
        ppg: Optional[np.ndarray] = None,
        fs: int = FS_EXPECTED,
        allow_resample: bool = False,
        pad_short: bool = False,
    ) -> List[PipelineSegmentOutput]:
        eeg4 = np.asarray(eeg4)
        if eeg4.ndim != 2:
            raise DeploymentError(f"eeg4 harus shape (4,T) atau (T,4), diterima {eeg4.shape}")
        if len(channel_names) != 4:
            raise DeploymentError("channel_names harus berisi 4 channel fisik.")
        # Terima (T,4) lalu transpose.
        if eeg4.shape[0] != 4 and eeg4.shape[1] == 4:
            eeg4 = eeg4.T
        if eeg4.shape[0] != 4:
            raise DeploymentError(f"eeg4 harus memiliki 4 channel, diterima {eeg4.shape}")

        df = pd.DataFrame({str(ch): eeg4[i] for i, ch in enumerate(channel_names)})
        if ppg is not None:
            df["PPG"] = np.asarray(ppg).reshape(-1)
        return self.run_clean_4ch_dataframe(df, fs=fs, allow_resample=allow_resample, pad_short=pad_short)

    @staticmethod
    def save_outputs(outputs: Sequence[PipelineSegmentOutput], out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_rows = []
        audit_rows = []
        feature_rows = []
        for out in outputs:
            sr_path = out_dir / f"segment_{out.segment_index:03d}_SR8.csv"
            out.sr8_df.to_csv(sr_path, index=False)

            row = {"segment_index": out.segment_index, "start_sample": out.start_sample, "end_sample": out.end_sample}
            for label, pred in out.predictions.items():
                row[f"{label}_status"] = pred["status"]
                row[f"{label}_risk_score_model"] = pred["risk_score_model"]
                row[f"{label}_decision_score"] = pred["decision_score"]
                row[f"{label}_threshold"] = pred["deployment_threshold"]
                row[f"{label}_direction"] = pred["deployment_threshold_direction"]
                row[f"{label}_raw_prob"] = pred["prob_positive_raw"]
            pred_rows.append(row)

            audit_rows.append({
                "segment_index": out.segment_index,
                **{f"sr_{k}": v for k, v in out.sr_audit.items() if not isinstance(v, (list, dict))},
                "n_features": out.feature_audit.get("n_features"),
                "n_ppg_peaks": out.feature_audit.get("n_ppg_peaks"),
                "nan_features_count": len(out.feature_audit.get("nan_features", [])),
            })
            feature_rows.append({"segment_index": out.segment_index, **out.features})

        pd.DataFrame(pred_rows).to_csv(out_dir / "predictions_summary.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out_dir / "audit_summary.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(out_dir / "features_per_segment.csv", index=False)




# =============================================================================
# GUI LAMA COBACLAUDE + PATCH TAB DEPLOYMENT
# =============================================================================
import time
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Menjadi:
from scipy.signal import butter, filtfilt, welch
import mne  # <--- TAMBAHKAN INI!
from matplotlib.colors import ListedColormap


# Pastikan file eeg_processor.py ada di folder yang sama
import eeg_processor_8
from eeg_processor_8 import reconstruct_from_ica

# --- KONSTANTA GLOBAL (Tanpa 'self') ---
FS = 250
WINDOW_SEC = 10
MAX_SAMPLES = FS * WINDOW_SEC
NUM_CHANNELS = 8
NUM_DIGITAL = 3   # D11, D12, D13
# ============================================================
# LIVE EEG DISPLAY 4CH + PPG
# Board tetap 8CH, tetapi GUI Live EEG hanya menampilkan:
# EXG0=Fp1, EXG1=Fp2, EXG4=T3, EXG5=T4
# ============================================================
LIVE_EEG_N = 4
LIVE_EEG_SOURCE_IDXS = [0, 1, 2, 3]
LIVE_EEG_NAMES = ["Fp1", "Fp2", "T3", "T4"]
LIVE_DIGITAL_BUFFER_IDX = LIVE_EEG_N

# ============================================================
# MODE CHANNEL PROCESSOR
# ============================================================
# 8CH mode : Fp1,Fp2,F3,F4,T3,T4,P3,P4 masuk Zapline/ICA/fitur lama.
# 4CH mode : hanya Fp1,Fp2,T3,T4 masuk Zapline/ICA, lalu SR+ML tetap bisa jalan.
PROCESSOR_8CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
PROCESSOR_4CH_NAMES = ["Fp1", "Fp2", "T3", "T4"]
PROCESSOR_PPG_COL = "Digital"


# # bisa ini, skalanya cuman gede bgt
# def calculate_impedance_openbci(raw_data, current_gain=1.0):
#     """
#     Menghitung impedansi menggunakan metode Lock-in pada 31.25 Hz.
#     current_gain: Gain yang sedang diset di hardware (default saat cek imp adalah 1.0)
#     """
#     if len(raw_data) < FS:
#         return 0.0

#     # KUNCINYA: BrainFlow selalu mengasumsikan Gain 24.0. 
#     # Jika kita set hardware ke Gain 1.0, kita harus mengalikan balik datanya.
#     data_uv = raw_data * (24.0 / current_gain)
    
#     # Konstanta OpenBCI
#     CURRENT_RMS = 6e-9          # 6 nA
#     R_SERIES = 2200             # 2.2 kOhm resistor internal
#     TARGET_FREQ = 31.25         # Frekuensi injeksi Cyton

#     # 1. Hilangkan DC Offset
#     data = data_uv - np.mean(data_uv)
    
#     # 2. Buat sinyal referensi
#     t = np.arange(len(data)) / FS
#     ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
#     ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)

#     # 3. Mixing (Demodulation)
#     mixed_i = data * ref_sin
#     mixed_q = data * ref_cos

#     # 4. Rata-rata (Low Pass Filter)
#     avg_i = np.mean(mixed_i)
#     avg_q = np.mean(mixed_q)

#     # 5. Hitung V peak-to-peak lalu ke RMS
#     v_peak_uV = 2 * np.sqrt(avg_i**2 + avg_q**2)
#     v_rms_volt = (v_peak_uV * 1e-6) / np.sqrt(2)

#     # 6. Hitung Z = (Vrms / Irms) - R_series
#     z = (v_rms_volt / CURRENT_RMS) - R_SERIES
    
    # return max(0, z / 1000.0) # Hasil dalam kOhm


# def calculate_impedance_openbci(raw_data):
#     """
#     Menghitung impedansi menggunakan metode Lock-in pada 31.25 Hz.
#     Sesuai standar ADS1299 OpenBCI.
#     """
#     if len(raw_data) < FS:
#         return 0.0

#     # Konstanta OpenBCI
#     CURRENT = 6e-9          # 6 nA
#     R_SERIES = 2200         # 2.2 kOhm resistor internal
#     TARGET_FREQ = 31.25     # Frekuensi injeksi Cyton

#     # 1. Hilangkan DC Offset
#     data = raw_data - np.mean(raw_data)
    
#     # 2. Buat sinyal referensi
#     t = np.arange(len(data)) / FS
#     ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
#     ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)

#     # 3. Mixing (Demodulation)
#     # BrainFlow mengembalikan data dalam mikrovolt (uV), 
#     # kita tidak perlu kalibrasi gain lagi karena BrainFlow sudah menghitungnya.
#     mixed_i = data * ref_sin
#     mixed_q = data * ref_cos

#     # 4. Low Pass Filter (Rata-rata)
#     avg_i = np.mean(mixed_i)
#     avg_q = np.mean(mixed_q)

#     # 5. Hitung V peak-to-peak
#     # Rumus: Vrms = sqrt(I^2 + Q^2) * sqrt(2), Vpp = Vrms * 2 * sqrt(2)
#     # Atau lebih sederhana untuk Lock-in:
#     v_peak_uV = 2 * np.sqrt(avg_i**2 + avg_q**2)
#     v_volt = v_peak_uV * 1e-6

#     # 6. Hitung Z = (V / I) - R_series
#     z = (v_volt / CURRENT) - R_SERIES
    
#     # Konversi ke kOhm
#     z_kohm = z / 1000.0
    
#     # Proteksi nilai negatif atau noise saat elektroda lepas (Rail)
#     if np.max(np.abs(data)) > 180000: # Jika sinyal saturasi/lewat batas uV
#         return 1000.0 # Anggap 1000 kOhm (elektroda lepas)

#     return max(0, z_kohm)

# calculate sebelumnya def calculate_impedance_openbci(raw_data, current_gain=1.0):
#     """
#     Menghitung impedansi menggunakan metode Lock-in pada 31.25 Hz.
#     current_gain: Gain yang sedang diset di hardware (default saat cek imp adalah 1.0)
#     """
#     if len(raw_data) < FS:
#         return 0.0

#     # KUNCINYA: BrainFlow selalu mengasumsikan Gain 24.0. 
#     # Jika kita set hardware ke Gain 1.0, kita harus mengalikan balik datanya.
#     data_uv = raw_data * (24.0 / current_gain)
    
#     # Konstanta OpenBCI
#     CURRENT_RMS = 6e-9          # 6 nA
#     R_SERIES = 2200             # 2.2 kOhm resistor internal
#     TARGET_FREQ = 31.25         # Frekuensi injeksi Cyton

#     # 1. Hilangkan DC Offset
#     data = data_uv - np.mean(data_uv)
    
#     # 2. Buat sinyal referensi
#     t = np.arange(len(data)) / FS
#     ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
#     ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)

#     # 3. Mixing (Demodulation)
#     mixed_i = data * ref_sin
#     mixed_q = data * ref_cos

#     # 4. Rata-rata (Low Pass Filter)
#     avg_i = np.mean(mixed_i)
#     avg_q = np.mean(mixed_q)

#     # 5. Hitung V peak-to-peak lalu ke RMS
#     v_peak_uV = 2 * np.sqrt(avg_i**2 + avg_q**2)
#     v_rms_volt = (v_peak_uV * 1e-6) / np.sqrt(2)

#     # 6. Hitung Z = (Vrms / Irms) - R_series
#     z = (v_rms_volt / CURRENT_RMS) - R_SERIES
    
#     return max(0, z / 1000.0) # Hasil dalam kOhm

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Hapus edge artifact dari sinyal ICA hasil rekonstruksi
# Spike di ujung IC terjadi karena filter/ICA edge effect (transien matematis).
# Solusi: trim beberapa sample di awal & akhir, lalu interpolasi linear agar
# sinyal tetap kontinu (tidak ada jump ke nol).
# ──────────────────────────────────────────────────────────────────────────────
def trim_ic_edges(ic_signal, trim_samples=None, fs=FS):
    """
    Hapus edge artifact pada satu sinyal IC 1-D.

    Parameters
    ----------
    ic_signal    : np.ndarray  shape (n_samples,)
    trim_samples : int | None  — jumlah sample yang di-trim di tiap ujung.
                                 Default = 0.2 detik (50 sample pada 250 Hz).
    fs           : int         — sampling frequency (untuk hitung default trim)

    Returns
    -------
    np.ndarray shape (n_samples,) — sinyal asli dengan tepi yang di-smooth,
    panjang TIDAK berubah (tidak di-crop) agar index tetap konsisten.
    """
    if trim_samples is None:
        trim_samples = int(fs * 0.2)  # default: 0.2 detik = 50 sample

    n = len(ic_signal)
    if n <= 2 * trim_samples:
        return ic_signal  # terlalu pendek, lewati

    result = ic_signal.copy()

    # Nilai "stabil" diambil dari rata-rata 10 sample setelah zona trim
    stable_start = np.mean(ic_signal[trim_samples : trim_samples + 10])
    stable_end   = np.mean(ic_signal[-(trim_samples + 10) : -trim_samples])

    # Interpolasi linear dari nol ke nilai stabil di tepi kiri
    result[:trim_samples] = np.linspace(0, stable_start, trim_samples)

    # Interpolasi linear dari nilai stabil ke nol di tepi kanan
    result[-trim_samples:] = np.linspace(stable_end, 0, trim_samples)

    return result


def trim_ica_clean(data, trim_samples=None, fs=FS):
    """
    Terapkan trim_ic_edges ke setiap channel pada data multidimensi.
    data : np.ndarray  shape (n_channels, n_samples)
    """
    if data is None or data.ndim != 2:
        return data
    return np.array([trim_ic_edges(data[ch], trim_samples, fs)
                     for ch in range(data.shape[0])])


def trim_ic_components(ics, trim_samples=None, fs=FS):
    """
    Terapkan trim_ic_edges ke field 'sig' pada setiap dict IC.
    ics : list of dict — masing-masing dict berisi key 'sig'
    """
    trimmed = []
    for ic in ics:
        ic_copy = dict(ic)
        ic_copy['sig'] = trim_ic_edges(ic['sig'], trim_samples, fs)
        trimmed.append(ic_copy)
    return trimmed
# ──────────────────────────────────────────────────────────────────────────────

def make_light_jet(minval=0.12, maxval=1.0, n=256):
    """
    Membuat colormap jet tanpa bagian biru tergelap.
    minval makin besar -> warna minimum makin terang.
    Coba 0.10 - 0.18.
    """
    base = plt.get_cmap("jet")
    new_colors = base(np.linspace(minval, maxval, n))
    return ListedColormap(new_colors, name="light_jet")

LIGHT_JET = make_light_jet(minval=0.2)
def calculate_impedance_openbci(raw_data):
    if len(raw_data) < FS:
        return 0.0

    CURRENT_RMS = 6e-9
    TARGET_FREQ = 31.25

    data = raw_data - np.mean(raw_data)

    t = np.arange(len(data)) / FS
    ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
    ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)

    mixed_i = data * ref_sin
    mixed_q = data * ref_cos

    avg_i = np.mean(mixed_i)
    avg_q = np.mean(mixed_q)

    # 🔥 mengikuti repo
    v_rms_uV = np.sqrt(avg_i**2 + avg_q**2)
    v_rms_volt = v_rms_uV * 1e-6

    z = v_rms_volt / CURRENT_RMS

    return max(0, z / 1000.0)


class EEGAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenBCI EEG - 6 Channel Research Suite")
        # Mengambil resolusi layar aktual milik Raspberry Pi
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        

        # Mengatur ukuran GUI sebesar 90% dari resolusi layar asli
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        # Menerapkan ukuran baru secara dinamis
        self.root.geometry(f"{window_width}x{window_height}+0+0")
        self.show_before_ica_var = tk.BooleanVar(value=True)
        self.show_ica_compare_legend_var = tk.BooleanVar(value=True)
        
        # Di dalam def __init__(self, root):
        self.loaded_data_from_file = None  # Tempat menyimpan data TXT
        self.current_source = "LIVE"       # Flag untuk menandai sumber data (LIVE atau FILE)
        
        # --- KONFIGURASI WARNA MODERN ---
        # Cari di dalam def __init__(self, root):
        self.colors = {
            "bg": "#ffffff",            # Putih Bersih (Background Utama)
            "bg_card": "#f8f9fa",       # Abu-abu sangat muda (Untuk Frame)
            "secondary_bg": "#f8f9fa",  # Alias agar tidak error
            "accent": "#1a73e8",        # Biru Google/Medical (Aksen Utama)
            "accent_blue": "#1a73e8",   # Alias biru
            "green": "#28a745",         # Hijau standar (Connect)
            "accent_green": "#28a745",  # Alias hijau
            "red": "#dc3545",           # Merah standar (Stop)
            "accent_red": "#dc3545",    # Alias merah
            "yellow": "#ffc107",        # Kuning standar
            "text": "#202124",          # Hitam/Abu gelap (Teks Utama)
            "text_dim": "#5f6368"       # Abu-abu (Teks Sekunder)
        }
        self.root.configure(bg=self.colors["bg"])

        self.board = None
        self.streaming = False
        self.is_configuring = False
        self.last_known_imp = [None] * NUM_CHANNELS
        self.is_recording = False
        self._countdown_job = None
        self._countdown_job = None  
        self.session_data = []      
        self.data_buffer = np.zeros((NUM_CHANNELS + 1, MAX_SAMPLES))
        self.time_axis = np.linspace(-5, 5, MAX_SAMPLES)
        self.last_filename = ""
        self.labels_imp_text = []   
        self.imp_states = [False] * NUM_CHANNELS 
        self.imp_buttons = [] 
        self.last_impedance_time = [0] * NUM_CHANNELS
        self.IMP_INTERVAL = 1.0  
        # Average topomap
        self.avg_topo_files = []
        self.avg_topo_start_var = tk.StringVar(value="0")
        self.avg_topo_end_var = tk.StringVar(value="")
        self.avg_topo_label_var = tk.StringVar(value="No files selected")
        # Tambahkan ini di bagian inisialisasi variabel (misal di bawah self.IMP_INTERVAL = 1.0)
        self.skip_ica_var = tk.BooleanVar(value=False)  # Default False: ICA tetap berjalan
        self.skip_zapline_var = tk.BooleanVar(value=True)
        self.channel_mode_var = tk.StringVar(value="8")  # pilihan: "4" atau "8"
        self.ica_kurt_var = tk.DoubleVar(value=1.6)
        self.ica_bcg_var  = tk.DoubleVar(value=0.065)   # ← TAMBAH
        self.ica_vbr_var  = tk.DoubleVar(value=15.0)    # ← TAMBAH
        # Topomap subtraction / difference
        self.diff_topo_files = []
        self.diff_topo_label_var = tk.StringVar(value="No files selected")

        self.diff1_start_var = tk.StringVar(value="0")
        self.diff1_end_var   = tk.StringVar(value="")

        self.diff2_start_var = tk.StringVar(value="")
        self.diff2_end_var   = tk.StringVar(value="")

        self._setup_styles()
        self._build_main_gui()
        self._build_realtime_plot()
    def _apply_raspy_compact_layout(self):
        """
        Patch compact layout untuk Raspberry Pi:
        1. Mengecilkan tab Notebook.
        2. Memendekkan nama tab.
        3. Mengecilkan figure Matplotlib yang sudah ada.
        4. Menyembunyikan toolbar Matplotlib agar area plot lebih lega.
        """

        # ============================================================
        # 1. COMPACT STYLE UNTUK NOTEBOOK / TAB / BUTTON
        # ============================================================
        try:
            style = ttk.Style()

            style.configure(
                "TNotebook.Tab",
                padding=(4, 1),
                font=("Segoe UI", 8, "bold")
            )

            style.configure(
                "TLabelframe.Label",
                font=("Segoe UI", 8, "bold")
            )

            style.configure(
                "TButton",
                font=("Segoe UI", 8)
            )

            style.configure(
                "TCheckbutton",
                font=("Segoe UI", 8)
            )

            style.configure(
                "TRadiobutton",
                font=("Segoe UI", 8)
            )

        except Exception as e:
            print(f"[RASPY UI] Gagal apply ttk compact style: {e}")

        # ============================================================
        # 2. PENDEKKAN NAMA TAB NOTEBOOK
        # ============================================================
        try:
            if hasattr(self, "notebook") and self.notebook is not None:
                rename_map = {
                    "Deep Learning SR": "SR",
                    "Machine Learning": "ML",
                    "Live EEG Stream": "Live",
                    "ICA Components": "ICA",
                    "ICA Compare": "ICA Cmp",
                    "Band Signals": "Bands",
                    "Power Bands": "Power",
                    "BPM Analysis": "BPM",
                    "Brain Mapping": "Map",
                    "Saving Feature": "Feat Save",
                    "Saving Signal": "Sig Save",
                    "Average Topomap": "Avg Map",
                    "Subtraction Topomap": "Sub Map",
                }

                for tab_id in self.notebook.tabs():
                    old_text = self.notebook.tab(tab_id, "text")
                    new_text = rename_map.get(old_text, old_text)

                    # Kalau masih terlalu panjang, potong otomatis
                    if len(new_text) > 9:
                        new_text = new_text[:9]

                    self.notebook.tab(tab_id, text=new_text)

        except Exception as e:
            print(f"[RASPY UI] Gagal rename notebook tabs: {e}")

        # ============================================================
        # 3. KECILKAN FIGURE MATPLOTLIB YANG SUDAH ADA
        # ============================================================
        # try:
        #     import matplotlib.pyplot as plt

        #     # Semua figure aktif dikecilkan agar muat di layar 1024x600
        #     # for num in plt.get_fignums():
        #     #     fig = plt.figure(num)

        #     #     w, h = fig.get_size_inches()

        #     #     # Jangan besarkan figure kecil; hanya kecilkan figure besar
        #     #     new_w = min(w, 9.0)
        #     #     new_h = min(h, 3.8)

        #     #     fig.set_size_inches(new_w, new_h, forward=True)
        #     #     fig.set_dpi(80)

        #         try:
        #             fig.tight_layout()
        #         except Exception:
        #             pass

        #         try:
        #             fig.canvas.draw_idle()
        #         except Exception:
        #             pass

        # except Exception as e:
        #     print(f"[RASPY UI] Gagal resize matplotlib figures: {e}")

        # ============================================================
        # 4. SEMBUNYIKAN TOOLBAR MATPLOTLIB
        # ============================================================
        # def hide_mpl_toolbar_recursive(widget):
        #     try:
        #         children = widget.winfo_children()
        #     except Exception:
        #         return

        #     for child in children:
        #         try:
        #             cls_name = child.winfo_class()
        #             str_child = str(child).lower()

        #             # Toolbar Matplotlib biasanya punya nama mengandung navigationtoolbar
        #             # atau berisi banyak button kecil.
        #             is_toolbar = (
        #                 "navigationtoolbar" in str_child
        #                 or "toolbar" in str_child
        #             )

        #             if is_toolbar:
        #                 child.pack_forget()
        #                 child.grid_forget()
        #                 child.place_forget()
        #                 continue

        #             hide_mpl_toolbar_recursive(child)

        #         except Exception:
        #             pass

        # try:
        #     hide_mpl_toolbar_recursive(self.root)
        # except Exception as e:
        #     print(f"[RASPY UI] Gagal hide matplotlib toolbar: {e}")

        # # ============================================================
        # # 5. UPDATE GEOMETRY WINDOW
        # # ============================================================
        # try:
        #     self.root.update_idletasks()
        # except Exception:
        #     pass
    def _make_sr8_topo_info(self):
        """
        Membuat MNE info untuk 8 channel hasil SR:
        Fp1, Fp2, F3, F4, T3/T7, T4/T8, P3, P4.
        Untuk MNE montage, T3/T4 dipetakan ke T7/T8.
        """
        import mne

        ch_names_mne = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]

        info = mne.create_info(
            ch_names=ch_names_mne,
            sfreq=FS,
            ch_types="eeg"
        )

        info.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            match_case=False,
            on_missing="ignore"
        )

        return info

    def _compute_brain_mapping_bands_from_eeg8(self, eeg8, fs=FS):
        """
        Membuat struktur bands seperti res['bands'] lama, tetapi dari hasil SR8.

        Input:
            eeg8 shape = (8, N)
            urutan = Fp1, Fp2, F3, F4, T3, T4, P3, P4

        Output:
            dict kompatibel dengan build_brain_mapping_tab() / plot_brain_mapping()
        """
        import numpy as np
        from scipy.signal import butter, filtfilt, welch

        eeg8 = np.asarray(eeg8, dtype=float)

        if eeg8.ndim != 2:
            raise ValueError(f"eeg8 harus 2D, ditemukan shape={eeg8.shape}")

        # Pastikan shape channel x sample
        if eeg8.shape[0] != 8 and eeg8.shape[1] == 8:
            eeg8 = eeg8.T

        if eeg8.shape[0] != 8:
            raise ValueError(f"Brain mapping SR8 butuh 8 channel, ditemukan shape={eeg8.shape}")

        ch_names = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]

        band_defs = {
            "Delta": (0.5, 4.0),
            "Theta": (4.0, 8.0),
            "Alpha": (8.0, 13.0),
            "Beta":  (13.0, 30.0),
        }

        band_names = list(band_defs.keys())
        abs_power = {}
        rel_power = {}

        nyq = fs / 2.0

        def bandpass_arr(x, low, high):
            low = max(low, 0.1)
            high = min(high, nyq - 0.1)
            b, a = butter(4, [low / nyq, high / nyq], btype="band")
            return filtfilt(b, a, x, axis=-1)

        # Hitung absolute power per band/channel
        for band, (lo, hi) in band_defs.items():
            x_band = bandpass_arr(eeg8, lo, hi)
            vals = []

            for ch in range(8):
                f, pxx = welch(
                    x_band[ch],
                    fs=fs,
                    nperseg=min(int(fs * 2), x_band.shape[1])
                )
                mask = (f >= lo) & (f <= hi)

                if np.any(mask):
                    vals.append(float(np.trapz(pxx[mask], f[mask])))
                else:
                    vals.append(0.0)

            abs_power[band] = vals

        # Relative power
        total_power = np.zeros(8, dtype=float)
        for band in band_names:
            total_power += np.asarray(abs_power[band], dtype=float)

        total_power = np.where(total_power == 0, 1e-10, total_power)

        for band in band_names:
            rel_power[band] = (
                np.asarray(abs_power[band], dtype=float) / total_power
            ).tolist()

        # Rasio power
        ratios = {}
        ratio_defs = [
            ("Delta/Theta", "Delta", "Theta"),
            ("Delta/Alpha", "Delta", "Alpha"),
            ("Delta/Beta",  "Delta", "Beta"),
            ("Theta/Alpha", "Theta", "Alpha"),
            ("Theta/Beta",  "Theta", "Beta"),
            ("Alpha/Beta",  "Alpha", "Beta"),
        ]

        for label, nb, db in ratio_defs:
            num = np.asarray(abs_power[nb], dtype=float)
            den = np.asarray(abs_power[db], dtype=float)
            den = np.where(den == 0, 1e-10, den)
            ratios[label] = (num / den).tolist()

        # Asimetri per band
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        ch_to_pair = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

        asym_by_band = {}

        for band in band_names:
            p = np.asarray(abs_power[band], dtype=float)
            asym_topo = np.zeros(8, dtype=float)

            for ch_idx, pair_idx in ch_to_pair.items():
                li, ri = pairs[pair_idx]
                L, R = p[li], p[ri]
                denom = L + R
                asym_topo[ch_idx] = (L - R) / denom if abs(denom) > 1e-12 else 0.0

            asym_by_band[band] = asym_topo.tolist()

        # SASI = (Beta - Alpha) / (Beta + Alpha)
        alpha = np.asarray(abs_power["Alpha"], dtype=float)
        beta = np.asarray(abs_power["Beta"], dtype=float)
        sasi_values = (beta - alpha) / (beta + alpha + 1e-10)

        return {
            "names": band_names,
            "abs_power": abs_power,
            "rel_power": rel_power,
            "ratios": ratios,
            "asym_by_band": asym_by_band,
            "sasi_values": sasi_values.tolist(),
            "ch_names": ch_names,
            "source": "SR8"
        }
        
    def read_offline_eeg_file(self, file_path):
        """
        Membaca file offline EEG dan menstandarkan formatnya agar kompatibel
        dengan eeg_processor_8.py.

        Format yang didukung:
        1. TXT OpenBCI/tab-separated:
        kolom EEG berada pada 8 kolom pertama, PPG/digital kolom ke-9 jika ada.

        2. CSV hasil linear mixing:
        time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG
        atau
        time_s,Fp1,Fp2,F3,F4,T7,T8,P3,P4,PPG

        3. CSV hasil save GUI:
        rec_id,EEG_ch1,EEG_ch2,...,EEG_ch8,PPG

        Output:
        DataFrame standar:
        EXG Channel 0..7, Digital
        """
        import os
        import numpy as np
        import pandas as pd

        ext = os.path.splitext(file_path)[1].lower()

        # ------------------------------------------------------------
        # 1. Baca file
        # ------------------------------------------------------------
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            # TXT 4CH/8CH hasil recording.
            # header lama diawali %, jadi harus header=None agar baris data pertama
            # tidak dianggap sebagai nama kolom.
            df = pd.read_csv(file_path, comment='%', sep='\t', header=None)
            df = df.dropna(axis=1, how="all")

            # Jika ternyata ada header tanpa %, buang baris header tersebut
            df_num = df.apply(pd.to_numeric, errors="coerce")
            if len(df_num) > 0 and df_num.iloc[0].isna().mean() > 0.5:
                df_num = df_num.iloc[1:].reset_index(drop=True)

            df = df_num
        if df.empty:
            raise ValueError("File kosong atau tidak terbaca.")

        # Normalisasi nama kolom agar robust terhadap kapital/spasi/underscore
        def norm_col(x):
            return str(x).strip().lower().replace(" ", "").replace("_", "")

        col_map = {norm_col(c): c for c in df.columns}

        eeg_cols = None
        ppg_col = None
        detected_format = None

        # ------------------------------------------------------------
        # 2. Format CSV hasil save GUI:
        # rec_id,EEG_ch1,...,EEG_ch8,PPG
        # ------------------------------------------------------------
        if all(f"eegch{i}" in col_map for i in range(1, 9)):
            detected_format = "CSV GUI saved signal: EEG_ch1..EEG_ch8"

            eeg_cols = [
                col_map["eegch1"],
                col_map["eegch2"],
                col_map["eegch3"],
                col_map["eegch4"],
                col_map["eegch5"],
                col_map["eegch6"],
                col_map["eegch7"],
                col_map["eegch8"],
            ]

            if "ppg" in col_map:
                ppg_col = col_map["ppg"]

        # ------------------------------------------------------------
        # 3. Format CSV linear mixing:
        # time_s,Fp1,Fp2,F3,F4,T3,T4,P3,P4,PPG
        # atau T7/T8
        # ------------------------------------------------------------
        elif (
            "fp1" in col_map and
            "fp2" in col_map and
            "f3" in col_map and
            "f4" in col_map and
            "p3" in col_map and
            "p4" in col_map and
            (
                ("t3" in col_map and "t4" in col_map) or
                ("t7" in col_map and "t8" in col_map)
            )
        ):
            detected_format = "CSV linear mixing: Fp1,Fp2,F3,F4,T3/T4,P3/P4"

            t_left = col_map["t3"] if "t3" in col_map else col_map["t7"]
            t_right = col_map["t4"] if "t4" in col_map else col_map["t8"]

            eeg_cols = [
                col_map["fp1"],
                col_map["fp2"],
                col_map["f3"],
                col_map["f4"],
                t_left,
                t_right,
                col_map["p3"],
                col_map["p4"],
            ]

            if "ppg" in col_map:
                ppg_col = col_map["ppg"]

        # ------------------------------------------------------------
        # 4. Fallback TXT/OpenBCI:
        # ambil 8 kolom numerik pertama sebagai EEG
        # kolom ke-9 sebagai Digital/PPG jika ada
        # ------------------------------------------------------------
        else:
            # Fallback TXT/OpenBCI.
            # Bisa menerima:
            # - 8 EEG + Digital
            # - 4 EEG + Digital
            # - 4 EEG tanpa Digital
            if df.shape[1] >= 9:
                detected_format = "Fallback TXT/OpenBCI: first 8 EEG + Digital"
                eeg_cols = list(df.columns[:8])
                ppg_col = df.columns[8]

            elif df.shape[1] >= 5:
                detected_format = "Fallback TXT/OpenBCI: first 4 EEG + Digital"
                eeg_cols = list(df.columns[:LIVE_EEG_N])
                ppg_col = df.columns[LIVE_EEG_N]

            elif df.shape[1] >= 4:
                detected_format = "Fallback TXT/OpenBCI: first 4 EEG, Digital kosong"
                eeg_cols = list(df.columns[:LIVE_EEG_N])
                ppg_col = None

            else:
                raise ValueError(
                    f"File tidak punya minimal 4 kolom EEG. "
                    f"Kolom tersedia: {list(df.columns)}"
                )

        # ------------------------------------------------------------
        # 5. Bangun dataframe standar untuk processor
        # ------------------------------------------------------------
        out = pd.DataFrame()

        for i, col in enumerate(eeg_cols):
            out[f"EXG Channel {i}"] = pd.to_numeric(df[col], errors="coerce")

        if ppg_col is not None:
            out["Digital"] = pd.to_numeric(df[ppg_col], errors="coerce")
        else:
            # Kalau tidak ada PPG/digital, isi nol agar processor tetap jalan
            out["Digital"] = 0.0

        # Isi NaN
        n_nan = int(out.isna().sum().sum())
        if n_nan > 0:
            print(f"[LOAD FILE] Ditemukan {n_nan} NaN, diisi 0.")
            out = out.fillna(0)

        # Pastikan urutan kolom fix
        n_loaded_eeg = len(eeg_cols)
        ordered_cols = [f"EXG Channel {i}" for i in range(n_loaded_eeg)] + ["Digital"]
        out = out[ordered_cols]
    

        print("\n[LOAD FILE] File berhasil distandarkan")
        print(f"  Path            : {file_path}")
        print(f"  Format terdeteksi: {detected_format}")
        print(f"  Shape original  : {df.shape}")
        print(f"  Shape standar   : {out.shape}")
        print(f"  EEG columns     : {eeg_cols}")
        print(f"  PPG/Digital col : {ppg_col}")

        return out, detected_format
    def _fit_matplotlib_figure_to_window(self, fig, parent=None, height_ratio=0.62):
        """
        Resize figure Matplotlib agar mengikuti ukuran window Raspberry Pi.
        Dipakai sebelum FigureCanvasTkAgg dibuat.
        """
        try:
            self.root.update_idletasks()

            if parent is not None:
                parent.update_idletasks()
                w_px = parent.winfo_width()
                h_px = parent.winfo_height()
            else:
                w_px = self.root.winfo_width()
                h_px = self.root.winfo_height()

            # fallback kalau widget belum punya ukuran valid
            if w_px < 400:
                w_px = self.root.winfo_width()
            if h_px < 250:
                h_px = self.root.winfo_height()

            # kurangi area kontrol atas GUI
            available_w = max(w_px - 20, 700)
            available_h = max(int(self.root.winfo_height() * height_ratio), 260)

            dpi = 80
            fig.set_dpi(dpi)
            fig.set_size_inches(
                available_w / dpi,
                available_h / dpi,
                forward=True
            )

            try:
                fig.tight_layout()
            except Exception:
                pass

        except Exception as e:
            print(f"[RASPY FIG] Gagal resize figure ke window: {e}")
            
    def _build_main_gui(self):
        # ============================================================
        # RASPBERRY PI COMPACT LAYOUT
        # Row 1 : Live Connection + Offline Data + Impedance
        # Row 2 : Acquisition + Analysis + Run
        # ============================================================

        # Ukuran aman untuk Raspberry Pi / RealVNC 1024x600
        try:
            self.root.geometry("980x560")
            self.root.minsize(900, 520)
        except Exception:
            pass

        # Style lebih compact
        try:
            style = ttk.Style()
            style.configure("TNotebook.Tab", padding=(5, 2), font=("Segoe UI", 8, "bold"))
            style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))
            style.configure("TCheckbutton", font=("Segoe UI", 8))
            style.configure("TRadiobutton", font=("Segoe UI", 8))
        except Exception:
            pass

        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=1)

        ctrl_row1 = ttk.Frame(ctrl_frame)
        ctrl_row1.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        ctrl_row2 = ttk.Frame(ctrl_frame)
        ctrl_row2.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ctrl_row2.columnconfigure(0, weight=0)
        ctrl_row2.columnconfigure(1, weight=1)

        # ============================================================
        # BARIS 1: LIVE CONNECTION + OFFLINE DATA + IMPEDANCE
        # ============================================================

        # --- LIVE CONNECTION ---
        conn_frame = ttk.LabelFrame(ctrl_row1, text=" LIVE CONNECTION ", padding=2)
        conn_frame.pack(side=tk.LEFT, padx=2, fill=tk.Y)

        ttk.Label(conn_frame, text="Port:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(2, 2))

        self.com_entry = ttk.Entry(conn_frame, width=7, justify="center", font=("Segoe UI", 8))
        self.com_entry.insert(0, "/dev/ttyACM0")
        self.com_entry.pack(side=tk.LEFT, padx=(1, 3))

        btn_conn = tk.Button(
            conn_frame,
            text="CONNECT",
            command=self.prepare_session,
            bg=self.colors["accent"],
            fg="white",
            activebackground="#174ea6",
            activeforeground="white",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=7,
            pady=2
        )
        btn_conn.pack(side=tk.LEFT, padx=(2, 2))

        # --- OFFLINE DATA ---
        off_frame = ttk.LabelFrame(ctrl_row1, text=" OFFLINE DATA ", padding=2)
        off_frame.pack(side=tk.LEFT, padx=2, fill=tk.Y)

        btn_load = tk.Button(
            off_frame,
            text="LOAD FILE",
            command=self.load_txt_file,
            bg="#f8f9fa",
            fg=self.colors["accent"],
            activebackground="#e8f0fe",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            highlightbackground=self.colors["accent"],
            highlightthickness=1,
            padx=7,
            pady=2
        )
        btn_load.pack(side=tk.LEFT, padx=(2, 3))

        self.file_label = ttk.Label(off_frame, text="No file", width=13, font=("Segoe UI", 8))
        self.file_label.pack(side=tk.LEFT, padx=(2, 2))

        # --- IMPEDANCE CHECK ---
        imp_frame = ttk.LabelFrame(ctrl_row1, text=" IMPEDANCE CHECK ", padding=2)
        imp_frame.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.imp_buttons = []
        for i, ch_name in enumerate(LIVE_EEG_NAMES):
            btn = tk.Button(
                imp_frame,
                text=f"CH{i+1}",
                width=4,
                bg="#f1f3f4",
                fg=self.colors["text"],
                relief=tk.FLAT,
                bd=1,
                highlightthickness=1,
                highlightbackground="#dadce0",
                font=("Segoe UI", 8),
                command=lambda idx=i: self.toggle_channel_impedance(idx)
            )
            btn.pack(side=tk.LEFT, padx=1)
            self.imp_buttons.append(btn)

        # ============================================================
        # BARIS 2: ACQUISITION + ANALYSIS
        # ============================================================

        # Row2 pakai grid supaya analysis bisa mengambil sisa lebar
        ctrl_row2.columnconfigure(0, weight=0)
        ctrl_row2.columnconfigure(1, weight=1)

        # --- ACQUISITION ---
        rec_frame = ttk.LabelFrame(ctrl_row2, text=" ACQUISITION ", padding=2)
        rec_frame.grid(row=0, column=0, sticky="w", padx=(2, 3))
        rec_frame = ttk.LabelFrame(ctrl_row2, text=" ACQUISITION ", padding=2)
        rec_frame.grid(row=0, column=0, sticky="w", padx=(2, 3))

        tk.Label(
            rec_frame,
            text="Dur(s):",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 1))

        self.duration_var = tk.IntVar(value=120)
        self.spinbox_duration = tk.Spinbox(
            rec_frame,
            from_=5,
            to=600,
            increment=5,
            textvariable=self.duration_var,
            width=4,
            font=("Segoe UI", 8)
        )
        self.spinbox_duration.pack(side=tk.LEFT, padx=(1, 4))

        tk.Label(
            rec_frame,
            text="File:",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 1))

        self.filename_var = tk.StringVar(value="EEG_Recording")
        self.filename_entry = tk.Entry(
            rec_frame,
            textvariable=self.filename_var,
            width=14,
            font=("Segoe UI", 8)
        )
        self.filename_entry.pack(side=tk.LEFT, padx=(1, 4))

        self.countdown_label = tk.Label(
            rec_frame,
            text="",
            font=("Segoe UI", 8, "bold"),
            fg="red",
            bg="#e8eaed",
            width=5
        )
        self.countdown_label.pack(side=tk.LEFT, padx=2)

        self.btn_record = tk.Button(
            rec_frame,
            text="▶ REC",
            command=self.start_recording,
            bg=self.colors["green"],
            fg="white",
            state=tk.DISABLED,
            activebackground="#1e7e34",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2
        )
        self.btn_record.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(
            rec_frame,
            text="STOP",
            command=self.stop_and_save,
            bg="#e8eaed",
            fg="#9aa0a6",
            state=tk.DISABLED,
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)
                # --- Input label manual (dipakai saat data direkam langsung) ---
        self.rec_stress_var = tk.StringVar(value="")
        self.rec_anxiety_var = tk.StringVar(value="")
        self.rec_depression_var = tk.StringVar(value="")

        for _lbl, _var in (("S:", self.rec_stress_var),
                           ("A:", self.rec_anxiety_var),
                           ("D:", self.rec_depression_var)):
            tk.Label(rec_frame, text=_lbl,
                     font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(4, 1))
            tk.Entry(rec_frame, textvariable=_var, width=4,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 2))

        # --- ANALYSIS COMPACT ---
        # ============================================================
        # ANALYSIS — SEJAJAR DENGAN ACQUISITION
        # ============================================================

        an_frame = ttk.LabelFrame(ctrl_row2, text=" ANALYSIS ", padding=2)
        an_frame.grid(row=0, column=1, sticky="ew", padx=(3, 2))

        analysis_row1 = ttk.Frame(an_frame)
        analysis_row1.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            analysis_row1,
            text="Start:",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(3, 0))

        self.analysis_start_var = tk.StringVar(value="0")
        tk.Entry(
            analysis_row1,
            textvariable=self.analysis_start_var,
            width=4,
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 4))

        tk.Label(
            analysis_row1,
            text="End:",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 0))

        self.analysis_end_var = tk.StringVar(value="")
        tk.Entry(
            analysis_row1,
            textvariable=self.analysis_end_var,
            width=4,
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 5))

        self.skip_ica_var = tk.BooleanVar(value=False)
        self.skip_zapline_var = tk.BooleanVar(value=True)
        self.channel_mode_var = tk.StringVar(value="4")

        self.chk_skip_ica = ttk.Checkbutton(
            analysis_row1,
            text="ICA",
            variable=self.skip_ica_var
        )
        self.chk_skip_ica.pack(side=tk.LEFT, padx=(2, 2))

        self.chk_skip_zapline = ttk.Checkbutton(
            analysis_row1,
            text="Zap",
            variable=self.skip_zapline_var
        )
        self.chk_skip_zapline.pack(side=tk.LEFT, padx=(0, 4))

        tk.Label(
            analysis_row1,
            text="Mode:",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(2, 1))

        self.rb_mode_4ch = ttk.Radiobutton(
            analysis_row1,
            text="4CH",
            variable=self.channel_mode_var,
            value="4"
        )
        self.rb_mode_4ch.pack(side=tk.LEFT, padx=(0, 1))

        self.rb_mode_8ch = ttk.Radiobutton(
            analysis_row1,
            text="8CH",
            variable=self.channel_mode_var,
            value="8"
        )
        self.rb_mode_8ch.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_analyze = tk.Button(
            analysis_row1,
            text="▶ RUN",
            command=self.start_analysis_thread,
            bg=self.colors["accent"],
            fg="white",
            state=tk.DISABLED,
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=(6, 2))

        # ============================================================
        # STATUS BAR
        # ============================================================
        self.status_label = tk.Label(
            self.root,
            text="STATUS: IDLE",
            bg=self.colors["bg"],
            fg=self.colors["accent"],
            font=("Consolas", 9, "bold")
        )
        self.status_label.pack(side=tk.TOP, anchor="w", padx=8, pady=(0, 1))
    
    # def read_offline_eeg_file(self, file_path):
    #     """
    #     Membaca file offline EEG.

    #     Mendukung:
    #     1. TXT OpenBCI lama:
    #        - komentar diawali %
    #        - tab-separated
    #        - minimal 8 EEG channel + optional digital/PPG

    #     2. CSV hasil linear mixing:
    #        - kolom: time_s, Fp1, Fp2, F3, F4, T3, T4, P3, P4, PPG
    #        - atau T7/T8 jika belum di-rename
    #     """

    #     import os

    #     ext = os.path.splitext(file_path)[1].lower()

    #     # =====================================================================
    #     # CASE 1 — CSV hasil linear mixing
    #     # =====================================================================
    #     if ext == ".csv":
    #         df = pd.read_csv(file_path)

    #         # Normalisasi nama kolom agar robust terhadap spasi/case
    #         col_map = {str(c).strip().lower(): c for c in df.columns}

    #         # Urutan channel yang diharapkan GUI/prosesor 8 channel
    #         # CSV linear mixing kamu: Fp1, Fp2, F3, F4, T3, T4, P3, P4
    #         # Saya tambahkan fallback T7/T8 kalau sewaktu-waktu belum di-rename.
    #         required_channels = [
    #             ("Fp1", ["fp1"]),
    #             ("Fp2", ["fp2"]),
    #             ("F3",  ["f3"]),
    #             ("F4",  ["f4"]),
    #             ("T3",  ["t3", "t7"]),
    #             ("T4",  ["t4", "t8"]),
    #             ("P3",  ["p3"]),
    #             ("P4",  ["p4"]),
    #         ]

    #         selected_cols = []

    #         for out_name, candidates in required_channels:
    #             found_col = None
    #             for cand in candidates:
    #                 if cand in col_map:
    #                     found_col = col_map[cand]
    #                     break

    #             if found_col is None:
    #                 raise ValueError(
    #                     f"Kolom channel {out_name} tidak ditemukan di CSV.\n"
    #                     f"Kolom tersedia: {list(df.columns)}"
    #                 )

    #             selected_cols.append(found_col)

    #         # Ambil 8 channel EEG
    #         eeg_df = df[selected_cols].apply(pd.to_numeric, errors="coerce")

    #         if eeg_df.isna().any().any():
    #             n_nan = int(eeg_df.isna().sum().sum())
    #             print(f"[WARNING] Ada {n_nan} NaN pada EEG CSV. Diisi dengan 0.")
    #             eeg_df = eeg_df.fillna(0)

    #         # Ambil PPG jika ada, kalau tidak ada isi nol
    #         if "ppg" in col_map:
    #             ppg = pd.to_numeric(df[col_map["ppg"]], errors="coerce").fillna(0).to_numpy()
    #         else:
    #             ppg = np.zeros(len(df))

    #         # Bentuk dataframe standar untuk pipeline GUI:
    #         # 8 EEG channel + Digital/PPG di kolom ke-9
    #         standard_cols = [f"EXG Channel {i}" for i in range(NUM_CHANNELS)] + ["Digital"]
    #         standard_df = pd.DataFrame(columns=standard_cols)

    #         for i, col in enumerate(selected_cols):
    #             standard_df[f"EXG Channel {i}"] = eeg_df[col].to_numpy(dtype=float)

    #         standard_df["Digital"] = ppg.astype(float)

    #         print("[LOAD CSV] File linear mixing terbaca.")
    #         print(f"  Original columns : {list(df.columns)}")
    #         print(f"  EEG columns used : {selected_cols}")
    #         print(f"  Output shape     : {standard_df.shape}")

    #         return standard_df

    #     # =====================================================================
    #     # CASE 2 — TXT OpenBCI lama
    #     # =====================================================================
    #     elif ext == ".txt":
    #         df = pd.read_csv(file_path, comment='%', sep='\t')

    #         # Pastikan semua numerik
    #         df = df.apply(pd.to_numeric, errors="coerce")

    #         if df.isna().any().any():
    #             n_nan = int(df.isna().sum().sum())
    #             print(f"[WARNING] Ada {n_nan} NaN pada TXT. Diisi dengan 0.")
    #             df = df.fillna(0)

    #         if df.shape[1] < NUM_CHANNELS:
    #             raise ValueError(
    #                 f"File TXT tidak berisi minimal {NUM_CHANNELS} channel EEG. "
    #                 f"Jumlah kolom terbaca: {df.shape[1]}"
    #             )

    #         # Kalau TXT hanya 8 channel, tambahkan Digital/PPG nol
    #         if df.shape[1] == NUM_CHANNELS:
    #             df["Digital"] = 0.0

    #         # Ambil hanya 8 EEG + 1 digital/PPG
    #         df = df.iloc[:, :NUM_CHANNELS + 1].copy()

    #         # Standarkan nama kolom
    #         df.columns = [f"EXG Channel {i}" for i in range(NUM_CHANNELS)] + ["Digital"]

    #         print("[LOAD TXT] File OpenBCI TXT terbaca.")
    #         print(f"  Output shape: {df.shape}")

    #         return df

    #     else:
    #         raise ValueError(
    #             f"Format file tidak didukung: {ext}\n"
    #             f"Gunakan .txt atau .csv"
    #         )
                
    def load_txt_file(self):
        from tkinter import filedialog
        import os

        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if file_path:
            try:
                self.last_filename = file_path

                # Baca TXT/CSV lalu ubah menjadi format internal standar
                data, detected_format = self.read_offline_eeg_file(file_path)

                eeg_cols_loaded = [c for c in data.columns if str(c).startswith("EXG Channel")]

                if len(eeg_cols_loaded) < LIVE_EEG_N:
                    messagebox.showerror(
                        "Error",
                        f"File tidak berisi minimal {LIVE_EEG_N} channel EEG."
                    )
                    return

                # Simpan dataframe standar agar run_analysis_pipeline tidak baca ulang format asli
                self.offline_df = data.copy()

                # Simpan data ke session_data: shape channel x sample
                self.session_data = data[eeg_cols_loaded[:LIVE_EEG_N]].values.T
                self.current_source = "FILE"

                # Optional: tampilkan di live buffer agar terlihat di tab Live EEG
                try:
                    n_show = min(MAX_SAMPLES, data.shape[0])
                    self.data_buffer[:] = 0
                    self.data_buffer[:NUM_CHANNELS, -n_show:] = data.iloc[-n_show:, :NUM_CHANNELS].values.T
                    self.data_buffer[NUM_CHANNELS, -n_show:] = data["Digital"].values[-n_show:]
                    self.canvas.draw_idle()
                except Exception as e:
                    print(f"[LOAD FILE] Preview buffer skipped: {e}")

                # Update UI
                self.file_label.config(text=os.path.basename(file_path)[:15] + "...")
                self.status_label.config(
                    text=f"STATUS: FILE LOADED ({detected_format})",
                    fg=self.colors["accent"]
                )

                # Aktifkan tombol analisis
                self.btn_analyze.config(
                    state=tk.NORMAL,
                    bg=self.colors["accent"],
                    fg="white"
                )

                messagebox.showinfo(
                    "Success",
                    "Data berhasil dimuat dan otomatis dikonversi internal.\n\n"
                    f"Format terdeteksi:\n{detected_format}\n\n"
                    "Silakan tekan RUN ANALYSIS."
                )

            except Exception as e:
                messagebox.showerror("Error", f"Gagal membaca file:\n{e}")
                
    def build_save_tab(self, parent):
        """Membangun tab Saving Feature berisi daftar fitur + tombol save CSV."""
        for widget in parent.winfo_children():
            widget.destroy()

        # --- Header info ---
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=6)

        if self.last_features:
            fi = self.last_features
            info_text = (
                f"File: {fi.get('filename','?')}   |   "
                f"Subject: {fi.get('subject','?')}   |   "
                f"Rec ID: {fi.get('rec_id','?')}   |   "
                f"Stress: {fi.get('stress','?')}   "
                f"Anxiety: {fi.get('anxiety','?')}   "
                f"Depression: {fi.get('depression','?')}"
            )
        else:
            info_text = "Belum ada fitur — jalankan analisis terlebih dahulu."

        tk.Label(info_frame, text=info_text, font=('Segoe UI', 9),
                 fg=self.colors["accent"], bg=self.colors["bg"],
                 wraplength=900, justify='left').pack(side=tk.LEFT)

        # --- Treeview daftar fitur ---
        # --- Input Rec ID ---
        recid_frame = ttk.Frame(parent)
        recid_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(recid_frame, text="Rec ID:", font=('Segoe UI', 10, 'bold'),
                 fg=self.colors["text"], bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(0, 6))

        if not hasattr(self, 'rec_id_var'):
            self.rec_id_var = tk.IntVar(value=1)

        tk.Spinbox(recid_frame, from_=1, to=9999, increment=1,
                   textvariable=self.rec_id_var,
                   width=6, font=('Segoe UI', 10)).pack(side=tk.LEFT)

        tk.Label(recid_frame,
                 text="  ← tentukan nomor rekaman sebelum menyimpan",
                 font=('Segoe UI', 9), fg=self.colors["text_dim"],
                 bg=self.colors["bg"]).pack(side=tk.LEFT, padx=8)

        # --- Treeview daftar fitur ---
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        

        tree = ttk.Treeview(tree_frame,
                            columns=("Feature", "Value"),
                            show='headings', height=25)
        tree.heading("Feature", text="Feature Name")
        tree.heading("Value",   text="Value")
        tree.column("Feature",  width=400)
        tree.column("Value",    width=200)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        # Isi treeview dari self.last_features
        if self.last_features:
            skip_keys = {'filename', 'subject', 'rec_id',
                         'stress', 'anxiety', 'depression', 'gender', 'seg_id'}
            for k, v in self.last_features.items():
                if k not in skip_keys:
                    val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                    tree.insert("", "end", values=(k, val_str))

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- Tombol Save ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        state = tk.NORMAL if self.last_features else tk.DISABLED
        self.btn_save_feat = tk.Button(
            btn_frame,
            text="💾  Save Features to CSV",
            command=self.save_features_to_csv,
            bg=self.colors["green"], fg="white",
            activebackground="#1e7e34", activeforeground="white",
            font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
            padx=20, pady=6,
            state=state
        )
        self.btn_save_feat.pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame,
                 text="Fitur akan ditambahkan ke baris baru di 'feature.csv'",
                 font=('Segoe UI', 9), fg=self.colors["text_dim"],
                 bg=self.colors["bg"]).pack(side=tk.LEFT, padx=15)
    
    def save_features_to_csv(self):
        """Simpan self.last_features ke feature.csv (append jika sudah ada)."""
        import os

        if not self.last_features:
            messagebox.showwarning("Tidak Ada Data",
                                   "Belum ada fitur yang diekstrak. Jalankan analisis dulu.")
            return

        csv_path = "feature.csv"

        try:
            # Ambil Rec ID dari spinbox dan masukkan ke baris sebelum disimpan
            self.last_features['rec_id'] = self.rec_id_var.get()

            new_row = pd.DataFrame([self.last_features])

            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                # Gabungkan kolom: kolom baru yang tidak ada di existing akan diisi NaN
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row

            # Susun urutan kolom: info dulu, lalu fitur, lalu label DASS
            info_cols  = ['filename', 'subject', 'rec_id', 'gender', 'seg_id']
            label_cols = ['stress', 'anxiety', 'depression']
            existing_info  = [c for c in info_cols  if c in combined.columns]
            existing_label = [c for c in label_cols if c in combined.columns]
            feat_cols = [c for c in combined.columns
                         if c not in existing_info + existing_label]
            combined = combined[existing_info + feat_cols + existing_label]

            combined.to_csv(csv_path, index=False)

            n_rows = len(combined)
            messagebox.showinfo(
                "Tersimpan ✓",
                f"Fitur berhasil disimpan!\n\n"
                f"File : {os.path.abspath(csv_path)}\n"
                f"Total baris di CSV: {n_rows}"
            )

            # Nonaktifkan tombol save setelah berhasil (agar tidak double-save)
            if self.btn_save_feat:
                self.btn_save_feat.config(state=tk.DISABLED,
                                          bg="#e8eaed", fg="#9aa0a6")
            # Reset last_features agar segmen berikutnya perlu Analyze dulu
            self.last_features = None

        except Exception as e:
            messagebox.showerror("Gagal Menyimpan", f"Error:\n{str(e)}")
         
    def build_signal_save_tab(self, parent):
        """Membangun tab Saving Signal: preview sinyal EEG+PPG preprocessed + tombol save CSV."""
        for widget in parent.winfo_children():
            widget.destroy()

        # --- Header info ---
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=6)

        has_data = (
            hasattr(self, 'analysis_results') and
            self.analysis_results and
            self.analysis_results.get('ica', {}).get('data') is not None
        )

        if has_data:
            info_text = (
                "Sinyal yang akan disimpan: EEG (ICA-cleaned, 8 ch) + PPG preprocessed. "
                "Setiap baris = 1 sampel (250 Hz). Kolom: rec_id, EEG_ch1..EEG_ch8, PPG."
            )
        else:
            info_text = "Belum ada sinyal — jalankan analisis terlebih dahulu."

        tk.Label(info_frame, text=info_text, font=('Segoe UI', 9),
                fg=self.colors["accent"], bg=self.colors["bg"],
                wraplength=900, justify='left').pack(side=tk.LEFT)

        # --- Input Rec ID ---
        recid_frame = ttk.Frame(parent)
        recid_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(recid_frame, text="Rec ID:", font=('Segoe UI', 10, 'bold'),
                fg=self.colors["text"], bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(0, 6))

        if not hasattr(self, 'signal_rec_id_var'):
            self.signal_rec_id_var = tk.IntVar(value=1)

        tk.Spinbox(recid_frame, from_=1, to=9999, increment=1,
                textvariable=self.signal_rec_id_var,
                width=6, font=('Segoe UI', 10)).pack(side=tk.LEFT)

        tk.Label(recid_frame,
                text="  ← tentukan nomor rekaman sebelum menyimpan",
                font=('Segoe UI', 9), fg=self.colors["text_dim"],
                bg=self.colors["bg"]).pack(side=tk.LEFT, padx=8)

        # --- Preview tabel (Treeview, 20 baris pertama) ---
        preview_frame = ttk.LabelFrame(parent, text=" Preview (20 baris pertama) ", padding=6)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Kolom preview dibuat dinamis mengikuti output processor: bisa 4CH atau 8CH.
        if has_data:
            eeg_preview = np.asarray(self.analysis_results['ica']['data'], dtype=float)
            if eeg_preview.ndim == 2 and eeg_preview.shape[0] > eeg_preview.shape[1] and eeg_preview.shape[1] <= 16:
                eeg_preview = eeg_preview.T
            n_ch_preview = int(eeg_preview.shape[0]) if eeg_preview.ndim == 2 else NUM_CHANNELS
        else:
            n_ch_preview = self.get_processor_n_channels() if hasattr(self, "get_processor_n_channels") else NUM_CHANNELS

        ch_names = [f"EEG_ch{i+1}" for i in range(n_ch_preview)]
        all_cols  = ["rec_id"] + ch_names + ["PPG"]

        tree = ttk.Treeview(preview_frame, columns=all_cols, show='headings', height=20)
        for col in all_cols:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor='center')

        sb = ttk.Scrollbar(preview_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        if has_data:
            eeg_clean = np.asarray(self.analysis_results['ica']['data'], dtype=float)
            if eeg_clean.ndim == 2 and eeg_clean.shape[0] > eeg_clean.shape[1] and eeg_clean.shape[1] <= 16:
                eeg_clean = eeg_clean.T
            bpm_data  = self.analysis_results.get('bpm_data', {})
            ppg_sig   = np.asarray(bpm_data.get('ppg_signal', []), dtype=float).reshape(-1)

            n_ch_actual, n_samples = eeg_clean.shape
            if len(ppg_sig) >= n_samples:
                ppg_aligned = ppg_sig[:n_samples]
            else:
                ppg_aligned = np.concatenate([ppg_sig, np.zeros(n_samples - len(ppg_sig))])

            rec_id_val = self.signal_rec_id_var.get()
            preview_n  = min(20, n_samples)
            for s in range(preview_n):
                eeg_vals = [f"{eeg_clean[ch, s]:.4f}" for ch in range(n_ch_actual)]
                row_vals  = [str(rec_id_val)] + eeg_vals + [f"{ppg_aligned[s]:.4f}"]
                tree.insert("", "end", values=row_vals)

        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- Tombol Save ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        state = tk.NORMAL if has_data else tk.DISABLED
        self.btn_save_signal = tk.Button(
            btn_frame,
            text="💾  Save Signal to CSV",
            command=self.save_signal_to_csv,
            bg=self.colors["accent"], fg="white",
            activebackground="#174ea6", activeforeground="white",
            font=('Segoe UI', 10, 'bold'), relief=tk.FLAT,
            padx=20, pady=6,
            state=state
        )
        self.btn_save_signal.pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame,
                text="Sinyal akan ditambahkan ke baris baru di 'signal.csv'",
                font=('Segoe UI', 9), fg=self.colors["text_dim"],
                bg=self.colors["bg"]).pack(side=tk.LEFT, padx=15)


    def save_signal_to_csv(self):
            """Simpan sinyal EEG-cleaned + PPG ke dalam folder processing_signal dengan format clean_<nama_file>.csv"""
            import os

            has_data = (
                hasattr(self, 'analysis_results') and
                self.analysis_results and
                self.analysis_results.get('ica', {}).get('data') is not None
            )
            if not has_data:
                messagebox.showwarning("Tidak Ada Data",
                                    "Belum ada sinyal. Jalankan analisis dulu.")
                return

            try:
                # 1. CEK & BUAT FOLDER: Jika folder 'processing_signal' belum ada, buat otomatis
                folder_name = "processing_signal"
                os.makedirs(folder_name, exist_ok=True)

                # =====================================================================
                # AMBIL NILAI SNR TERBARU: Dari hasil analisis/recompute terakhir
                # =====================================================================
                avg_snr = 0.0
                if hasattr(self, 'analysis_results') and self.analysis_results:
                    avg_snr = self.analysis_results.get('ica', {}).get('avg_snr', 0.0)
                # =====================================================================

                # 2. FORMAT PENAMAAN OTOMATIS BARU: Menambahkan (snr)dB ke dalam judul file
                if hasattr(self, 'last_filename') and self.last_filename:
                    base_name = os.path.basename(self.last_filename)  # contoh: "Abel1.txt"
                    name_without_ext, _ = os.path.splitext(base_name) # menghasilkan: "Abel1"
                    
                    # Format hasil akhir menjadi: clean_7.20dB_Abel1.csv
                    csv_filename = f"clean_{avg_snr:.2f}dB_{name_without_ext}.csv"     
                else:
                    # Antisipasi jika analisis dijalankan dari live streaming/recording
                    csv_filename = f"clean_{avg_snr:.2f}dB_recorded_signal_{int(time.time())}.csv"
                    
                # Gabungkan folder dengan nama file baru
                csv_path = os.path.join(folder_name, csv_filename)

                # 3. Ekstraksi data sinyal seperti biasa, tetapi jumlah channel dinamis.
                eeg_clean = np.asarray(self.analysis_results['ica']['data'], dtype=float)
                if eeg_clean.ndim == 2 and eeg_clean.shape[0] > eeg_clean.shape[1] and eeg_clean.shape[1] <= 16:
                    eeg_clean = eeg_clean.T
                bpm_data  = self.analysis_results.get('bpm_data', {})
                ppg_sig   = np.asarray(bpm_data.get('ppg_signal', []), dtype=float).reshape(-1)

                n_ch_actual, n_samples = eeg_clean.shape
                if len(ppg_sig) >= n_samples:
                    ppg_aligned = ppg_sig[:n_samples]
                else:
                    ppg_aligned = np.concatenate([ppg_sig, np.zeros(n_samples - len(ppg_sig))])

                rec_id_val = self.signal_rec_id_var.get()

                ch_names = [f"EEG_ch{i+1}" for i in range(n_ch_actual)]
                data_dict = {"rec_id": np.full(n_samples, rec_id_val, dtype=int)}
                for i, ch in enumerate(ch_names):
                    data_dict[ch] = eeg_clean[i]
                data_dict["PPG"] = ppg_aligned

                new_df = pd.DataFrame(data_dict)

                # 4. SIMPAN MANDIRI: Langsung disave ke file unik masing-masing (bukan di-append ke file global)
                new_df.to_csv(csv_path, index=False)
                n_rows = len(new_df)

                # Tampilkan pop-up informasi lokasi penyimpanan baru
                messagebox.showinfo(
                    "Tersimpan ✓",
                    f"Sinyal berhasil disimpan ke file terpisah!\n\n"
                    f"Folder : {folder_name}/\n"
                    f"File   : {csv_filename}\n"
                    f"Path   : {os.path.abspath(csv_path)}\n"
                    f"Sampel : {n_samples} baris berhasil diekspor."
                )

                # Nonaktifkan tombol agar user tahu proses simpan sukses dilakukan
                if self.btn_save_signal:
                    self.btn_save_signal.config(state=tk.DISABLED,
                                                bg="#e8eaed", fg="#9aa0a6")

            except Exception as e:
                messagebox.showerror("Gagal Menyimpan", f"Error:\n{str(e)}")                            
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Styling Notebook (Tabs) - Putih dengan indikator biru
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", 
                        background="#e8f0fe", 
                        foreground=self.colors["accent"], 
                        padding=[20, 8], font=('Segoe UI', 9, 'bold'))
        style.map("TNotebook.Tab", 
                background=[("selected", self.colors["accent"])],
                foreground=[("selected", "#ffffff")])

        # Styling LabelFrame - Border biru tipis
        style.configure("TLabelframe", 
                        background=self.colors["bg"], 
                        foreground=self.colors["accent"])
        style.configure("TLabelframe.Label", 
                        background=self.colors["bg"], 
                        foreground=self.colors["accent"], 
                        font=('Segoe UI', 10, 'bold'))

        # Styling Label Umum
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        


    def display_feature_table(self, df, parent):
            for widget in parent.winfo_children():
                widget.destroy()
            # Frame untuk Treeview
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.BOTH, expand=True)

            tree = ttk.Treeview(frame, columns=("Feature", "Value"), show='headings')
            tree.heading("Feature", text="Feature Name")
            tree.heading("Value", text="Value")
            
            # Scrollbar
            sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)

            for _, row in df.iterrows():
                val = f"{row['Value']:.4f}"
                tree.insert("", "end", values=(row['Feature Name'], val))

            tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            
    def _build_realtime_plot(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text=" LIVE EEG STREAM ")

        # Frame penampung Tabs Analisis (Lazy initialization)
        self.zap_tab = ttk.Frame(self.notebook)
        self.ica_tab = ttk.Frame(self.notebook)
        self.mif_tab = ttk.Frame(self.notebook)
        self.band_tab = ttk.Frame(self.notebook)
        self.bpm_tab = ttk.Frame(self.notebook)
        self.topo_tab = ttk.Frame(self.notebook)
        # Sub-tab di dalam Brain Mapping (diisi saat show_visual_results dipanggil)
        self.topo_abs_tab  = None
        self.topo_rel_tab  = None
        self.topo_ratio_tab = None
        self.topo_asym_tab = None
        self.topo_sasi_tab = None
        self.ica_compare_tab = ttk.Frame(self.notebook)
        self.feat_tab = ttk.Frame(self.notebook)
        self.save_tab = ttk.Frame(self.notebook)   # === TAB BARU ===
        self.signal_save_tab = ttk.Frame(self.notebook)  # === TAB SIGNAL ===
        self.signal_save_tab = ttk.Frame(self.notebook)  # TAB BARU: Saving Signal
        if not hasattr(self, 'signal_rec_id_var'):
            self.signal_rec_id_var = tk.IntVar(value=1)
        self.btn_save_signal = None  # referensi tombol, dipakai untuk disable setelah save
        
        # --- UPDATE WARNA UNTUK MEDICAL CLEAN STYLE ---
        
        # --- UPDATE WARNA UNTUK MEDICAL CLEAN STYLE ---
        bg_color = self.colors["bg"]         # #ffffff (Putih)
        plot_bg = "#ffffff"                  # Putih bersih untuk area grafik
        grid_color = "#f0f0f0"               # Abu-abu sangat muda untuk grid
        
        self.fig, self.axs = plt.subplots(3, 2, figsize=(9.5, 5.2), dpi=80)
        self.fig.patch.set_facecolor(bg_color) 
        self.ax_flat = self.axs.flatten()
        
        self.lines = []
        self.labels_imp_text = [] 
        
        for i, ch_name in enumerate(LIVE_EEG_NAMES):
            ax = self.ax_flat[i]
            # Sinyal EEG menggunakan Biru Royal (#1a73e8)
            line, = ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), 
                            color=self.colors["accent"], lw=1.2)
            self.lines.append(line)
            
            ax.set_ylabel(ch_name, color=self.colors["accent"], fontsize=9, fontweight='bold')
            ax.set_facecolor(plot_bg)
            
            # Warna angka sumbu menggunakan abu-abu gelap
            ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
            
            # Grid yang halus/clean
            ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
            
            # Hilangkan border atas dan kanan agar terlihat modern
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_edgecolor('#dddddd')
            ax.spines['bottom'].set_edgecolor('#dddddd')
            
            # Impedance Label Styling (Putih dengan border biru)
            lbl = ax.text(0.98, 0.90, "", transform=ax.transAxes, 
                        fontsize=9, color=self.colors["accent"], fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.9, 
                                 edgecolor=self.colors["accent"], boxstyle='round,pad=0.2'))
            self.labels_imp_text.append(lbl)

        # Digital Plot (Warna kontras tapi tetap di bg putih)
        # --- Subplot index 8: DIGITAL ---
        self.dig_ax = self.ax_flat[4]
        self.dig_line, = self.dig_ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color="#e91e63", lw=1.5)
        self.dig_ax.set_ylabel("DIGITAL", fontsize=9, color='#e91e63', fontweight='bold')
        self.dig_ax.set_facecolor(plot_bg)
        self.dig_ax.set_ylim(-0.2, 1.2)
        self.dig_ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
        self.dig_ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
        self.dig_ax.spines['top'].set_visible(False)
        self.dig_ax.spines['right'].set_visible(False)

        # --- Subplot index 9: PPG (tersembunyi sampai analysis selesai) ---
        self.ppg_ax = self.ax_flat[5]
        self.ppg_line, = self.ppg_ax.plot([], [], color="#ff9800", lw=1.2)
        self.ppg_ax.set_ylabel("PPG", fontsize=9, color="#ff9800", fontweight='bold')
        self.ppg_ax.set_facecolor(plot_bg)
        self.ppg_ax.tick_params(colors=self.colors["text_dim"], labelsize=8)
        self.ppg_ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
        self.ppg_ax.spines['top'].set_visible(False)
        self.ppg_ax.spines['right'].set_visible(False)
        self.ppg_ax.set_visible(False)   # sembunyikan dulu
        # self.ax_flat[7].axis('off')

        plt.tight_layout()

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        # ============================================================
        # Canvas live EEG
        # ============================================================
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.live_tab)
        self.canvas.draw()

        # Canvas diletakkan di atas toolbar
        self.canvas.get_tk_widget().pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True
        )

        # ============================================================
        # Toolbar zoom/pan/save untuk live EEG
        # ============================================================
        self.live_toolbar_frame = ttk.Frame(self.live_tab)
        self.live_toolbar_frame.pack(
            side=tk.BOTTOM,
            fill=tk.X
        )

        try:
            self.live_toolbar = NavigationToolbar2Tk(
                self.canvas,
                self.live_toolbar_frame,
                pack_toolbar=False
            )
            self.live_toolbar.update()
            self.live_toolbar.pack(side=tk.LEFT, fill=tk.X)

        except TypeError:
            # Fallback untuk versi matplotlib lama
            self.live_toolbar = NavigationToolbar2Tk(
                self.canvas,
                self.live_toolbar_frame
            )
            self.live_toolbar.update()
            

    def display_last_5s(self):
        """Tampilkan SELURUH rekaman EEG + PPG ke live plot setelah analysis selesai."""
        if self.session_data is None or len(self.session_data) == 0:
            return

        # --- Normalisasi session_data ---
        if isinstance(self.session_data, list):
            raw = np.concatenate(self.session_data, axis=1)
            eeg_all_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            eeg_ch = [eeg_all_ch[j] for j in LIVE_EEG_SOURCE_IDXS]
            eeg_data = raw[eeg_ch, :]
        else:
            raw = np.asarray(self.session_data)

            # Jika data offline/hasil load berisi 8 EEG, ambil Fp1,Fp2,T3,T4
            if raw.shape[0] >= 8:
                eeg_data = raw[LIVE_EEG_SOURCE_IDXS, :]

            # Jika data sudah 4CH, ambil langsung 4 channel pertama
            else:
                eeg_data = raw[:LIVE_EEG_N, :]

        n_samples = eeg_data.shape[1]
        t_plot = np.linspace(0, n_samples / FS, n_samples)
        t_max  = t_plot[-1]

        # --- Plot seluruh EEG ---
        n_available = eeg_data.shape[0]
        n_plot = min(LIVE_EEG_N, n_available, len(self.lines))

        for i in range(n_plot):
            sig = eeg_data[i]
            self.lines[i].set_data(t_plot, sig)

            ax = self.ax_flat[i]
            ax.set_xlim(0, t_max)                        # ← kunci: paksa batas X
            y_margin = max(np.ptp(sig) * 0.1, 1.0)
            ax.set_ylim(sig.min() - y_margin, sig.max() + y_margin)
            ax.set_xlabel("Time (s)", fontsize=7)

        # --- Plot Digital ---
        # (biarkan dig_line apa adanya, hanya sesuaikan xlim)
        self.dig_ax.set_xlim(0, t_max)

        # --- Plot PPG jika tersedia ---
        if hasattr(self, 'analysis_results') and self.analysis_results:
            bpm_data = self.analysis_results.get('bpm_data', {})
            # KODE BARU — tambah scatter untuk midpoint peaks:
            if 'ppg_signal' in bpm_data and 'ppg_time' in bpm_data:
                ppg_t   = np.array(bpm_data['ppg_time'])
                ppg_sig = np.array(bpm_data['ppg_signal'])

                self.ppg_ax.set_visible(True)
                self.ppg_ax.cla()  # bersihkan dulu agar scatter tidak numpuk

                self.ppg_ax.plot(ppg_t, ppg_sig, color="#ff9800", lw=1.2, label="PPG")

                # Tampilkan midpoint peaks jika tersedia
                ppg_peaks = bpm_data.get('ppg_peaks', np.array([]))
                if len(ppg_peaks) > 0:
                    self.ppg_ax.scatter(
                        ppg_t[ppg_peaks], ppg_sig[ppg_peaks],
                        color='blue', s=40, zorder=5, label='Peaks'
                    )

                self.ppg_ax.set_xlim(ppg_t[0], ppg_t[-1])
                y_ppg = max(np.ptp(ppg_sig) * 0.1, 0.1)
                self.ppg_ax.set_ylim(ppg_sig.min() - y_ppg, ppg_sig.max() + y_ppg)
                self.ppg_ax.set_xlabel("Time (s)", fontsize=7)
                self.ppg_ax.set_ylabel("PPG", fontsize=9, color="#ff9800", fontweight='bold')
                self.ppg_ax.grid(True, color='#eeeeee', linestyle='-', linewidth=0.5)

        self.fig.tight_layout()
        self.canvas.draw_idle()       # draw_idle lebih aman dari draw() di konteks after()
        self.notebook.select(0)
        
         
    def prepare_session(self):
        """Connect ke board di background thread agar GUI tidak freeze"""
        # Nonaktifkan tombol connect selama proses
        for widget in self.root.winfo_children():
            pass  # tidak perlu disable semua
        self.status_label.config(text="Status: Connecting...", foreground="orange")
        self.root.update_idletasks()

        def _do_connect():
            try:
                params = BrainFlowInputParams()
                params.serial_port = self.com_entry.get()
                board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                board.prepare_session()

                board.config_board("v")

                for ch in range(1, NUM_CHANNELS + 1):
                    cmd = f"x{ch}060101X"
                    board.config_board(cmd)
                    time.sleep(0.2)

                board.config_board("c")
                time.sleep(0.2)
                board.config_board("/3")
                time.sleep(1.0)

                # Simpan board ke self di main thread
                self.board = board
                self.root.after(0, self._on_connect_success)

            except Exception as e:
                err_msg = str(e)   # ← simpan dulu sebelum 'e' hilang
                self.root.after(0, lambda: self._on_connect_error(err_msg))

        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_connect_success(self):
        self.board_id =  BoardIds.CYTON_BOARD.value   # ← tambahkan ini
        self.status_label.config(text="Status: Connected (Ready)", foreground="green")
        self.start_monitoring_stream()

    def _on_connect_error(self, msg):
        """Dipanggil di main thread jika connect gagal"""
        self.board = None
        messagebox.showerror("Error Connection", msg)
        self.status_label.config(text="Status: Connection Failed", foreground="red")
        
    # def _build_realtime_plot(self):
    #         self.notebook = ttk.Notebook(self.root)
    #         self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
            
    #         self.live_tab = ttk.Frame(self.notebook)
    #         self.notebook.add(self.live_tab, text=" LIVE EEG STREAM ")

    #         # Frame penampung Tabs Analisis (Lazy initialization)
    #         self.zap_tab = ttk.Frame(self.notebook)
    #         self.ica_tab = ttk.Frame(self.notebook)
    #         self.mif_tab = ttk.Frame(self.notebook)
    #         self.band_tab = ttk.Frame(self.notebook)
    #         self.bpm_tab = ttk.Frame(self.notebook)
    #         self.topo_tab = ttk.Frame(self.notebook)
    #         self.feat_tab = ttk.Frame(self.notebook)

    #         # Warna plot background disamakan dengan UI background
    #         bg_color, plot_bg = self.colors["bg"], "#ffffff"
            
    #         self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 10))
    #         self.fig.patch.set_facecolor(bg_color) # Ini kuncinya agar menyatu
    #         self.ax_flat = self.axs.flatten()
            
    #         self.lines = []
    #         self.labels_imp_text = [] 
            
    #         for i in range(NUM_CHANNELS):
    #             ax = self.ax_flat[i]
    #             line, = ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color=self.colors["green"], lw=1)
    #             self.lines.append(line)
                
    #             ax.set_ylabel(f"CH {i+1}", color=self.colors["accent"], fontsize=9, fontweight='bold')
    #             ax.set_facecolor(plot_bg)
    #             ax.tick_params(colors='#5f6368', labelsize=8)
    #             ax.grid(True, color='#f0f0f0', linestyle='--', alpha=0.5)
                
    #             # Impedance Label Styling
    #             lbl = ax.text(0.98, 0.90, "", transform=ax.transAxes, 
    #                         fontsize=9, color='white', fontweight='bold',
    #                         ha='right', va='top',
    #                         bbox=dict(facecolor='#252526', alpha=0.8, edgecolor=self.colors["accent"], boxstyle='round,pad=0.2'))
    #             self.labels_imp_text.append(lbl)

    #         # Digital Plot
    #         self.dig_ax = self.ax_flat[6]
    #         self.dig_line, = self.dig_ax.plot(self.time_axis, np.zeros(MAX_SAMPLES), color="#ff00ff", lw=1.5)
    #         self.dig_ax.set_ylabel("DIGITAL", fontsize=9, color='#ff00ff', fontweight='bold')
    #         self.dig_ax.set_facecolor(plot_bg)
    #         self.dig_ax.set_ylim(-0.2, 1.2)
    #         self.dig_ax.tick_params(colors='#444444', labelsize=8)

    #         self.ax_flat[7].axis('off')

    #         plt.tight_layout()
    #         self.canvas = FigureCanvasTkAgg(self.fig, master=self.live_tab)
    #         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        
    def start_monitoring_stream(self):
        """Memulai stream untuk monitoring (belum merekam ke file)"""
        if self.board and not self.streaming:
            self.board.start_stream()
            self.streaming = True
            self.is_recording = False
            
        # SESUDAH
        if hasattr(self, 'btn_record'):
            self.btn_record.config(state=tk.NORMAL)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.config(state=tk.NORMAL)
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Status: Monitoring (Not Recording)", foreground="orange")
            self.update_live_plot()

    # ==============================
    # LOGIKA IMPEDANSI INDIVIDU
    # ==============================
    def toggle_channel_impedance(self, ch_idx):
        if not self.board:
            messagebox.showwarning("Warning", "Connect to board first.")
            return

        # Gunakan thread agar GUI tidak membeku saat menunggu time.sleep()
        def run_toggle():
            current_state = self.imp_states[ch_idx]
            new_state = not current_state
            self.imp_states[ch_idx] = new_state
            ch_char = str(ch_idx + 1) 

            if new_state:
                self.imp_buttons[ch_idx].config(bg="yellow")
                # Cek Impedansi: Gain 1x (0), Bias On (1), SRB2 On (1)
                cmd_config = f"x{ch_char}000101X"
                cmd_inject = f"z{ch_char}10Z"
                self._send_impedance_commands(cmd_config, cmd_inject)
            else:
                self.imp_buttons[ch_idx].config(bg="SystemButtonFace")
                # Mode Normal: Gain 24x (6), Bias On (1), SRB2 On (1)
                cmd_inject_off = f"z{ch_char}00Z"
                cmd_restore = f"x{ch_char}060101X"
                self._send_impedance_commands(cmd_inject_off, cmd_restore)
                
            # Bersihkan buffer channel ini agar data impedansi tidak mencemari EEG normal
            self.data_buffer[ch_idx].fill(0)
            self.last_known_imp[ch_idx] = None
            self.last_impedance_time[ch_idx] = 0  # Reset timer impedansi juga
            if hasattr(self, 'labels_imp_text') and ch_idx < len(self.labels_imp_text):
                # Hapus label impedansi dari plot agar tidak misleading
                self.root.after(0, lambda idx=ch_idx: self.labels_imp_text[idx].set_text(""))
                 
        threading.Thread(target=run_toggle, daemon=True).start()
                
    # Fungsi Helper untuk kirim command dengan jeda aman
    def _send_impedance_commands(self, cmd1, cmd2):
        if self.board:
            try:
                self.is_configuring = True
                
                # 1. Matikan stream jika aktif
                if self.streaming:
                    print("Stopping stream for hardware configuration...")
                    self.board.stop_stream()
                    self.streaming = False
                    time.sleep(0.5) # Jeda agar paket terakhir selesai terkirim

                # 2. AGGRESSIVE FLUSH (Kunci utama)
                # Ambil semua data sisa dari buffer internal BrainFlow
                for _ in range(3):
                    self.board.get_board_data()
                    time.sleep(0.1)

                # 3. Kirim perintah dengan Try-Except untuk UTF-8
                for cmd in [cmd1, cmd2]:
                    try:
                        print(f"Sending Command: {cmd}")
                        # Beberapa versi firmware butuh awalan 's' untuk sinkronisasi teks
                        # Namun x dan z standar biasanya langsung bisa
                        response = self.board.config_board(cmd)
                        print(f"Board Response: {response}")
                        time.sleep(0.3) # Jeda antar perintah agar firmware tidak hang
                    except Exception as e:
                        if "utf-8" in str(e).lower():
                            print(f"Terjadi sisa byte biner, tapi perintah {cmd} kemungkinan berhasil.")
                        else:
                            raise e

                # 4. Nyalakan kembali stream
                # 4. Nyalakan kembali stream
                print("Restarting stream...")
                self.board.start_stream()
                self.streaming = True

                # Flush agresif: tunggu 1 detik lalu buang semua data yang masuk
                # selama periode transisi gain (dari mode impedansi ke mode EEG normal)
                time.sleep(1.0)
                for _ in range(5):
                    flushed = self.board.get_board_data()
                    print(f"  Flush: {flushed.shape[1]} samples discarded")
                    time.sleep(0.1)

                print("Stream stable, ready.")
                
            except Exception as e:
                print(f"Error Konfigurasi: {e}")
                messagebox.showerror("Hardware Error", f"Gagal konfigurasi board: {e}")
            finally:
                self.is_configuring = False      
            
             
    def reset_all_impedance(self):
        """Mematikan semua impedansi (dipanggil saat mau rekam)"""
        for i, ch_name in enumerate(LIVE_EEG_NAMES):
            if self.imp_states[i]:
                self.toggle_channel_impedance(i) # Turn OFF

    # ==============================
    # RECORDING
    # ==============================
    def start_recording(self):
        """Mulai merekam dan aktifkan aliran data jika belum jalan"""

        # Jika board sudah di-release (setelah stop_and_save), reconnect dulu
        if not self.board:
            try:
                self.status_label.config(text="Status: Reconnecting...", foreground="orange")
                self.root.update_idletasks()

                params = BrainFlowInputParams()
                params.serial_port = self.com_entry.get()
                self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                self.board.prepare_session()
                self.board.config_board("v")
                for ch in range(1, NUM_CHANNELS + 1):
                    self.board.config_board(f"x{ch}060101X")
                    time.sleep(0.1)
                self.board.config_board("c")
                time.sleep(0.2)
                self.board.config_board("/3")
                time.sleep(0.5)
                self.board.start_stream()
                self.streaming = True
                
                self.update_live_plot()
            except Exception as e:
                err_msg = str(e)
                messagebox.showerror("Reconnect Error", f"Gagal reconnect ke board:\n{err_msg}")
                self.status_label.config(text="Status: Reconnect Failed", foreground="red")
                self.board = None
                return

        # Jalankan stream board jika masih idle
        if not self.streaming:
            self.board.start_stream()
            self.streaming = True
            self.update_live_plot()

        # ✅ BAGIAN INI YANG HILANG — set state dan mulai countdown
        self.is_recording = True
        self.session_data = []
        self.data_buffer.fill(0)

        self.reset_all_impedance()

        for i, ch_name in enumerate(LIVE_EEG_NAMES):
            
            self.ax_flat[i].set_ylim(-100, 100)

        self.status_label.config(text="Status: RECORDING DATA...", foreground="red")
        self.btn_record.config(state=tk.DISABLED, text="▶ Recording...")
        self.record_start_time = time.time()
        self._start_countdown(self.duration_var.get())
        
        
    def _start_countdown(self, total_seconds):
        """Mulai countdown dari total_seconds."""
        if self._countdown_job:
            self.root.after_cancel(self._countdown_job)
        self._tick_countdown(total_seconds)


    def _tick_countdown(self, total_seconds):
        if not self.is_recording:
            self.countdown_label.config(text="")
            return
        
        # Hitung sisa waktu dari waktu nyata, bukan dari counter
        elapsed = time.time() - self.record_start_time
        remaining = total_seconds - elapsed
        
        if remaining <= 0:
            self.countdown_label.config(text="00:00")
            self.stop_and_save()
            return
        
        mins, secs = divmod(int(remaining), 60)
        self.countdown_label.config(text=f"{mins:02d}:{secs:02d}")
        # Jadwalkan ulang setiap ~100ms agar lebih responsif dan akurat
        self._countdown_job = self.root.after(100, self._tick_countdown, total_seconds)  
    def stop_and_save(self):
            # 1. Pastikan kita sedang merekam
            if not self.is_recording:
                messagebox.showinfo("Info", "Tidak ada proses rekaman yang berjalan.")
                return

# SESUDAH (aman):
            if getattr(self, '_countdown_job', None):
                self.root.after_cancel(self._countdown_job)
                self._countdown_job = None
            if hasattr(self, 'countdown_label'):
                self.countdown_label.config(text="")
            
            self.is_recording = False
            # time.sleep(0.2)
            
            # 2. Ambil data terakhir dari board jika masih ada
            if self.board:
                try:
                    # Ambil sisa data yang belum terbaca
                    remaining_data = self.board.get_board_data()
                    if remaining_data.shape[1] > 0:
                        self.session_data.append(remaining_data)
                    
                    # Stop stream dan release agar port COM bersih
                    self.board.stop_stream()
                    self.board.release_session()
                    self.streaming = False
                    print("Stream stopped and session released.")
                except Exception as e:
                    print(f"Error closing board: {e}")
                
                self.board = None # Reset object board

            # 3. Proses Penggabungan dan Penyimpanan
            if self.session_data:
                try:
                    full_data = np.concatenate(self.session_data, axis=1)
                    custom_name = self.filename_var.get().strip()
                    if not custom_name:
                        custom_name = "EEG_Recording"
                    # Bersihkan karakter ilegal dari nama file
                    import re
                    custom_name = re.sub(r'[\\/*?:"<>|]', "_", custom_name)
                    self.last_filename = f"{custom_name}_{int(time.time())}.txt"
                    self.current_source = "LIVE"

                    eeg_all = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

                    # Ambil hanya EXG0, EXG1, EXG4, EXG5
                    eeg_ch = [eeg_all[i] for i in LIVE_EEG_SOURCE_IDXS]

                    DIGITAL_CHANNEL = 13
                    save_data = full_data[eeg_ch + [DIGITAL_CHANNEL], :]

                    # Simpan ke TXT
                    with open(self.last_filename, 'w') as f:
                        f.write("% OpenBCI EEG Recording\n")
                        f.write(f"% FS: {FS} Hz | MODE: 4CH\n")

                        header_cols = [f"EXG Channel {i}" for i in range(LIVE_EEG_N)] + ["Digital"]
                        header = "% " + "\t".join(header_cols) + "\n"

                        f.write(header)
                        np.savetxt(f, save_data.T, delimiter='\t', fmt='%.4f')
                    # TAMPILKAN STATUS & AKTIFKAN TOMBOL ANALISIS
                    self.status_label.config(text=f"Saved: {self.last_filename}", foreground="green")
                    self.btn_analyze.config(state=tk.NORMAL) # AKTIFKAN DI SINI
                    self.btn_record.config(state=tk.NORMAL, text="Start Recording") # Siap rekam lagi
                    
                    messagebox.showinfo("Success", f"Data berhasil disimpan ke {self.last_filename}\nSekarang Anda bisa menjalankan Analysis.")

                except Exception as e:
                    messagebox.showerror("Save Error", f"Gagal menyimpan data: {e}")
            else:
                messagebox.showwarning("Warning", "Rekaman terlalu pendek, tidak ada data untuk disimpan.")
                self.status_label.config(text="Status: No Data Recorded", foreground="red")
                self.btn_record.config(state=tk.NORMAL, text="Start Recording")
                
    def compute_sasi(self, psd, freqs, fc):
        f1 = fc - 6
        f2 = fc - 2
        f3 = fc + 2
        f4 = min(fc + 26, 30)  # limit 30 Hz

        low_mask  = (freqs >= f1) & (freqs <= f2)
        high_mask = (freqs >= f3) & (freqs <= f4)

        Wl = np.sum(psd[low_mask])
        Wh = np.sum(psd[high_mask])

        if (Wh + Wl) == 0:
            return 0

        return (Wh - Wl) / (Wh + Wl)

# =========================================================
# HELPER: Embed topomap + tabel angka dalam satu frame
# =========================================================
    def _plot_topo_with_table(self, parent, fig, data_dict, band_names, ch_names):
        """
        Tampilkan figure topomap di atas dan tabel angka di bawah.
        data_dict: {band_name: list_nilai_per_channel}
        """
        # Frame atas: topomap
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True)
        self._embed_canvas(fig, top_frame)

        # Frame bawah: tabel angka (scrollable)
        bot_frame = ttk.Frame(parent)
        bot_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        cols = ["Channel"] + band_names
        tree = ttk.Treeview(bot_frame, columns=cols, show='headings', height=len(ch_names))
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=80, anchor='center')

        for ch_i, ch in enumerate(ch_names):
            row_vals = [ch] + [f"{data_dict[b][ch_i]:.4f}" for b in band_names]
            tree.insert("", "end", values=row_vals)

        sb = ttk.Scrollbar(bot_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")


    def plot_brain_mapping(self, data, parent):
        if not self.analysis_results:
            return

        for widget in parent.winfo_children():
            widget.destroy()

        # --- Data & konstanta ---
        abs_power  = data['abs_power']
        rel_power  = data['rel_power']
        band_names = data['names']
        info       = self.analysis_results['topo_info']
        ch_names   = info['ch_names']

        # ============================================================
        # TOPO COMPAT FIX
        # ============================================================
        # Beberapa versi eeg_processor_8 / MNE dapat mengembalikan topo_info
        # dengan jumlah channel tidak sama dengan data band-power.
        # Error yang muncul:
        # "Number of channels in the Info object (2) and the data array (8) do not match"
        # Solusi aman: rebuild Info object sesuai jumlah nilai topomap.
        # Untuk posisi MNE, T3/T4 dipetakan ke T7/T8, tetapi tabel tetap memakai
        # nama sistem lama T3/T4.
        try:
            first_band_for_topo = band_names[0]
            n_data_ch = len(np.asarray(abs_power[first_band_for_topo], dtype=float).reshape(-1))
        except Exception:
            n_data_ch = len(ch_names)

        default_display_names = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
        display_ch_names = default_display_names[:n_data_ch]
        plot_ch_names = ["T7" if ch == "T3" else "T8" if ch == "T4" else ch for ch in display_ch_names]

        need_rebuild_info = False
        try:
            need_rebuild_info = (len(info["ch_names"]) != n_data_ch)
        except Exception:
            need_rebuild_info = True

        # Rebuild juga jika masih memakai T3/T4, karena standard_1020 MNE lebih aman
        # memakai T7/T8 untuk posisi temporal.
        try:
            if any(ch in ["T3", "T4"] for ch in info["ch_names"]):
                need_rebuild_info = True
        except Exception:
            pass

        if need_rebuild_info:
            try:
                print(f"[TOPO COMPAT] Rebuild topo_info: info={len(info['ch_names']) if hasattr(info, 'ch_names') or isinstance(info, dict) else 'unknown'} ch, data={n_data_ch} ch")
            except Exception:
                print(f"[TOPO COMPAT] Rebuild topo_info untuk data={n_data_ch} ch")

            info = mne.create_info(ch_names=plot_ch_names, sfreq=FS, ch_types="eeg")
            montage = mne.channels.make_standard_montage("standard_1020")
            try:
                info.set_montage(montage, match_case=False, on_missing="ignore")
            except TypeError:
                info.set_montage(montage, match_case=False)

            self.analysis_results["topo_info"] = info
            ch_names = display_ch_names
        else:
            # Nama tabel tetap disamakan dengan urutan fitur lama.
            ch_names = display_ch_names if n_data_ch == len(default_display_names) else list(info["ch_names"])

        n_bands    = len(band_names)
        def local_vlim(values, symmetric=False):
            """
            Membuat vmin-vmax lokal untuk satu topomap.
            symmetric=True dipakai kalau nilai bisa negatif-positif,
            misalnya asymmetry/SASI.
            """
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                return 0.0, 1.0

            if symmetric:
                m = np.max(np.abs(vals))
                if m < 1e-12:
                    m = 1.0
                return -m, m

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            # Proteksi kalau semua nilai sama
            if np.isclose(vmin, vmax):
                pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
                vmin -= pad
                vmax += pad

            return vmin, vmax

        ratio_defs = [
            ("Delta/Theta", "Delta", "Theta"),
            ("Delta/Alpha", "Delta", "Alpha"),
            ("Delta/Beta",  "Delta", "Beta"),
            ("Theta/Alpha", "Theta", "Alpha"),
            ("Theta/Beta",  "Theta", "Beta"),
            ("Alpha/Beta",  "Alpha", "Beta"),
        ]
        asym_pairs = [
            ("Fp1-Fp2", 0, 1),
            ("F3-F4",   2, 3),
            ("T3-T4",   4, 5),
            ("P3-P4",   6, 7),
        ]
        ch_to_pair = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3}

        # --- Hitung SASI ---
        # --- Hitung / ambil SASI ---
        if "sasi_values" in data:
            sasi_values = np.array(data["sasi_values"], dtype=float)
        else:
            from scipy.signal import welch as sp_welch

            if isinstance(self.session_data, list):
                eeg_data = np.concatenate(self.session_data, axis=1)
                eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)[:NUM_CHANNELS]
                eeg_data = eeg_data[eeg_ch, :]
            else:
                eeg_data = self.session_data[:NUM_CHANNELS, :]

            freqs, psd_all = sp_welch(eeg_data, 250, nperseg=500, axis=1)
            alpha_mask = (freqs >= 8) & (freqs <= 12)

            if np.any(alpha_mask):
                fc = freqs[alpha_mask][np.argmax(psd_all[:, alpha_mask].mean(axis=0))]
                sasi_values = np.array([
                    self.compute_sasi(psd_all[ch], freqs, fc)
                    for ch in range(psd_all.shape[0])
                ])
            else:
                sasi_values = np.zeros(NUM_CHANNELS)
                
                
        # --- Hitung rasio ---
        # --- Hitung / ambil rasio ---
        if "ratios" in data:
            ratios = {k: np.array(v, dtype=float) for k, v in data["ratios"].items()}
        else:
            ratios = {}
            for label, nb, db in ratio_defs:
                num = np.array(abs_power[nb], dtype=float)
                den = np.where(
                    np.array(abs_power[db], dtype=float) == 0,
                    1e-10,
                    np.array(abs_power[db], dtype=float)
                )
                ratios[label] = num / den

        # --- Hitung asimetri per band ---
        # --- Hitung / ambil asimetri per band ---
        if "asym_by_band" in data:
            asym_by_band = {
                k: np.array(v, dtype=float).tolist()
                for k, v in data["asym_by_band"].items()
            }
        else:
            asym_by_band = {}
            for b_idx, band in enumerate(band_names):
                asym_topo = np.zeros(len(ch_names))
                p = np.array(abs_power[band], dtype=float)

                for ch_idx, pair_idx in ch_to_pair.items():
                    if ch_idx < len(p):
                        _, li, ri = asym_pairs[pair_idx]
                        L, R = p[li], p[ri]
                        denom = L + R
                        asym_topo[ch_idx] = (L - R) / denom if denom != 0 else 0.0

                asym_by_band[band] = asym_topo.tolist()
                
        # -----------------------------------------------
        # Buat Sub-Notebook di dalam topo_tab
        # -----------------------------------------------
        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill=tk.BOTH, expand=True)

        frames = {}
        for label in [
            "Absolute Power",
            "Relative Power",
            "Power Ratio",
            "Asimetri Amplitudo",
            "SASI",
            "LZC Topomap",
            "Pengurangan Topomap"
        ]:
            f = ttk.Frame(sub_nb)
            sub_nb.add(f, text=f" {label} ")
            frames[label] = f

        # =====================
        # TAB 1: Absolute Power
                # =====================
        fig1, axs1 = plt.subplots(1, n_bands, figsize=(3.8*n_bands, 4))
        if n_bands == 1:
            axs1 = [axs1]

        for i, band in enumerate(band_names):
            vals = np.array(abs_power[band], dtype=float)
            vmin_b, vmax_b = local_vlim(vals)

            im, _ = mne.viz.plot_topomap(
                vals, info, axes=axs1[i],
                show=False, cmap='jet', extrapolate='head',
                vlim=(vmin_b, vmax_b),
                sphere=0.1
            )

            axs1[i].set_title(
                f"{band}\n[{vmin_b:.3g}, {vmax_b:.3g}]",
                fontsize=9,
                fontweight='bold'
            )

            fig1.colorbar(im, ax=axs1[i], shrink=0.65, label='µV²/Hz')
        plt.tight_layout()
        self._plot_topo_with_table(frames["Absolute Power"], fig1, abs_power, band_names, ch_names)

        # =====================
        # TAB 2: Relative Power
                # =====================
        fig2, axs2 = plt.subplots(1, n_bands, figsize=(3.8*n_bands, 4))
        if n_bands == 1:
            axs2 = [axs2]

        for i, band in enumerate(band_names):
            vals = np.array(rel_power[band], dtype=float)
            vmin_b, vmax_b = local_vlim(vals)

            im2, _ = mne.viz.plot_topomap(
                vals, info, axes=axs2[i],
                show=False, cmap='jet', extrapolate='head',
                vlim=(vmin_b, vmax_b)
            )

            axs2[i].set_title(
                f"{band}\n[{vmin_b:.3g}, {vmax_b:.3g}]",
                fontsize=9,
                fontweight='bold'
            )

            fig2.colorbar(im2, ax=axs2[i], shrink=0.65, label='Relative')
            
        plt.tight_layout()
        self._plot_topo_with_table(frames["Relative Power"], fig2, rel_power, band_names, ch_names)

        # =====================
        # TAB 3: Power Ratio
        # =====================
        n_ratios  = len(ratio_defs)
        cols_r    = min(n_ratios, 4)
        rows_r    = int(np.ceil(n_ratios / cols_r))
        fig3, axs3 = plt.subplots(rows_r, cols_r,
                                figsize=(3.5*cols_r, 4*rows_r),
                                squeeze=False)
        for r_idx, (label, _, _) in enumerate(ratio_defs):
            ax = axs3[r_idx // cols_r][r_idx % cols_r]

            vals = np.array(ratios[label], dtype=float)
            vmin_r, vmax_r = local_vlim(vals)

            im3, _ = mne.viz.plot_topomap(
                vals, info, axes=ax,
                show=False, cmap='jet', extrapolate='head',
                vlim=(vmin_r, vmax_r)
            )

            ax.set_title(
                f"{label}\n[{vmin_r:.3g}, {vmax_r:.3g}]",
                fontsize=8,
                fontweight='bold'
            )

            fig3.colorbar(im3, ax=ax, shrink=0.65)
        # Matikan axis kosong
        for r_idx in range(n_ratios, rows_r * cols_r):
            axs3[r_idx // cols_r][r_idx % cols_r].axis('off')
        # if last_im3:
        #     plt.colorbar(last_im3, ax=axs3.flatten()[-1], shrink=0.7)
        plt.tight_layout()
        # Tabel rasio
        ratio_dict = {lbl: ratios[lbl].tolist() for lbl, _, _ in ratio_defs}
        ratio_labels = [lbl for lbl, _, _ in ratio_defs]
        # Tabel: baris=channel, kolom=rasio
        bot3 = ttk.Frame(frames["Power Ratio"])
        top3 = ttk.Frame(frames["Power Ratio"])
        top3.pack(fill=tk.BOTH, expand=True)
        bot3.pack(fill=tk.X, padx=10, pady=(0,8))
        self._embed_canvas(fig3, top3)
        cols3 = ["Channel"] + ratio_labels
        tree3 = ttk.Treeview(bot3, columns=cols3, show='headings', height=len(ch_names))
        for c in cols3:
            tree3.heading(c, text=c); tree3.column(c, width=85, anchor='center')
        for ci, ch in enumerate(ch_names):
            row3 = [ch] + [f"{ratio_dict[lb][ci]:.4f}" for lb in ratio_labels]
            tree3.insert("", "end", values=row3)
        sb3 = ttk.Scrollbar(bot3, orient="vertical", command=tree3.yview)
        tree3.configure(yscrollcommand=sb3.set)
        tree3.pack(side="left", fill="x", expand=True)
        sb3.pack(side="right", fill="y")
        # ============================================================
        # LZC TOPO MAP
        # ============================================================
        if "lzc_maps" in data and data["lzc_maps"]:
            lzc_frame = frames["LZC Topomap"]

            fig_lzc, axs_lzc = plt.subplots(1, len(data["lzc_maps"]), figsize=(4 * len(data["lzc_maps"]), 4))
            axs_lzc = np.asarray(axs_lzc).reshape(-1)

            lzc_names = list(data["lzc_maps"].keys())

            vlims = {}
            try:
                vlims = self.analysis_results.get("brainmap_vlims", {}).get("lzc_maps", {})
            except Exception:
                vlims = {}

            info_lzc = self._make_sr8_topo_info()

            for i, name in enumerate(lzc_names):
                vals = np.asarray(data["lzc_maps"][name], dtype=float)

                if name in vlims:
                    vlim = vlims[name]
                else:
                    vmin = float(np.nanmin(vals))
                    vmax = float(np.nanmax(vals))
                    if np.isclose(vmin, vmax):
                        vmin -= 0.01
                        vmax += 0.01
                    vlim = (vmin, vmax)

                im, _ = mne.viz.plot_topomap(
                    vals,
                    info_lzc,
                    axes=axs_lzc[i],
                    show=False,
                    cmap="viridis",
                    extrapolate="head",
                    vlim=vlim
                )

                axs_lzc[i].set_title(name, fontsize=9, fontweight="bold")
                fig_lzc.colorbar(im, ax=axs_lzc[i], shrink=0.65)

            fig_lzc.tight_layout()

            canvas_lzc = FigureCanvasTkAgg(fig_lzc, master=lzc_frame)
            canvas_lzc.draw()
            canvas_lzc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        else:
            ttk.Label(
                frames["LZC Topomap"],
                text="LZC topomap belum tersedia. Jalankan SR + ML terlebih dahulu.",
                foreground="gray"
            ).pack(padx=20, pady=20)
            
    # =====================
        # TAB 4: Asimetri Amplitudo (bentuk asli: sensor map + garis)
        # =====================
        from itertools import combinations
        from mne.channels.layout import _find_topomap_coords
        from matplotlib.lines import Line2D # Tambahkan import ini di bagian atas fungsi

        cols_a  = min(n_bands, 4)
        rows_a  = int(np.ceil(n_bands / cols_a))
        fig4, axs4 = plt.subplots(rows_a, cols_a,
                                figsize=(3.5*cols_a, 4*rows_a),
                                squeeze=False)

        cmap_a = plt.cm.viridis
        norm_a = plt.Normalize(vmin=0.1, vmax=0.8)
        pos2d  = _find_topomap_coords(info, picks=np.arange(len(ch_names)))
        ch_index = {ch: i for i, ch in enumerate(ch_names)}
        all_pairs = list(combinations(ch_names, 2))

        for b_idx, band in enumerate(band_names):
            ax = axs4[b_idx // cols_a][b_idx % cols_a]
            ax.set_title(f"Asym {band}", fontsize=8, fontweight='bold')

            # 1. Plot kepala + sensor (sama persis seperti kode asli)
            mne.viz.plot_sensors(info, axes=ax, show=False, kind='topomap')

            p = np.array(abs_power[band], dtype=float)

            # 2. Gambar garis antar channel (sama persis seperti kode asli)
            for ch1, ch2 in all_pairs:
                i = ch_index[ch1]
                j = ch_index[ch2]
                denom = p[i] + p[j]
                val  = (p[i] - p[j]) / denom if denom != 0 else 0
                coh  = abs(val)
                if coh > 0.2:
                    x1, y1 = pos2d[i]
                    x2, y2 = pos2d[j]
                    line_color = 'red' if val > 0 else 'blue'
                    ax.plot([x1, x2], [y1, y2],
                            color=line_color,
                            linewidth=0.5 + 1.5*coh,
                            alpha=0.7,
                            solid_capstyle='round')

    # --- TAMBAHKAN LEGENDA SKALA DISINI ---
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='(-) Asimetri'),
            Line2D([0], [0], color='red', lw=2, label='(+) Asimetri')
        ]
        fig4.legend(handles=legend_elements, loc='lower center', 
                    ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))

    # --- MASUKKAN KODE INI SEBAGAI GANTINYA ---
        import matplotlib.colors as mcolors

        # 1. Tambahkan bar skala menyambung di posisi tengah bawah
        ax_bar = fig4.add_axes([0.4, 0.07, 0.2, 0.015]) 
        cmap_scale = mcolors.ListedColormap(['blue', 'red'])
        norm_scale = mcolors.Normalize(vmin=-1, vmax=1)
        
        cb = fig4.colorbar(
            plt.cm.ScalarMappable(norm=norm_scale, cmap=cmap_scale),
            cax=ax_bar, orientation='horizontal', ticks=[]
        )
        cb.outline.set_visible(False) 

        # 2. Tambahkan simbol '-' dan '+'
        ax_bar.text(-0.2, 0.5, '-', transform=ax_bar.transAxes, 
                    va='center', ha='right', fontsize=14, fontweight='bold', color='blue')
        ax_bar.text(1.2, 0.5, '+', transform=ax_bar.transAxes, 
                    va='center', ha='left', fontsize=12, fontweight='bold', color='red')

        # 3. Matikan axis kosong
        for b_idx in range(n_bands, rows_a * cols_a):
            axs4[b_idx // cols_a][b_idx % cols_a].axis('off')

        # 4. Atur layout agar bar skala tidak terpotong (rect=[left, bottom, right, top])
        plt.tight_layout(rect=[0, 0.1, 1, 1]) 
        # ============================================================
        # SIMPAN FIGURE ASIMETRI GUI UNTUK REPORT PDF
        # ============================================================
        try:
            if not hasattr(self, "_report_fig_cache"):
                self._report_fig_cache = {}

            self._report_fig_cache["Asimetri Amplitudo"] = fig4
            print("[REPORT FIG] Figure Asimetri Amplitudo dari GUI disimpan untuk PDF.")
        except Exception as e:
            print(f"[REPORT FIG] Gagal menyimpan fig asimetri GUI: {e}")
    
        self._plot_topo_with_table(frames["Asimetri Amplitudo"], fig4,
                                    asym_by_band, band_names, ch_names)
        # TAB 5: SASI
        # =====================
        fig5, ax5 = plt.subplots(1, 1, figsize=(4, 4))
        # ============================================================
        # SASI TOPO SAFE LENGTH
        # ============================================================
        sasi_values = np.asarray(sasi_values, dtype=float).reshape(-1)

        # Kalau sedang ada hasil SR8, pakai topo_info SR8 dan paksa panjang 8
        if (
            hasattr(self, "analysis_results")
            and isinstance(self.analysis_results, dict)
            and "sr8" in self.analysis_results
        ):
            topo_info_sasi = self._make_sr8_topo_info()
            target_len = 8
        else:
            topo_info_sasi = info
            try:
                target_len = len(topo_info_sasi["ch_names"])
            except Exception:
                target_len = len(sasi_values)

        # Jika sasi_values mengandung PPG atau extra row, potong.
        # Jika kurang, pad nol.
        if len(sasi_values) != target_len:
            print(
                f"[TOPO COMPAT] SASI length {len(sasi_values)} "
                f"disesuaikan ke {target_len}"
            )

            if len(sasi_values) > target_len:
                sasi_values = sasi_values[:target_len]
            else:
                sasi_values = np.pad(
                    sasi_values,
                    (0, target_len - len(sasi_values)),
                    mode="constant",
                    constant_values=0.0
                )

        im5, _ = mne.viz.plot_topomap(
            sasi_values,
            topo_info_sasi,
            axes=ax5,
            cmap="jet",
            vlim=(-1, 1),
            show=False,
            extrapolate="head"
        )

        plt.colorbar(im5, ax=ax5, shrink=0.7, label="SASI")
        ax5.set_title(
            "SASI\n(Spectral Asymmetry Index)",
            fontsize=9,
            fontweight="bold"
        )

        plt.tight_layout()
        # Tabel SASI
        top5 = ttk.Frame(frames["SASI"])
        bot5 = ttk.Frame(frames["SASI"])
        top5.pack(fill=tk.BOTH, expand=True)
        bot5.pack(fill=tk.X, padx=10, pady=(0,8))
        self._embed_canvas(fig5, top5)
        tree5 = ttk.Treeview(bot5, columns=["Channel", "SASI"], show='headings', height=len(ch_names))
        for c in ["Channel", "SASI"]:
            tree5.heading(c, text=c); tree5.column(c, width=120, anchor='center')
        for ci, ch in enumerate(ch_names):
            tree5.insert("", "end", values=(ch, f"{sasi_values[ci]:.4f}"))
        sb5 = ttk.Scrollbar(bot5, orient="vertical", command=tree5.yview)
        tree5.configure(yscrollcommand=sb5.set)
        tree5.pack(side="left", fill="x", expand=True)
        sb5.pack(side="right", fill="y")
        # =====================
        # TAB 6: Pengurangan Topomap Absolute Power
        # =====================
        self.build_topomap_subtraction_tab(frames["Pengurangan Topomap"])
        
    # =========================================
    # REVISI FINAL: METODE FFT WINDOWING (MIRIP OPENBCI JAVA)
    # =========================================
    # def _calculate_impedance_ui(self, raw_data):
    #         """
    #         Metode Lock-in Amplifier: Paling robust terhadap noise 50Hz.
    #         Sangat cocok untuk riset medis TEOAE/EEG.
    #         """
    #         if len(raw_data) < FS: return 0.0

    #         # --- KONSTANTA ---
    #         CURRENT_AMP = 6.0e-9 
    #         R_SERIES = 2200.0     
    #         TARGET_FREQ = 31.25 # Frekuensi injeksi ADS1299
            
    #         # 1. Pre-Processing & Scaling
    #         data_uv = raw_data * 24.0
    #         data_ac = data_uv - np.mean(data_uv)

    #         # 2. LAPIS 1: COMB FILTER (NOTCH 50Hz SEMPURNA)
    #         # Karena 250Hz / 50Hz = 5, rata-rata 5 sampel akan menghilangkan 50Hz.
    #         data_comb = np.convolve(data_ac, np.ones(5)/5, mode='same')

    #         # 3. LAPIS 2: LOCK-IN AMPLIFIER
    #         t = np.arange(len(data_comb)) / FS
            
    #         # Referensi Sinus & Kosinus (In-Phase & Quadrature)
    #         ref_sin = np.sin(2 * np.pi * TARGET_FREQ * t)
    #         ref_cos = np.cos(2 * np.pi * TARGET_FREQ * t)
            
    #         # Perkalian (Mixing)
    #         mixed_i = data_comb * ref_sin
    #         mixed_q = data_comb * ref_cos

    #         # 4. LAPIS 3: LPF EKSTREM (Sangat Robust)
    #         # Kita ambil rata-rata dari seluruh window (ini LPF paling tajam)
    #         avg_i = np.mean(mixed_i)
    #         avg_q = np.mean(mixed_q)

    #         # Hitung Amplitudo: A = 2 * sqrt(I^2 + Q^2)
    #         # Faktor 2 berasal dari identitas trigonometri sin^2(x)
    #         v_peak_uv = 2 * np.sqrt(avg_i**2 + avg_q**2)
    #         v_peak_volts = v_peak_uv * 1e-6

    #         # 5. Hitung Impedansi
    #         z_kohm = ((v_peak_volts / CURRENT_AMP) - R_SERIES) / 1000.0

    #         # --- VALIDASI KONTAK (HEURISTIC) ---
    #         # Jika sinyal mentok (Rail), kembalikan nilai tinggi
    #         if np.abs(np.mean(data_uv)) > 1000000.0:
    #             return 99999.0

    #         # Jika elektroda mengambang, noise acak tidak akan fase-locked dengan 31.25Hz.
    #         # Namun tegangan total (STD) akan sangat tinggi.
    #         total_std = np.std(data_comb)
    #         if total_std > 800.0 and z_kohm < 100.0:
    #             return 5800.0 + np.random.uniform(-100, 100)

    #         # Clamping
    #         if z_kohm > 6000.0: return 6000.0 + np.random.uniform(-100, 100)
            
    #         return max(0.0, z_kohm)
    def plot_topomap_subtraction_result(
        self,
        parent,
        abs_1,
        abs_2,
        diff_abs,
        band_names,
        d1_label="Eyes open",
        d2_label="Eyes closed"
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        import mne
        from tkinter import ttk

        for widget in parent.winfo_children():
            widget.destroy()

        info = self.analysis_results["topo_info"]
        ch_names = info["ch_names"]
        n_bands = len(band_names)

    def _fit_values_to_info(vals, info_obj, label="topomap"):
        vals = np.asarray(vals, dtype=float).reshape(-1)

        try:
            n_info = len(info_obj["ch_names"])
        except Exception:
            n_info = len(vals)

        if len(vals) != n_info:
            print(f"[TOPO COMPAT] {label}: values={len(vals)}, info={n_info}. Disesuaikan.")

            if len(vals) > n_info:
                vals = vals[:n_info]
            else:
                vals = np.pad(
                    vals,
                    (0, n_info - len(vals)),
                    mode="constant",
                    constant_values=0.0
                )

        return vals
        # =====================================================
        # Helper skala warna
        # =====================================================
        def safe_vlim(vals, symmetric=False):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                return 0.0, 1.0

            if symmetric:
                m = float(np.max(np.abs(vals)))
                if m < 1e-12:
                    m = 1.0
                return -m, m

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if np.isclose(vmin, vmax):
                pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
                return vmin - pad, vmax + pad

            return vmin, vmax

        # =====================================================
        # Buat figure 3 x n_bands = 12 topomap kalau band = 4
        # =====================================================
        fig, axs = plt.subplots(
            3,
            n_bands,
            figsize=(4.2 * n_bands, 12.5),
            squeeze=False
        )

        fig.patch.set_facecolor("#ffffff")
        # fig.suptitle(
        #     f"Absolute Power Topomap Subtraction: {d2_label} - {d1_label}",
        #     fontsize=12,
        #     fontweight="bold",
        #     y=0.985
        # )

        row_labels = [
            f"Eyes open\n{d1_label}",
            f"Eyes closed\n{d2_label}",
            "Selisih\nClosed - Open"
        ]

        # =====================================================
        # Plot 12 topomap
        # Setiap topomap punya vmin/vmax sendiri
        # =====================================================
        for b_idx, band in enumerate(band_names):
            vals1 = np.array(abs_1[band], dtype=float)
            vals2 = np.array(abs_2[band], dtype=float)
            valsd = np.array(diff_abs[band], dtype=float)

            # Skala masing-masing topomap
            vmin1, vmax1 = safe_vlim(vals1, symmetric=False)
            vmin2, vmax2 = safe_vlim(vals2, symmetric=False)

            # Untuk difference, saya sarankan tetap symmetric=True
            # supaya nilai negatif/positif seimbang terhadap nol.
            vmind, vmaxd = safe_vlim(valsd, symmetric=True)

            # =========================
            # Row 1: Durasi 1
            # =========================
            im1, _ = mne.viz.plot_topomap(
                vals1,
                info,
                axes=axs[0, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmin1, vmax1),
                sphere=0.1
            )

            axs[0, b_idx].set_title(
                f"{band}\n[{vmin1:.3g}, {vmax1:.3g}]",
                fontsize=8,
                fontweight="bold",
                pad=10
            )

            fig.colorbar(im1, ax=axs[0, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

            # =========================
            # Row 2: Durasi 2
            # =========================
            im2, _ = mne.viz.plot_topomap(
                vals2,
                info,
                axes=axs[1, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmin2, vmax2),
                sphere=0.1
            )

            axs[1, b_idx].set_title(
                f"{band}\n[{vmin2:.3g}, {vmax2:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig.colorbar(im2, ax=axs[1, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

            # =========================
            # Row 3: Difference D2 - D1
            # =========================
            imd, _ = mne.viz.plot_topomap(
                valsd,
                info,
                axes=axs[2, b_idx],
                show=False,
                cmap=LIGHT_JET,
                extrapolate="head",
                vlim=(vmind, vmaxd),
                sphere=0.1
            )

            axs[2, b_idx].set_title(
                f"{band} Δ\n[{vmind:.3g}, {vmaxd:.3g}]",
                fontsize=9,
                fontweight="bold"
            )

            fig.colorbar(imd, ax=axs[2, b_idx], shrink=0.55, fraction=0.035, pad=0.02)

        # =====================================================
        # Label baris
        # =====================================================
        for r in range(3):
            axs[r, 0].set_ylabel(
                row_labels[r],
                fontsize=10,
                fontweight="bold",
                rotation=0,
                labelpad=45,
                va="center"
            )

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            bottom=0.06,
            top=0.90,
            wspace=0.55,
            hspace=0.80
        )

        # =====================================================
        # Embed figure ke GUI
        # =====================================================
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self._embed_canvas(fig, top_frame)

        # =====================================================
        # Tabel nilai
        # =====================================================
        table_frame = ttk.LabelFrame(parent, text=" Nilai Topomap per Channel ", padding=6)
        table_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        cols = ["Channel"]
        for band in band_names:
            cols += [f"{band}_D1", f"{band}_D2", f"{band}_D2-D1"]

        tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            height=len(ch_names)
        )

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=85, anchor="center")

        for ch_i, ch in enumerate(ch_names):
            row = [ch]

            for band in band_names:
                row += [
                    f"{abs_1[band][ch_i]:.5f}",
                    f"{abs_2[band][ch_i]:.5f}",
                    f"{diff_abs[band][ch_i]:.5f}",
                ]

            tree.insert("", "end", values=row)

        sb_y = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb_y.set)

        tree.pack(side="left", fill="x", expand=True)
        sb_y.pack(side="right", fill="y")
            
        
    def start_topomap_subtraction_thread(self):
        import threading

        if not self.diff_topo_files:
            messagebox.showwarning(
                "Topomap Subtraction",
                "Pilih satu atau beberapa file terlebih dahulu."
            )
            return

        threading.Thread(target=self.run_topomap_subtraction, daemon=True).start()


    def run_topomap_subtraction(self):
        import traceback
        import numpy as np

        try:
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: RUNNING TOPO MAP SUBTRACTION...",
                fg="#f29900"
            ))

            # Ambil durasi 1
            d1_start = float(self.diff1_start_var.get())
            d1_end_str = self.diff1_end_var.get().strip()
            d1_end = float(d1_end_str) if d1_end_str else None

            # Ambil durasi 2
            d2_start = float(self.diff2_start_var.get())
            d2_end_str = self.diff2_end_var.get().strip()
            d2_end = float(d2_end_str) if d2_end_str else None

            if d1_end is None or d2_end is None:
                raise ValueError("Untuk pengurangan topomap, D1 End dan D2 End wajib diisi.")

            if d1_end <= d1_start:
                raise ValueError("Durasi 1 tidak valid: D1 End harus lebih besar dari D1 Start.")

            if d2_end <= d2_start:
                raise ValueError("Durasi 2 tidak valid: D2 End harus lebih besar dari D2 Start.")

            # Proses Durasi 1 dan Durasi 2
            avg_abs_1, band_names, topo_info = self.compute_average_abs_power_for_window(
                self.diff_topo_files,
                d1_start,
                d1_end
            )

            avg_abs_2, band_names_2, topo_info_2 = self.compute_average_abs_power_for_window(
                self.diff_topo_files,
                d2_start,
                d2_end
            )

            if list(band_names) != list(band_names_2):
                raise ValueError("Band names Durasi 1 dan Durasi 2 tidak sama.")

            # Hitung difference: Durasi 2 - Durasi 1
            diff_abs = {}

            for band in band_names:
                diff_abs[band] = np.array(avg_abs_2[band], dtype=float) - np.array(avg_abs_1[band], dtype=float)

            def update_ui():
                if not hasattr(self, "analysis_results") or self.analysis_results is None:
                    self.analysis_results = {}

                self.analysis_results["topo_info"] = topo_info

                self.plot_topomap_subtraction_result(
                    parent=self.diff_topo_plot_frame,
                    abs_1=avg_abs_1,
                    abs_2=avg_abs_2,
                    diff_abs=diff_abs,
                    band_names=band_names,
                    d1_label=f"{d1_start:g}-{d1_end:g}s",
                    d2_label=f"{d2_start:g}-{d2_end:g}s"
                )

                self.status_label.config(
                    text=f"STATUS: TOPO MAP SUBTRACTION DONE — {len(self.diff_topo_files)} files",
                    fg="green"
                )

                messagebox.showinfo(
                    "Topomap Subtraction",
                    "Pengurangan topomap selesai.\n\n"
                    f"Jumlah file: {len(self.diff_topo_files)}\n"
                    f"Eyes Open: {d1_start:g}-{d1_end:g}s\n"
                    f"Eyes Closed: {d2_start:g}-{d2_end:g}s\n"
                    "Arah: Durasi 2 - Durasi 1"
                )

            self.root.after(0, update_ui)

        except Exception as e:
            err = traceback.format_exc()
            print("[TOPO DIFF ERROR]")
            print(err)

            self.root.after(0, lambda: messagebox.showerror(
                "Topomap Subtraction Error",
                f"Gagal membuat pengurangan topomap:\n{e}"
            ))

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: TOPO MAP SUBTRACTION ERROR",
                fg="red"
            ))
            
    def compute_average_abs_power_for_window(self, file_paths, t_start, t_end):
        """
        Proses beberapa file pada satu window waktu,
        lalu rata-ratakan absolute power per channel dan per band.
        """
        import numpy as np
        import os

        abs_list = []
        first_topo_info = None
        band_names_ref = None

        for file_path in file_paths:
            res, eeg_crop = self.process_one_file_for_average_topomap(
                file_path=file_path,
                t_start=t_start,
                t_end=t_end
            )

            if first_topo_info is None:
                first_topo_info = res["topo_info"]

            abs_power = res["bands"]["abs_power"]
            band_names = res["bands"]["names"]

            if band_names_ref is None:
                band_names_ref = band_names
            else:
                if list(band_names) != list(band_names_ref):
                    raise ValueError(
                        f"Band names tidak konsisten pada file {os.path.basename(file_path)}"
                    )

            abs_list.append(abs_power)

        avg_abs = {}

        for band in band_names_ref:
            avg_abs[band] = np.mean(
                [np.array(abs_power[band], dtype=float) for abs_power in abs_list],
                axis=0
            )

        return avg_abs, band_names_ref, first_topo_info
    
    def select_subtract_topomap_files(self):
        from tkinter import filedialog
        import os

        paths = filedialog.askopenfilenames(
            title="Select EEG files for topomap subtraction",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if not paths:
            return

        self.diff_topo_files = list(paths)
        self.diff_topo_label_var.set(f"{len(self.diff_topo_files)} files selected")

        print("\n[TOPO DIFF] Files selected:")
        for p in self.diff_topo_files:
            print(" ", os.path.basename(p))


    def clear_subtract_topomap_files(self):
        self.diff_topo_files = []
        self.diff_topo_label_var.set("No files selected")

        if hasattr(self, "diff_topo_plot_frame"):
            for widget in self.diff_topo_plot_frame.winfo_children():
                widget.destroy()
                
    def build_topomap_subtraction_tab(self, parent):
        """
        Sub-tab Brain Mapping untuk pengurangan topomap absolute power.

        Mekanisme:
        - pilih satu / beberapa file
        - proses durasi 1
        - proses durasi 2
        - hitung selisih absolute power: Durasi 2 - Durasi 1
        - tampilkan 12 topomap: 4 band x 3 baris
        """
        for widget in parent.winfo_children():
            widget.destroy()

        ctrl = ttk.LabelFrame(parent, text=" TOPO MAP SUBTRACTION — ABSOLUTE POWER ", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            ctrl,
            text="📂 Select Files",
            command=self.select_subtract_topomap_files,
            bg="#f8f9fa",
            fg=self.colors["accent"],
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, textvariable=self.diff_topo_label_var, width=22).pack(side=tk.LEFT, padx=5)

        # Durasi 1
        ttk.Label(ctrl, text="D1 Start:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(ctrl, textvariable=self.diff1_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="D1 End:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(ctrl, textvariable=self.diff1_end_var, width=6).pack(side=tk.LEFT, padx=2)

        # Durasi 2
        ttk.Label(ctrl, text="D2 Start:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.diff2_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="D2 End:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(ctrl, textvariable=self.diff2_end_var, width=6).pack(side=tk.LEFT, padx=2)

        tk.Button(
            ctrl,
            text="RUN SUBTRACT TOPO",
            command=self.start_topomap_subtraction_thread,
            bg=self.colors["accent"],
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            ctrl,
            text="Clear",
            command=self.clear_subtract_topomap_files,
            bg="#e8eaed",
            fg=self.colors["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=8
        ).pack(side=tk.LEFT, padx=4)

        info = (
            "Arah pengurangan: Durasi 2 - Durasi 1. "
            "Nilai positif berarti absolute power lebih tinggi pada Durasi 2."
        )
        ttk.Label(parent, text=info).pack(anchor="w", padx=12, pady=(0, 4))

        self.diff_topo_plot_frame = ttk.Frame(parent)
        self.diff_topo_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
    # =========================================
    # FUNGSI UTAMA: UPDATE PLOT
    # =========================================
    def update_live_plot(self):
            if self.is_configuring or not self.board:
                self.root.after(100, self.update_live_plot)
                return

            if self.board and self.streaming:
                try:
                    data = self.board.get_board_data()
                    if data.shape[1] > 0:
                        if self.is_recording:
                            self.session_data.append(data)
                        
                        # SESUDAH:
                        eeg_all_idx = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                        eeg_idx = [eeg_all_idx[j] for j in LIVE_EEG_SOURCE_IDXS]

                        current_time = time.time()

                        for i, board_row in enumerate(eeg_idx):
                            # i = index tampilan live 0..3
                            # source_ch = index channel asli Cyton 0,1,4,5
                            source_ch = LIVE_EEG_SOURCE_IDXS[i]

                            # Update data buffer
                            new_sig = data[board_row]
                            self.data_buffer[i] = np.roll(self.data_buffer[i], -len(new_sig))
                            self.data_buffer[i, -len(new_sig):] = new_sig

                            ax = self.ax_flat[i]

                            # Logika Cek Impedansi tetap mengacu ke channel asli Cyton
                            if self.imp_states[source_ch]:
                                            # Update hitungan tiap interval
                                if current_time - self.last_impedance_time[source_ch] > self.IMP_INTERVAL:
                                    raw_chunk = self.data_buffer[i, -FS:]
                                    z_val = calculate_impedance_openbci(raw_chunk)
                                    self.last_known_imp[source_ch] = z_val
                                    self.last_impedance_time[source_ch] = current_time
                                self.lines[i].set_data(self.time_axis, self.data_buffer[i])
                                ax.set_ylim(-200000, 200000)
                            else:
                                # Mode EEG Normal
                                plot_sig = self.data_buffer[i] - np.mean(self.data_buffer[i])
                                self.lines[i].set_data(self.time_axis, plot_sig)
                                
                                # Auto-scale
                                max_val = np.max(np.abs(plot_sig))
                                limit = max(100, max_val * 1.2)
                                ax.set_ylim(-limit, limit)

                            # 3. UPDATE TEKS (Pastikan ini selalu terpanggil)
                            # 3. UPDATE TEKS (Ubah bagian ini)
                            if self.last_known_imp[source_ch] is not None:
                                z_val = self.last_known_imp[source_ch]
                                
                                # Hapus logika "if z_val > 500" yang sebelumnya menulis RAIL
                                # Sekarang kita langsung masukkan angka kilo ohm
                                self.labels_imp_text[i].set_text(f"{z_val:.1f} kΩ")
                                
                                # Kita tetap gunakan warna untuk indikator kualitas
                                if z_val < 750:
                                    color = "lime"      # Bagus (0 - 15 kOhm)
                                elif z_val < 2500:
                                    color = "yellow"    # Cukup (15 - 50 kOhm)
                                else:
                                    color = "red"       # Sangat Buruk / Lepas (> 100 kOhm)
                                
                                self.labels_imp_text[i].set_color(color)

                        # Update Digital (Index 6)
                        new_dig = (data[13] > 0).astype(float)
                        self.data_buffer[LIVE_DIGITAL_BUFFER_IDX] = np.roll(
                            self.data_buffer[LIVE_DIGITAL_BUFFER_IDX],
                            -len(new_dig)
                        )
                        self.data_buffer[LIVE_DIGITAL_BUFFER_IDX, -len(new_dig):] = new_dig

                        self.dig_line.set_data(
                            self.time_axis,
                            self.data_buffer[LIVE_DIGITAL_BUFFER_IDX]
                        )
                        self.canvas.draw_idle()

                except Exception as e:
                    # Jika ada error, print agar kita tahu di baris mana
                    print(f"Update Plot Error: {e}")

            self.root.after(100, self.update_live_plot)

    # ANALISIS (Sama seperti sebelumnya)
    # ==============================
    def start_analysis_thread(self):
            """
            Versi Revisi: Mendukung analisis dari data LIVE maupun FILE TXT.
            """
            # 1. Validasi keberadaan data berdasarkan sumbernya
            if self.current_source == "LIVE":
                # Jika sumbernya LIVE, pastikan user sudah merekam data sebelumnya
                if not hasattr(self, 'session_data') or len(self.session_data) == 0:
                    messagebox.showwarning("Analysis Warning", 
                                        "Tidak ada data rekaman live. Silakan Connect dan Record data terlebih dahulu.")
                    return
            
            elif self.current_source == "FILE":
                # Jika sumbernya FILE, pastikan data session_data sudah terisi dari load_txt_file
                if not hasattr(self, 'session_data') or len(self.session_data) == 0:
                    messagebox.showwarning("Analysis Warning", 
                                        "Data dari file belum dimuat dengan benar.")
                    return

            # 2. Membersihkan tampilan grafik Real-time sebelum menampilkan hasil analisis
            # Hapus label impedansi agar tidak menumpuk di grafik analisis nantinya
            self.last_known_imp = [None] * NUM_CHANNELS
            for i, ch_name in enumerate(LIVE_EEG_NAMES):
                if i < len(self.labels_imp_text):
                    self.labels_imp_text[i].set_text("")
            
            # 3. Update Status UI (Tema Medical Clean)
            # Gunakan warna Orange (#f29900) agar terlihat mencolok di latar putih sebagai tanda 'Proses'
            source_name = "OFFLINE FILE" if self.current_source == "FILE" else "LIVE RECORDING"
            self.status_label.config(
                text=f"STATUS: ANALYZING {source_name}... (Please Wait)", 
                fg="#f29900" 
            )
            
            # Nonaktifkan tombol sementara agar tidak diklik dua kali saat proses berjalan
            self.btn_analyze.config(state=tk.DISABLED, bg="#e8eaed", fg="#9aa0a6")

            # 4. Jalankan pipeline analisis di background thread agar GUI tidak membeku (freeze)
            threading.Thread(target=self.run_analysis_pipeline, daemon=True).start()
            
    def run_analysis_pipeline(self):
        import os
        import tempfile
        import re
        # Import fungsi fitur dari olah_data.py (pastikan file ada di folder yang sama)
        from olah_data_lengkap import extract_bands, extract_eeg_features, extract_ppg_features

        _tmp_path = None  # Untuk cleanup di finally

        try:
            # 1. VALIDASI file sumber
            if not hasattr(self, 'last_filename') or not self.last_filename:
                raise FileNotFoundError("Path file kosong.")
            if not os.path.exists(self.last_filename):
                raise FileNotFoundError(f"File tidak ditemukan: {self.last_filename}")

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: PROCESSING...", fg="orange"))

            # 2. Baca file sumber (skip baris komentar %)
            # 2. Baca file sumber
            # Kalau file sudah pernah dimuat dari tombol LOAD FILE, gunakan offline_df
            # agar TXT dan CSV sama-sama masuk sebagai format standar:
            # 8 EEG channel + Digital/PPG.
            if self.current_source == "FILE" and hasattr(self, "offline_df") and self.offline_df is not None:
                raw_df = self.offline_df.copy()
            else:
                loaded = self.read_offline_eeg_file(self.last_filename)

                if isinstance(loaded, tuple):
                    raw_df, detected_format = loaded
                else:
                    raw_df = loaded
                    detected_format = "unknown"
                    
            total_samples = len(raw_df)
            total_duration = total_samples / FS

            # 3. Parse Start / End time dari input GUI
            try:
                t_start = float(self.analysis_start_var.get())
            except (ValueError, AttributeError):
                t_start = 0.0

            try:
                end_str = self.analysis_end_var.get().strip()
                t_end = float(end_str) if end_str else total_duration
            except (ValueError, AttributeError):
                t_end = total_duration

            # Clamp agar tidak keluar batas
            t_start = max(0.0, min(t_start, total_duration - 1.0))
            t_end   = max(t_start + 1.0, min(t_end, total_duration))

            s_start = int(t_start * FS)
            s_end   = int(t_end   * FS)

            print(f"[ANALYSIS] Crop: {t_start:.1f}s – {t_end:.1f}s "
                f"({s_end - s_start} samples dari total {total_samples})")
            # Simpan durasi aktual untuk report PDF
            self._report_len = {
                "record_s": float(total_duration),
                "edit_s": float(t_end - t_start),
            }
            
            # 4. Crop dataframe
            cropped_df = raw_df.iloc[s_start:s_end].reset_index(drop=True)

            # 4b. Mode channel: paksa dataframe sebelum masuk eeg_processor_8.
            # 8CH -> Fp1,Fp2,F3,F4,T3,T4,P3,P4
            # 4CH -> Fp1,Fp2,T3,T4
            cropped_df = self._force_dataframe_for_channel_mode(
                cropped_df,
                context=f"run_analysis_pipeline crop {t_start:.1f}-{t_end:.1f}s"
            )
            n_processor_ch = self.get_processor_n_channels()

            # 5. Simpan ke file TXT sementara dengan format yang sama persis
            #    seperti file asli (header komentar % + kolom tab-separated)
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.txt', mode='w', encoding='utf-8')
            _tmp_path = tmp_file.name

            # 5. Simpan ke file TXT sementara — format IDENTIK dengan file asli
            #    eeg_processor_8 pakai skiprows=2, jadi:
            #    Baris 1 = komentar bebas (di-skip)
            #    Baris 2 = komentar bebas (di-skip)
            #    Baris 3 = nama kolom (WAJIB, langsung dibaca sebagai header)
            #    Baris 4+ = data angka
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.txt', mode='w', encoding='utf-8')
            _tmp_path = tmp_file.name

            tmp_file.write(f"% OpenBCI EEG Recording (Cropped {t_start:.1f}s-{t_end:.1f}s)\n")  # baris 1 — di-skip
            tmp_file.write(f"% FS: {FS} Hz\n")                                                    # baris 2 — di-skip
            # Baris 3 = header kolom (TANPA %, langsung terbaca sebagai nama kolom)
            tmp_file.write("\t".join(cropped_df.columns) + "\n")
            tmp_file.close()

            # Tambahkan data angka (append)
            with open(_tmp_path, 'a') as f:
                np.savetxt(f, cropped_df.values, delimiter='\t', fmt='%.4f')
                
            skip_ica_value = self.skip_ica_var.get()
            skip_zapline_value = self.skip_zapline_var.get()

            k_thresh = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())

            n_processor_ch = self.get_processor_n_channels()
            maica_n_components = 8 if int(n_processor_ch) == 4 else 16

            results = _call_process_eeg_data_compat(
                eeg_processor_8,
                _tmp_path,
                fs=FS,
                skip_ica=skip_ica_value,
                skip_zapline=skip_zapline_value,
                k_thresh=k_thresh,
                bcg_thresh=bcg_thresh,
                vbr_thresh=vbr_thresh,
                n_eeg_channels=n_processor_ch,
                channel_names=self.get_processor_channel_names(),
                maica_n_components=maica_n_components
            )

            # ── PATCH: Hapus edge artifact (spike di ujung) dari hasil ICA ──
            # Spike terjadi karena transien matematis filter/ICA di batas data.
            # trim_ica_clean  → smooth tepi pada sinyal rekonstruksi (n_ch × n_samp)
            # trim_ic_components → smooth tepi pada setiap komponen IC untuk plot
            if 'ica' in results:
                if results['ica'].get('data') is not None:
                    results['ica']['data'] = trim_ica_clean(results['ica']['data'])
                if results['ica'].get('after') is not None:
                    results['ica']['after'] = trim_ica_clean(results['ica']['after'])
                if results['ica'].get('ics') is not None:
                    results['ica']['ics'] = trim_ic_components(results['ica']['ics'])
            # ── AKHIR PATCH ──────────────────────────────────────────────────

            self.analysis_results = results

            # =====================================================================
            # FORCE DEBUG SNR AWAL (Dipaksa muncul dengan Fallback Perhitungan di GUI)
            # =====================================================================
            print("\n[DEBUG GUI] Memulai pengecekan SNR Awal...")
            
            if 'ica' in results:
                # Cek apakah eeg_processor_8 sudah sukses mengirim nilai avg_snr
                if 'avg_snr' in results['ica']:
                    init_snr = results['ica']['avg_snr']
                    print(f"[DEBUG GUI] SNR ditemukan dari backend.")
                else:
                    # FALLBACK: Jika eeg_processor_8 belum ke-save, kita hitung langsung di GUI
                    print("[DEBUG GUI] Warning: 'avg_snr' tidak ditemukan di file processor. Menghitung mandiri di GUI...")
                    from eeg_processor_8 import calculate_eeg_snr
                    # raw_sig = results['ica']['before']   # Ini adalah subset_clean (setelah Zapline)
                    raw_sig = results['zapline']['raw']  # Ini adalah subset_bpf (sinyal asli sebelum Zapline)
                    clean_sig = results['ica']['after']  # Sinyal sesudah ICA
                    
                    _, init_snr = calculate_eeg_snr(raw_sig, clean_sig)
                    # Masukkan ke dictionary agar sinkron
                    results['ica']['avg_snr'] = init_snr

                # Cetak ke terminal secara paksa (ditambahkan flush=True agar langsung keluar)
                print(f"[INITIAL RUN] SNR Rata-rata Awal: {init_snr:.2f} dB", flush=True)
            else:
                print("[DEBUG GUI] Error: Key 'ica' tidak ditemukan pada output hasil analisis.", flush=True)
            print("=====================================================\n")
            # =====================================================================
            # =============================================================== 

            # === TAMBAHAN: Ekstrak fitur lengkap (EEG + PPG) dari olah_data.py ===
            try:
                # a. Parse label dari nama file (format: Nama<ID>_S<n>_A<n>_D<n>_<G>)
                import re as _re
                basename = os.path.basename(self.last_filename)
                base_no_ext = os.path.splitext(basename)[0]
                m = _re.match(
                    r'^([A-Za-z]+)(\d+)_S(\d+)_A(\d+)_D(\d+)(?:_(M|F))?',
                    base_no_ext
                )
                if m:
                    label_info = {
                        'filename'  : basename,
                        'subject'   : m.group(1),
                        'rec_id'    : int(m.group(2)),
                        'stress'    : int(m.group(3)),
                        'anxiety'   : int(m.group(4)),
                        'depression': int(m.group(5)),
                        'gender'    : m.group(6) if m.group(6) else 'N/A',
                    }
                else:
                    label_info = {
                        'filename'  : basename,
                        'subject'   : base_no_ext,
                        'rec_id'    : 0,
                        'stress'    : None,
                        'anxiety'   : None,
                        'depression': None,
                        'gender'    : 'N/A',
                    }

                # b. Ambil sinyal ICA-clean dari hasil processor
                ica_clean = results['ica']['data']   # shape: (n_ch, n_samp)

                # c. Ekstrak band signals dan EEG features
                band_sig  = extract_bands(ica_clean, FS)
                eeg_feat  = extract_eeg_features(band_sig, FS)

                # d. Ekstrak PPG/HRV features dari kolom digital cropped_df
                if cropped_df.shape[1] > 8:
                    ppg_col = cropped_df.iloc[:, 8].values.astype(float)
                else:
                    ppg_col = np.zeros(len(cropped_df))
                ppg_feat = extract_ppg_features(ppg_col, FS)

                # e. Gabungkan semua ke self.last_features
                self.last_features = {**label_info, **eeg_feat, **ppg_feat}

            except Exception as feat_err:
                print(f"[WARNING] Ekstraksi fitur olah_data gagal: {feat_err}")
                # Fallback: gunakan fitur dari eeg_processor_8 saja
                self.last_features = {
                    'filename': os.path.basename(self.last_filename),
                    **results['features']['dict']
                }
            # === AKHIR TAMBAHAN ===

            # 7. Simpan juga cropped data ke session_data

            # 7. Simpan juga cropped data ke session_data
            #    agar display_last_5s menampilkan rentang yang sama
          #  self.session_data = cropped_df.iloc[:, :NUM_CHANNELS].values.T
            #self.session_data = cropped_df.iloc[:
            self.session_data = cropped_df.iloc[:, :n_processor_ch].values.T
            # 8. Tampilkan hasil
            self.root.after(0, lambda: self.show_visual_results(results))
            self.root.after(0, self.display_last_5s)
            _n_rej_initial = results.get('ica', {}).get('n_rejected', 0)
            _k_init = self.ica_kurt_var.get() if hasattr(self, 'ica_kurt_var') else 2.1
            self.root.after(0, lambda: self.status_label.config(
                text=(f"STATUS: ANALYSIS COMPLETE [{t_start:.1f}s – {t_end:.1f}s] | "
                    f"K≥{_k_init:.1f} | {_n_rej_initial} IC ditolak"),
                fg="green"))

        except FileNotFoundError as fnf_e:
            self.root.after(0, lambda: messagebox.showerror("File Error", str(fnf_e)))
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: FILE NOT FOUND", fg="red"))

        except Exception as e:
            final_error = str(e)
            print(f"Pipeline Error: {final_error}")
            self.root.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"Terjadi kesalahan:\n{final_error}"))
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: ERROR IN ANALYSIS", fg="red"))

        finally:
            # Hapus file temp
            if _tmp_path and os.path.exists(_tmp_path):
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_record.config(
                state=tk.NORMAL, text="▶ START RECORDING"))
                    
    def reset_session(self):
        """Reset semua state agar bisa mulai ulang tanpa restart GUI.
        CATATAN: last_features TIDAK direset agar fitur yang sudah diekstrak
        tetap bisa disimpan, atau akan ditimpa oleh analisis berikutnya."""

        # 1. Stop stream kalau masih jalan
        try:
            if self.board and self.streaming:
                self.board.stop_stream()
                time.sleep(1.0) 
        except:
            pass

        # 2. Reset status internal
        self.streaming = False
        self.is_recording = False
        # Batalkan countdown jika sedang berjalan
        if getattr(self, '_countdown_job', None):
            self.root.after_cancel(self._countdown_job)
            self._countdown_job = None
        if hasattr(self, 'countdown_label'):
            self.countdown_label.config(text="")
            
        self.session_data = []
        self.last_filename = ""
        # === PERUBAHAN: last_features SENGAJA tidak direset di sini ===
        # last_features akan di-None-kan secara otomatis setelah berhasil disimpan
        # atau akan ditimpa saat analisis berikutnya dijalankan
        
    def show_visual_results(self, res):
            """Mengaktifkan tab analisis dan mengisi datanya."""
            # 1. Tambahkan "Features" ke dalam list tabs agar frame-nya muncul di notebook
            tabs = [
                ("Zapline", self.zap_tab), 
                ("ICA", self.ica_tab), 
                ("ICA Compare",  self.ica_compare_tab),
                ("Band Signals", self.mif_tab), 
                ("Power Bands", self.band_tab),
                ("BPM Analysis", self.bpm_tab),
                ("Brain Mapping", self.topo_tab),
                ("Features", self.feat_tab),
                ("Saving Feature", self.save_tab),   # === TAB BARU ===
                ("Saving Signal",   self.signal_save_tab)  # ← TAMBAHKAN INI
            ]
            
            
            for name, frame in tabs:
                try: 
                    self.notebook.add(frame, text=f" {name} ")
                except: 
                    pass 

            # 2. Panggil fungsi plotting masing-masing
            # 2. Panggil fungsi plotting masing-masing
            self.plot_multi_zapline(res['zapline'], self.zap_tab)
            self.plot_multi_ica(res['ica'], self.ica_tab)
            self.plot_ica_compare(res['ica'], self.ica_compare_tab)
            self.plot_multi_mif(res['mif'], self.mif_tab)
            self.plot_band_power(res['bands'], self.band_tab)
            self.plot_bpm_analysis(res['bpm_data'], self.bpm_tab)
            self.build_brain_mapping_tab(self.topo_tab, res['bands']) 
            self.display_feature_table(res['features']['table'], self.feat_tab)
            self.build_save_tab(self.save_tab)   # === TAMBAHAN BARU ===
            self.build_signal_save_tab(self.signal_save_tab)  # ← TAMBAHKAN INI
            
            # Fokuskan ke tab pertama hasil analisis
            self.notebook.select(self.zap_tab)
            
    def build_brain_mapping_tab(self, parent, bands_data=None):
        """
        Membuat tab Brain Mapping dengan:
        1. Panel kontrol average topomap
        2. Area plot topomap
        """
        for widget in parent.winfo_children():
            widget.destroy()

        # =========================
        # CONTROL PANEL
        # =========================
        ctrl = ttk.LabelFrame(parent, text=" AVERAGE TOPO MAP ", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            ctrl,
            text="📂 Select Multiple Files",
            command=self.select_average_topomap_files,
            bg="#f8f9fa",
            fg=self.colors["accent"],
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, textvariable=self.avg_topo_label_var, width=28).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Start (s):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(ctrl, textvariable=self.avg_topo_start_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="End (s):").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(ctrl, textvariable=self.avg_topo_end_var, width=6).pack(side=tk.LEFT, padx=2)

        tk.Button(
            ctrl,
            text="RUN AVERAGE TOPOMAP",
            command=self.start_average_topomap_thread,
            bg=self.colors["accent"],
            fg="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            ctrl,
            text="Clear",
            command=self.clear_average_topomap_files,
            bg="#e8eaed",
            fg=self.colors["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=8
        ).pack(side=tk.LEFT, padx=4)

        # =========================
        # PLOT AREA
        # =========================
        self.topo_plot_frame = ttk.Frame(parent)
        self.topo_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Kalau ada data bands dari analisis normal, tampilkan seperti biasa
        if bands_data is not None:
            self.plot_brain_mapping(bands_data, self.topo_plot_frame)

    def select_average_topomap_files(self):
        from tkinter import filedialog
        import os

        paths = filedialog.askopenfilenames(
            title="Select EEG files for average topomap",
            filetypes=[
                ("EEG files", "*.txt *.csv"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if not paths:
            return

        self.avg_topo_files = list(paths)

        self.avg_topo_label_var.set(
            f"{len(self.avg_topo_files)} files selected"
        )

        print("\n[AVERAGE TOPO] Files selected:")
        for p in self.avg_topo_files:
            print(" ", os.path.basename(p))


    def clear_average_topomap_files(self):
        self.avg_topo_files = []
        self.avg_topo_label_var.set("No files selected")
        

    def get_channel_mode(self):
        """Return mode channel preprocessing: '4' atau '8'."""
        try:
            mode = str(self.channel_mode_var.get()).strip()
        except Exception:
            mode = "8"
        return "4" if mode == "4" else "8"

    def get_processor_n_channels(self):
        return 4 if self.get_channel_mode() == "4" else 8

    def get_processor_channel_names(self):
        return PROCESSOR_4CH_NAMES if self.get_channel_mode() == "4" else PROCESSOR_8CH_NAMES

    def _norm_col_name(self, col):
        return str(col).strip().lower().replace(" ", "").replace("_", "")

    def _audit_processor_dataframe(self, df, tag=""):
        """Audit ringan agar terlihat apakah input processor benar 4/8 channel."""
        try:
            print("\n" + "=" * 80)
            print(f"[CHANNEL MODE AUDIT] {tag}")
            print("=" * 80)
            print("Mode channel:", self.get_channel_mode(), "CH")
            print("Shape       :", df.shape)
            print("Columns     :", list(df.columns))
            for c in df.columns:
                arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(arr)
                if finite.any():
                    print(
                        f"  {c:16s} | finite={finite.sum():6d}/{len(arr):6d} "
                        f"| std={np.nanstd(arr): .5f} | min={np.nanmin(arr): .5f} | max={np.nanmax(arr): .5f}"
                    )
                else:
                    print(f"  {c:16s} | tidak ada nilai finite")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"[CHANNEL MODE AUDIT] gagal audit: {e}")

    def _force_dataframe_for_channel_mode(self, df, context=""):
        """
        Standarkan dataframe untuk eeg_processor_8 sesuai mode GUI.

        Mode 8CH:
            output: EXG Channel 0..7 + Digital
            urutan: Fp1,Fp2,F3,F4,T3,T4,P3,P4

        Mode 4CH:
            output: EXG Channel 0..3 + Digital
            urutan: Fp1,Fp2,T3,T4
            Jika sumber 8CH, channel yang dipakai adalah index [0,1,4,5].
        """
        import numpy as _np
        import pandas as _pd

        if df is None or len(df) == 0:
            raise ValueError(f"[{context}] Dataframe kosong.")

        df = df.copy()
        mode = self.get_channel_mode()
        n_out = 4 if mode == "4" else 8
        target_names = PROCESSOR_4CH_NAMES if mode == "4" else PROCESSOR_8CH_NAMES

        col_map = {self._norm_col_name(c): c for c in df.columns}

        def _col_by_candidates(cands):
            for cand in cands:
                key = self._norm_col_name(cand)
                if key in col_map:
                    return col_map[key]
            return None

        # Ambil semua channel bernama jika tersedia.
        named_sources = {
            "Fp1": _col_by_candidates(["Fp1"]),
            "Fp2": _col_by_candidates(["Fp2"]),
            "F3" : _col_by_candidates(["F3"]),
            "F4" : _col_by_candidates(["F4"]),
            "T3" : _col_by_candidates(["T3", "T7"]),
            "T4" : _col_by_candidates(["T4", "T8"]),
            "P3" : _col_by_candidates(["P3"]),
            "P4" : _col_by_candidates(["P4"]),
        }

        out = _pd.DataFrame(index=df.index)
        used_sources = []

        # Case A: dataframe sudah standar EXG Channel 0..7.
        has_exg8 = all(self._norm_col_name(f"EXG Channel {i}") in col_map for i in range(8))
        has_exg4 = all(self._norm_col_name(f"EXG Channel {i}") in col_map for i in range(4))

        # Case B: format GUI saved EEG_ch1..EEG_ch8 atau EEG_ch1..EEG_ch4.
        has_eegch8 = all(f"eegch{i}" in col_map for i in range(1, 9))
        has_eegch4 = all(f"eegch{i}" in col_map for i in range(1, 5))

        if mode == "8":
            if all(named_sources[name] is not None for name in PROCESSOR_8CH_NAMES):
                for i, name in enumerate(PROCESSOR_8CH_NAMES):
                    src = named_sources[name]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_exg8:
                for i, name in enumerate(PROCESSOR_8CH_NAMES):
                    src = col_map[self._norm_col_name(f"EXG Channel {i}")]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_eegch8:
                for i, name in enumerate(PROCESSOR_8CH_NAMES):
                    src = col_map[f"eegch{i+1}"]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            else:
                if df.shape[1] < 8:
                    raise ValueError(
                        f"[{context}] Mode 8CH dipilih, tetapi input hanya punya {df.shape[1]} kolom. "
                        f"Pilih mode 4CH jika file memang hanya Fp1,Fp2,T3,T4. Kolom: {list(df.columns)}"
                    )
                for i, name in enumerate(PROCESSOR_8CH_NAMES):
                    src = df.columns[i]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))

        else:  # mode 4CH
            if all(named_sources[name] is not None for name in PROCESSOR_4CH_NAMES):
                for i, name in enumerate(PROCESSOR_4CH_NAMES):
                    src = named_sources[name]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_exg8:
                # Asumsi urutan 8CH lama: Fp1,Fp2,F3,F4,T3,T4,P3,P4.
                source_indices = [0, 1, 2, 3]
                for i, src_i in enumerate(source_indices):
                    src = col_map[self._norm_col_name(f"EXG Channel {src_i}")]
                    name = PROCESSOR_4CH_NAMES[i]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_eegch8:
                # Asumsi urutan 8CH lama: EEG_ch1=Fp1, EEG_ch2=Fp2, EEG_ch5=T3, EEG_ch6=T4.
                source_indices = [1, 2, 5, 6]
                for i, src_i in enumerate(source_indices):
                    src = col_map[f"eegch{src_i}"]
                    name = PROCESSOR_4CH_NAMES[i]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_exg4:
                for i, name in enumerate(PROCESSOR_4CH_NAMES):
                    src = col_map[self._norm_col_name(f"EXG Channel {i}")]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            elif has_eegch4:
                for i, name in enumerate(PROCESSOR_4CH_NAMES):
                    src = col_map[f"eegch{i+1}"]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))
            else:
                if df.shape[1] < 4:
                    raise ValueError(
                        f"[{context}] Mode 4CH dipilih, tetapi input hanya punya {df.shape[1]} kolom. "
                        f"Butuh minimal Fp1,Fp2,T3,T4. Kolom: {list(df.columns)}"
                    )
                # Fallback 4 kolom pertama diasumsikan Fp1,Fp2,T3,T4.
                for i, name in enumerate(PROCESSOR_4CH_NAMES):
                    src = df.columns[i]
                    out[f"EXG Channel {i}"] = _pd.to_numeric(df[src], errors="coerce")
                    used_sources.append((f"EXG Channel {i}", name, src))

        # Ambil PPG/Digital.
        ppg_src = _col_by_candidates(["Digital", "PPG", "Marker Channel", "Digital Channel 0 (D11)"])
        if ppg_src is not None:
            out[PROCESSOR_PPG_COL] = _pd.to_numeric(df[ppg_src], errors="coerce")
        elif df.shape[1] > n_out:
            out[PROCESSOR_PPG_COL] = _pd.to_numeric(df.iloc[:, n_out], errors="coerce")
        else:
            out[PROCESSOR_PPG_COL] = 0.0

        out = out.replace([_np.inf, -_np.inf], _np.nan)
        n_bad = int(out.isna().sum().sum())
        if n_bad > 0:
            print(f"[CHANNEL MODE AUDIT] {context}: {n_bad} NaN/inf diperbaiki dengan interpolasi + fill 0.")
            out = out.interpolate(limit_direction="both").fillna(0.0)

        expected_cols = [f"EXG Channel {i}" for i in range(n_out)] + [PROCESSOR_PPG_COL]
        out = out[expected_cols]

        self.current_processor_channel_names = list(target_names)
        self.current_processor_n_channels = n_out
        self.current_processor_mode = mode

        print("\n[CHANNEL MODE] Mapping input → processor")
        print(f"  Mode        : {mode}CH")
        for dst, logical, src in used_sources:
            print(f"  {dst:14s} ({logical:3s}) <= {src}")
        print(f"  {PROCESSOR_PPG_COL:14s} <= {ppg_src if ppg_src is not None else 'zeros/fallback'}")

        self._audit_processor_dataframe(out, tag=f"{context} | processor input {mode}CH")
        return out

    def make_processor_temp_file(self, cropped_df):
        import tempfile
        import numpy as np

        cropped_df = self._force_dataframe_for_channel_mode(cropped_df, context="make_processor_temp_file")
        n_ch = self.get_processor_n_channels()
        expected_cols = [f"EXG Channel {i}" for i in range(n_ch)] + [PROCESSOR_PPG_COL]

        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".txt",
            mode="w",
            encoding="utf-8"
        )

        _tmp_path = tmp_file.name
        tmp_file.write("% OpenBCI EEG Recording\n")
        tmp_file.write(f"% FS: {FS} Hz | MODE: {n_ch}CH\n")
        tmp_file.write("\t".join(expected_cols) + "\n")
        tmp_file.close()

        with open(_tmp_path, "a", encoding="utf-8") as f:
            np.savetxt(
                f,
                cropped_df[expected_cols].to_numpy(dtype=float),
                delimiter="	",
                fmt="%.8f"
            )

        print(f"[CHANNEL MODE] Temporary processor file: {_tmp_path}")
        print(f"[CHANNEL MODE] Temporary shape: {cropped_df[expected_cols].shape}")
        return _tmp_path

    def process_one_file_for_average_topomap(self, file_path, t_start, t_end):
        import os

        # Pakai loader universal yang sama dengan LOAD FILE
        loaded = self.read_offline_eeg_file(file_path)

        # Antisipasi kalau fungsi loader masih ada versi lama yang return 1 nilai
        if isinstance(loaded, tuple):
            raw_df, detected_format = loaded
        else:
            raw_df = loaded
            detected_format = "unknown"

        total_samples = len(raw_df)
        total_duration = total_samples / FS

        if t_end is None:
            t_end_use = total_duration
        else:
            t_end_use = float(t_end)

        t_start_use = float(t_start)

        if t_start_use < 0:
            raise ValueError(f"Start time negatif pada file {os.path.basename(file_path)}")

        if t_end_use <= t_start_use:
            raise ValueError(f"End time harus lebih besar dari start pada file {os.path.basename(file_path)}")

        if t_end_use > total_duration:
            raise ValueError(
                f"File {os.path.basename(file_path)} terlalu pendek.\n"
                f"Durasi file = {total_duration:.2f}s, tetapi diminta sampai {t_end_use:.2f}s."
            )

        s_start = int(round(t_start_use * FS))
        s_end = int(round(t_end_use * FS))

        cropped_df = raw_df.iloc[s_start:s_end].reset_index(drop=True)

        _tmp_path = self.make_processor_temp_file(cropped_df)

        try:
            skip_ica_value = self.skip_ica_var.get()
            skip_zapline_value = self.skip_zapline_var.get()

            k_thresh = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())

            n_processor_ch = self.get_processor_n_channels()
            maica_n_components = 8 if int(n_processor_ch) == 4 else 16

            res = _call_process_eeg_data_compat(
                eeg_processor_8,
                _tmp_path,
                fs=FS,
                skip_ica=skip_ica_value,
                skip_zapline=skip_zapline_value,
                k_thresh=k_thresh,
                bcg_thresh=bcg_thresh,
                vbr_thresh=vbr_thresh,
                n_eeg_channels=n_processor_ch,
                channel_names=self.get_processor_channel_names(),
                maica_n_components=maica_n_components
            )

        finally:
            try:
                os.remove(_tmp_path)
            except Exception:
                pass

        # EEG yang dipakai untuk SASI average
        eeg_crop = cropped_df.iloc[:, :NUM_CHANNELS].values.T.astype(float)

        print(
            f"[AVERAGE TOPO] Done: {os.path.basename(file_path)} | "
            f"format={detected_format} | crop={t_start_use:.1f}-{t_end_use:.1f}s"
        )

        return res, eeg_crop


    def compute_topomap_metric_bundle(self, res, eeg_crop):
        from scipy.signal import welch as sp_welch
        import numpy as np

        bands = res["bands"]
        abs_power = bands["abs_power"]
        rel_power = bands["rel_power"]
        band_names = bands["names"]

        # =========================
        # Ratio
        # =========================
        ratio_defs = [
            ("Delta/Theta", "Delta", "Theta"),
            ("Delta/Alpha", "Delta", "Alpha"),
            ("Delta/Beta",  "Delta", "Beta"),
            ("Theta/Alpha", "Theta", "Alpha"),
            ("Theta/Beta",  "Theta", "Beta"),
            ("Alpha/Beta",  "Alpha", "Beta"),
        ]

        ratios = {}

        for label, nb, db in ratio_defs:
            if nb in abs_power and db in abs_power:
                num = np.array(abs_power[nb], dtype=float)
                den = np.array(abs_power[db], dtype=float)
                den = np.where(den == 0, 1e-10, den)
                ratios[label] = num / den

        # =========================
        # Asymmetry
        # =========================
        asym_pairs = [
            ("Fp1-Fp2", 0, 1),
            ("F3-F4",   2, 3),
            ("T3-T4",   4, 5),
            ("P3-P4",   6, 7),
        ]
        ch_to_pair = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

        asym_by_band = {}

        for band in band_names:
            p = np.array(abs_power[band], dtype=float)
            asym_topo = np.zeros(NUM_CHANNELS)

            for ch_idx, pair_idx in ch_to_pair.items():
                _, li, ri = asym_pairs[pair_idx]
                L, R = p[li], p[ri]
                denom = L + R
                asym_topo[ch_idx] = (L - R) / denom if denom != 0 else 0.0

            asym_by_band[band] = asym_topo

        # =========================
        # SASI
        # =========================
        freqs, psd_all = sp_welch(
            eeg_crop[:NUM_CHANNELS],
            FS,
            nperseg=min(500, eeg_crop.shape[1]),
            axis=1
        )

        alpha_mask = (freqs >= 8) & (freqs <= 12)

        if np.any(alpha_mask):
            fc = freqs[alpha_mask][np.argmax(psd_all[:, alpha_mask].mean(axis=0))]
            sasi_values = np.array([
                self.compute_sasi(psd_all[ch], freqs, fc)
                for ch in range(NUM_CHANNELS)
            ])
        else:
            sasi_values = np.zeros(NUM_CHANNELS)

        return {
            "abs_power": abs_power,
            "rel_power": rel_power,
            "ratios": ratios,
            "asym_by_band": asym_by_band,
            "sasi_values": sasi_values,
            "band_names": band_names
        }
        
    def average_topomap_metric_bundles(self, bundles):
        import numpy as np

        band_names = bundles[0]["band_names"]

        # =========================
        # Average absolute & relative power
        # =========================
        avg_abs = {}
        avg_rel = {}

        for band in band_names:
            avg_abs[band] = np.mean(
                [np.array(b["abs_power"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

            avg_rel[band] = np.mean(
                [np.array(b["rel_power"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

        # =========================
        # Average ratio
        # =========================
        ratio_labels = bundles[0]["ratios"].keys()
        avg_ratios = {}

        for label in ratio_labels:
            avg_ratios[label] = np.mean(
                [np.array(b["ratios"][label], dtype=float) for b in bundles],
                axis=0
            )

        # =========================
        # Average asymmetry
        # =========================
        avg_asym = {}

        for band in band_names:
            avg_asym[band] = np.mean(
                [np.array(b["asym_by_band"][band], dtype=float) for b in bundles],
                axis=0
            ).tolist()

        # =========================
        # Average SASI
        # =========================
        avg_sasi = np.mean(
            [np.array(b["sasi_values"], dtype=float) for b in bundles],
            axis=0
        )

        avg_bands = {
            "abs_power": avg_abs,
            "rel_power": avg_rel,
            "names": band_names,
            "ranges": bundles[0].get("ranges", {}),
            "powers": {band: float(np.mean(avg_rel[band])) for band in band_names},

            # Tambahan untuk plot_brain_mapping
            "ratios": avg_ratios,
            "asym_by_band": avg_asym,
            "sasi_values": avg_sasi,
            "is_average": True,
            "n_files": len(bundles)
        }

        return avg_bands

    def start_average_topomap_thread(self):
        import threading

        if not self.avg_topo_files:
            messagebox.showwarning(
                "Average Topomap",
                "Pilih beberapa file terlebih dahulu."
            )
            return

        threading.Thread(target=self.run_average_topomap, daemon=True).start()


    def run_average_topomap(self):
        import os
        import traceback

        try:
            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: RUNNING AVERAGE TOPO MAP...",
                fg="#f29900"
            ))

            try:
                t_start = float(self.avg_topo_start_var.get())
            except Exception:
                t_start = 0.0

            end_str = self.avg_topo_end_var.get().strip()
            t_end = float(end_str) if end_str else None

            bundles = []
            first_topo_info = None

            for idx, file_path in enumerate(self.avg_topo_files):
                res, eeg_crop = self.process_one_file_for_average_topomap(
                    file_path=file_path,
                    t_start=t_start,
                    t_end=t_end
                )

                if first_topo_info is None:
                    first_topo_info = res["topo_info"]

                bundle = self.compute_topomap_metric_bundle(res, eeg_crop)

                # Simpan ranges kalau ada
                bundle["ranges"] = res["bands"].get("ranges", {})

                bundles.append(bundle)

            avg_bands = self.average_topomap_metric_bundles(bundles)

            def update_ui():
                if not hasattr(self, "analysis_results") or self.analysis_results is None:
                    self.analysis_results = {}

                # plot_brain_mapping butuh topo_info dari self.analysis_results
                self.analysis_results["topo_info"] = first_topo_info

                # Tampilkan average ke area plot brain map
                self.plot_brain_mapping(avg_bands, self.topo_plot_frame)

                self.status_label.config(
                    text=f"STATUS: AVERAGE TOPO MAP DONE — {len(self.avg_topo_files)} files",
                    fg="green"
                )

                messagebox.showinfo(
                    "Average Topomap",
                    f"Average topomap selesai.\n\n"
                    f"Jumlah file: {len(self.avg_topo_files)}\n"
                    f"Window: {t_start:.1f}s - {t_end if t_end is not None else 'end'}s"
                )

            self.root.after(0, update_ui)

        except Exception as e:
            err = traceback.format_exc()
            print("[AVERAGE TOPO ERROR]")
            print(err)

            self.root.after(0, lambda: messagebox.showerror(
                "Average Topomap Error",
                f"Gagal membuat average topomap:\n{e}"
            ))

            self.root.after(0, lambda: self.status_label.config(
                text="STATUS: AVERAGE TOPO MAP ERROR",
                fg="red"
            ))
            
            
            
    def plot_bpm_analysis(self, data, parent): # Pastikan namanya 'data'
        for widget in parent.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#0f172a')

        # Sekarang variabel 'data' akan terbaca dengan benar
        # KODE BARU — tambah marker peaks:
        if "ppg_signal" in data:
            axs[0].plot(data["ppg_time"], data["ppg_signal"], color='cyan', lw=1)
            axs[0].set_title("PPG Pulse Signal (Midpoint Peaks)")
            
            ppg_peaks = data.get("ppg_peaks", np.array([]))
            if len(ppg_peaks) > 0:
                axs[0].scatter(
                    data["ppg_time"][ppg_peaks],
                    data["ppg_signal"][ppg_peaks],
                    color='blue', s=50, zorder=5, label='Midpoint Peak'
                )
            axs[0].legend(fontsize=8)
            axs[0].set_ylabel("Binary")

        # =========================
        # 2. RR INTERVAL
        # =========================
        if "rr_values" in data:
            axs[1].plot(data["rr_times"], data["rr_values"], color='yellow', marker='o')
            axs[1].set_title("RR Interval (ms)")
            axs[1].set_ylabel("ms")

        # =========================
        # 3. BPM INSTANT
        # =========================
        if "bpm_values" in data:
            axs[2].plot(data["bpm_times"], data["bpm_values"], color='lime', marker='o')
            axs[2].set_title("Instant BPM")
            axs[2].set_ylabel("BPM")

        # =========================
        # 4. SUMMARY BOX
        # =========================
        axs[3].axis('off')

        avg_bpm = data.get('mean_bpm', 0)
        sdnn = data.get('sdnn', 0)
        rmssd = data.get('rmssd', 0)
  
        
        info_text = (
            f"Heart Rate Summary\n\n"
            f"Mean BPM : {avg_bpm:.2f}\n"
            f"SDNN     : {data.get('sdnn', 0):.2f} ms\n"
            f"RMSSD    : {data.get('rmssd', 0):.2f} ms\n"
            f"Status   : {data.get('status', 'N/A')}"
        )

        axs[3].text(
            0.5, 0.5, info_text,
            color='white',
            ha='center', va='center',
            fontsize=12,
            bbox=dict(facecolor='#020617', edgecolor='cyan', boxstyle='round,pad=0.5')
        )

        # =========================
        # STYLE
        # =========================
        for ax in axs[:3]:
            ax.set_facecolor('#020617')
            ax.grid(True, linestyle=':', alpha=0.3)
            ax.tick_params(colors='white')

            for spine in ax.spines.values():
                spine.set_color('#334155')

        plt.tight_layout()
        self._embed_canvas(fig, parent)
    
            
        
    def plot_multi_zapline(self, data, parent):
        for widget in parent.winfo_children():
            widget.destroy()

        t, fs = data['t'], 250
        n_ch = data['raw'].shape[0]

        fig, axs = plt.subplots(4, 4, figsize=(18, 10))
        fig.patch.set_facecolor("#ffffff")

        for i in range(4):
            # =========================
            # KOLOM 1: CH1–CH4 (TIME)
            # =========================
            if i < n_ch:
                axs[i, 0].plot(t, data['raw'][i], color='red', alpha=0.4)
                axs[i, 0].plot(t, data['clean'][i], color='green', lw=0.8)
                axs[i, 0].set_ylabel(f"CH {i+1}", fontsize=8)

            # =========================
            # KOLOM 2: CH5–CH8 (TIME)
            # =========================
            ch_idx = i + 4
            if ch_idx < n_ch:
                axs[i, 1].plot(t, data['raw'][ch_idx], color='red', alpha=0.4)
                axs[i, 1].plot(t, data['clean'][ch_idx], color='green', lw=0.8)
                axs[i, 1].set_ylabel(f"CH {ch_idx+1}", fontsize=8)

            # =========================
            # KOLOM 3: CH1–CH4 (PSD)
            # =========================
            if i < n_ch:
                f_raw, p_raw = welch(data['raw'][i], fs, nperseg=fs*2)
                f_cln, p_cln = welch(data['clean'][i], fs, nperseg=fs*2)

                axs[i, 2].semilogy(f_raw, p_raw, color='red', alpha=0.5)
                axs[i, 2].semilogy(f_cln, p_cln, color='green', lw=1)
                axs[i, 2].set_xlim(2, 80)

            # =========================
            # KOLOM 4: CH5–CH8 (PSD)
            # =========================
            if ch_idx < n_ch:
                f_raw, p_raw = welch(data['raw'][ch_idx], fs, nperseg=fs*2)
                f_cln, p_cln = welch(data['clean'][ch_idx], fs, nperseg=fs*2)

                axs[i, 3].semilogy(f_raw, p_raw, color='red', alpha=0.5)
                axs[i, 3].semilogy(f_cln, p_cln, color='green', lw=1)
                axs[i, 3].set_xlim(2, 80)

        # =========================
        # STYLE (BIAR CONSISTENT)
        # =========================
        for row in range(4):
            for col in range(4):
                ax = axs[row, col]
                ax.grid(True, linestyle=':', alpha=0.3)
                ax.tick_params(labelsize=7)

        # TITLE KOLOM
        axs[0, 0].set_title("EEG CH1–CH4")
        axs[0, 1].set_title("EEG CH5–CH8")
        axs[0, 2].set_title("PSD CH1–CH4")
        axs[0, 3].set_title("PSD CH5–CH8")

        plt.tight_layout()
        self._embed_canvas(fig, parent)
        
    def plot_multi_ica(self, data, parent):
        """Plot SEMUA ICA components dengan scroll + threshold yang bisa diatur."""
        for widget in parent.winfo_children():
            widget.destroy()
        try:
            if hasattr(self, "_ica_fig") and self._ica_fig is not None:
                plt.close(self._ica_fig)
                self._ica_fig = None
        except Exception:
            pass

        ics = data['ics']
        n_ic = len(ics)

        # ── Frame kontrol threshold (di atas, tidak ikut scroll) ─────
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(ctrl_frame, text="Kurtosis ≥",
                 font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(4, 2))
        tk.Spinbox(ctrl_frame, from_=0.0, to=50.0, increment=0.5,
                   textvariable=self.ica_kurt_var,
                   width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

# TAMBAHKAN INI:
        tk.Label(ctrl_frame, text="  BCG ≥",
                font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(ctrl_frame, from_=0.01, to=1.0, increment=0.01,
                textvariable=self.ica_bcg_var,
                width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl_frame, text="  VBR ≥",
                font=('Segoe UI', 9), bg=self.colors["bg"]).pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(ctrl_frame, from_=1.0, to=20.0, increment=0.5,
                textvariable=self.ica_vbr_var,
                width=5, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

        tk.Button(ctrl_frame, text="Apply & Recompute Features",
                  command=lambda: self.apply_ica_threshold(parent),
                  bg=self.colors["accent"], fg="white",
                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT,
                  padx=10).pack(side=tk.LEFT, padx=10)
        
        rejected_set = set(data.get('rejected_indices') or [])
        n_rejected = len(rejected_set)

        tk.Label(
            ctrl_frame,
            text=f"Total IC: {n_ic}  |  Rejected: {n_rejected}",
            font=('Segoe UI', 8),
            fg=self.colors["text_dim"],
            bg=self.colors["bg"]
        ).pack(side=tk.LEFT, padx=5)
               

        # ── Container scroll ─────────────────────────────────────────
        scroll_container = ttk.Frame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        # Scrollbar vertikal dan horizontal
        vbar = tk.Scrollbar(scroll_container, orient=tk.VERTICAL)
        hbar = tk.Scrollbar(scroll_container, orient=tk.HORIZONTAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)

        canvas_scroll = tk.Canvas(scroll_container,
                                  yscrollcommand=vbar.set,
                                  xscrollcommand=hbar.set,
                                  bg=self.colors["bg"],
                                  highlightthickness=0)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=canvas_scroll.yview)
        hbar.config(command=canvas_scroll.xview)

        inner_frame = ttk.Frame(canvas_scroll)
        canvas_scroll.create_window((0, 0), window=inner_frame, anchor='nw')

        def _on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        inner_frame.bind("<Configure>", _on_frame_configure)

        # Scroll vertikal dengan mouse wheel, horizontal dengan Shift+wheel
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_h(event):
            canvas_scroll.xview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        canvas_scroll.bind_all("<Shift-MouseWheel>", _on_mousewheel_h)

        # ── Baca threshold saat ini ───────────────────────────────────
        k_thresh = self.ica_kurt_var.get()

        # ── Plot SEMUA IC ─────────────────────────────────────────────
        n_rows = int(np.ceil(n_ic / 2))

        fig, axs = plt.subplots(
            n_rows,
            4,
            figsize=(16, max(2.8 * n_rows, 7)),   # lebih proporsional untuk Raspberry Pi
            dpi=80,
            squeeze=False,
            gridspec_kw={
                'width_ratios': [4.2, 1.4, 4.2, 1.4],
                'wspace': 0.35,
                'hspace': 0.75
            }
        )

        fig.patch.set_facecolor("#ffffff")

        

        for i in range(n_ic):
            row = i % n_rows
            col_offset = 0 if i < n_rows else 2
            ic = ics[i]
            # z = ic.get('z_score', 0.0)
            corr_bcg = ic.get('corr_pulse', 0.0)

            # Gunakan rejected_indices sebagai ground truth jika tersedia
            # Ground truth reject dari processor
            is_rejected = i in rejected_set
            color_sig = '#e53935' if is_rejected else "#174ea6"
            text_col  = '#e53935' if is_rejected else '#2e7d32'
            status    = "✖ REJECTED" if is_rejected else "✔ KEPT"

            # Time domain
            ax_t = axs[row, col_offset]
            ax_t.plot(ic['sig'], lw=0.6, color=color_sig)
            ax_t.set_facecolor("#ffffff")

            vbr_val = ic.get('vbr', 0.0)
            info = (f"IC {i+1}  {status}\n"
                    f"K:{ic['kurt']:.2f} BCG:{corr_bcg:.3f} VBR:{vbr_val:.1f}")

            # info = (f"IC {i+1}  {status}\n"
            #         f"K: {ic['kurt']:.2f}   BCG: {corr_bcg:.3f}") 
            ax_t.set_ylabel(
                info,
                color=text_col,
                fontsize=6,
                rotation=0,
                labelpad=30,
                va='center',
                multialignment='left'
            )

            # Frequency domain
            ax_f = axs[row, col_offset + 1]
            ax_f.plot(ic['freqs'],
                      10 * np.log10(ic['psd'] + 1e-12),
                      color='indianred', lw=0.8)
            ax_f.set_facecolor("#ffffff")
            ax_f.set_xlim(0, 80)

            for ax in [ax_t, ax_f]:
                ax.grid(True, linestyle=':', color='#cccccc', alpha=0.5)
                ax.tick_params(axis='both', colors='#555555', labelsize=7, pad=1)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#dddddd')

        # Sembunyikan axes yang tidak terpakai (jika n_ic ganjil)
        if n_ic % 2 != 0:
            for c in range(2, 4):
                axs[n_rows - 1, c].axis('off')

        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.96,
            bottom=0.08,
            wspace=0.40,
            hspace=0.95
        )

        # ============================================================
        # EMBED CANVAS ICA — KHUSUS ICA JANGAN FIT-TO-WINDOW
        # ============================================================
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        # Simpan referensi agar canvas lama tidak liar
        try:
            if hasattr(self, "_ica_fig") and self._ica_fig is not None:
                plt.close(self._ica_fig)
        except Exception:
            pass

        self._ica_fig = fig

        fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        fig_canvas.draw()

        self._ica_canvas = fig_canvas

        fig_widget = fig_canvas.get_tk_widget()
        fig_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar di luar area scroll, supaya zoom/pan tetap terlihat
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        try:
            toolbar = NavigationToolbar2Tk(
                fig_canvas,
                toolbar_frame,
                pack_toolbar=False
            )
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)
        except TypeError:
            toolbar = NavigationToolbar2Tk(fig_canvas, toolbar_frame)
            toolbar.update()

        self._ica_toolbar = toolbar

        # Penting:
        # Tutup figure dari pyplot manager agar tidak ikut di-resize global
        # oleh _apply_raspy_compact_layout().
        plt.close(fig)

    
    def plot_ica_compare(self, data, parent):
        """Perbandingan sinyal per channel sebelum dan sesudah ICA, overlay transparan."""
        for widget in parent.winfo_children():
            widget.destroy()
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(
            ctrl_frame,
            text="Show Before ICA",
            variable=self.show_before_ica_var,
            command=lambda: self.plot_ica_compare(data, parent)
        ).pack(side=tk.LEFT)
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(
            ctrl_frame,
            text="Show legend",
            variable=self.show_ica_compare_legend_var,
            command=lambda: self.plot_ica_compare(data, parent)
        ).pack(side=tk.LEFT)
        before = data.get('before')   # shape: (n_ch, n_samp)
        after  = data.get('after')    # shape: (n_ch, n_samp)

        if before is None or after is None:
            ttk.Label(parent, text="Data before/after ICA tidak tersedia.").pack()
            return

        n_ch   = before.shape[0]
        n_samp = min(before.shape[1], after.shape[1])
        t      = np.arange(n_samp) / FS

        MAX_PTS = 6000
        if n_samp > MAX_PTS:
            step   = n_samp // MAX_PTS
            t      = t[::step]
            before = before[:, ::step]
            after  = after[:,  ::step]

        ch_labels = ['Fp1','Fp2','F3','F4','T3','T4','P3','P4']

        fig, axs = plt.subplots(n_ch, 1, figsize=(14, 2.2 * n_ch), sharex=True)
        fig.patch.set_facecolor('#ffffff')

        if n_ch == 1:
            axs = [axs]

        for i in range(n_ch):
            ax = axs[i]
            sig_before = before[i] - np.mean(before[i])
            sig_after  = after[i]  - np.mean(after[i])

            show_before = self.show_before_ica_var.get()

            if show_before:
                ax.plot(t, sig_before, color='#e53935', lw=0.8, alpha=0.35, label='Before ICA')

            ax.plot(t, sig_after, color='#1a73e8', lw=0.9, alpha=0.90, label='After ICA')
            y_max = max(np.max(np.abs(sig_after)) * 1.3, 10.0)
            ax.set_ylim(-y_max, y_max)

            ch_name = ch_labels[i] if i < len(ch_labels) else f"CH{i+1}"
            ax.set_ylabel(ch_name, fontsize=9, fontweight='bold',
                          color='#1a73e8', rotation=0, labelpad=30, va='center')
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle=':', color='#e0e0e0', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7, colors='#555555')

            if i == 0:
                handles = []

                if show_before:
                    handles.append(
                        plt.Line2D([0], [0], color='#e53935', alpha=0.35, lw=1.5, label='Before ICA')
                    )

                handles.append(
                    plt.Line2D([0], [0], color='#1a73e8', alpha=0.90, lw=1.5, label='After ICA')
                )

                ax.legend(loc='upper right', fontsize=8, framealpha=0.7, handles=handles)

        axs[-1].set_xlabel("Time (s)", fontsize=9)
        fig.suptitle("ICA Artifact Removal — Before vs After (per Channel)",
                     fontsize=11, fontweight='bold', color='#202124', y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        self._embed_canvas(fig, parent)
        
    def apply_ica_threshold(self, ica_parent):
        if not hasattr(self, 'analysis_results') or self.analysis_results is None:
            messagebox.showwarning("Belum Ada Data",
                                "Jalankan Run Analysis terlebih dahulu.")
            return

        intermediates = self.analysis_results.get('ica_intermediates')
        if intermediates is None:
            messagebox.showwarning(
                "Data Tidak Lengkap",
                "Intermediate ICA tidak tersedia.\n"
                "Pastikan menggunakan eeg_processor_8.py versi terbaru.")
            return

        # Ambil nilai threshold terbaru dari GUI slider Anda
        try:
            k_thresh   = float(self.ica_kurt_var.get())
            bcg_thresh = float(self.ica_bcg_var.get())
            vbr_thresh = float(self.ica_vbr_var.get())
        except Exception:
            k_thresh = 1.6

        print(f"[DEBUG GUI] Threshold terbaca dari GUI -> Kurtosis: {k_thresh}")

        self.status_label.config(
            text=f"STATUS: RECOMPUTING (K≥{k_thresh:.1f} | BCG≥{bcg_thresh:.2f} | VBR≥{vbr_thresh:.1f})...",
            fg="#f29900"
        )
        self.root.update_idletasks()

        # Fungsi pekerja yang akan berjalan di dalam Thread baru
        def _do_recompute():
            try:
                # Kirim threshold terbaru ke eeg_processor_8
                # recon = reconstruct_from_ica(intermediates, k_thresh)
                recon = reconstruct_from_ica(intermediates, k_thresh, bcg_thresh, vbr_thresh)
                print(f"[RECOMPUTE SUCCESS] k_thresh={k_thresh}")
                print(f"[RECOMPUTE SUCCESS] n_rejected={recon.get('n_rejected', 'N/A')}")

                if recon.get('ica_clean') is not None:
                    if 'trim_ica_clean' in globals() or hasattr(self, 'trim_ica_clean'):
                        recon['ica_clean'] = trim_ica_clean(recon['ica_clean'])

                # Menimpa hasil analisis lama dengan rekonstruksi threshold baru
                self.analysis_results['ica']['data']             = recon['ica_clean']
                self.analysis_results['ica']['after']            = recon['ica_clean']
                self.analysis_results['ica']['ics_updated']      = recon['ics_updated']

                # TAMBAHKAN INI:
                self.analysis_results['ica']['ics']              = recon['ics_updated']

                self.analysis_results['ica']['n_rejected']       = recon['n_rejected']
                self.analysis_results['ica']['rejected_indices'] = recon.get('rejected_indices', [])

                self.analysis_results['bands']                   = recon['bands']
                self.analysis_results['features']                = recon['features']
                self.analysis_results['mif']                     = recon['mif']

                if self.last_features is not None:
                    label_keys = {'filename', 'subject', 'rec_id',
                                'stress', 'anxiety', 'depression',
                                'gender', 'seg_id'}
                    label_part = {k: v for k, v in self.last_features.items()
                                if k in label_keys}
                    self.last_features = {**label_part, **recon['features']['dict']}
                else:
                    import os
                    self.last_features = {
                        'filename': os.path.basename(
                            getattr(self, 'last_filename', 'unknown')),
                        **recon['features']['dict']
                    }

                n_rej = recon['n_rejected']

                # Mengirim perintah update grafik ke Main Thread Tkinter
                def _update_ui():
                    self.plot_multi_ica(self.analysis_results['ica'], ica_parent)
                    self.plot_ica_compare(self.analysis_results['ica'], self.ica_compare_tab)
                    self.plot_multi_mif(self.analysis_results['mif'], self.mif_tab)
                    self.plot_band_power(self.analysis_results['bands'], self.band_tab)
                    self.plot_brain_mapping(self.analysis_results['bands'], self.topo_tab)
                    self.display_feature_table(
                        self.analysis_results['features']['table'], self.feat_tab)
                    self.build_save_tab(self.save_tab)
                    self.status_label.config(
                        text=(f"STATUS: RECOMPUTE DONE — "
                            f"K≥{k_thresh:.1f} | "
                            f"{n_rej} IC ditolak"),
                        fg="green"
                    )

                self.root.after(0, _update_ui)

            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                print(f"[ERROR] apply_ica_threshold:\n{err_msg}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Recompute Error", f"Gagal rekonstruksi ulang:\n{str(e)}"))
                self.root.after(0, lambda: self.status_label.config(
                    text="STATUS: RECOMPUTE ERROR", fg="red"))

        # Jalankan proses recompute di latar belakang agar GUI tidak hang/freeze
        threading.Thread(target=_do_recompute, daemon=True).start()
            
    def plot_multi_mif(self, data, parent):
            """Plot MIF dengan Time Series yang diperlebar (Rasio 3:1)."""
            for widget in parent.winfo_children(): widget.destroy()
            mimfs = data['mimfs']
            n_mimfs = len(mimfs)
            n_rows = int(np.ceil(n_mimfs / 2))
            
            # Gunakan width_ratios untuk memperlebar kolom 0 dan 2
            fig, axs = plt.subplots(n_rows, 4, figsize=(18, 1.8 * n_rows), squeeze=False,
                                    gridspec_kw={'width_ratios': [3, 1, 3, 1]})
            fig.patch.set_facecolor("#fffefe")

            for i in range(n_mimfs):
                row = i % n_rows
                col_offset = 0 if i < n_rows else 2
                m = mimfs[i]
                
                # --- Kolom Time Domain (WIDE) ---
                ax_t = axs[row, col_offset]
                ax_t.plot(m['sig'], lw=0.8, color='#174ea6')
                ax_t.set_facecolor("#FFFFFF")
                band_name = m.get('name', f"Band {i+1}")
                ax_t.set_ylabel(band_name, color='#174ea6', fontsize=8, rotation=0, labelpad=20, va='center')

                # --- Kolom Freq Domain (COMPACT) ---
                ax_f = axs[row, col_offset + 1]
                ax_f.fill_between(m['f'], m['p'], color='green', alpha=0.4)
                ax_f.set_facecolor("#FFFDFD")
                ax_f.set_xlim(0, 80)

                for ax in [ax_t, ax_f]:
                    ax.grid(True, linestyle=':', color='#333333', alpha=0.3)
                    ax.tick_params(axis='both', colors='#555555', labelsize=7, pad=1)
                    for spine in ax.spines.values(): spine.set_edgecolor('#333333')

            if n_mimfs < (n_rows * 2):
                for c in range(2, 4): axs[n_rows-1, c].axis('off')

            plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.1, rect=[0.02, 0.02, 0.99, 0.98])
            self._embed_canvas(fig, parent)
            
    def plot_band_power(self, data, parent):
        """Plot distribusi power band dengan layout 2 kolom."""
        # Bersihkan widget sebelumnya
        for widget in parent.winfo_children(): 
            widget.destroy()

        powers, signals = data['powers'], data['signals']
        ranges, imf_map = data['ranges'], data['imf_map']
        band_names = list(powers.keys())
        num_bands = len(band_names)
        
        # Hitung jumlah baris yang dibutuhkan untuk 2 kolom (setelah bar chart)
        # math.ceil(num_bands / 2)
        rows_needed = (num_bands + 1) // 2 
        
        fig = plt.figure(figsize=(14, 12))
        # Bar chart di baris 0, sinyal mulai dari baris 1
        gs = fig.add_gridspec(rows_needed + 1, 2)
        
        # --- 1. Bar Chart (Spans both columns at the top) ---
        ax_bar = fig.add_subplot(gs[0, :]) 
        ax_bar.bar(band_names, list(powers.values()), color='skyblue', edgecolor='navy')
        ax_bar.set_yscale('log')
        ax_bar.set_title("Relative Power Distribution (Butterworth Band Extraction)", fontsize=12, pad=15)
        ax_bar.set_ylabel("Power (Log)")

        # --- 2. Sinyal Tiap Band (2-Column Layout) ---
        for i, band in enumerate(band_names):
            # Hitung posisi grid: 
            # i // 2 menentukan baris (ditambah 1 karena baris 0 adalah bar chart)
            # i % 2 menentukan kolom (0 atau 1)
            row_idx = (i // 2) + 1
            col_idx = i % 2
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Ambil rata-rata sinyal antar channel
            mean_sig = np.mean(signals[i], axis=0) 
            n_points = len(mean_sig)
            t_plot = np.linspace(0, n_points / FS, n_points) 
            
            ax.plot(t_plot, mean_sig, lw=0.6, color='#174ea6')
            
            # Metadata Formatting
            hz_range = ranges[band]
            hz_text = f"{hz_range[0]:.1f}-{hz_range[1]:.1f} Hz" if hz_range[0] is not None else "N/A"
            hz_range = ranges[band]
            imf_text = f"{hz_range[0]:.1f}–{hz_range[1]:.1f} Hz" if hz_range[0] is not None else "N/A"
                
            # Styling Subplot
            ax.set_title(f"Band: {band} ({hz_text})", fontsize=10, loc='left')
            ax.text(0.01, 0.05, f"Composition: {imf_text}", 
                    transform=ax.transAxes, fontsize=8, color='darkred', alpha=0.7)
            
            ax.set_ylabel("Amp")
            ax.set_xlabel("Time (s)")
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._embed_canvas(fig, parent)
        plt.close(fig)
    

    def _embed_canvas(self, fig, parent, fit_to_window=True):
        """
        Embed figure Matplotlib ke Tkinter.
        fit_to_window=True dipakai untuk plot biasa.
        Untuk ICA yang banyak subplot, jangan pakai fungsi ini,
        atau panggil dengan fit_to_window=False.
        """
        if fit_to_window:
            try:
                self._fit_matplotlib_figure_to_window(fig, parent)
            except Exception as e:
                print(f"[EMBED] Resize figure gagal: {e}")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        try:
            toolbar = NavigationToolbar2Tk(
                canvas,
                toolbar_frame,
                pack_toolbar=False
            )
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)
        except TypeError:
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

        try:
            plt.close(fig)
        except Exception:
            pass

        return canvas
    
            


# =============================================================================
# PATCH DEPLOYMENT P16 SR + SVM KE GUI COBACLAUDE
# =============================================================================
# Patch ini menjaga semua tab lama dari EEGAnalysisGUI, lalu menambahkan:
# 1) Tab "Deep Learning SR" untuk menampilkan 4 channel fisik + 4 channel virtual.
# 2) Tab "Machine Learning" untuk hasil stress/anxiety/depression dari boundary SVM NumPy.


def _deploy_clear_frame(parent):
    for widget in parent.winfo_children():
        widget.destroy()


def _deploy_parse_float_or_none(text):
    text = str(text).strip().lower()
    if text in ["", "auto", "none", "null", "nan"]:
        return None
    return float(text)


def _deploy_ensure_vars(self):
    base = Path(__file__).resolve().parent
    if not hasattr(self, "deploy_output_dir_var"):
        self.deploy_output_dir_var = tk.StringVar(value=str(base / "DEPLOY_OUTPUT_GUI"))
    if not hasattr(self, "deploy_sr_ckpt_var"):
        self.deploy_sr_ckpt_var = tk.StringVar(value=str(base / "DEPLOY_EEG_SR_P16_C10B" / "models" / "p16_sr_model_deploy.pt"))
    if not hasattr(self, "deploy_sr_arch_var"):
        self.deploy_sr_arch_var = tk.StringVar(value=str(base / "DEPLOY_EEG_SR_P16_C10B" / "scripts" / "eeg_sr_hybrid_fusion.py"))
    if not hasattr(self, "deploy_sr_config_var"):
        self.deploy_sr_config_var = tk.StringVar(value=str(base / "DEPLOY_EEG_SR_P16_C10B" / "config" / "p16_sr_deploy_config.json"))
    if not hasattr(self, "deploy_olah_var"):
        self.deploy_olah_var = tk.StringVar(value=str(base / "olah_data_lengkap.py"))
    if not hasattr(self, "deploy_stress_var"):
        self.deploy_stress_var = tk.StringVar(value=str(base / DEFAULT_MODEL_PATHS["stress"]))
    if not hasattr(self, "deploy_anxiety_var"):
        self.deploy_anxiety_var = tk.StringVar(value=str(base / DEFAULT_MODEL_PATHS["anxiety"]))
    if not hasattr(self, "deploy_depression_var"):
        self.deploy_depression_var = tk.StringVar(value=str(base / DEFAULT_MODEL_PATHS["depression"]))
    if not hasattr(self, "deploy_pad_short_var"):
        self.deploy_pad_short_var = tk.BooleanVar(value=False)
    if not hasattr(self, "deploy_temp_stress_var"):
        self.deploy_temp_stress_var = tk.StringVar(value="auto")
    if not hasattr(self, "deploy_temp_anxiety_var"):
        self.deploy_temp_anxiety_var = tk.StringVar(value="auto")
    if not hasattr(self, "deploy_temp_depression_var"):
        self.deploy_temp_depression_var = tk.StringVar(value="auto")
    if not hasattr(self, "deploy_status_var"):
        self.deploy_status_var = tk.StringVar(value="Siap. Jalankan analisis MAICA dahulu, lalu klik Run SR + ML.")
    if not hasattr(self, "deploy_outputs"):
        self.deploy_outputs = None
    if not hasattr(self, "deploy_pipeline"):
        self.deploy_pipeline = None


def _deploy_row(parent, label, var, browse=False, is_dir=False):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, padx=8, pady=2)
    ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
    ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    if browse:
        def choose():
            from tkinter import filedialog
            p = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
            if p:
                var.set(p)
        ttk.Button(row, text="Browse", command=choose).pack(side=tk.LEFT, padx=4)

def _deploy_build_deep_learning_sr_tab(self):
    _deploy_ensure_vars(self)

    if not hasattr(self, "deep_learning_tab") or self.deep_learning_tab is None:
        self.deep_learning_tab = ttk.Frame(self.notebook)

    for w in self.deep_learning_tab.winfo_children():
        w.destroy()

    container = ttk.Frame(self.deep_learning_tab)
    container.pack(fill=tk.BOTH, expand=True, padx=4, pady=3)

    # ============================================================
    # HEADER COMPACT
    # ============================================================
    header = ttk.Frame(container)
    header.pack(fill=tk.X, pady=(0, 3))

    ttk.Label(
        header,
        text="Deep Learning EEG SR P16: Fp1,Fp2,T3/T7,T4/T8 → F3,F4,P3,P4",
        font=("Segoe UI", 9, "bold")
    ).pack(side=tk.LEFT, anchor="w")

    ttk.Button(
        header,
        text="Run SR + ML",
        command=self.run_p16_sr_svm_from_current_analysis
    ).pack(side=tk.RIGHT, padx=4)

    # ============================================================
    # CEK OUTPUT
    # ============================================================
    if not hasattr(self, "deploy_outputs") or not self.deploy_outputs:
        msg = (
            "Belum ada hasil SR.\n\n"
            "Langkah: load/acquire data → Analyze → klik Run SR + ML."
        )
        ttk.Label(
            container,
            text=msg,
            foreground="gray",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=8, pady=12)
        return

    out = self.deploy_outputs[0]

    if not hasattr(out, "sr8_df") or out.sr8_df is None:
        ttk.Label(
            container,
            text="Hasil SR8 tidak tersedia.",
            foreground="red",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=8, pady=12)
        return

    sr8_df = out.sr8_df.copy()

    # ============================================================
    # AMBIL CHANNEL
    # ============================================================
    ch_order = ["Fp1", "Fp2", "T3", "T4", "F3", "F4", "P3", "P4"]

    # fallback kalau masih pakai T7/T8
    if "T3" not in sr8_df.columns and "T7" in sr8_df.columns:
        sr8_df["T3"] = sr8_df["T7"]
    if "T4" not in sr8_df.columns and "T8" in sr8_df.columns:
        sr8_df["T4"] = sr8_df["T8"]

    available = [ch for ch in ch_order if ch in sr8_df.columns]

    if len(available) == 0:
        ttk.Label(
            container,
            text=f"Tidak ada channel SR8 yang cocok. Kolom tersedia: {list(sr8_df.columns)}",
            foreground="red"
        ).pack(anchor="w", padx=8, pady=12)
        return

    data = sr8_df[available].to_numpy(dtype=float).T

    # Batasi tampilan agar plotting ringan di Raspberry Pi
    fs = 250
    max_sec_show = 120
    max_samples_show = min(data.shape[1], fs * max_sec_show)

    data = data[:, :max_samples_show]
    t = np.arange(max_samples_show) / fs

    # ============================================================
    # PLOT COMPACT
    # ============================================================
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    n_ch = len(available)
    n_rows = 4
    n_cols = 2

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.5, 3.6),
        dpi=80,
        sharex=True
    )

    axs = np.asarray(axs).reshape(-1)

    for i, ch in enumerate(available):
        ax = axs[i]
        ax.plot(t, data[i], linewidth=0.8)

        if ch in ["Fp1", "Fp2", "T3", "T4", "T7", "T8"]:
            label_type = "Fisik"
        else:
            label_type = "Virtual SR"

        ax.set_title(f"{ch} ({label_type})", fontsize=8, loc="left")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.25)

        # label y dibuat kecil agar tidak makan tempat
        ax.set_ylabel("µV", fontsize=7)

    for j in range(len(available), len(axs)):
        axs[j].axis("off")

    fig.suptitle(
        "Hasil Deep Learning EEG Super-Resolution P16",
        fontsize=10,
        fontweight="bold",
        y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plot_frame = ttk.Frame(container)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    self._fit_matplotlib_figure_to_window(fig, container)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Toolbar sengaja tidak ditampilkan di Raspberry Pi agar area plot lebih luas

def _deploy_build_ml_result_tab(self):
    _deploy_ensure_vars(self)
    _deploy_clear_frame(self.ml_result_tab)

    container = ttk.Frame(self.ml_result_tab)
    container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    cfg = ttk.LabelFrame(container, text="Konfigurasi Deployment", padding=6)
    cfg.pack(fill=tk.X)
    _deploy_row(cfg, "Output folder", self.deploy_output_dir_var, browse=True, is_dir=True)
    _deploy_row(cfg, "SR checkpoint", self.deploy_sr_ckpt_var, browse=True)
    _deploy_row(cfg, "SR architecture", self.deploy_sr_arch_var, browse=True)
    _deploy_row(cfg, "SR config", self.deploy_sr_config_var, browse=True)
    _deploy_row(cfg, "olah_data_lengkap", self.deploy_olah_var, browse=True)
    _deploy_row(cfg, "Stress artifact", self.deploy_stress_var, browse=True)
    _deploy_row(cfg, "Anxiety artifact", self.deploy_anxiety_var, browse=True)
    _deploy_row(cfg, "Depression artifact", self.deploy_depression_var, browse=True)

    opt = ttk.Frame(container)
    opt.pack(fill=tk.X, pady=6)
    ttk.Checkbutton(opt, text="Pad jika durasi <120s", variable=self.deploy_pad_short_var).pack(side=tk.LEFT, padx=4)
    ttk.Label(opt, text="Temp stress:").pack(side=tk.LEFT, padx=(18, 2))
    ttk.Entry(opt, textvariable=self.deploy_temp_stress_var, width=8).pack(side=tk.LEFT)
    ttk.Label(opt, text="Temp anxiety:").pack(side=tk.LEFT, padx=(10, 2))
    ttk.Entry(opt, textvariable=self.deploy_temp_anxiety_var, width=8).pack(side=tk.LEFT)
    ttk.Label(opt, text="Temp depression:").pack(side=tk.LEFT, padx=(10, 2))
    ttk.Entry(opt, textvariable=self.deploy_temp_depression_var, width=8).pack(side=tk.LEFT)
    ttk.Button(opt, text="Run SR + ML", command=self.run_p16_sr_svm_from_current_analysis).pack(side=tk.RIGHT, padx=4)

    self.deploy_tree = ttk.Treeview(
        container,
        columns=("seg", "label", "status", "risk", "decision", "threshold", "margin", "temp", "direction"),
        show="headings",
        height=9
    )
    headings = [
        ("seg", "Seg", 55),
        ("label", "Label", 105),
        ("status", "Status", 120),
        ("risk", "Skor Risiko Model", 145),
        ("decision", "Decision Score", 135),
        ("threshold", "Threshold", 120),
        ("margin", "Margin", 120),
        ("temp", "Temp", 90),
        ("direction", "Dir", 55),
    ]
    for col, text, width in headings:
        self.deploy_tree.heading(col, text=text)
        self.deploy_tree.column(col, width=width, anchor="center")
    self.deploy_tree.pack(fill=tk.BOTH, expand=True, pady=6)

    ttk.Label(container, textvariable=self.deploy_status_var, foreground=self.colors.get("accent", "blue"), wraplength=1100).pack(fill=tk.X)
    ttk.Label(
        container,
        text="Disclaimer: Hasil ini merupakan estimasi model berbasis sinyal EEG/PPG dan bukan diagnosis klinis.",
        foreground=self.colors.get("text_dim", "gray")
    ).pack(fill=tk.X, pady=(4, 0))

    if self.deploy_outputs:
        self._fill_deploy_ml_tree(self.deploy_outputs)


def _deploy_make_pipeline(self):
    _deploy_ensure_vars(self)
    temps = {
        "stress": _deploy_parse_float_or_none(self.deploy_temp_stress_var.get()),
        "anxiety": _deploy_parse_float_or_none(self.deploy_temp_anxiety_var.get()),
        "depression": _deploy_parse_float_or_none(self.deploy_temp_depression_var.get()),
    }
    temps = {k: v for k, v in temps.items() if v is not None}
    svm_paths = {
        "stress": self.deploy_stress_var.get(),
        "anxiety": self.deploy_anxiety_var.get(),
        "depression": self.deploy_depression_var.get(),
    }
    return DeploymentPipeline(
        sr_ckpt_path=self.deploy_sr_ckpt_var.get(),
        sr_arch_path=self.deploy_sr_arch_var.get(),
        sr_config_path=self.deploy_sr_config_var.get() or None,
        olah_data_path=self.deploy_olah_var.get(),
        svm_artifact_paths=svm_paths,
        risk_temperatures=temps,
    )


def _deploy_extract_current_clean_4ch_and_ppg(self):
    if not hasattr(self, "analysis_results") or self.analysis_results is None:
        raise DeploymentError("Belum ada analysis_results. Jalankan Analyze terlebih dahulu.")
    if "ica" not in self.analysis_results or self.analysis_results["ica"].get("data") is None:
        raise DeploymentError("Hasil ICA/MAICA belum tersedia. Jalankan preprocessing dahulu.")

    clean = np.asarray(self.analysis_results["ica"]["data"], dtype=np.float32)
    if clean.ndim != 2:
        raise DeploymentError(f"Data ICA harus 2D (channel,time), diterima {clean.shape}")

    # Jika mode preprocessing 8CH: clean urutan Fp1,Fp2,F3,F4,T3,T4,P3,P4,
    # maka SR mengambil channel fisik Fp1,Fp2,T3,T4 = [0,1,4,5].
    # Jika mode preprocessing 4CH: clean sudah urutan Fp1,Fp2,T3,T4.
    try:
        mode = self.get_channel_mode()
    except Exception:
        mode = "8"

    if clean.shape[0] >= 8:
        eeg4 = clean[[0, 1, 2, 3], :]
        names = ["Fp1", "Fp2", "T3", "T4"]
    elif clean.shape[0] == 4:
        eeg4 = clean
        names = ["Fp1", "Fp2", "T3", "T4"]
    else:
        raise DeploymentError(
            f"Jumlah channel clean tidak didukung: {clean.shape[0]}. "
            "SR P16 membutuhkan 4 channel fisik Fp1,Fp2,T3,T4 atau data 8CH lama."
        )

    print(f"[SR+ML] Mode preprocessing={mode}CH | input SR shape={eeg4.shape} | names={names}")

    bpm_data = self.analysis_results.get("bpm_data", {}) or {}
    ppg = bpm_data.get("ppg_signal", None)
    if ppg is not None:
        ppg = np.asarray(ppg, dtype=np.float32).reshape(-1)
        min_len = min(eeg4.shape[1], len(ppg))
        eeg4 = eeg4[:, :min_len]
        ppg = ppg[:min_len]
    return eeg4, names, ppg


def _deploy_fill_ml_tree(self, outputs):
    if not hasattr(self, "deploy_tree"):
        return
    for item in self.deploy_tree.get_children():
        self.deploy_tree.delete(item)
    for out in outputs:
        for label in ["stress", "anxiety", "depression"]:
            if label not in out.predictions:
                continue
            pred = out.predictions[label]
            self.deploy_tree.insert("", "end", values=(
                out.segment_index,
                label.capitalize(),
                pred["status"],
                f"{pred['risk_score_model']:.1f}%",
                f"{pred['decision_score']:.8g}",
                f"{pred['deployment_threshold']:.8g}",
                f"{pred['margin_to_threshold']:.8g}",
                f"{pred.get('risk_temperature', np.nan):.3g}",
                pred["deployment_threshold_direction"],
            ))

def _compute_brainmap_vlims(self, bands):
    """
    Hitung skala vlim yang konsisten untuk GUI dan PDF.
    """
    import numpy as np

    vlims = {
        "abs_power": {},
        "rel_power": {},
        "ratios": {},
        "asym_by_band": {},
        "sasi": (-1.0, 1.0),
    }

    def finite_range(vals, symmetric=False):
        arr = np.asarray(vals, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return (0.0, 1.0)

        if symmetric:
            m = float(np.max(np.abs(arr)))
            if m < 1e-12:
                m = 1.0
            return (-m, m)

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

        if np.isclose(vmin, vmax):
            pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
            vmin -= pad
            vmax += pad

        return (vmin, vmax)

    for band, vals in bands.get("abs_power", {}).items():
        vlims["abs_power"][band] = finite_range(vals, symmetric=False)

    for band, vals in bands.get("rel_power", {}).items():
        vlims["rel_power"][band] = finite_range(vals, symmetric=False)

    for ratio_name, vals in bands.get("ratios", {}).items():
        vlims["ratios"][ratio_name] = finite_range(vals, symmetric=False)

    for band, vals in bands.get("asym_by_band", {}).items():
        vlims["asym_by_band"][band] = finite_range(vals, symmetric=True)

    if "sasi_values" in bands:
        vlims["sasi"] = finite_range(bands["sasi_values"], symmetric=True)

    return vlims

def _deploy_run_from_current_analysis(self):
    try:
        _deploy_ensure_vars(self)
        self.deploy_status_var.set("Memuat model SR P16 dan boundary SVM, lalu menjalankan inference...")
        self.root.update_idletasks()

        eeg4, names, ppg = self._extract_current_clean_4ch_and_ppg()
        self.deploy_pipeline = self._make_p16_svm_pipeline()
        outputs = self.deploy_pipeline.run_clean_4ch_array(
            eeg4,
            channel_names=names,
            ppg=ppg,
            fs=FS_EXPECTED,
            allow_resample=False,
            pad_short=bool(self.deploy_pad_short_var.get()),
        )
        self.deploy_outputs = outputs

        out_dir = Path(self.deploy_output_dir_var.get())
        self.deploy_pipeline.save_outputs(outputs, out_dir)

        self._fill_deploy_ml_tree(outputs)
        try:
            self._build_report_tab()
        except Exception as e:
            print(f"[REPORT] Gagal refresh tab report: {e}")

        # Update tab Brain Mapping menggunakan hasil 8CH dari SR
        self._refresh_brain_mapping_from_sr8(outputs)

        self._build_deep_learning_sr_tab()
        self._build_deep_learning_sr_tab()
        # Jangan rebuild ML tab penuh setelah tree terisi agar tidak flicker, cukup update status.
        self.deploy_status_var.set(
            f"Selesai. {len(outputs)} segmen diproses. Output SR8, fitur, audit, dan prediksi disimpan ke: {out_dir}"
        )
        try:
            self.notebook.select(self.ml_result_tab)
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        _deploy_ensure_vars(self)
        self.deploy_status_var.set(f"ERROR: {e}")
        messagebox.showerror("P16 SR + SVM Error", str(e))


# Simpan method lama lalu bungkus agar tab lama tetap sama.
_ORIG_BUILD_REALTIME_PLOT = EEGAnalysisGUI._build_realtime_plot
_ORIG_SHOW_VISUAL_RESULTS = EEGAnalysisGUI.show_visual_results


def _deploy_build_realtime_plot_wrapper(self):
    _ORIG_BUILD_REALTIME_PLOT(self)
    _deploy_ensure_vars(self)
    self.deep_learning_tab = ttk.Frame(self.notebook)
    self.ml_result_tab = ttk.Frame(self.notebook)


def _deploy_show_visual_results_wrapper(self, res):
    """
    Wrapper robust: tab SR+ML harus tetap muncul walaupun salah satu tab lama
    (misalnya Brain Mapping atau Saving Signal) error karena mismatch jumlah channel.
    Pada versi sebelumnya, tab SR+ML ditambahkan SETELAH _ORIG_SHOW_VISUAL_RESULTS().
    Kalau fungsi lama crash di tengah, kode penambah tab tidak pernah dieksekusi.
    """
    import traceback as _traceback

    _deploy_ensure_vars(self)

    # Pastikan frame tab deploy ada, meskipun _build_realtime_plot wrapper tidak terpanggil.
    if not hasattr(self, "deep_learning_tab") or self.deep_learning_tab is None:
        self.deep_learning_tab = ttk.Frame(self.notebook)
    if not hasattr(self, "ml_result_tab") or self.ml_result_tab is None:
        self.ml_result_tab = ttk.Frame(self.notebook)

    def _add_deploy_tabs():
        if not hasattr(self, "report_tab") or self.report_tab is None:
            self.report_tab = ttk.Frame(self.notebook)

        for name, frame in [
            ("Deep Learning SR", self.deep_learning_tab),
            ("Machine Learning", self.ml_result_tab),
            ("Report", self.report_tab),
        ]:
            
            try:
                if str(frame) not in self.notebook.tabs():
                    self.notebook.add(frame, text=f" {name} ")
            except Exception:
                try:
                    self.notebook.add(frame, text=f" {name} ")
                except Exception:
                    pass

    # Tambahkan lebih dulu agar tetap terlihat walaupun tab lama crash.
    _add_deploy_tabs()

    try:
        _ORIG_SHOW_VISUAL_RESULTS(self, res)
    except Exception as e:
        print(f"[WARN] show_visual_results lama error, tetapi tab SR+ML tetap dibuat: {e}")
        _traceback.print_exc()
    try:
        if isinstance(self.analysis_results, dict) and "bands" in self.analysis_results:
            self.analysis_results["brainmap_vlims"] = self._compute_brainmap_vlims(
                self.analysis_results["bands"]
            )
            print("[TOPO VLIM] brainmap_vlims disimpan dari hasil analisis.")
    except Exception as e:
        print(f"[TOPO VLIM] Gagal menyimpan vlim hasil analisis: {e}")
        
    # Tambahkan lagi setelah fungsi lama, karena fungsi lama bisa mengubah susunan tab/selection.
    _add_deploy_tabs()

    try:
        self._build_deep_learning_sr_tab()
    except Exception as e:
        print(f"[WARN] Gagal build tab Deep Learning SR: {e}")
        _traceback.print_exc()

    try:
        self._build_ml_result_tab()
        self._build_report_tab()
    except Exception as e:
        print(f"[WARN] Gagal build tab Machine Learning: {e}")
        _traceback.print_exc()
    try:
        self._apply_raspy_compact_layout()
    except Exception as e:
        print(f"[RASPY UI] Gagal apply compact layout setelah visual results: {e}")

def _deploy_make_sr8_topo_info(self):
    """
    Topo info khusus hasil SR8.
    Urutan GUI: Fp1, Fp2, F3, F4, T3, T4, P3, P4
    Untuk MNE: T3/T4 dipetakan ke T7/T8.
    """
    ch_names_mne = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]

    info = mne.create_info(
        ch_names=ch_names_mne,
        sfreq=FS,
        ch_types="eeg"
    )

    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="ignore"
    )

    return info

def _deploy_compute_brainmap_vlims(self, bands):
    """
    Menghitung skala warna brain mapping agar GUI dan PDF konsisten.
    Disimpan di self.analysis_results["brainmap_vlims"].
    """
    import numpy as np

    vlims = {
        "abs_power": {},
        "rel_power": {},
        "ratios": {},
        "asym_by_band": {},
        "sasi": (-1.0, 1.0),
        "lzc_maps": {},
    }

    def finite_range(vals, symmetric=False):
        arr = np.asarray(vals, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return (0.0, 1.0)

        if symmetric:
            m = float(np.max(np.abs(arr)))
            if m < 1e-12:
                m = 1.0
            return (-m, m)

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

        if np.isclose(vmin, vmax):
            pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
            vmin -= pad
            vmax += pad

        return (vmin, vmax)

    for band, vals in bands.get("abs_power", {}).items():
        vlims["abs_power"][band] = finite_range(vals, symmetric=False)

    for band, vals in bands.get("rel_power", {}).items():
        vlims["rel_power"][band] = finite_range(vals, symmetric=False)

    for ratio_name, vals in bands.get("ratios", {}).items():
        vlims["ratios"][ratio_name] = finite_range(vals, symmetric=False)

    for band, vals in bands.get("asym_by_band", {}).items():
        vlims["asym_by_band"][band] = finite_range(vals, symmetric=True)

    if "sasi_values" in bands:
        vlims["sasi"] = finite_range(bands["sasi_values"], symmetric=True)

    for name, vals in bands.get("lzc_maps", {}).items():
        vlims["lzc_maps"][name] = finite_range(vals, symmetric=False)

    return vlims

def _deploy_lzc_binary_complexity(self, binary_seq):
    """
    Lempel-Ziv Complexity sederhana untuk sequence biner.
    Output dinormalisasi kasar agar antar channel lebih comparable.
    """
    import numpy as np
    import math

    s = "".join(binary_seq.astype(int).astype(str).tolist())
    n = len(s)

    if n < 2:
        return 0.0

    i = 0
    c = 1
    l = 1
    k = 1
    k_max = 1

    while True:
        if i + k >= n or l + k >= n:
            c += 1
            break

        if s[i + k] == s[l + k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k

            i += 1

            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    # normalisasi umum: c / (n/log2(n))
    norm = n / math.log2(n) if n > 1 else 1.0
    return float(c / norm)


def _deploy_compute_lzc_maps_from_eeg8(self, eeg8, fs=FS):
    """
    Hitung LZC untuk semua band dan semua channel dari EEG 8CH.
    Input:
        eeg8 shape = (8, N)
        urutan = Fp1,Fp2,F3,F4,T3,T4,P3,P4

    Output:
        {
            "LZC_Delta": [8 nilai],
            "LZC_Theta": [8 nilai],
            "LZC_Alpha": [8 nilai],
            "LZC_Beta": [8 nilai]
        }
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    eeg8 = np.asarray(eeg8, dtype=float)

    if eeg8.ndim != 2:
        raise ValueError(f"eeg8 harus 2D, ditemukan shape={eeg8.shape}")

    if eeg8.shape[0] != 8 and eeg8.shape[1] == 8:
        eeg8 = eeg8.T

    if eeg8.shape[0] != 8:
        raise ValueError(f"LZC maps butuh 8 channel, ditemukan shape={eeg8.shape}")

    band_defs = {
        "LZC_Delta": (0.5, 4.0),
        "LZC_Theta": (4.0, 8.0),
        "LZC_Alpha": (8.0, 13.0),
        "LZC_Beta":  (13.0, 30.0),
    }

    nyq = fs / 2.0
    lzc_maps = {}

    for band_name, (lo, hi) in band_defs.items():
        low = max(lo, 0.1)
        high = min(hi, nyq - 0.1)

        b, a = butter(
            4,
            [low / nyq, high / nyq],
            btype="band"
        )

        x_band = filtfilt(b, a, eeg8, axis=-1)

        vals = []

        for ch in range(8):
            x = x_band[ch]
            x = np.asarray(x, dtype=float)

            # Hilangkan NaN/inf
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Binarisasi median, umum untuk LZC EEG
            med = np.median(x)
            binary = (x > med).astype(int)

            vals.append(self._lzc_binary_complexity(binary))

        lzc_maps[band_name] = vals

    return lzc_maps

def _deploy_compute_bands_from_sr8(self, eeg8, fs=FS):
    """
    Hitung struktur bands untuk Brain Mapping dari hasil SR8.

    Input eeg8:
        shape = (8, N)
        urutan = Fp1, Fp2, F3, F4, T3, T4, P3, P4
    """
    eeg8 = np.asarray(eeg8, dtype=float)

    if eeg8.ndim != 2:
        raise ValueError(f"eeg8 harus 2D, ditemukan shape={eeg8.shape}")

    if eeg8.shape[0] != 8 and eeg8.shape[1] == 8:
        eeg8 = eeg8.T

    if eeg8.shape[0] != 8:
        raise ValueError(f"Brain Mapping SR8 butuh 8 channel, ditemukan shape={eeg8.shape}")

    band_defs = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta":  (13.0, 30.0),
    }

    band_names = list(band_defs.keys())
    abs_power = {}
    rel_power = {}

    for band, (lo, hi) in band_defs.items():
        vals = []

        for ch in range(8):
            freqs, pxx = welch(
                eeg8[ch],
                fs=fs,
                nperseg=min(int(fs * 2), eeg8.shape[1])
            )

            mask = (freqs >= lo) & (freqs <= hi)

            if np.any(mask):
                vals.append(float(np.trapz(pxx[mask], freqs[mask])))
            else:
                vals.append(0.0)

        abs_power[band] = vals

    total_power = np.zeros(8, dtype=float)

    for band in band_names:
        total_power += np.asarray(abs_power[band], dtype=float)

    total_power = np.where(total_power == 0, 1e-10, total_power)

    for band in band_names:
        rel_power[band] = (
            np.asarray(abs_power[band], dtype=float) / total_power
        ).tolist()

    ratios = {}
    ratio_defs = [
        ("Delta/Theta", "Delta", "Theta"),
        ("Delta/Alpha", "Delta", "Alpha"),
        ("Delta/Beta",  "Delta", "Beta"),
        ("Theta/Alpha", "Theta", "Alpha"),
        ("Theta/Beta",  "Theta", "Beta"),
        ("Alpha/Beta",  "Alpha", "Beta"),
    ]

    for label, nb, db in ratio_defs:
        num = np.asarray(abs_power[nb], dtype=float)
        den = np.asarray(abs_power[db], dtype=float)
        den = np.where(den == 0, 1e-10, den)
        ratios[label] = (num / den).tolist()

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    ch_to_pair = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
    }

    asym_by_band = {}

    for band in band_names:
        p = np.asarray(abs_power[band], dtype=float)
        asym_topo = np.zeros(8, dtype=float)

        for ch_idx, pair_idx in ch_to_pair.items():
            li, ri = pairs[pair_idx]
            L, R = p[li], p[ri]
            denom = L + R
            asym_topo[ch_idx] = (L - R) / denom if abs(denom) > 1e-12 else 0.0

        asym_by_band[band] = asym_topo.tolist()

    alpha = np.asarray(abs_power["Alpha"], dtype=float)
    beta = np.asarray(abs_power["Beta"], dtype=float)
    sasi_values = (beta - alpha) / (beta + alpha + 1e-10)

    return {
        "names": band_names,
        "abs_power": abs_power,
        "rel_power": rel_power,
        "ratios": ratios,
        "asym_by_band": asym_by_band,
        "sasi_values": sasi_values.tolist(),
        "ch_names": ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"],
        "source": "SR8"
    }

def _deploy_make_pdf_topomap_info(self, n_ch=8):
    if n_ch == 4:
        ch_names_mne = ["Fp1", "Fp2", "T7", "T8"]
    else:
        ch_names_mne = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]

    info = mne.create_info(
        ch_names=ch_names_mne,
        sfreq=FS,
        ch_types="eeg"
    )

    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="ignore"
    )

    return info


def _deploy_fit_values_to_info(self, values, info):
    vals = np.asarray(values, dtype=float).reshape(-1)
    n_info = len(info["ch_names"])

    if len(vals) > n_info:
        vals = vals[:n_info]
    elif len(vals) < n_info:
        vals = np.pad(vals, (0, n_info - len(vals)), mode="constant", constant_values=0.0)

    return vals

def _deploy_save_existing_gui_figure_png(self, fig, out_png, dpi=250):
    """
    Menyimpan figure Matplotlib yang sudah tampil di GUI ke PNG.
    Ini membuat PDF memakai gambar yang sama persis dengan tab GUI.
    """
    if fig is None:
        raise ValueError("Figure GUI kosong, tidak bisa disimpan.")

    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15
    )
    
def _deploy_save_topomap_grid_png(
    self,
    title,
    data_dict,
    band_names,
    out_png,
    cmap="jet",
    symmetric=False,
    vlim_map=None,
    cbar_label=None
):
    """
    Simpan grid topomap ke PNG untuk PDF.
    vlim_map dipakai agar skala GUI dan PDF konsisten.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not band_names:
        raise ValueError(f"Tidak ada band_names untuk {title}")

    first_key = band_names[0]
    n_ch = len(np.asarray(data_dict[first_key]).reshape(-1))

    info = self._make_pdf_topomap_info(n_ch=n_ch)

    n = len(band_names)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.0 * cols, 3.7 * rows)
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes = np.asarray(axes).reshape(-1)

    for i, key in enumerate(band_names):
        vals = self._fit_values_to_info(data_dict[key], info)

        # =====================================================
        # PAKAI SKALA YANG SUDAH DISIMPAN
        # =====================================================
        if vlim_map is not None and key in vlim_map:
            vlim = vlim_map[key]
        else:
            finite = vals[np.isfinite(vals)]

            if finite.size == 0:
                vlim = (0.0, 1.0)

            elif symmetric:
                m = float(np.max(np.abs(finite)))
                if m < 1e-12:
                    m = 1.0
                vlim = (-m, m)

            else:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))

                if np.isclose(vmin, vmax):
                    pad = abs(vmin) * 0.05 if abs(vmin) > 1e-12 else 1.0
                    vmin -= pad
                    vmax += pad

                vlim = (vmin, vmax)

        im, _ = mne.viz.plot_topomap(
            vals,
            info,
            axes=axes[i],
            show=False,
            cmap=cmap,
            extrapolate="head",
            vlim=vlim
        )

        axes[i].set_title(str(key), fontsize=10)
        fig.colorbar(im, ax=axes[i], shrink=0.68, label=cbar_label or "")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(
        out_png,
        dpi=250,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15
    )

    plt.close(fig)

def _deploy_save_sasi_topomap_png(self, out_png):
    bands = self.analysis_results.get("bands", {})
    sasi_values = np.asarray(bands.get("sasi_values", []), dtype=float)

    if sasi_values.size == 0:
        raise RuntimeError("SASI values tidak tersedia untuk report.")

    n_ch = len(sasi_values)
    info = self._make_pdf_topomap_info(n_ch=n_ch)
    vals = self._fit_values_to_info(sasi_values, info)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im, _ = mne.viz.plot_topomap(
        vals,
        info,
        axes=ax,
        show=False,
        cmap="jet",
        extrapolate="head",
        vlim=(-1, 1)
    )
    ax.set_title("SASI", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="SASI (unitless)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _deploy_save_report_pdf_dialog(self):
    try:
        default_name = f"EEG_Report_{self._get_subject_name_from_file()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_path = filedialog.asksaveasfilename(
            title="Save EEG Report as PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")]
        )

        if not pdf_path:
            return

        self._save_report_pdf(pdf_path)
        messagebox.showinfo("Report Saved", f"PDF report berhasil disimpan:\n{pdf_path}")

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Report PDF Error", str(e))
        
        
# def _deploy_save_asymmetry_connectivity_png(self, asym_by_band, out_png, vlim_map=None):
#     """
#     Membuat PNG Asimetri Amplitudo dengan style connectivity seperti GUI:
#     - kepala penuh
#     - node 8 channel
#     - garis biru-merah (bwr)
#     - skala simetris negatif/positif
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import Normalize
#     from matplotlib.patches import Circle, Polygon, Arc
#     import matplotlib.cm as cm

#     band_names = list(asym_by_band.keys())
#     n_bands = len(band_names)

#     if n_bands == 0:
#         raise ValueError("asym_by_band kosong.")

#     fig, axes = plt.subplots(1, n_bands, figsize=(4.0 * n_bands, 4.4))
#     if n_bands == 1:
#         axes = [axes]

#     # ============================================================
#     # POSISI CHANNEL — samakan dengan layout GUI 8CH
#     # ============================================================
#     pos = {
#         "Fp1": (0.38, 0.86),
#         "Fp2": (0.62, 0.86),
#         "F3":  (0.36, 0.68),
#         "F4":  (0.64, 0.68),
#         "T3":  (0.18, 0.52),
#         "T4":  (0.82, 0.52),
#         "P3":  (0.34, 0.34),
#         "P4":  (0.66, 0.34),
#     }

#     # Pasangan homolog kiri-kanan
#     # Nilai asym di array disimpan duplikat per pair:
#     # index 0/1 = Fp1-Fp2
#     # index 2/3 = F3-F4
#     # index 4/5 = T3-T4
#     # index 6/7 = P3-P4
#     pairs = [
#         ("Fp1", "Fp2", 0),
#         ("F3",  "F4",  2),
#         ("T3",  "T4",  4),
#         ("P3",  "P4",  6),
#     ]

#     cmap = cm.get_cmap("bwr")

#     for ax, band in zip(axes, band_names):
#         vals = np.asarray(asym_by_band[band], dtype=float).reshape(-1)

#         # ========================================================
#         # VLIM — pakai yang sama dengan GUI kalau tersedia
#         # ========================================================
#         if vlim_map is not None and band in vlim_map:
#             vmin, vmax = vlim_map[band]
#         else:
#             finite = vals[np.isfinite(vals)]
#             if finite.size == 0:
#                 vmin, vmax = -1.0, 1.0
#             else:
#                 m = float(np.max(np.abs(finite)))
#                 if m < 1e-12:
#                     m = 1.0
#                 vmin, vmax = -m, m

#         norm = Normalize(vmin=vmin, vmax=vmax)

#         ax.set_title(f"Asym {band}", fontsize=11, fontweight="bold")
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect("equal")
#         ax.axis("off")

#         # ========================================================
#         # GAMBAR KEPALA
#         # ========================================================
#         head = Circle((0.50, 0.50), 0.43, facecolor="none", edgecolor="black", lw=1.2)
#         ax.add_patch(head)

#         nose = Polygon(
#             [[0.45, 0.93], [0.50, 0.98], [0.55, 0.93]],
#             closed=False,
#             fill=False,
#             edgecolor="black",
#             lw=1.2
#         )
#         ax.add_patch(nose)

#         left_ear = Arc((0.06, 0.50), 0.10, 0.22, theta1=70, theta2=290, lw=1.0, color="black")
#         right_ear = Arc((0.94, 0.50), 0.10, 0.22, theta1=-110, theta2=110, lw=1.0, color="black")
#         ax.add_patch(left_ear)
#         ax.add_patch(right_ear)

#         # ========================================================
#         # NODE CHANNEL
#         # ========================================================
#         for ch, (x, y) in pos.items():
#             circ = Circle((x, y), 0.015, facecolor="black", edgecolor="black", lw=0.8, zorder=3)
#             ax.add_patch(circ)
#             ax.text(x, y + 0.035, ch, ha="center", va="center", fontsize=7)

#         # ========================================================
#         # GARIS ASIMETRI
#         # ========================================================
#         denom = max(abs(vmin), abs(vmax), 1e-12)

#         for chL, chR, idx in pairs:
#             val = float(vals[idx]) if idx < len(vals) else 0.0

#             x1, y1 = pos[chL]
#             x2, y2 = pos[chR]

#             color = cmap(norm(val))
#             lw = 1.8 + 3.2 * (abs(val) / denom)

#             ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.95, zorder=2)

#             ax.text(
#                 (x1 + x2) / 2,
#                 (y1 + y2) / 2 + 0.025,
#                 f"{val:+.2f}",
#                 fontsize=7,
#                 ha="center",
#                 va="center",
#                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.80)
#             )

#         # ========================================================
#         # COLORBAR per subplot
#         # ========================================================
#         sm = cm.ScalarMappable(norm=norm, cmap=cmap)
#         sm.set_array([])
#         cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
#         cbar.ax.tick_params(labelsize=7)

#     fig.suptitle("Asimetri Amplitudo", fontsize=14, fontweight="bold")
#     plt.tight_layout(rect=[0, 0, 1, 0.92])

#     fig.savefig(
#         out_png,
#         dpi=250,
#         bbox_inches="tight",
#         facecolor="white",
#         pad_inches=0.15
#     )
#     plt.close(fig)
        
def _deploy_save_report_pdf(self, pdf_path):
    summary = self._collect_report_summary()

    # ============================================================
    # PALET WARNA REPORT — LAVENDER / PURPLE
    # ============================================================
    C = {
        "bg": "#F7F1FB",
        "panel": "#EFE7F8",
        "panel2": "#E5D7F5",
        "accent": "#8E6CCF",
        "accent_dark": "#5F3FA3",
        "accent_soft": "#B99AE8",
        "text": "#2D2242",
        "muted": "#6E6485",
        "white": "#FFFFFF",
        "line": "#D7C8ED",
        "green": "#5BC0A8",
        "yellow": "#F2B84B",
        "red": "#E66B8A",
        "lavender_dark": "#3F2E63",
    }

    def hexcol(x):
        return colors.HexColor(x)

    # ============================================================
    # HELPER LOCAL: GAMBAR TIDAK TERMAMPATKAN
    # ============================================================
    def rl_image_keep_aspect(img_path, max_w, max_h):
        """
        Membuat RLImage dengan aspect ratio asli.
        Ini mencegah topomap menjadi gepeng/tertarik.
        """
        try:
            img = Image.open(img_path)
            w, h = img.size
            if w <= 0 or h <= 0:
                return RLImage(img_path, width=max_w, height=max_h)

            scale = min(max_w / float(w), max_h / float(h))
            new_w = w * scale
            new_h = h * scale
            return RLImage(img_path, width=new_w, height=new_h)
        except Exception:
            return RLImage(img_path, width=max_w, height=max_h)

    # ============================================================
    # HELPER LOCAL: RISK BAR
    # ============================================================
    def make_risk_bar(label, risk_value, status="", width=8.4 * cm):
        risk_value = float(np.clip(risk_value, 0, 100))

        if risk_value < 40:
            bar_color = hexcol(C["green"])
        elif risk_value < 70:
            bar_color = hexcol(C["yellow"])
        else:
            bar_color = hexcol(C["red"])

        d = Drawing(width + 2.3 * cm, 1.45 * cm)

        d.add(String(
            0,
            1.07 * cm,
            f"{label}",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=hexcol(C["text"])
        ))

        if status:
            d.add(String(
                2.7 * cm,
                1.07 * cm,
                f"{status}",
                fontName="Helvetica",
                fontSize=8,
                fillColor=hexcol(C["muted"])
            ))

        # background bar
        d.add(Rect(
            0,
            0.35 * cm,
            width,
            0.38 * cm,
            rx=4,
            ry=4,
            fillColor=hexcol(C["panel2"]),
            strokeColor=hexcol(C["line"])
        ))

        # filled bar
        d.add(Rect(
            0,
            0.35 * cm,
            width * (risk_value / 100.0),
            0.38 * cm,
            rx=4,
            ry=4,
            fillColor=bar_color,
            strokeColor=bar_color
        ))

        # percent label
        d.add(String(
            width + 0.20 * cm,
            0.38 * cm,
            f"{risk_value:.1f}%",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=hexcol(C["text"])
        ))

        return d

    # ============================================================
    # HELPER LOCAL: PAGE BACKGROUND
    # ============================================================
    def pdf_page_bg(canvas, doc):
        page_w, page_h = doc.pagesize
        canvas.saveState()

        # background dasar lavender
        canvas.setFillColor(hexcol(C["bg"]))
        canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        # header strip
        canvas.setFillColor(hexcol(C["panel2"]))
        canvas.rect(0, page_h - 2.6 * cm, page_w, 2.6 * cm, fill=1, stroke=0)

        canvas.setFillColor(colors.white)
        canvas.roundRect(
            0.7 * cm,
            0.7 * cm,
            page_w - 1.4 * cm,
            page_h - 3.6 * cm,
            radius=10,
            fill=1,
            stroke=0
        )

        canvas.restoreState()

    # ============================================================
    # DOCUMENT SETUP
    # ============================================================
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=0.9 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "EEGTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=hexcol(C["text"]),
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=hexcol(C["accent_dark"]),
        spaceBefore=4,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        textColor=hexcol(C["text"]),
        leading=11,
    )

    small_style = ParagraphStyle(
        "SmallMuted",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=hexcol(C["muted"]),
        leading=9,
    )

    story = []

    subject = summary["subject_name"]
    session_dt = summary["session_datetime"]

    # ============================================================
    # PAGE 1 — SUMMARY
    # ============================================================
    story.append(Paragraph("EEG Report", title_style))
    story.append(Spacer(1, 0.10 * cm))

    meta_table = Table(
        [
            [
                "Date",
                datetime.now().strftime("%Y-%m-%d"),
                "Name",
                subject,
                "Session",
                session_dt
            ],
        ],
        colWidths=[2.0 * cm, 3.3 * cm, 2.0 * cm, 4.0 * cm, 2.0 * cm, 6.5 * cm]
    )

    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), hexcol(C["accent"])),
        ("BACKGROUND", (2, 0), (2, 0), hexcol(C["accent"])),
        ("BACKGROUND", (4, 0), (4, 0), hexcol(C["accent"])),

        ("BACKGROUND", (1, 0), (1, 0), hexcol(C["panel"])),
        ("BACKGROUND", (3, 0), (3, 0), hexcol(C["panel"])),
        ("BACKGROUND", (5, 0), (5, 0), hexcol(C["panel"])),

        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
        ("TEXTCOLOR", (2, 0), (2, 0), colors.white),
        ("TEXTCOLOR", (4, 0), (4, 0), colors.white),
        ("TEXTCOLOR", (1, 0), (5, 0), hexcol(C["text"])),

        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("GRID", (0, 0), (-1, -1), 0.35, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.6, hexcol(C["accent_soft"])),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(meta_table)
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph("EEG Test Results", section_style))

    # ============================================================
    # MODEL SUMMARY TABLE
    # ============================================================
    result_data = [["Model", "Status", "Skor Risiko Model"]]

    for row in summary["model_rows"]:
        result_data.append([
            row["label"],
            row["status"],
            f"{row['risk']:.1f}%"
        ])

    result_table = Table(
        result_data,
        colWidths=[5.0 * cm, 6.0 * cm, 5.0 * cm]
    )

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), hexcol(C["accent"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),

        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            hexcol(C["panel"])
        ]),

        ("TEXTCOLOR", (0, 1), (-1, -1), hexcol(C["text"])),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),

        ("GRID", (0, 0), (-1, -1), 0.3, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.7, hexcol(C["accent_soft"])),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    # Risk bar dibuat di kanan agar tampilan lebih menarik
    risk_bar_items = []
    for row in summary["model_rows"]:
        risk_bar_items.append([
            make_risk_bar(
                row["label"],
                row["risk"],
                status=row["status"],
                width=8.2 * cm
            )
        ])

    risk_bar_table = Table(
        risk_bar_items,
        colWidths=[11.3 * cm]
    )
    risk_bar_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, hexcol(C["line"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    summary_layout = Table(
        [[result_table, risk_bar_table]],
        colWidths=[16.5 * cm, 11.5 * cm]
    )

    summary_layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(summary_layout)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph(
        "Hasil ini merupakan estimasi model berbasis sinyal EEG/PPG dan bukan diagnosis klinis.",
        small_style
    ))

    # ============================================================
    # PAGE 2 — FEATURE VALUES
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("Feature Values Used by Each Model", section_style))
    story.append(Spacer(1, 0.15 * cm))

    feat_data = [["Model", "Feature", "Value"]]

    for row in summary["feature_rows"]:
        try:
            val_str = f"{float(row['value']):.6g}"
        except Exception:
            val_str = str(row["value"])

        feat_data.append([
            row["label"],
            row["feature"],
            val_str
        ])

    feat_table = Table(
        feat_data,
        colWidths=[3.4 * cm, 16.0 * cm, 6.0 * cm],
        repeatRows=1
    )

    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), hexcol(C["accent"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            hexcol(C["panel"])
        ]),

        ("TEXTCOLOR", (0, 1), (-1, -1), hexcol(C["text"])),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.8),

        ("GRID", (0, 0), (-1, -1), 0.25, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.7, hexcol(C["accent_soft"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(feat_table)

    # ============================================================
    # PAGE 3 — BRAIN MAPPING SUMMARY DALAM 1 LEMBAR
    # ============================================================
    if not hasattr(self, "analysis_results") or "bands" not in self.analysis_results:
        story.append(PageBreak())
        story.append(Paragraph("Brain Mapping Summary", section_style))
        story.append(Paragraph("Topomap tidak tersedia.", normal_style))

    else:
        bands = self.analysis_results["bands"]
        band_names = list(bands.get("names", []))
        bands = self.analysis_results["bands"]
# Ambil skala warna yang sudah disimpan dari GUI/SR8.
# Jika belum ada, hitung ulang agar PDF tetap bisa dibuat.
        vlims = self.analysis_results.get("brainmap_vlims", None)

        if vlims is None:
            try:
                vlims = self._compute_brainmap_vlims(bands)
                self.analysis_results["brainmap_vlims"] = vlims
                print("[PDF TOPO] brainmap_vlims belum ada, dihitung ulang untuk PDF.")
            except Exception as e:
                print(f"[PDF TOPO] Gagal hitung brainmap_vlims, pakai default kosong: {e}")
                vlims = {}
        tmp_dir = tempfile.mkdtemp(prefix="eeg_report_topomap_")
        image_paths = []

        try:
            # -------------------------
            # Absolute Power
            # -------------------------
            if "abs_power" in bands and band_names:
                abs_png = os.path.join(tmp_dir, "topomap_absolute.png")
                self._save_topomap_grid_png(
                    "Absolute Power",
                    bands["abs_power"],
                    band_names,
                    abs_png,
                    cmap="jet",
                    symmetric=False,
                    vlim_map=vlims.get("abs_power", {})
                )
                image_paths.append(("Absolute Power", abs_png))

            # -------------------------
            # Relative Power
            # -------------------------
            if "rel_power" in bands and band_names:
                rel_png = os.path.join(tmp_dir, "topomap_relative.png")
                self._save_topomap_grid_png(
                "Relative Power",
                bands["rel_power"],
                band_names,
                rel_png,
                cmap="jet",
                symmetric=False,
                vlim_map=vlims.get("rel_power", {})
            )
                image_paths.append(("Relative Power", rel_png))

            # -------------------------
            # Power Ratio
            # -------------------------
            if "ratios" in bands and bands["ratios"]:
                ratio_png = os.path.join(tmp_dir, "topomap_ratio.png")
                ratio_names = list(bands["ratios"].keys())
                self._save_topomap_grid_png(
                    "Power Ratio",
                    bands["ratios"],
                    ratio_names,
                    ratio_png,
                    cmap="jet",
                    symmetric=False,
                    vlim_map=vlims.get("ratios", {})
                )
                image_paths.append(("Power Ratio", ratio_png))

            # -------------------------
            # Asimetri Amplitudo
            # -------------------------
            # -------------------------
            # Asimetri Amplitudo
            # -------------------------
            if "asym_by_band" in bands and bands["asym_by_band"]:
                asym_png = os.path.join(tmp_dir, "topomap_asymmetry.png")

                # Prioritas utama: pakai figure asimetri yang sudah tampil di GUI
                asym_fig = None
                try:
                    asym_fig = getattr(self, "_report_fig_cache", {}).get("Asimetri Amplitudo", None)
                except Exception:
                    asym_fig = None

                if asym_fig is not None:
                    print("[PDF] Asimetri Amplitudo memakai figure langsung dari GUI.")
                    self._save_existing_gui_figure_png(
                        asym_fig,
                        asym_png,
                        dpi=250
                    )
                else:
                    print("[PDF] Figure GUI Asimetri tidak tersedia, fallback ke renderer PDF.")
                    self._save_asymmetry_gui_style_png(
                        bands["abs_power"],
                        band_names,
                        asym_png
                    )

                image_paths.append(("Asimetri Amplitudo", asym_png))
            # -------------------------
            # LZC Maps
            # -------------------------
            if "lzc_maps" in bands and bands["lzc_maps"]:
                lzc_png = os.path.join(tmp_dir, "topomap_lzc.png")
                lzc_names = list(bands["lzc_maps"].keys())

                self._save_topomap_grid_png(
                    "LZC Band Complexity",
                    bands["lzc_maps"],
                    lzc_names,
                    lzc_png,
                    cmap="viridis",
                    symmetric=False,
                    vlim_map=vlims.get("lzc_maps", {})
                )

                image_paths.append(("LZC Band Complexity", lzc_png))
    
            # -------------------------
            # SASI
            # -------------------------
            if "sasi_values" in bands:
                sasi_png = os.path.join(tmp_dir, "topomap_sasi.png")
                self._save_sasi_topomap_png(sasi_png)
                image_paths.append(("SASI", sasi_png))

            story.append(PageBreak())
            story.append(Paragraph("Brain Mapping Summary", section_style))
            story.append(Spacer(1, 0.1 * cm))

            path_dict = dict(image_paths)

            # Ukuran per cell sengaja dibuat tidak terlalu tinggi agar semua muat 1 page landscape
            cell_w = 13.0 * cm
            cell_h_top = 4.1 * cm
            cell_h_bottom = 3.6 * cm

            def img_block(title, key, max_h):
                img_path = path_dict.get(key)

                if not img_path:
                    return [
                        Paragraph(f"<b>{title}</b>", small_style),
                        Spacer(1, 0.05 * cm),
                        Paragraph("Tidak tersedia", small_style),
                    ]

                return [
                    Paragraph(f"<b>{title}</b>", small_style),
                    Spacer(1, 0.05 * cm),
                    rl_image_keep_aspect(
                        img_path,
                        max_w=cell_w - 0.6 * cm,
                        max_h=max_h
                    )
                ]

            brainmap_table = Table(
                [
                    [
                        img_block("Absolute Power", "Absolute Power", cell_h_top),
                        img_block("Relative Power", "Relative Power", cell_h_top),
                    ],
                    [
                        img_block("Power Ratio", "Power Ratio", cell_h_top),
                        img_block("Asimetri Amplitudo", "Asimetri Amplitudo", cell_h_top),
                    ],
                    [
                        img_block("SASI", "SASI", cell_h_bottom),
                        img_block("LZC Band Complexity", "LZC Band Complexity", cell_h_bottom),
                    ],
                ],
                colWidths=[13.2 * cm, 13.2 * cm],
                rowHeights=[4.7 * cm, 4.7 * cm, 4.2 * cm]
            )

            brainmap_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, hexcol(C["accent_soft"])),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, hexcol(C["line"])),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))

            story.append(brainmap_table)

        except Exception as e:
            story.append(PageBreak())
            story.append(Paragraph("Brain Mapping Summary", section_style))
            story.append(Paragraph(f"Gagal membuat topomap untuk PDF: {e}", normal_style))

    # ============================================================
    # BUILD DOCUMENT
    # ============================================================
    doc.build(
        story,
        onFirstPage=pdf_page_bg,
        onLaterPages=pdf_page_bg
    )
      
def _deploy_save_asymmetry_gui_style_png(self, abs_power, band_names, out_png):
    """
    Fallback jika figure GUI belum tersedia.
    Renderer ini sengaja meniru kode Asimetri Amplitudo di plot_brain_mapping():
    - mne.viz.plot_sensors
    - semua kombinasi pasangan channel
    - val = (p[i] - p[j]) / (p[i] + p[j])
    - garis merah jika positif, biru jika negatif
    - threshold abs(val) > 0.2
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import combinations
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    from mne.channels.layout import _find_topomap_coords

    # Pakai info SR8 kalau tersedia
    try:
        info = self.analysis_results.get("topo_info", None)
        if info is None or len(info["ch_names"]) != 8:
            info = self._make_sr8_topo_info()
    except Exception:
        info = self._make_sr8_topo_info()

    # Nama MNE bisa T7/T8; untuk indexing tidak masalah
    ch_names = list(info["ch_names"])
    n_ch = len(ch_names)

    n_bands = len(band_names)
    cols_a = min(n_bands, 4)
    rows_a = int(np.ceil(n_bands / cols_a))

    fig4, axs4 = plt.subplots(
        rows_a,
        cols_a,
        figsize=(3.5 * cols_a, 4.0 * rows_a),
        squeeze=False
    )

    pos2d = _find_topomap_coords(info, picks=np.arange(n_ch))
    ch_index = {ch: i for i, ch in enumerate(ch_names)}
    all_pairs = list(combinations(ch_names, 2))

    for b_idx, band in enumerate(band_names):
        ax = axs4[b_idx // cols_a][b_idx % cols_a]
        ax.set_title(f"Asym {band}", fontsize=8, fontweight="bold")

        mne.viz.plot_sensors(
            info,
            axes=ax,
            show=False,
            kind="topomap"
        )

        p = np.array(abs_power[band], dtype=float).reshape(-1)

        # Samakan panjang p dengan jumlah channel info
        if len(p) > n_ch:
            p = p[:n_ch]
        elif len(p) < n_ch:
            p = np.pad(p, (0, n_ch - len(p)), mode="constant", constant_values=0.0)

        for ch1, ch2 in all_pairs:
            i = ch_index[ch1]
            j = ch_index[ch2]

            denom = p[i] + p[j]
            val = (p[i] - p[j]) / denom if denom != 0 else 0.0
            coh = abs(val)

            if coh > 0.2:
                x1, y1 = pos2d[i]
                x2, y2 = pos2d[j]

                line_color = "red" if val > 0 else "blue"

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=line_color,
                    linewidth=0.5 + 1.5 * coh,
                    alpha=0.7,
                    solid_capstyle="round"
                )

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="(-) Asimetri"),
        Line2D([0], [0], color="red", lw=2, label="(+) Asimetri"),
    ]

    fig4.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02)
    )

    ax_bar = fig4.add_axes([0.4, 0.07, 0.2, 0.015])
    cmap_scale = mcolors.ListedColormap(["blue", "red"])
    norm_scale = mcolors.Normalize(vmin=-1, vmax=1)

    cb = fig4.colorbar(
        plt.cm.ScalarMappable(norm=norm_scale, cmap=cmap_scale),
        cax=ax_bar,
        orientation="horizontal",
        ticks=[]
    )
    cb.outline.set_visible(False)

    ax_bar.text(
        -0.2,
        0.5,
        "-",
        transform=ax_bar.transAxes,
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
        color="blue"
    )

    ax_bar.text(
        1.2,
        0.5,
        "+",
        transform=ax_bar.transAxes,
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color="red"
    )

    for b_idx in range(n_bands, rows_a * cols_a):
        axs4[b_idx // cols_a][b_idx % cols_a].axis("off")

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    fig4.savefig(
        out_png,
        dpi=250,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15
    )
    plt.close(fig4)
      
def _deploy_refresh_brain_mapping_from_sr8(self, outputs):
    """
    Setelah tombol Run SR + ML ditekan, hasil SR8 langsung dipakai
    untuk memperbarui tab Brain Mapping.
    """
    try:
        if outputs is None or len(outputs) == 0:
            raise ValueError("outputs SR kosong.")

        # outputs adalah list PipelineSegmentOutput
        sr8_dfs = []

        for out in outputs:
            if hasattr(out, "sr8_df"):
                sr8_dfs.append(out.sr8_df.copy())

        if not sr8_dfs:
            raise ValueError("Tidak menemukan sr8_df pada outputs.")

        sr8_df = pd.concat(sr8_dfs, ignore_index=True)

        # Pastikan nama channel sesuai GUI lama
        if "T7" in sr8_df.columns and "T3" not in sr8_df.columns:
            sr8_df = sr8_df.rename(columns={"T7": "T3"})

        if "T8" in sr8_df.columns and "T4" not in sr8_df.columns:
            sr8_df = sr8_df.rename(columns={"T8": "T4"})

        sr8_order = ["Fp1", "Fp2", "F3", "F4", "T3", "T4", "P3", "P4"]
        missing = [c for c in sr8_order if c not in sr8_df.columns]

        if missing:
            raise ValueError(f"Kolom SR8 tidak lengkap untuk Brain Mapping: {missing}")

        eeg8_sr = sr8_df[sr8_order].to_numpy(dtype=float).T

        self.analysis_results["sr8"] = {
            "data": eeg8_sr,
            "df": sr8_df,
            "ch_names": sr8_order,
            "source": "P16 EEG Super-Resolution"
        }

        self.analysis_results["topo_info"] = self._make_sr8_topo_info()
        self.analysis_results["bands"] = self._compute_bands_from_sr8(eeg8_sr, fs=FS)
        # Tambahkan LZC semua band dan semua channel untuk topomap/report
        self.analysis_results["bands"]["lzc_maps"] = self._compute_lzc_maps_from_eeg8(
            eeg8_sr,
            fs=FS
        )

        self.analysis_results["brainmap_vlims"] = self._compute_brainmap_vlims(
            self.analysis_results["bands"]
        )
        print("[SR8 TOPO] Brain Mapping diupdate memakai hasil SR8.")
        print("[SR8 TOPO] eeg8_sr shape:", eeg8_sr.shape)

        if hasattr(self, "topo_tab"):
            self.build_brain_mapping_tab(
                self.topo_tab,
                self.analysis_results["bands"]
            )

    except Exception as e:
        import traceback
        print(f"[SR8 TOPO] Gagal update Brain Mapping dari SR8: {e}")
        traceback.print_exc()
        
EEGAnalysisGUI._build_realtime_plot = _deploy_build_realtime_plot_wrapper
EEGAnalysisGUI.show_visual_results = _deploy_show_visual_results_wrapper
EEGAnalysisGUI._build_deep_learning_sr_tab = _deploy_build_deep_learning_sr_tab
EEGAnalysisGUI._build_ml_result_tab = _deploy_build_ml_result_tab
EEGAnalysisGUI._make_p16_svm_pipeline = _deploy_make_pipeline
EEGAnalysisGUI._extract_current_clean_4ch_and_ppg = _deploy_extract_current_clean_4ch_and_ppg
EEGAnalysisGUI._fill_deploy_ml_tree = _deploy_fill_ml_tree

EEGAnalysisGUI.run_p16_sr_svm_from_current_analysis = _deploy_run_from_current_analysis
EEGAnalysisGUI._make_sr8_topo_info = _deploy_make_sr8_topo_info
EEGAnalysisGUI._compute_bands_from_sr8 = _deploy_compute_bands_from_sr8
EEGAnalysisGUI._refresh_brain_mapping_from_sr8 = _deploy_refresh_brain_mapping_from_sr8

EEGAnalysisGUI._get_subject_name_from_file = _deploy_get_subject_name_from_file
EEGAnalysisGUI._collect_report_summary = _deploy_collect_report_summary
EEGAnalysisGUI._build_report_tab = _deploy_build_report_tab

EEGAnalysisGUI._make_pdf_topomap_info = _deploy_make_pdf_topomap_info
EEGAnalysisGUI._fit_values_to_info = _deploy_fit_values_to_info
EEGAnalysisGUI._save_topomap_grid_png = _deploy_save_topomap_grid_png
EEGAnalysisGUI._save_sasi_topomap_png = _deploy_save_sasi_topomap_png
EEGAnalysisGUI._save_report_pdf_dialog = _deploy_save_report_pdf_dialog
EEGAnalysisGUI._save_report_pdf = _deploy_save_report_pdf

EEGAnalysisGUI._rl_image_keep_aspect = _deploy_rl_image_keep_aspect
EEGAnalysisGUI._pdf_page_bg = _deploy_pdf_page_bg
EEGAnalysisGUI._make_risk_bar = _deploy_make_risk_bar

EEGAnalysisGUI._compute_brainmap_vlims = _deploy_compute_brainmap_vlims
EEGAnalysisGUI._lzc_binary_complexity = _deploy_lzc_binary_complexity
EEGAnalysisGUI._compute_lzc_maps_from_eeg8 = _deploy_compute_lzc_maps_from_eeg8

EEGAnalysisGUI._save_existing_gui_figure_png = _deploy_save_existing_gui_figure_png
EEGAnalysisGUI._save_asymmetry_gui_style_png = _deploy_save_asymmetry_gui_style_png


# ============================================================================
# PATCH TAMBAHAN — PREPARE 3 DETIK SEBELUM RECORD + PDF CLINICAL STYLE
# ============================================================================

def _deploy_parse_filename_identity(self):
    """Ambil identitas dari nama file yang sedang aktif."""
    import os, re
    fname = ""
    for attr in ["last_filename", "loaded_file_path", "current_file_path"]:
        if hasattr(self, attr):
            val = getattr(self, attr)
            if val:
                fname = str(val)
                break
    if not fname and hasattr(self, "file_label"):
        try:
            fname = self.file_label.cget("text")
        except Exception:
            fname = ""

    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    out = {
        "filename": base or "Unknown",
        "subject_display": self._get_subject_name_from_file() if hasattr(self, '_get_subject_name_from_file') else "Unknown",
        "subject": None,
        "rec_id": None,
        "snr_db": None,
        "stress": None,
        "anxiety": None,
        "depression": None,
        "gender": None,
        "segment": None,
        "tag": None,
    }

    pat = re.compile(
        r'(?P<subject>[A-Za-z]+)_rec(?P<rec_id>\d+)_clean_(?P<snr>[-]?\d+(?:\.\d+)?)dB_'
        r'(?P<subject2>[A-Za-z]+)(?P<rec_id2>\d+)_S(?P<stress>\d+)_A(?P<anxiety>\d+)_D(?P<depression>\d+)_'
        r'(?P<gender>[FM])(?:_seg(?P<segment>\d+))?(?:_(?P<tag>.+))?$',
        re.IGNORECASE
    )
    m = pat.search(stem)
    if m:
        gd = m.groupdict()
        out.update({
            "subject": gd.get("subject"),
            "rec_id": gd.get("rec_id"),
            "snr_db": gd.get("snr"),
            "stress": gd.get("stress"),
            "anxiety": gd.get("anxiety"),
            "depression": gd.get("depression"),
            "gender": gd.get("gender"),
            "segment": gd.get("segment"),
            "tag": gd.get("tag"),
        })
        out["subject_display"] = f"{gd.get('subject','')} {gd.get('rec_id','')}".strip()
        return out

    # fallback lebih longgar
    parts = stem.split('_')
    if parts:
        out["subject"] = parts[0]
    return out


def _deploy_start_recording_with_prep(self):
    """Mulai tahap persiapan 3 detik, lalu baru benar-benar merekam sinyal."""
    import time
    from tkinter import messagebox
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    except Exception:
        # jika import global sudah ada, biarkan pakai global
        pass

    # hindari klik ganda
    if getattr(self, '_prep_recording_active', False) or getattr(self, 'is_recording', False):
        return

    # Jika board sudah di-release, reconnect dulu
    if not self.board:
        try:
            self.status_label.config(text="Status: Reconnecting...", foreground="orange")
            self.root.update_idletasks()

            params = BrainFlowInputParams()
            params.serial_port = self.com_entry.get()
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            self.board.config_board("v")
            for ch in range(1, NUM_CHANNELS + 1):
                self.board.config_board(f"x{ch}060101X")
                time.sleep(0.1)
            self.board.config_board("c")
            time.sleep(0.2)
            self.board.config_board("/3")
            time.sleep(0.5)
            self.board.start_stream()
            self.streaming = True
            self.update_live_plot()
        except Exception as e:
            err_msg = str(e)
            messagebox.showerror("Reconnect Error", f"Gagal reconnect ke board:\n{err_msg}")
            self.status_label.config(text="Status: Reconnect Failed", foreground="red")
            self.board = None
            return

    if not self.streaming:
        self.board.start_stream()
        self.streaming = True
        self.update_live_plot()

    self._prep_recording_active = True
    self._prep_remaining = 3
    self.status_label.config(text="Status: PERSIAPAN... rekam dimulai dalam 3 detik", foreground="#8E6CCF")
    self.btn_record.config(state=tk.DISABLED, text="Prepare...")
    if hasattr(self, 'btn_stop'):
        self.btn_stop.config(state=tk.DISABLED)
    if hasattr(self, 'countdown_label'):
        self.countdown_label.config(text="PREP 00:03", fg="#8E6CCF")

    if getattr(self, '_prep_countdown_job', None):
        self.root.after_cancel(self._prep_countdown_job)
        self._prep_countdown_job = None
    self._tick_recording_prep_countdown()


def _deploy_tick_recording_prep_countdown(self):
    if not getattr(self, '_prep_recording_active', False):
        return

    remaining = int(getattr(self, '_prep_remaining', 0))
    if hasattr(self, 'countdown_label'):
        self.countdown_label.config(text=f"PREP 00:0{max(remaining,0)}")

    if remaining <= 0:
        self._prep_recording_active = False
        self._prep_countdown_job = None
        self._begin_recording_after_prep()
        return

    self._prep_remaining = remaining - 1
    self._prep_countdown_job = self.root.after(1000, self._tick_recording_prep_countdown)


def _deploy_begin_recording_after_prep(self):
    """Bagian recording asli dijalankan setelah fase persiapan selesai."""
    import time
    self.is_recording = True
    self.session_data = []
    self.data_buffer.fill(0)

    try:
        self.reset_all_impedance()
    except Exception:
        pass

    for i, ch_name in enumerate(LIVE_EEG_NAMES):
        try:
            self.ax_flat[i].set_ylim(-100, 100)
        except Exception:
            pass

    self.status_label.config(text="Status: RECORDING DATA...", foreground="red")
    self.btn_record.config(state=tk.DISABLED, text="▶ Recording...")
    if hasattr(self, 'btn_stop'):
        self.btn_stop.config(state=tk.NORMAL)
    self.record_start_time = time.time()
    self._start_countdown(self.duration_var.get())


def _deploy_get_record_and_edit_length(self):
    """Record length & edit length untuk halaman Technical Information."""
    def fmt_sec(seconds):
        try:
            seconds = max(0, int(round(float(seconds))))
        except Exception:
            seconds = 0
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"

    meta = getattr(self, "_report_len", None)
    if meta:
        record_seconds = meta.get("record_s", SEGMENT_SECONDS)
        edit_seconds = meta.get("edit_s", record_seconds)
    else:
        # fallback: semua data diasumsikan 2 menit
        record_seconds = SEGMENT_SECONDS
        edit_seconds = SEGMENT_SECONDS
    return fmt_sec(record_seconds), fmt_sec(edit_seconds)

def _deploy_save_report_pdf_clinical(self, pdf_path):
    """PDF report dengan halaman awal technical information dan layout brain map bernuansa klinis."""
    import os, math, tempfile
    import numpy as np
    from datetime import datetime
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage
    )
    from reportlab.graphics.shapes import Drawing, Rect, String

    summary = self._collect_report_summary()
    ident = self._parse_filename_identity() if hasattr(self, '_parse_filename_identity') else {}
    record_len, edit_len = self._get_record_and_edit_length()

    C = {
        "bg": "#F7F1FB",
        "panel": "#F3ECFA",
        "panel2": "#E9DBF7",
        "accent": "#8E6CCF",
        "accent_dark": "#5F3FA3",
        "accent_soft": "#B99AE8",
        "text": "#2D2242",
        "muted": "#6E6485",
        "white": "#FFFFFF",
        "line": "#D7C8ED",
        "green": "#59B79A",
        "yellow": "#F0B54A",
        "red": "#DE6D86",
    }
    hexcol = colors.HexColor

    def rl_image_keep_aspect(img_path, max_w, max_h):
        try:
            img = Image.open(img_path)
            w, h = img.size
            if w <= 0 or h <= 0:
                return RLImage(img_path, width=max_w, height=max_h)
            scale = min(max_w / float(w), max_h / float(h))
            return RLImage(img_path, width=w * scale, height=h * scale)
        except Exception:
            return RLImage(img_path, width=max_w, height=max_h)

    W_CONTENT = 26.4 * cm   # lebar area konten (A4 landscape - margin kiri/kanan)

    def page_bg(canvas, doc):
        page_w, page_h = doc.pagesize
        band_h = 1.9 * cm
        canvas.saveState()
        # latar halaman
        canvas.setFillColor(hexcol(C["bg"]))
        canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        # header band + judul laporan
        canvas.setFillColor(hexcol(C["panel2"]))
        canvas.rect(0, page_h - band_h, page_w, band_h, fill=1, stroke=0)
        canvas.setStrokeColor(hexcol(C["accent"]))
        canvas.setLineWidth(1.4)
        canvas.line(0, page_h - band_h, page_w, page_h - band_h)
        canvas.setFillColor(hexcol(C["accent_dark"]))
        canvas.setFont("Helvetica-Bold", 15)
        canvas.drawString(1.4 * cm, page_h - 1.17 * cm, "EEG ANALYSIS REPORT")
        canvas.setFillColor(hexcol(C["muted"]))
        canvas.setFont("Helvetica", 9.5)
        canvas.drawRightString(page_w - 1.4 * cm, page_h - 1.12 * cm,
                               f"{subject}   |   Page {doc.page}")
        # panel putih
        canvas.setFillColor(colors.white)
        canvas.roundRect(0.7 * cm, 0.7 * cm, page_w - 1.4 * cm, page_h - 2.9 * cm,
                         radius=10, fill=1, stroke=0)
        # footer
        canvas.setFillColor(hexcol(C["muted"]))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.9 * cm, 0.3 * cm,
                          "Model-based estimation from EEG/PPG signals. Not a clinical diagnosis.")
        canvas.restoreState()

    def _status_en(s):
        s = str(s or "").strip()
        return {"tidak normal": "Abnormal"}.get(s.lower(), s or "-")

    def make_risk_bar(label, risk_value, status="", width=9.6 * cm):
        risk_value = float(np.clip(np.nan_to_num(risk_value), 0, 100))
        st = str(status or "").strip().lower()
        if st == "normal":
            bar_color = hexcol(C["green"])
        elif st in ("tidak normal", "abnormal"):
            bar_color = hexcol(C["red"])
        elif risk_value < 40:
            bar_color = hexcol(C["green"])
        elif risk_value < 70:
            bar_color = hexcol(C["yellow"])
        else:
            bar_color = hexcol(C["red"])
        d = Drawing(width + 2.8 * cm, 1.6 * cm)
        d.add(String(0, 1.15 * cm, label, fontName="Helvetica-Bold", fontSize=10.5, fillColor=hexcol(C["text"])))
        d.add(String(3.0 * cm, 1.15 * cm, _status_en(status), fontName="Helvetica", fontSize=9, fillColor=hexcol(C["muted"])))
        d.add(Rect(0, 0.35 * cm, width, 0.42 * cm, rx=4, ry=4, fillColor=hexcol(C["panel2"]), strokeColor=hexcol(C["line"])))
        d.add(Rect(0, 0.35 * cm, width * (risk_value / 100.0), 0.42 * cm, rx=4, ry=4, fillColor=bar_color, strokeColor=bar_color))
        d.add(String(width + 0.25 * cm, 0.40 * cm, f"{risk_value:.1f}%", fontName="Helvetica-Bold", fontSize=10.5, fillColor=hexcol(C["text"])))
        return d

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=2.5 * cm,
        bottomMargin=1.1 * cm,
        title="EEG Analysis Report",
    )

    styles = getSampleStyleSheet()
    page_title_style = ParagraphStyle("RptPageTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                                      fontSize=18, leading=22, textColor=hexcol(C["text"]),
                                      alignment=TA_LEFT, spaceAfter=2)
    title_style = page_title_style
    section_style = ParagraphStyle("RptSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=13.5, leading=16, textColor=hexcol(C["accent_dark"]),
                                   spaceBefore=4, spaceAfter=6)
    normal_style = ParagraphStyle("RptNormal", parent=styles["Normal"], fontName="Helvetica",
                                  fontSize=9.5, leading=12, textColor=hexcol(C["text"]))
    small_style = ParagraphStyle("RptSmall", parent=styles["Normal"], fontName="Helvetica",
                                 fontSize=8.5, leading=10.5, textColor=hexcol(C["muted"]))
    cell_style = ParagraphStyle("RptCell", parent=styles["Normal"], fontName="Helvetica-Bold",
                                fontSize=9.5, leading=11.5, textColor=hexcol(C["text"]))

    story = []
    subject = summary["subject_name"]
    session_dt = summary["session_datetime"]

    def _v(x):
        return str(x) if x not in (None, "") else "-"

    # ------------------------------------------------------------------
    # PAGE 1 — SUBJECT & RECORDING SUMMARY
    # ------------------------------------------------------------------
    story.append(Paragraph("Subject &amp; Recording Summary", page_title_style))
    story.append(Paragraph("Montage: 8-Channel EEG + PPG", small_style))
    story.append(Spacer(1, 0.35 * cm))

    snr_txt = f"{ident.get('snr_db')} dB" if ident.get("snr_db") else "-"
    info_rows = [
        ["Subject", _v(ident.get("subject_display") or subject), "Stress Score", _v(ident.get("stress")), "Gender", _v(ident.get("gender"))],
        ["File Name", Paragraph(_v(ident.get("filename")), cell_style), "Anxiety Score", _v(ident.get("anxiety")), "Sampling Rate", f"{FS} Hz"],
        ["Recording Length", record_len, "Depression Score", _v(ident.get("depression")), "Analyzed Length", edit_len],
        ["Report Generated", session_dt, "SNR", snr_txt, "Device", "OpenBCI Cyton"],
    ]
    extra = [("Record ID", ident.get("rec_id")), ("Segment", ident.get("segment")), ("Tag", ident.get("tag"))]
    if any(v not in (None, "") for _, v in extra):
        info_rows.append([extra[0][0], _v(extra[0][1]), extra[1][0], _v(extra[1][1]), extra[2][0], _v(extra[2][1])])

    info_colwidths = [3.6 * cm, 7.8 * cm, 3.8 * cm, 3.2 * cm, 3.6 * cm, 4.4 * cm]
    info_table = Table(info_rows, colWidths=info_colwidths, hAlign="LEFT")
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), hexcol(C["accent"])),
        ("BACKGROUND", (2, 0), (2, -1), hexcol(C["accent"])),
        ("BACKGROUND", (4, 0), (4, -1), hexcol(C["accent"])),
        ("BACKGROUND", (1, 0), (1, -1), hexcol(C["panel"])),
        ("BACKGROUND", (3, 0), (3, -1), hexcol(C["panel"])),
        ("BACKGROUND", (5, 0), (5, -1), hexcol(C["panel"])),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.white),
        ("TEXTCOLOR", (4, 0), (4, -1), colors.white),
        ("TEXTCOLOR", (1, 0), (1, -1), hexcol(C["text"])),
        ("TEXTCOLOR", (3, 0), (3, -1), hexcol(C["text"])),
        ("TEXTCOLOR", (5, 0), (5, -1), hexcol(C["text"])),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.35, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.6, hexcol(C["accent_soft"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph("Model Risk Estimation", section_style))
    risk_items = [[make_risk_bar(row["label"], row["risk"], row["status"], width=9.6 * cm)] for row in summary["model_rows"]]
    risk_table = Table(risk_items, colWidths=[13.4 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.55, hexcol(C["line"])),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    result_data = [["Condition", "Model Status", "Risk Score"]] + [
        [r["label"], _status_en(r["status"]), f"{r['risk']:.1f}%"] for r in summary["model_rows"]
    ]
    result_table = Table(result_data, colWidths=[4.2 * cm, 4.4 * cm, 3.4 * cm],
                         rowHeights=[1.0 * cm] + [1.5 * cm] * len(summary["model_rows"]))
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), hexcol(C["accent"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, hexcol(C["panel"])]),
        ("TEXTCOLOR", (0, 1), (-1, -1), hexcol(C["text"])),
        ("GRID", (0, 0), (-1, -1), 0.30, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.60, hexcol(C["accent_soft"])),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    combo = Table([[result_table, risk_table]], colWidths=[12.6 * cm, 13.8 * cm], hAlign="LEFT")
    combo.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(combo)
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(
        "Stress / Anxiety / Depression Score are the reference values supplied for this recording; "
        "Model Risk Score is the estimate produced by the model.", small_style))
    story.append(Paragraph(
        "These results are model-based estimates from EEG/PPG signals and do not replace a clinical diagnosis.",
        small_style))

    # ------------------------------------------------------------------
    # PAGE 2 — MODEL INPUT FEATURES
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Model Input Features", page_title_style))
    story.append(Paragraph("Feature values extracted from this recording and used as input by each model.", small_style))
    story.append(Spacer(1, 0.3 * cm))
    feat_data = [["Model", "Feature", "Value"]]
    for row in summary["feature_rows"]:
        try:
            val_str = f"{float(row['value']):.6g}"
        except Exception:
            val_str = str(row["value"])
        feat_data.append([row["label"], row["feature"], val_str])
    feat_table = Table(feat_data, colWidths=[3.8 * cm, 16.6 * cm, 6.0 * cm], repeatRows=1, hAlign="LEFT")
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), hexcol(C["accent"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, hexcol(C["panel"])]),
        ("GRID", (0, 0), (-1, -1), 0.25, hexcol(C["line"])),
        ("BOX", (0, 0), (-1, -1), 0.6, hexcol(C["accent_soft"])),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(feat_table)
    # ------------------------------------------------------------------
    # PAGE 3 — BRAIN MAPPING SUMMARY (clinical vibe)
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("EEG Topographic Brain Maps", page_title_style))
    story.append(Paragraph("Scalp distribution of power, asymmetry, complexity and spectral indices across the 8 channels.", small_style))
    story.append(Spacer(1, 0.25 * cm))
    
    if not hasattr(self, "analysis_results") or "bands" not in self.analysis_results:
        story.append(Paragraph("Topomap tidak tersedia.", normal_style))
    else:
        bands = self.analysis_results["bands"]
        band_names = list(bands.get("names", []))
        vlims = self.analysis_results.get("brainmap_vlims", None)
        if vlims is None:
            try:
                vlims = self._compute_brainmap_vlims(bands)
            except Exception:
                vlims = {}
        tmp_dir = tempfile.mkdtemp(prefix="eeg_report_topomap_")
        image_paths = []
        try:
            if "abs_power" in bands and band_names:
                p = os.path.join(tmp_dir, "topomap_abs.png")
                self._save_topomap_grid_png("Absolute Power", bands["abs_power"], band_names, p, cmap="jet", symmetric=False, vlim_map=vlims.get("abs_power", {}), cbar_label="µV²")
                image_paths.append(("Absolute Power", p))
            if "rel_power" in bands and band_names:
                p = os.path.join(tmp_dir, "topomap_rel.png")
                self._save_topomap_grid_png("Relative Power", bands["rel_power"], band_names, p, cmap="jet", symmetric=False, vlim_map=vlims.get("rel_power", {}), cbar_label="relative (0-1)")
                image_paths.append(("Relative Power", p))
            if "ratios" in bands and bands["ratios"]:
                p = os.path.join(tmp_dir, "topomap_ratio.png")
                ratio_names = list(bands["ratios"].keys())
                self._save_topomap_grid_png("Power Ratio", bands["ratios"], ratio_names, p, cmap="jet", symmetric=False, vlim_map=vlims.get("ratios", {}), cbar_label="ratio")
                image_paths.append(("Power Ratio", p))
            if "asym_by_band" in bands and bands["asym_by_band"]:
                p = os.path.join(tmp_dir, "topomap_asym.png")
                asym_fig = None
                try:
                    asym_fig = getattr(self, "_report_fig_cache", {}).get("Asimetri Amplitudo", None)
                except Exception:
                    asym_fig = None
                if asym_fig is not None:
                    self._save_existing_gui_figure_png(asym_fig, p, dpi=250)
                else:
                    self._save_asymmetry_gui_style_png(bands["abs_power"], band_names, p)
                image_paths.append(("Amplitude Asymmetry", p))
            if "lzc_maps" in bands and bands["lzc_maps"]:
                p = os.path.join(tmp_dir, "topomap_lzc.png")
                lzc_names = list(bands["lzc_maps"].keys())
                self._save_topomap_grid_png("LZC Band Complexity", bands["lzc_maps"], lzc_names, p, cmap="viridis", symmetric=False, vlim_map=vlims.get("lzc_maps", {}), cbar_label="normalized LZC")
                image_paths.append(("LZC Topomap", p))
            if "sasi_values" in bands:
                p = os.path.join(tmp_dir, "topomap_sasi.png")
                self._save_sasi_topomap_png(p)
                image_paths.append(("SASI", p))

            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            try:
                pdfmetrics.registerFont(TTFont("Vera", "Vera.ttf"))
                pdfmetrics.registerFont(TTFont("VeraBd", "VeraBd.ttf"))
                _uv2_b = '<font name="VeraBd">\u00b5V\u00b2</font>'
                _uv2 = '<font name="Vera">\u00b5V\u00b2</font>'
            except Exception:
                _uv2_b = _uv2 = "uV<super>2</super>"

            path_dict = dict(image_paths)
            panel_title_style = ParagraphStyle("PanelTitle", parent=small_style, fontName="Helvetica-Bold",
                                               fontSize=9.5, leading=11, textColor=hexcol(C["accent_dark"]))

            def make_panel(title, key, max_w=12.7 * cm, max_h=4.2 * cm):
                img_path = path_dict.get(key)
                if not img_path:
                    return [Paragraph(title, panel_title_style), Spacer(1, 0.05 * cm), Paragraph("Not available", small_style)]
                return [Paragraph(title, panel_title_style), Spacer(1, 0.05 * cm),
                        rl_image_keep_aspect(img_path, max_w=max_w, max_h=max_h)]

            clinical_table = Table([
                [make_panel("Absolute Power (" + _uv2_b + ")", "Absolute Power"),
                 make_panel("Relative Power (unitless, 0 to 1)", "Relative Power")],
                [make_panel("Amplitude Asymmetry (L-R)/(L+R), unitless", "Amplitude Asymmetry"),
                 make_panel("Power Ratio (unitless)", "Power Ratio")],
                [make_panel("LZC Complexity (normalized, unitless)", "LZC Topomap", max_h=3.8 * cm),
                 make_panel("SASI - Spectral Asymmetry Index (unitless, -1 to +1)", "SASI", max_h=3.8 * cm)],
            ], colWidths=[13.2 * cm, 13.2 * cm], rowHeights=[4.8 * cm, 4.8 * cm, 4.4 * cm], hAlign="LEFT")
            clinical_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.55, hexcol(C["line"])),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, hexcol(C["line"])),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(clinical_table)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "Units: Absolute Power is band power in " + _uv2 + " (integral of the PSD over the band); "
                "Relative Power, Power Ratio, Asymmetry, LZC and SASI are dimensionless indices.", small_style))
            
        except Exception as e:
            story.append(Paragraph(f"Gagal membuat topomap untuk PDF: {e}", normal_style))

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)

def _deploy_parse_filename_identity_v2(self):
    """Identitas report: S/A/D + SNR dari dua jalur (LIVE / FILE)."""
    import os, re
    ident = _deploy_parse_filename_identity(self)   # fungsi lama tetap dipakai
    stem = os.path.splitext(os.path.basename(str(ident.get("filename") or "")))[0]

    def _clean_num(v):
        try:
            f = float(str(v).strip())
            return str(int(f)) if f.is_integer() else f"{f:g}"
        except Exception:
            return None

    source = getattr(self, "current_source", "FILE")

    if source == "LIVE":
        # JALUR 1: data direkam langsung
        for key, var_name in (("stress", "rec_stress_var"),
                              ("anxiety", "rec_anxiety_var"),
                              ("depression", "rec_depression_var")):
            var = getattr(self, var_name, None)
            ident[key] = _clean_num(var.get()) if var is not None else None
        try:
            snr = self.analysis_results.get("ica", {}).get("avg_snr", None)
            ident["snr_db"] = f"{float(snr):.2f}" if snr is not None else None
        except Exception:
            ident["snr_db"] = None
    else:
        # JALUR 2: data dari file CSV (nilai ada di nama file)
        if not ident.get("stress"):
            m = re.search(r'(?:^|_)S(\d+)_A(\d+)_D(\d+)(?:_|$)', stem, re.IGNORECASE)
            if m:
                ident["stress"], ident["anxiety"], ident["depression"] = m.group(1), m.group(2), m.group(3)
        if not ident.get("gender"):
            m = re.search(r'_D\d+_([FM])(?:_|$)', stem, re.IGNORECASE)
            if m:
                ident["gender"] = m.group(1).upper()
        if not ident.get("snr_db"):
            m = re.search(r'(-?\d+(?:\.\d+)?)\s*dB', stem, re.IGNORECASE)
            if m:
                ident["snr_db"] = m.group(1)
            else:
                # fallback: file mentah tanpa "dB" di nama -> pakai hasil hitung analisis
                try:
                    snr = self.analysis_results.get("ica", {}).get("avg_snr", None)
                    if snr is not None:
                        ident["snr_db"] = f"{float(snr):.2f}"
                except Exception:
                    pass
    return ident

EEGAnalysisGUI._parse_filename_identity = _deploy_parse_filename_identity_v2
EEGAnalysisGUI.start_recording = _deploy_start_recording_with_prep
EEGAnalysisGUI._tick_recording_prep_countdown = _deploy_tick_recording_prep_countdown
EEGAnalysisGUI._begin_recording_after_prep = _deploy_begin_recording_after_prep
EEGAnalysisGUI._get_record_and_edit_length = _deploy_get_record_and_edit_length
EEGAnalysisGUI._save_report_pdf = _deploy_save_report_pdf_clinical

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalysisGUI(root)
    root.mainloop()
