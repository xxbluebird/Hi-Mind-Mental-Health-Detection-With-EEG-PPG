import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch, iirnotch, hilbert, find_peaks
from scipy.linalg import eigh
from scipy.ndimage import convolve1d
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import mne
# ============================================================
# COMPATIBILITY PATCH NUMPY 2.x
# np.trapz lama diganti menjadi np.trapezoid di NumPy baru.
# Patch ini membuat kode lama tetap jalan tanpa mengganti semua trapz.
# ============================================================
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

# ============================================================
# 1. HELPER FUNCTIONS (FILTERS & ZAPLINE)
# ============================================================

def bandpass_filter(x, fs, low=1.0, high=40.0, order=4):
    """Bandpass butterworth, x shape: (n_ch, n_samp)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def bandpass_band(x, fs, low, high, order=4):
    """Bandpass untuk satu band spesifik, x shape: (n_ch, n_samp)"""
    nyq = fs / 2.0
    low  = max(low,  0.5)
    high = min(high, nyq - 0.5)
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def post_notch(x, fs, freq, Q=50):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, x, axis=-1)

def zapline_strong_auto_plus1(data, fs, fline=50, width=1.5):
    n_ch, n_time = data.shape
    harmonics = [fline, fline * 2]
    ref = data.copy()
    for f in harmonics:
        if f < fs / 2:
            q = f / width
            b, a = iirnotch(f, q, fs)
            ref = filtfilt(b, a, ref, axis=-1)
    
    noise_ref = data - ref
    C1 = (noise_ref @ noise_ref.T) / n_time
    C0 = (data @ data.T) / n_time
    C0 += np.eye(n_ch) * (1e-6 * np.trace(C0) / n_ch)
    
    vals, vecs = eigh(C1, C0)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    
    threshold = np.mean(vals) + np.std(vals)
    n_auto = np.sum(vals > threshold)
    nremove = min(n_auto, max(1, n_ch // 2))
    
    W = vecs[:, :nremove]
    S = W.T @ data
    T = (data @ S.T) @ np.linalg.inv(S @ S.T + np.eye(nremove) * 1e-9)
    return data - T @ S, nremove

# ============================================================
# 2. HELPER FUNCTIONS (ICA & MIF)
# ============================================================

def create_virtual_channel(x, N=2):
    b = np.ones(N) / N
    return filtfilt(b, 1, x)

def get_extrema_count(x):
    peaks, _ = signal.find_peaks(x)
    troughs, _ = signal.find_peaks(-x)
    return len(peaks) + len(troughs)

def double_average_filter(x, L):
    if L < 3: return x
    kernel = np.ones(L) / L
    y = convolve1d(x, kernel, mode='reflect')
    y = convolve1d(y, kernel, mode='reflect')
    return y
def run_mif(data, alpha=1.6, max_imfs=10, sift_iter=8):
    n_ch, N = data.shape
    residual = data.copy()
    mimfs = []
    
    for k in range(max_imfs):
        imf = residual.copy()
        
        # 1. PERBAIKAN: Hitung E_max & L (Sesuai Paper: 2L+1)
        E_max = max([get_extrema_count(imf[c]) for c in range(n_ch)])
        if E_max <= 2: break
        
        # Hitung L dasar
        L_val = int(np.floor(alpha * N / (E_max + 1)))
        
        # --- PERBAIKAN: Gamma Protection untuk MIMF 1 ---
        if k == 0:
            L_val = min(L_val, 11) # Menjamin MIMF1 menangkap Gamma/50Hz
        
        # Panjang filter sesuai paper adalah 2L+1
        filter_len = 2 * L_val + 1
        filter_len = max(3, min(filter_len, N // 2))

        # Sifting Loop
        for _ in range(sift_iter):
            imf_old = imf.copy()
            trend = np.zeros_like(imf)
            for c in range(n_ch): 
                trend[c] = double_average_filter(imf[c], filter_len)
            
            imf = imf - trend
            
            # 2. PERBAIKAN: Cek Konvergensi Sifting (Agar stabil)
            norm_diff = np.linalg.norm(imf - imf_old) / (np.linalg.norm(imf_old) + 1e-12)
            if norm_diff < 0.01:
                break

        # 3. PERBAIKAN: Stop Condition Multivariate
        # Harus cek SEMUA channel, bukan cuma channel [0]
        if max([get_extrema_count(imf[c]) for c in range(n_ch)]) <= 2: 
            break
            
        mimfs.append(imf)
        residual = residual - imf
        
        # Stop jika energi sisa sudah sangat kecil
        if np.std(residual) < 0.05 * np.std(data): 
            break
            
    mimfs.append(residual)
    return np.array(mimfs)

# ============================================================
# 3. HELPER FUNCTIONS (RHYTHM & POWER)
# ============================================================

def get_robust_mf(imf, fs):
    analytic = hilbert(imf) 
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.diff(phase) / (2*np.pi) * fs
    inst_f = np.insert(inst_f, 0, inst_f[0])
    mask = amp > 0.1 * np.max(amp)
    return np.median(inst_f[mask]) if np.any(mask) else 0.0

def compute_total_power(x, fs, f_low, f_high):
    f, Pxx = welch(x, fs=fs, nperseg=int(fs*2))
    mask = (f >= f_low) & (f <= f_high)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 1e-10

# ============================================================
# 4. COMPREHENSIVE FEATURE EXTRACTION
# ============================================================

def extract_features_comprehensive(rhythm_signals, band_names, band_ranges, phase_signals, fs):
    eps = 1e-10
    feature_dict = {}
    n_ch = rhythm_signals.shape[1]
    
    # Absolute & Relative Power
    abs_power = {band: [] for band in band_names} 
    rel_power = {band: [] for band in band_names}
    for b_idx, band in enumerate(band_names):
        lo, hi = band_ranges[band]
        if lo is None: 
            abs_power[band] = [0.0] * n_ch
            continue
        abs_power[band] = []
        for ch in range(n_ch):
            p = compute_total_power(rhythm_signals[b_idx, ch], fs, lo, hi)
            abs_power[band].append(p)
            feature_dict[f"Abs_{band}_ch{ch}"] = p

    for ch in range(n_ch):
        total = sum(abs_power[b][ch] for b in abs_power)
        for band in abs_power:
            rel_power.setdefault(band, [])
            val = abs_power[band][ch] / (total + eps)
            rel_power[band].append(val)
            feature_dict[f"Rel_{band}_ch{ch}"] = val

    # Ratios
    theta_key = next((b for b in abs_power if b.lower()=="theta"), None)
    alpha_key = next((b for b in abs_power if b.lower()=="alpha"), None)
    beta_key  = next((b for b in abs_power if b.lower()=="beta"), None)

    for ch in range(n_ch):
        if theta_key and alpha_key:
            feature_dict[f"Ratio_theta_alpha_ch{ch}"] = abs_power[theta_key][ch] / (abs_power[alpha_key][ch] + eps)
        if alpha_key and beta_key:
            feature_dict[f"Ratio_alpha_beta_ch{ch}"] = abs_power[alpha_key][ch] / (abs_power[beta_key][ch] + eps)

    # Asymmetry (Pairs: 0-1, 2-3, 4-5)
    pairs = [(0,1), (2,3), (4,5), (6,7)]

    valid_pairs = [(L, R) for L, R in pairs if L < n_ch and R < n_ch]

    for band in abs_power:
        for L, R in valid_pairs:
            val = (abs_power[band][L] - abs_power[band][R]) / (abs_power[band][L] + abs_power[band][R] + eps)
            feature_dict[f"Asym_{band}_ch{L}_{R}"] = val

    # SASI & Phase Connectivity
    for ch in range(n_ch):
        if beta_key and alpha_key:
            feature_dict[f"SASI_ch{ch}"] = (abs_power[beta_key][ch] - abs_power[alpha_key][ch]) / (abs_power[beta_key][ch] + abs_power[alpha_key][ch] + eps)

    if alpha_key:
        alpha_idx = band_names.index(alpha_key)
        for L, R in valid_pairs:
            dtheta = phase_signals[alpha_idx, L] - phase_signals[alpha_idx, R]
            feature_dict[f"PhaseLag_ch{L}_{R}"] = np.mean(dtheta)
            feature_dict[f"PhaseCoh_ch{L}_{R}"] = np.abs(np.mean(np.exp(1j * dtheta)))

    return feature_dict, pd.DataFrame(list(feature_dict.items()), columns=['Feature Name', 'Value']), abs_power, rel_power

def reconstruct_from_ica(intermediates, k_thresh=1.6, bcg_thresh=0.065, vbr_thresh=15.0):
    """
    Rekonstruksi sinyal EEG bersih dari intermediate ICA yang sudah ada,
    menggunakan threshold kurtosis baru.

    Parameter:
        intermediates : dict dari 'ica_intermediates' hasil process_eeg_data()
        k_thresh      : threshold kurtosis (IC ditolak jika kurt >= k_thresh)

    Return:
        dict berisi 'ica_clean', 'ics_updated', 'bands', 'features'
    """
    S_ica    = intermediates['S_ica']
    scaler   = intermediates['scaler']
    ica_obj  = intermediates['ica_obj']
    n_ch     = intermediates['n_ch']
    L_bank   = intermediates['L_bank']
    fs       = intermediates['fs']

    S_clean_ica = S_ica.copy()
    rejected_indices = []

    # Ambil pulse_smooth dari intermediates jika tersedia
    pulse_smooth = intermediates.get('pulse_smooth', None)

    ica_list_new = []
    S_clean_new  = S_ica.copy()

    for i in range(S_ica.shape[1]):
        sig_ic = S_ica[:, i]
        k = kurtosis(sig_ic)

        f_ic, p_ic = welch(
            sig_ic,
            fs,
            nperseg=min(int(fs * 2), len(sig_ic) // 2)
        )

        # Hitung corr_pulse jika pulse_smooth tersedia
        corr_pulse = 0.0
        if pulse_smooth is not None:
            corr_matrix = np.corrcoef(pulse_smooth, sig_ic)
            corr_pulse = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Hanya kurtosis sebagai penentu reject
        is_bcg = corr_pulse >= bcg_thresh
        # Tambah VBR di loop IC
        vbr = variance_burst_ratio(sig_ic, fs)
        is_emg = vbr > vbr_thresh   # vbr_thresh dikirim sebagai parameter baru

        is_rejected = (k >= k_thresh) or is_bcg or is_emg

        ica_list_new.append({
            'sig'       : sig_ic,
            'kurt'      : k,
            'corr_pulse': corr_pulse,
            'vbr'       : vbr,
            'rejected'  : is_rejected,
            'freqs'     : f_ic,
            'psd'       : p_ic
        })

        if is_rejected:
            S_clean_new[:, i] = 0
            rejected_indices.append(i)

    rejected_indices = sorted(list(set(rejected_indices)))
    # 3. Rekonstruksi sinyal EEG
    # 3. Rekonstruksi sinyal EEG
    # Satukan logika rekonstruksi menggunakan inverse_transform bawaan Scikit-Learn
    # Ini menjamin inversi PCA internal tetap berjalan aman meskipun n_components < jumlah kolom augmented
    X_rec = scaler.inverse_transform(ica_obj.inverse_transform(S_clean_new))
    
    # Hitung jumlah komponen yang dihapus untuk dilempar ke output
    removed = np.any(S_clean_new != S_ica, axis=0).sum()
    
    orig_col_indices = intermediates.get('orig_col_indices',
                                         list(range(0, n_ch * (len(L_bank)+1), len(L_bank)+1)))
    ica_clean = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])
    # ica_clean shape: (n_ch, n_samp)

    # 4. Ekstraksi band + fitur (sama persis dengan process_eeg_data)
    EEG_BANDS = {
        'Delta': (0.5, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta' : (13.0, 30.0),
    }
    band_names  = list(EEG_BANDS.keys())
    n_ch_r      = ica_clean.shape[0]
    n_samp_r    = ica_clean.shape[1]

    rhythm_signals = np.zeros((len(band_names), n_ch_r, n_samp_r))
    band_ranges    = {}
    band_list      = []

    for b_idx, (band, (lo, hi)) in enumerate(EEG_BANDS.items()):
        filtered = bandpass_band(ica_clean, fs, lo, hi)
        rhythm_signals[b_idx] = filtered
        band_ranges[band]     = [lo, hi]
        sig_b  = filtered[0]
        f_b, p_b = welch(sig_b, fs, nperseg=min(int(fs * 2), len(sig_b)))
        band_list.append({'name': band, 'sig': sig_b, 'f': f_b, 'p': p_b,
                          'lo': lo, 'hi': hi})

    phase_signals = np.zeros_like(rhythm_signals)
    for b_idx in range(len(band_names)):
        for ch in range(n_ch_r):
            seg = rhythm_signals[b_idx, ch]
            if np.any(seg):
                phase_signals[b_idx, ch] = np.angle(hilbert(seg))

    feat_dict, feat_table, abs_p, rel_p = extract_features_comprehensive(
        rhythm_signals, band_names, band_ranges, phase_signals, fs
    )
    # =====================================================================
    # PERBAIKAN: Hitung SNR dengan merekonstruksi S_ica tanpa nol (Sinyal Asli)
    # =====================================================================
    # 1. Rekonstruksi seluruh matriks augmented sebelum ada komponen yang di-zero-out
    X_orig_full = scaler.inverse_transform(ica_obj.inverse_transform(S_ica))
    
    # 2. Ekstrak hanya 8 channel EEG fisik asli menggunakan orig_col_indices yang tersimpan
    X_original = np.array([X_orig_full[:, col_idx] for col_idx in orig_col_indices])
    
    # 3. Hitung SNR menggunakan sinyal asli rekonstruksi vs sinyal bersih
    snr_channels, snr_total = calculate_eeg_snr(X_original, ica_clean)
    # =====================================================================
    return {
        'ica_clean'   : ica_clean,
        'ics_updated' : ica_list_new,
        'bands': {
            'signals'  : rhythm_signals,
            'abs_power': abs_p,
            'rel_power': rel_p,
            'powers'   : {b: np.mean(rel_p[b]) if b in rel_p else 0.0 for b in band_names},
            'ranges'   : {b: tuple(v) for b, v in band_ranges.items()},
            'imf_map'  : {b: [] for b in band_names},
            'names'    : band_names
        },
        'features': {
            'table': feat_table,
            'dict' : feat_dict
        },
        'mif': {
            'mimfs': band_list
        },
        'n_rejected': int(removed),
        'rejected_indices': rejected_indices,
        'snr_per_channel': snr_channels,
        'avg_snr': snr_total
    }

def compute_maica_L(x1, fs, min_orig=0.6, max_orig=0.98, max_outputs=0.4):
    """
    Algoritma Figure 2 dari paper MAICA (Rejer & Górski, 2019).
    Mencari vector L (window sizes) secara adaptif berdasarkan korelasi.

    Returns
    -------
    L : list of int   — window sizes yang valid
    """
    from numpy import corrcoef

    def zero_phase_ma(x, N):
        """Zero-phase moving average (filtfilt style manual)."""
        b = np.ones(N) / N
        return filtfilt(b, [1.0], x)

    def pearson_abs(a, b):
        c = corrcoef(a, b)
        return abs(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

    L = []
    previous_input = None

    for N in range(2, len(x1) // 4):  # batas atas wajar
        new_input = zero_phase_ma(x1, N)

        # Stop condition 1: korelasi dengan sinyal asli terlalu rendah
        if pearson_abs(new_input, x1) < min_orig:
            break

        # Skip: terlalu mirip dengan sinyal asli atau previous input
        if pearson_abs(new_input, x1) > max_orig:
            continue
        if previous_input is not None and pearson_abs(new_input, previous_input) > max_orig:
            continue

        previous_input = new_input
        L.append(N)

        # Stop condition 2: jika sudah cukup banyak filter, cek redundansi
        # (simplified: maksimal 10 filter agar tidak terlalu berat komputasi)
        if len(L) >= 2:
            break

    return L if L else [2, 3, 5]  # fallback minimal jika tidak ada yang lolos

def calculate_eeg_snr(raw_signal, clean_signal):
    """
    Menghitung nilai SNR (in dB) untuk setiap channel EEG.
    
    Parameter:
        raw_signal   : numpy array shape (n_channels, n_samples) -> sinyal kotor sebelum ICA
        clean_signal : numpy array shape (n_channels, n_samples) -> sinyal bersih setelah ICA
    
    Return:
        snr_per_channel : list berisi nilai SNR (dB) untuk masing-masing channel (1-8)
        avg_snr         : rata-rata SNR dari seluruh channel
    """
    # Kebisingan/Artefak didapat dari selisih sinyal sebelum dan sesudah ICA
    noise_signal = raw_signal - clean_signal
    
    snr_per_channel = []
    
    for ch in range(raw_signal.shape[0]):
        # Hitung Root Mean Square (RMS) dari sinyal bersih
        rms_signal = np.sqrt(np.mean(clean_signal[ch] ** 2))
        
        # Hitung Root Mean Square (RMS) dari noise yang dibuang
        rms_noise = np.sqrt(np.mean(noise_signal[ch] ** 2))
        
        # Antisipasi pembagian dengan nol jika tidak ada komponen yang di-reject (noise = 0)
        if rms_noise < 1e-12:
            snr_db = 40.0  # Batas atas default (sinyal sangat bersih)
        else:
            # Rumus SNR berbasis Amplitudo (RMS)
            snr_db = 20 * np.log10(rms_signal / rms_noise)
            
        snr_per_channel.append(snr_db)
        
    avg_snr = np.mean(snr_per_channel)
    return snr_per_channel, avg_snr

def variance_burst_ratio(sig, fs, win_sec=0.5, step_frac=0.5):
    """
    Hitung rasio variansi lokal maksimum vs median.
    EMG bursty → rasio tinggi. EEG stasioner → rasio rendah.
    """
    win  = int(win_sec * fs)
    step = max(1, int(win * step_frac))
    if len(sig) < win:
        return 1.0
    local_vars = np.array([
        np.var(sig[i : i + win])
        for i in range(0, len(sig) - win, step)
    ])
    median_var = np.median(local_vars)
    return float(np.max(local_vars) / (median_var + 1e-12))


def kustom_proses_eeg_anda(self):
        # 1. Tahap prapemrosesan awal / Zapline
        data_filtered = lakukan_zapline(self.session_data)
        
        # 2. Cek apakah user memilih untuk Skip ICA atau tidak
        if self.skip_ica_var.get():
            print("ICA dilewati sesuai permintaan user.")
            # Jika skip ICA, data hasil rekonstruksi langsung disamakan dengan data filter awal
            eeg_clean = data_filtered 
        else:
            print("Menjalankan pipeline ICA...")
            # Jalankan proses ICA normal Anda di sini
            # contoh: ics = jalankan_ica(data_filtered)
            #         eeg_clean = reconstruct_from_ica(ics)
            pass

        # 3. Tahap berikutnya: Ekstraksi Band Frekuensi
        # Lanjutkan ekstraksi fitur menggunakan variable 'eeg_clean'
        fitur = lakukan_ekstraksi_band_frekuensi(eeg_clean)
        

def apply_hann_taper(data, fs, taper_sec=1.0):
    """
    Terapkan Hann window taper di tepi data untuk mengurangi edge artifact
    sebelum ICA, tanpa membuang data.

    data : (n_ch, n_samples)
    Returns data yang sudah di-taper, shape sama.
    """
    n_samp = data.shape[1]
    taper_len = int(taper_sec * fs)
    taper_len = min(taper_len, n_samp // 4)

    window = np.ones(n_samp)
    # Ramp naik di awal
    ramp_up = np.hanning(taper_len * 2)[:taper_len]
    # Ramp turun di akhir
    ramp_down = np.hanning(taper_len * 2)[taper_len:]

    window[:taper_len] = ramp_up
    window[-taper_len:] = ramp_down

    return data * window[np.newaxis, :]  # broadcast ke semua channel

# ============================================================
# 5. FUNGSI UTAMA: process_eeg_data
# ============================================================
def process_eeg_data(
    filename,
    fs=250,
    skip_ica=False,
    skip_zapline=False,
    k_thresh=1.6,
    bcg_thresh=0.055,
    vbr_thresh=15.0,
    n_eeg_channels=8,
    channel_names=None,
    maica_n_components=None
):
    # 1. LOAD DATA
    data = pd.read_csv(filename, sep='	', skiprows=2)

    print("\n" + "=" * 80)
    print("[PROCESSOR AUDIT] LOAD INPUT")
    print("=" * 80)
    print("File              :", filename)
    print("Requested EEG ch  :", n_eeg_channels)
    print("DATA SHAPE awal   :", data.shape)
    print("COLUMNS awal      :", data.columns.tolist())

    n_eeg_channels = int(n_eeg_channels)
    if n_eeg_channels not in (4, 8):
        raise ValueError("n_eeg_channels harus 4 atau 8")

    if channel_names is None:
        if n_eeg_channels == 4:
            ch_names_orig = ['Fp1', 'Fp2', 'T3', 'T4']
        else:
            if channel_names is not None:
                ch_names_orig = list(channel_names)
            elif n_eeg_channels == 4:
                ch_names_orig = ['Fp1', 'Fp2', 'T3', 'T4']
            else:
                ch_names_orig = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4']
    else:
        ch_names_orig = list(channel_names)
        if len(ch_names_orig) != n_eeg_channels:
            raise ValueError(
                f"Panjang channel_names ({len(ch_names_orig)}) tidak sama dengan n_eeg_channels ({n_eeg_channels})."
            )

    # Jangan drop row agresif. Satu NaN di PPG/channel lain bisa menghapus seluruh baris.
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.interpolate(limit_direction='both').fillna(0.0).reset_index(drop=True)

    expected_cols = [f'EXG Channel {i}' for i in range(n_eeg_channels)]
    if all(c in data.columns for c in expected_cols):
        eeg_cols_used = expected_cols
    else:
        if data.shape[1] < n_eeg_channels:
            raise ValueError(
                f"[PROCESSOR ERROR] Input hanya punya {data.shape[1]} kolom, "
                f"tidak cukup untuk mode {n_eeg_channels}CH. Kolom: {list(data.columns)}"
            )
        eeg_cols_used = list(data.columns[:n_eeg_channels])
        print("[PROCESSOR WARN] EXG Channel 0..N tidak lengkap. Fallback memakai kolom pertama.")

    raw_eeg = data[eeg_cols_used].values.T.astype(np.float64)

    if 'Digital' in data.columns:
        ppg_digital = data['Digital'].values.astype(float)
    elif 'PPG' in data.columns:
        ppg_digital = data['PPG'].values.astype(float)
    elif data.shape[1] > n_eeg_channels:
        ppg_digital = data.iloc[:, n_eeg_channels].values.astype(float)
    else:
        ppg_digital = np.zeros(data.shape[0])

    print("DATA SHAPE bersih:", data.shape)
    print("EEG cols used    :", eeg_cols_used)
    print("Channel names    :", ch_names_orig)
    print("raw_eeg shape    :", raw_eeg.shape)
    print("ppg shape        :", ppg_digital.shape)
    print("sample pertama ch0:", raw_eeg[0, 0])
    print("sample terakhir ch0:", raw_eeg[0, -1])
    print("=" * 80 + "\n")

    if raw_eeg.shape[0] != n_eeg_channels:
        raise RuntimeError(
            f"[PROCESSOR ERROR] raw_eeg harus {n_eeg_channels} channel, tetapi terbaca {raw_eeg.shape[0]}."
        )

    
    # 2. PRE-PROCESSING (ZAPLINE & FILTERS)
    # Bandpass 2-80 Hz dan pembersihan derau listrik 50 Hz
    subset_bpf = bandpass_filter(raw_eeg, fs)

    if skip_zapline:
        print("[PROCESSOR] Skip Zapline aktif — memakai sinyal bandpass sebagai input tahap berikutnya.")
        subset_zap = subset_bpf.copy()
        n_zap = 0
    else:
        print("[PROCESSOR] Menjalankan Zapline...")
        subset_zap, n_zap = zapline_strong_auto_plus1(subset_bpf, fs)

    # Tetap jalankan post-notch 50 Hz
    subset_clean = post_notch(subset_zap, fs, 50, Q=50)
    # subset_clean = subset_zap
    # Menyiapkan vektor waktu untuk sinkronisasi grafik
    t_vec = np.arange(subset_clean.shape[1]) / fs
# ============================================================
    # PPG PROCESSING — Metode Midpoint + Debounce (dari notebook)
    # ============================================================
    ppg_raw = ppg_digital

    unique_vals = np.unique(ppg_raw)
    if len(unique_vals) <= 10:
        threshold_ppg = (unique_vals.max() + unique_vals.min()) / 2
        print(f"[PPG] Mode: DIGITAL diskrit | Threshold: {threshold_ppg}")
    else:
        threshold_ppg = np.median(ppg_raw)
        print(f"[PPG] Mode: ANALOG kontinu | Threshold median: {threshold_ppg:.2f}")

    ppg_binary = (ppg_raw > threshold_ppg).astype(int)
    diff_binary = np.diff(ppg_binary)
    rising_edges  = np.where(diff_binary > 0)[0] + 1
    falling_edges = np.where(diff_binary < 0)[0] + 1

    raw_midpoints = []
    for r in rising_edges:
        f_candidates = falling_edges[falling_edges > r]
        if len(f_candidates) == 0:
            continue
        f = f_candidates[0]
        pulse_width = f - r
        if pulse_width < int(0.020 * fs) or pulse_width > int(0.600 * fs):
            continue
        raw_midpoints.append((r + f) // 2)
    raw_midpoints = np.array(raw_midpoints)

    min_rr_samples = int(0.35 * fs)
    if len(raw_midpoints) == 0:
        print("[PPG] ⚠️ Tidak ada midpoint terdeteksi.")
        ppg_peaks = np.array([])
    else:
        ppg_peaks = [raw_midpoints[0]]
        for pk in raw_midpoints[1:]:
            if pk - ppg_peaks[-1] >= min_rr_samples:
                ppg_peaks.append(pk)
        ppg_peaks = np.array(ppg_peaks)

    # Alias 'beats' dipertahankan agar kompatibel dengan kode BCG di bawah
    beats = ppg_peaks

    print(f"[PPG] Rising edges    : {len(rising_edges)}")
    print(f"[PPG] Setelah validasi: {len(raw_midpoints)}")
    print(f"[PPG] Setelah debounce: {len(ppg_peaks)}")

    if len(ppg_peaks) > 2:
        rr_samples = np.diff(ppg_peaks)
        rr_ms      = rr_samples / fs * 1000
        bpm_inst   = 60000 / rr_ms
        bpm_val    = np.mean(bpm_inst)

        bpm_data = {
            "mean_bpm"  : bpm_val,
            "sdnn"      : np.std(rr_ms),
            "rmssd"     : np.sqrt(np.mean(np.diff(rr_ms)**2)),
            "status"    : "Success",
            "ppg_signal": ppg_raw,
            "ppg_time"  : np.arange(len(ppg_raw)) / fs,
            "ppg_peaks" : ppg_peaks,
            "rr_times"  : ppg_peaks[1:] / fs,
            "rr_values" : rr_ms,
            "bpm_times" : ppg_peaks[1:] / fs,
            "bpm_values": bpm_inst,
            "avg"       : bpm_val,
        }
    else:
        bpm_data = {
            "mean_bpm"  : 0,
            "status"    : "No Pulse Detected",
            "ppg_signal": ppg_raw,
            "ppg_time"  : np.arange(len(ppg_raw)) / fs,
            "ppg_peaks" : np.array([]),
            "times": [], "values": [], "avg": 0
        }

# ====================================================================
    # 4. PARTIAL MAICA (Dengan Opsi Skip ICA)
    # ====================================================================
    X_raw_data = subset_clean.T          # Langsung gunakan data clean tanpa taper
    n_samples, n_ch = X_raw_data.shape

    # Inisialisasi variabel penampung default untuk mencegah error saat skip_ica=True
    ica_list = []
    rejected_indices = []
    S_ica = np.zeros((n_samples, 1))     # dummy array untuk metadata ica_intermediates
    scaler = None
    ica = None
    L_bank = []
    orig_col_indices = list(range(n_ch)) # Indeks default channel asli
    pulse_smooth = None

    if skip_ica:
        print("[PROCESSOR] Skip ICA aktif. Melewati proses dekomposisi MAICA...")
        # Sinyal bersih langsung disamakan dengan sinyal hasil filter sebelumnya (Zapline + Post Notch)
        subset_ica_clean = subset_clean.copy()
        
        # Mengisi ica_list dengan data dummy per channel agar GUI/Plotting ICA tidak crash
        for i in range(n_ch):
            sig_dummy = subset_clean[i, :]
            f_ic, p_ic = welch(sig_dummy, fs, nperseg=min(int(fs * 2), len(sig_dummy) // 2))
            ica_list.append({
                'sig': sig_dummy, 
                'kurt': 0.0, 
                'freqs': f_ic, 
                'psd': p_ic
            })
    else:
        print("[PROCESSOR] Menjalankan pipeline MAICA normal...")
        # Hitung L adaptif dari channel pertama (sesuai paper Fig. 2)
        L_bank = compute_maica_L(X_raw_data[:, 0], fs,
                                 min_orig=0.6, max_orig=0.98, max_outputs=0.4)
        print(f"[MAICA] L_bank adaptif: {L_bank} ({len(L_bank)} filter)")

        # Cek korelasi antar virtual channel — buang jika ada pasangan > max_orig
        def zero_phase_ma(x, N):
            b = np.ones(N) / N
            return filtfilt(b, [1.0], x)

        X_aug_list = []
        orig_col_indices = []   # simpan indeks kolom X_aug yang merupakan channel asli

        for i in range(n_ch):
            orig = X_raw_data[:, i]
            orig_col_indices.append(len(X_aug_list))   # posisi kolom channel asli ini
            X_aug_list.append(orig)
            for N in L_bank:
                virt = zero_phase_ma(orig, N)
                corr_ok = True
                for existing in X_aug_list:
                    c = abs(np.corrcoef(virt, existing)[0, 1])
                    if c > 0.98:
                        corr_ok = False
                        break
                if corr_ok:
                    X_aug_list.append(virt)

        X_aug = np.column_stack(X_aug_list)
        print(f"[MAICA] Augmented matrix: {X_aug.shape}")
        print(f"[MAICA] Kolom channel asli: {orig_col_indices}")

        # Normalisasi dan ICA
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_aug)

        if maica_n_components is None:
            maica_n_components = 8 if n_ch <= 4 else 16

        n_components = min(int(maica_n_components), X_norm.shape[1])

        print(
            f"[MAICA] Mode {n_ch}CH -> target IC={maica_n_components}, "
            f"dipakai n_components={n_components}, X_aug={X_aug.shape}"
        )

        ica = FastICA(
            n_components=n_components,
            algorithm='parallel',
            whiten='unit-variance',
            whiten_solver='eigh',
            max_iter=3000,
            tol=1e-4,
            random_state=42
        )
        S_ica = ica.fit_transform(X_norm)

        # === PREPARASI PULSE REFERENCE UNTUK DETEKSI BCG ===
        pulse_smooth = None
        if len(beats) > 2:
            pulse_ref = np.zeros(n_samples)
            for bt in beats:
                if bt < n_samples:
                    pulse_ref[bt] = 1.0
            window_len = int(fs * 0.3)
            pulse_smooth = np.convolve(pulse_ref, np.ones(window_len) / window_len, 'same')

        # Rejection IC berdasarkan kurtosis + BCG
        S_clean_ica = S_ica.copy()
        for i in range(S_ica.shape[1]):
            sig_ic = S_ica[:, i]
            k = kurtosis(sig_ic)
            f_ic, p_ic = welch(sig_ic, fs, nperseg=min(int(fs * 2), len(sig_ic) // 2))

            # === LOGIKA DETEKSI BCG ===
            # === LOGIKA DETEKSI BCG ===
            corr_pulse = 0.0

            if pulse_smooth is not None:
                corr_matrix = np.corrcoef(pulse_smooth, sig_ic)
                corr_pulse = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

            # Pakai threshold dari parameter, bukan hardcode
            is_bcg = corr_pulse >= bcg_thresh

            # === LOGIKA DETEKSI EMG / BURST ===
            vbr = variance_burst_ratio(sig_ic, fs)
            is_emg = vbr >= vbr_thresh

            # === LOGIKA REJECT FINAL ===
            is_rejected = (k >= k_thresh) or is_bcg or is_emg

            ica_list.append({
                'sig'       : sig_ic,
                'kurt'      : k,
                'corr_pulse': corr_pulse,
                'vbr'       : vbr,
                'rejected'  : is_rejected,
                'freqs'     : f_ic,
                'psd'       : p_ic
            })

            print(
                f"[INITIAL IC {i+1:02d}] "
                f"K={k:.3f} >= {k_thresh:.3f}: {k >= k_thresh} | "
                f"BCG={corr_pulse:.5f} >= {bcg_thresh:.5f}: {is_bcg} | "
                f"VBR={vbr:.3f} >= {vbr_thresh:.3f}: {is_emg} | "
                f"REJECT={is_rejected}"
            )

            if is_rejected:
                S_clean_ica[:, i] = 0
                rejected_indices.append(i)

                reasons = []
                if k >= k_thresh:
                    reasons.append("Kurtosis")
                if is_bcg:
                    reasons.append("BCG")
                if is_emg:
                    reasons.append("VBR/EMG")

                print(
                    f"IC {i+1} REMOVED ({' + '.join(reasons)}) | "
                    f"kurt={k:.3f} | BCG={corr_pulse:.5f} | VBR={vbr:.3f}"
                )
        # Rekonstruksi — ambil kolom channel asli berdasarkan indeks yang sudah direkam
        X_rec = scaler.inverse_transform(ica.inverse_transform(S_clean_ica))
        subset_ica_clean = np.array([X_rec[:, col_idx] for col_idx in orig_col_indices])
            
        
    EEG_BANDS = {
        'Delta': (0.5, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta' : (13.0, 30.0),
    }
    band_names  = list(EEG_BANDS.keys())
    n_ch_ica    = subset_ica_clean.shape[0]
    n_samp_ica  = subset_ica_clean.shape[1]

    rhythm_signals = np.zeros((len(band_names), n_ch_ica, n_samp_ica))
    band_ranges    = {}
    imf_map_dict   = {b: [] for b in band_names}   # kosong, tidak ada MIMF lagi
    band_list      = []   # untuk tab plot (pengganti mif_list)

    for b_idx, (band, (lo, hi)) in enumerate(EEG_BANDS.items()):
        filtered = bandpass_band(subset_ica_clean, fs, lo, hi)   # (n_ch, n_samp)
        rhythm_signals[b_idx] = filtered
        band_ranges[band]     = [lo, hi]

        # Simpan sinyal channel-0 + PSD untuk tab plot
        sig_b       = filtered[0]
        f_b, p_b    = welch(sig_b, fs, nperseg=min(int(fs * 2), len(sig_b)))
        band_list.append({'name': band, 'sig': sig_b, 'f': f_b, 'p': p_b,
                          'lo': lo, 'hi': hi})
        
            

    # Ekstraksi Fase menggunakan Hilbert untuk fitur konektivitas
# Ekstraksi Fase menggunakan Hilbert (DENGAN PROTEKSI SINYAL KOSONG)
    phase_signals = np.zeros_like(rhythm_signals)
    for b_idx in range(len(band_names)):
        for ch in range(n_ch):
            sig_segment = rhythm_signals[b_idx, ch]
            if np.any(sig_segment): # Cek jika sinyal tidak kosong (all zeros)
                phase_signals[b_idx, ch] = np.angle(hilbert(sig_segment))
            else:
                phase_signals[b_idx, ch] = 0.0
                
    # 7. FEATURE EXTRACTION
    feat_dict, feat_table, abs_p, rel_p = extract_features_comprehensive(
        rhythm_signals, band_names, band_ranges, phase_signals, fs
    )

    # 8. PREPARE MNE INFO
    # MNE standard_1020 lebih aman memakai T7/T8 daripada T3/T4,
    # tetapi nama fitur/GUI tetap boleh T3/T4 di layer lain.
    ch_names_mne = [
        'T7' if ch == 'T3' else
        'T8' if ch == 'T4' else ch
        for ch in ch_names_orig
    ]
    info_mne = mne.create_info(ch_names=ch_names_mne, sfreq=fs, ch_types='eeg')
    try:
        info_mne.set_montage(
            mne.channels.make_standard_montage('standard_1020'),
            match_case=False,
            on_missing='ignore'
        )
    except TypeError:
        info_mne.set_montage(mne.channels.make_standard_montage('standard_1020'))

    # Final Return dengan struktur yang tepat untuk GUI.py
    return {
        'zapline': {
            't': t_vec,
            'raw': subset_bpf,
            'clean': subset_clean,
            'n_removed': n_zap,
            'skipped': skip_zapline
        },
        'ica': {
            'ics': ica_list,
            'data': subset_ica_clean,
            'before': subset_clean,
            'after': subset_ica_clean,

            # TAMBAHAN
            'rejected_indices': rejected_indices,
            'n_rejected': len(rejected_indices),
        },
        # === TAMBAHAN: simpan intermediate agar bisa rekonstruksi ulang ===
        'ica_intermediates': {
            'S_ica'            : S_ica,
            'scaler'           : scaler,
            'ica_obj'          : ica,
            'n_ch'             : n_ch,
            'L_bank'           : L_bank,
            'orig_col_indices' : orig_col_indices,
            'fs'               : fs,
            'rejected_indices' : rejected_indices,
            'pulse_smooth'     : pulse_smooth,   # untuk BCG detection saat recompute
        },
        # ====================================================================
        'mif': {
            'mimfs': band_list
        },
        'bpm_data': bpm_data,
        'bands': {
            'signals': rhythm_signals,
            'abs_power': abs_p,
            'rel_power': rel_p,
            'powers': {b: np.mean(rel_p[b]) if b in rel_p else 0.0 for b in band_names},
            'ranges': {b: tuple(v) for b, v in band_ranges.items()},
            'imf_map': imf_map_dict,
            'names': band_names
        },
        'features': {
            'table': feat_table,
            'dict': feat_dict
        },
        'topo_info': info_mne
    }
