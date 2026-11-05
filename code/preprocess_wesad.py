"""Preprocess WESAD chest ECG/EDA/Resp into 30 s windows (stride 5 s).

Usage (cmd):
    python preprocess_wesad.py "C:\\Users\\DELL\\Downloads\\Q1_Scopus\\WESAD"

Output (same folder as this script):
    wesad_windows.npz   X (N,3,3500) float32 [ECG, EDA, Resp], y (0/1/2), sid, t0
    preprocess_summary.txt   <- send this file back

Pipeline (documented for the paper):
  1. Zero-phase Butterworth filtering on the FULL continuous recording
     (ECG band-pass 3-45 Hz, EDA low-pass 5 Hz, Resp band-pass 0.1-0.35 Hz),
     i.e. the same pass-bands as the BioSPPy defaults used before.
  2. Per-subject z-score (label-free statistics of that subject's recording).
  3. Windows of 30 s, stride 5 s, only inside runs with a single label in {1,2,3}.
     Labels are mapped 1->0 Baseline, 2->1 Stress, 3->2 Amusement.
  4. Anti-aliased downsampling 21000 -> 3500 samples (700 Hz -> ~116.7 Hz).
"""
import os
import pickle
import sys
import warnings

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

warnings.filterwarnings("ignore")

FS = 700
WIN = 30 * FS
STRIDE = 5 * FS
DOWN = 6  # 21000 / 6 = 3500
LABEL_MAP = {1: 0, 2: 1, 3: 2}

root = sys.argv[1] if len(sys.argv) > 1 else "WESAD"
here = os.path.dirname(os.path.abspath(__file__))
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def bandpass(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def lowpass(x, hi, order=4):
    sos = butter(order, hi, btype="lowpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


X_all, y_all, sid_all, t0_all = [], [], [], []

sids = [i for i in range(2, 18) if i != 12]
for s in sids:
    path = os.path.join(root, f"S{s}", f"S{s}.pkl")
    if not os.path.exists(path):
        log(f"S{s}: NOT FOUND at {path}")
        continue
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    ch = d["signal"]["chest"]
    ecg = np.asarray(ch["ECG"], dtype=np.float64).ravel()
    eda = np.asarray(ch["EDA"], dtype=np.float64).ravel()
    resp = np.asarray(ch["Resp"], dtype=np.float64).ravel()
    lab = np.asarray(d["label"]).ravel().astype(int)

    ecg = zscore(bandpass(ecg, 3.0, 45.0))
    eda = zscore(lowpass(eda, 5.0))
    resp = zscore(bandpass(resp, 0.1, 0.35))

    # contiguous runs of constant label
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]

    counts = {0: 0, 1: 0, 2: 0}
    for a, b in zip(starts, ends):
        l = lab[a]
        if l not in LABEL_MAP:
            continue
        pos = a
        while pos + WIN <= b:
            seg = np.stack([ecg[pos:pos + WIN], eda[pos:pos + WIN], resp[pos:pos + WIN]])
            seg = resample_poly(seg, 1, DOWN, axis=1).astype(np.float32)
            X_all.append(seg)
            y_all.append(LABEL_MAP[l])
            sid_all.append(s)
            t0_all.append(pos / FS)
            counts[LABEL_MAP[l]] += 1
            pos += STRIDE
    log(f"S{s}: windows baseline={counts[0]} stress={counts[1]} amusement={counts[2]}")

X = np.stack(X_all)
y = np.array(y_all, dtype=np.int64)
sid = np.array(sid_all, dtype=np.int64)
t0 = np.array(t0_all, dtype=np.float32)

log("")
log(f"X shape {X.shape} dtype {X.dtype} | size {X.nbytes/1e6:.0f} MB")
log(f"class counts (0/1/2): {np.bincount(y).tolist()}")
log(f"any NaN: {bool(np.isnan(X).any())} | X mean/std: {X.mean():.3f}/{X.std():.3f}")

out = os.path.join(here, "wesad_windows.npz")
np.savez_compressed(out, X=X, y=y, sid=sid, t0=t0)
log(f"Saved {out} ({os.path.getsize(out)/1e6:.0f} MB)")

with open(os.path.join(here, "preprocess_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
