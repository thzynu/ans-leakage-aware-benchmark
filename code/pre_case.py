import glob, os, re, sys, warnings
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly
warnings.filterwarnings("ignore")

FS = 1000
WIN, STRIDE = 30 * FS, 5 * FS
# video ID -> lop: 0 Baseline (chan + thu gian), 1 Kich hoat cao/valence thap (dang so), 2 Amusement (gay cuoi)
LABEL = {1: 2, 2: 2, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1}

root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(root, "**", "interpolated", "physiological", "sub_*.csv"), recursive=True)
files = sorted(files, key=lambda p: int(re.findall(r"sub_(\d+)", os.path.basename(p))[0]))
print("Found", len(files), "files", flush=True)
lines = []

def log(s):
    print(s, flush=True); lines.append(s)

def bp(x, lo, hi, order=4):
    return sosfiltfilt(butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos"), x)

def lp(x, hi, order=4):
    return sosfiltfilt(butter(order, hi, btype="lowpass", fs=FS, output="sos"), x)

def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)

X_all, y_all, sid_all, vid_all, t0_all = [], [], [], [], []
for p in files:
    s = int(re.findall(r"sub_(\d+)", os.path.basename(p))[0])
    d = pd.read_csv(p, usecols=["ecg", "gsr", "rsp", "video"])
    ecg = z(bp(d.ecg.to_numpy(np.float64), 3.0, 45.0))
    eda = z(lp(d.gsr.to_numpy(np.float64), 5.0))
    rsp = z(bp(d.rsp.to_numpy(np.float64), 0.1, 0.35))
    vid = d.video.to_numpy().astype(int)
    change = np.flatnonzero(np.diff(vid)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(vid)]
    cnt = [0, 0, 0]
    for a, b in zip(starts, ends):
        v = vid[a]
        if v not in LABEL: continue
        pos = a
        while pos + WIN <= b:
            seg = np.stack([ecg[pos:pos + WIN], eda[pos:pos + WIN], rsp[pos:pos + WIN]])
            X_all.append(resample_poly(seg, 7, 60, axis=1).astype(np.float32))   # 30000 -> 3500 (anti-alias)
            y_all.append(LABEL[v]); sid_all.append(s); vid_all.append(v); t0_all.append(pos / FS)
            cnt[LABEL[v]] += 1
            pos += STRIDE
    log(f"sub_{s}: rows={len(d)} windows baseline={cnt[0]} scary={cnt[1]} amusing={cnt[2]} "
        f"| finite={bool(np.isfinite(ecg).all() and np.isfinite(eda).all() and np.isfinite(rsp).all())}")

X = np.stack(X_all); y = np.array(y_all); sid = np.array(sid_all); vid = np.array(vid_all)
log(f"X {X.shape} {X.dtype} | classes (0/1/2): {np.bincount(y).tolist()} | NaN: {bool(np.isnan(X).any())} | mean/std {X.mean():.3f}/{X.std():.3f}")
out = os.path.join(here, "case_windows.npz")
np.savez_compressed(out, X=X, y=y, sid=sid, vid=vid, t0=np.array(t0_all, dtype=np.float32))
log(f"Saved {out} ({os.path.getsize(out)/1e6:.0f} MB)")
open(os.path.join(here, "case_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))