"""WESAD leave-one-subject-out benchmark (3 seeds): Hybrid, modality subsets, deep baselines,
attention/recurrence ablations and fusion rules.

Usage:  python wesad_benchmark.py --data DATA_DIR --out OUT_DIR --group {main,modality,baselines,ablation,fusion,all} [--smoke]
Output: results.pkl (all groups except fusion) and results_fusion.pkl; keys are (model, seed, subject).
Results are saved after every fold; re-running resumes. The SVM / Random Forest baselines are in ml_baseline.py.

Note: the cell that produced the first runs of 'hybrid' and the modality subsets was not available when this file was
assembled; the loop below applies the same run_fold protocol (same seeds, split and training settings) to them.
"""
import argparse, os, pickle, time
import numpy as np, torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import common
from common import Hybrid, CNNOnly, RNNOnly, State, set_data, run_fold
from models_extra import TransformerMM, CrossModal, HybridFusion, CFG0

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--group", default="all", choices=["main", "modality", "baselines", "ablation", "fusion", "all"])
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); SMOKE = a.smoke
d = np.load(os.path.join(a.data, "wesad_windows.npz"))
set_data(d["X"], d["y"], d["sid"], np.unique(d["sid"]))
SIDS = State.SIDS
SEEDS = (0,) if SMOKE else (0, 1, 2)
EPOCHS = 1 if SMOKE else 30
subjects = SIDS[:1] if SMOKE else SIDS

MODALITY = {"eda": (1,), "ecg+eda": (0, 1), "eda+resp": (1, 2), "resp": (2,), "ecg+resp": (0, 2), "ecg": (0,)}
BASELINES = {"cnn_only": CNNOnly, "rnn_only": RNNOnly, "transformer": TransformerMM, "crossmodal": CrossModal}
ABLATION = [("hyb_noattn", CFG0, "none"),
            ("hyb_uniform", CFG0, "uniform"),
            ("hyb_allbi", {0: (7, True), 1: (5, True), 2: (7, True)}, "mha"),
            ("hyb_alluni", {0: (7, False), 1: (5, False), 2: (7, False)}, "mha")]

RES = os.path.join(a.out, "smoke_main.pkl" if SMOKE else "results.pkl")
FUS = os.path.join(a.out, "smoke_fusion.pkl" if SMOKE else "results_fusion.pkl")
results = pickle.load(open(RES, "rb")) if os.path.exists(RES) else {}
resf = pickle.load(open(FUS, "rb")) if os.path.exists(FUS) else {}


def loso(store, path, name, **kw):
    for sd in SEEDS:
        for s in subjects:
            key = (name, sd, s)
            if key in store: continue
            r = run_fold(s, seed=sd, epochs=EPOCHS, verbose=False, **kw)
            store[key] = {k: r[k] for k in ("probs", "y", "epochs", "sec")}
            pickle.dump(store, open(path, "wb"))
            print(f"{name} seed{sd} S{s}: acc {r['acc']:.3f} F1 {r['f1']:.3f} AUC {r['auc']:.3f} "
                  f"ep{r['epochs']} {r['sec']:.0f}s", flush=True)
            del r; torch.cuda.empty_cache()


def loso_cfg(name, cfg, attn):
    Hybrid.CFG = cfg                      # branch configuration is a class attribute of Hybrid
    try:
        loso(results, RES, name, attn=attn)
    finally:
        Hybrid.CFG = CFG0


g = a.group
if g in ("main", "all"):
    loso(results, RES, "hybrid", mods=(0, 1, 2)); print("=== hybrid done", flush=True)
if g in ("modality", "all"):
    for nm, mods in MODALITY.items():
        loso(results, RES, nm, mods=mods); print(f"=== {nm} done", flush=True)
if g in ("baselines", "all"):
    for nm, cls in BASELINES.items():
        loso(results, RES, nm, model_fn=cls); print(f"=== {nm} done", flush=True)
if g in ("ablation", "all"):
    for nm, cfg, attn in ABLATION:
        loso_cfg(nm, cfg, attn); print(f"=== {nm} done", flush=True)
if g in ("fusion", "all"):
    print("params:", {k: sum(p.numel() for p in HybridFusion(k).parameters()) for k in ("sum", "gated", "xattn")})
    for kind in ("sum", "gated", "xattn"):
        loso(resf, FUS, f"fus_{kind}", model_fn=lambda k=kind: HybridFusion(k)); print(f"=== fus_{kind} done", flush=True)


def metrics_df(store, name):
    rows = []
    for (n, sd, s), v in store.items():
        if n != name: continue
        p, yt = v["probs"], v["y"]
        rows.append(dict(seed=sd, sid=s, acc=accuracy_score(yt, p.argmax(1)), f1=f1_score(yt, p.argmax(1), average="macro"),
                         auc=roc_auc_score(yt, p, multi_class="ovr", average="macro")))
    return pd.DataFrame(rows)


print("\nMean over subjects, averaged over seeds; (std across seeds)")
for store in (results, resf):
    for n in sorted({k[0] for k in store}):
        df = metrics_df(store, n); ps = df.groupby("seed")[["acc", "f1", "auc"]].mean()
        print(f"{n:12s} n={len(df):2d} | acc {ps.acc.mean():.3f}({ps.acc.std():.3f}) | "
              f"f1 {ps.f1.mean():.3f}({ps.f1.std():.3f}) | auc {ps.auc.mean():.3f}({ps.auc.std():.3f})")
