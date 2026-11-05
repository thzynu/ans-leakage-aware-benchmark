"""External validation on CASE: (A) LOSO within CASE, (B) zero-shot transfer from WESAD.

Usage: python case_validation.py --data DATA_DIR --out OUT_DIR [--smoke]
Results are saved after every fold; re-running resumes from the saved file.
"""
import argparse, os, pickle, time
import numpy as np, torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import common
from common import CNNOnly, RNNOnly, Hybrid, State, set_data, seed_all, split_ids, train, predict, run_fold, dev

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
SMOKE = a.smoke
dw = np.load(os.path.join(a.data, "wesad_windows.npz")); dc = np.load(os.path.join(a.data, "case_windows.npz"))
X = np.concatenate([dw["X"], dc["X"]]); y = np.concatenate([dw["y"], dc["y"]])
sid = np.concatenate([dw["sid"], dc["sid"] + 100])          # CASE subjects get IDs 101..130
SIDS_W = [int(s) for s in np.unique(dw["sid"])]
SIDS_C = [int(s) for s in np.unique(dc["sid"] + 100)]
set_data(X, y, sid, SIDS_C)                                  # run_fold iterates over CASE subjects
MODELS = {"hybrid": None, "cnn_only": CNNOnly, "rnn_only": RNNOnly}
RESC = os.path.join(a.out, "smoke.pkl" if SMOKE else "results_case.pkl")
resc = pickle.load(open(RESC, "rb")) if os.path.exists(RESC) else {}
EP = 1 if SMOKE else 30
subjects = SIDS_C[:1] if SMOKE else SIDS_C
seeds = (0,) if SMOKE else (0, 1, 2)
T0 = time.time()

for nm, cls in MODELS.items():                               # A) LOSO within CASE
    for sd in seeds:
        for s in subjects:
            key = (nm, sd, s)
            if key in resc: continue
            r = run_fold(s, seed=sd, epochs=EP, verbose=False, model_fn=cls)
            resc[key] = {k: r[k] for k in ("probs", "y", "epochs", "sec")}
            pickle.dump(resc, open(RESC, "wb"))
            print(f"{nm} seed{sd} S{s}: acc {r['acc']:.3f} F1 {r['f1']:.3f} AUC {r['auc']:.3f} "
                  f"ep{r['epochs']} {r['sec']:.0f}s", flush=True)
            del r; torch.cuda.empty_cache()


def run_transfer(cls, seed, epochs):
    """Train on WESAD (validation subjects S5 and S13), apply unchanged to every CASE subject."""
    seed_all(seed)
    val_s = [SIDS_W[3], SIDS_W[10]]
    tr_s = [s for s in SIDS_W if s not in val_s]
    idx = lambda ss: torch.from_numpy(np.where(np.isin(State.sid, ss))[0]).to(dev)
    tr, va = idx(tr_s), idx(val_s)
    model = (cls() if cls is not None else Hybrid((0, 1, 2), "mha")).to(dev)
    n_ep = train(model, tr, va, epochs)
    return {s: dict(probs=predict(model, idx([s])), y=State.y[idx([s])].cpu().numpy())
            for s in subjects}, n_ep


for nm, cls in MODELS.items():                               # B) WESAD -> CASE
    for sd in seeds:
        if ("tr_" + nm, sd, subjects[0]) in resc: continue
        out, ne = run_transfer(cls, sd, EP)
        for s, v in out.items(): resc[("tr_" + nm, sd, s)] = v
        pickle.dump(resc, open(RESC, "wb"))
        print(f"transfer {nm} seed{sd}: {ne} epochs ({(time.time() - T0) / 60:.0f} min)", flush=True)


def table(names):
    for n in names:
        rows = []
        for (nm, sd, s), v in resc.items():
            if nm != n: continue
            p, yt = v["probs"], v["y"]
            rows.append(dict(seed=sd, sid=s, acc=accuracy_score(yt, p.argmax(1)),
                             f1=f1_score(yt, p.argmax(1), average="macro"),
                             auc=roc_auc_score(yt, p, multi_class="ovr", average="macro")))
        df = pd.DataFrame(rows); ps = df.groupby("seed")[["acc", "f1", "auc"]].mean()
        ss = df.groupby("sid")[["acc", "f1", "auc"]].mean()
        print(f"{n:14s} n={len(df):3d} | acc {ss.acc.mean():.3f}+-{ss.acc.std():.3f} | f1 {ss.f1.mean():.3f}+-{ss.f1.std():.3f} "
              f"| auc {ss.auc.mean():.3f}+-{ss.auc.std():.3f} | sd between seeds (f1) {ps.f1.std():.3f}")


print("\n== LOSO within CASE =="); table(list(MODELS))
print("\n== Train on WESAD, test on CASE (zero-shot) =="); table(["tr_" + n for n in MODELS])
