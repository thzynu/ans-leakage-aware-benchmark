import numpy as np, pickle, time
from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

FS = 3500 / 30.0
d = np.load("wesad_windows.npz")
X, y, sid = d["X"], d["y"], d["sid"]
SIDS = [int(s) for s in np.unique(sid)]
t_ax = np.arange(3500) / FS

def feats(w):
    ecg, eda, rsp = w
    pk, _ = find_peaks(ecg, height=2.0, distance=int(0.4 * FS))
    rr = np.diff(pk) / FS
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) >= 3:
        drr = np.diff(rr)
        hrv = [60.0 / rr.mean(), rr.std(), np.sqrt(np.mean(drr ** 2)), np.mean(np.abs(drr) > 0.05)]
    else:
        hrv = [0, 0, 0, 0]
    f = hrv + [len(pk), ecg.std(), np.abs(np.diff(ecg)).mean()]
    ep, _ = find_peaks(eda, prominence=0.05, distance=int(1.0 * FS))
    f += [eda.mean(), eda.std(), eda.min(), eda.max(), eda.max() - eda.min(),
          np.polyfit(t_ax, eda, 1)[0], len(ep), np.abs(np.diff(eda)).mean()]
    rp, _ = find_peaks(rsp, prominence=0.3 * rsp.std(), distance=int(1.5 * FS))
    ri = np.diff(rp) / FS
    f += [rsp.std(), np.abs(np.diff(rsp)).mean(), len(rp) * 2.0,
          ri.mean() if len(ri) else 0, ri.std() if len(ri) > 1 else 0]
    return f

t0 = time.time()
F = np.nan_to_num(np.array([feats(w) for w in X], dtype=np.float64))
print("features:", F.shape, f"({time.time()-t0:.0f}s)", flush=True)

def split(s):
    rem = [k for k in SIDS if k != s]; i = SIDS.index(s)
    val_s = [rem[i % 14], rem[(i + 7) % 14]]
    tr_s = [k for k in rem if k not in val_s]
    return np.isin(sid, tr_s), np.isin(sid, val_s), sid == s

def run(name, make, grid, seed):
    out = {}
    for s in SIDS:
        tr, va, te = split(s)
        sc = StandardScaler().fit(F[tr])
        Xtr, Xva, Xte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
        best = None
        for p in grid:                                   # chon sieu tham so tren 2 subject validation
            m = make(p, seed, False).fit(Xtr, y[tr])
            f1v = f1_score(y[va], m.predict(Xva), average="macro")
            if best is None or f1v > best[0]: best = (f1v, p)
        m = make(best[1], seed, True).fit(Xtr, y[tr])
        pr = m.predict_proba(Xte)
        out[(name, seed, s)] = dict(probs=pr, y=y[te], params=best[1])
        yt = y[te]
        print(f"{name} seed{seed} S{s}: acc {accuracy_score(yt, pr.argmax(1)):.3f} "
              f"F1 {f1_score(yt, pr.argmax(1), average='macro'):.3f} {best[1]}", flush=True)
    return out

svm = lambda p, sd, prob: SVC(kernel="rbf", C=p["C"], gamma=p["g"], class_weight="balanced",
                              probability=prob, random_state=sd)
rf = lambda p, sd, prob: RandomForestClassifier(n_estimators=300, max_depth=p["d"], class_weight="balanced",
                                                n_jobs=-1, random_state=sd)
svm_grid = [dict(C=c, g=g) for c in (0.1, 1, 10) for g in ("scale", 0.01)]
rf_grid = [dict(d=dd) for dd in (5, 10, None)]

res = {}
res.update(run("svm", svm, svm_grid, 0))
for sd in (0, 1, 2):
    res.update(run("rf", rf, rf_grid, sd))
pickle.dump(res, open("results_ml.pkl", "wb"))

print("\nMEAN over 15 subjects (acc / f1 / auc) +- std")
rows = {}
for name in ("svm", "rf"):
    a, f, u = [], [], []
    for sd in (0, 1, 2):
        for s in SIDS:
            k = (name, sd, s)
            if k not in res: continue
            pr, yt = res[k]["probs"], res[k]["y"]
            a.append(accuracy_score(yt, pr.argmax(1)))
            f.append(f1_score(yt, pr.argmax(1), average="macro"))
            u.append(roc_auc_score(yt, pr, multi_class="ovr", average="macro"))
    print(f"{name:4s} acc {np.mean(a):.3f}+-{np.std(a, ddof=1):.3f} | f1 {np.mean(f):.3f}+-{np.std(f, ddof=1):.3f} | auc {np.mean(u):.3f}+-{np.std(u, ddof=1):.3f}")
print("\nPER-SUBJECT macro-F1 (rf: mean of 3 seeds)")
for s in SIDS:
    fs = lambda nm, sds: np.mean([f1_score(res[(nm, q, s)]["y"], res[(nm, q, s)]["probs"].argmax(1), average="macro") for q in sds])
    print(f"S{s}: svm {fs('svm', (0,)):.2f} | rf {fs('rf', (0, 1, 2)):.2f}")