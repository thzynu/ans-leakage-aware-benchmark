"""Statistics behind the tables of Chapter 4, computed from the saved result files.

Usage: python analysis.py --results RESULTS_DIR
Expected files: results.pkl, results_ml.pkl, results_fusion.pkl, results_case.pkl, xai_results.pkl, xai_stability.pkl
Unit of analysis = subject (seed results are averaged within subject). Bootstrap: 10,000 resamples over subjects, seed 0.
"""
import argparse, itertools, os, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
load = lambda f: pickle.load(open(os.path.join(a.results, f), "rb"))
rng = np.random.RandomState(0)


def ci(x):
    x = np.asarray(x); return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(10000)], [2.5, 97.5])


def holm(ps):
    m = len(ps); o = np.argsort(ps); adj = np.zeros(m); prev = 0
    for r, i in enumerate(o): prev = max(prev, min(1, (m - r) * ps[i])); adj[i] = prev
    return adj


def metrics(v):
    p, y = v["probs"], v["y"]
    return (accuracy_score(y, p.argmax(1)), f1_score(y, p.argmax(1), average="macro"),
            roc_auc_score(y, p, multi_class="ovr", average="macro"))


def per_subject(R, name, col=1):
    seeds = sorted({k[1] for k in R if k[0] == name}); subs = sorted({k[2] for k in R if k[0] == name})
    return np.array([[metrics(R[(name, sd, s)])[col] for sd in seeds] for s in subs]).mean(1)


# ---------------------------------------------------------------- WESAD benchmark, ablations, fusion
R = {**load("results.pkl"), **load("results_ml.pkl")}; Fu = load("results_fusion.pkl"); ALL = {**R, **Fu}
names = sorted({k[0] for k in ALL}); subs = sorted({k[2] for k in R})
F1 = {n: per_subject(ALL, n) for n in names}
print("== WESAD: macro-F1 per model (mean over subjects, 95% CI), acc, AUC, SD across seeds")
for n in names:
    seeds = sorted({k[1] for k in ALL if k[0] == n})
    arr = np.array([[metrics(ALL[(n, sd, s)]) for sd in seeds] for s in subs])
    c = ci(F1[n]); sdseed = arr[:, :, 1].mean(0).std(ddof=1) if len(seeds) > 1 else float("nan")
    print(f"{n:12s} F1 {F1[n].mean():.3f} [{c[0]:.3f};{c[1]:.3f}] acc {arr[:, :, 0].mean():.3f} auc {arr[:, :, 2].mean():.3f} sdseed {sdseed:.3f}")
fus = [n for n in names if n.startswith("fus_")]
others = [n for n in names if n != "hybrid" and n not in fus]
ps = [wilcoxon(F1["hybrid"], F1[n]).pvalue for n in others]; adj = holm(np.array(ps))
print(f"\n== Hybrid vs {len(others)} other models (Holm over {len(others)})")
for n, p, q in zip(others, ps, adj): print(f"{n:12s} diff {(F1['hybrid'] - F1[n]).mean():+.3f} p={p:.3f} Holm={q:.3f}")
main = ["hybrid", "cnn_only", "rnn_only", "transformer", "crossmodal", "svm", "rf"]
print("Friedman (7 main models): p = %.3f" % friedmanchisquare(*[F1[n] for n in main]).pvalue)
ps = [wilcoxon(F1[n], F1["hybrid"]).pvalue for n in fus]; adj = holm(np.array(ps))
print("\n== Fusion rules vs concatenation (Holm over 3)")
for n, p, q in zip(fus, ps, adj): print(f"{n:10s} F1 {F1[n].mean():.3f} diff {(F1[n] - F1['hybrid']).mean():+.3f} wins {(F1[n] > F1['hybrid']).sum()}/15 p={p:.2f} Holm={q:.2f}")
print("\n== Per-subject macro-F1 (main models)")
print(pd.DataFrame({n: F1[n] for n in main}, index=[f"S{s}" for s in subs]).round(2).to_string())

# ---------------------------------------------------------------- CASE
C = load("results_case.pkl"); csubs = sorted({k[2] for k in C})
FC = {n: per_subject(C, n) for n in ["hybrid", "cnn_only", "rnn_only", "tr_hybrid", "tr_cnn_only", "tr_rnn_only"]}
print("\n== CASE (30 subjects): macro-F1 [95% CI]")
for n, v in FC.items(): c = ci(v); print(f"{n:12s} {v.mean():.3f} [{c[0]:.3f};{c[1]:.3f}]")
for grp in (["hybrid", "cnn_only", "rnn_only"], ["tr_hybrid", "tr_cnn_only", "tr_rnn_only"]):
    pr = list(itertools.combinations(grp, 2)); ps = [wilcoxon(FC[x], FC[y]).pvalue for x, y in pr]; ad = holm(np.array(ps))
    print(f"Friedman p={friedmanchisquare(*[FC[n] for n in grp]).pvalue:.3f}", *[f"| {x} vs {y}: Holm {q:.3f}" for (x, y), q in zip(pr, ad)])

# ---------------------------------------------------------------- RQ5: oracle class offsets tuned on test labels
print("\n== RQ5: macro-F1 with offsets tuned on test labels (upper bound)")
grid = np.linspace(-2, 2, 17)
for name in ["hybrid", "cnn_only", "rnn_only"]:
    b, o = [], []
    for s in subs:
        bs, os_ = [], []
        for sd in (0, 1, 2):
            v = R[(name, sd, s)]; p = np.log(v["probs"] + 1e-9); y = v["y"]
            bs.append(f1_score(y, p.argmax(1), average="macro"))
            os_.append(max(f1_score(y, (p + np.array([0, b1, b2])).argmax(1), average="macro") for b1, b2 in itertools.product(grid, grid)))
        b.append(np.mean(bs)); o.append(np.mean(os_))
    b, o = np.array(b), np.array(o); d = o - b
    print(f"{name:9s} arg max {b.mean():.3f} | tuned {o.mean():.3f} | inflation {d.mean():+.3f} [{ci(d)[0]:.3f};{ci(d)[1]:.3f}]")

# ---------------------------------------------------------------- robustness and explanation (xai_results.pkl)
x = load("xai_results.pkl"); xs = sorted(x)
conds = ["clean", "gauss0.2", "gauss0.5", "gauss1.0", "wander", "motion", "drop_ecg", "drop_eda", "drop_resp"]
mods = ["hybrid", "cnn_only", "rnn_only"]
FR = {m: {c: np.array([x[s]["rob"][m][c][1] for s in xs]) for c in conds} for m in mods}   # rob = (accuracy, macro-F1)
lab, ps = [], []
for c in conds:
    for m in ["cnn_only", "rnn_only"]: lab.append((c, m)); ps.append(wilcoxon(FR["hybrid"][c], FR[m][c]).pvalue)
adj = holm(np.array(ps))
print("\n== Robustness: macro-F1 (Holm over 18 Hybrid-vs-baseline comparisons)")
for c in conds:
    print(f"{c:10s}", "  ".join(f"{m} {FR[m][c].mean():.3f}" for m in mods),
          *[f"| vs {m}: Holm {adj[lab.index((c, m))]:.3f}" for m in ["cnn_only", "rnn_only"]])
ks = [0.05, 0.1, 0.2, 0.3, 0.5]
curve = lambda kind, mode: np.array([[x[s]["delins"][(kind, mode, k)].mean() for k in ks] for s in xs])
auc = lambda c: np.trapezoid(c, ks, axis=1) / (ks[-1] - ks[0])
print("\n== Faithfulness (AUC over k = 5..50%), Integrated Gradients vs random ranking")
for mode in ["del", "ins"]:
    g, r_ = auc(curve("ig", mode)), auc(curve("rand", mode))
    print(f"{mode}: IG {g.mean():.3f} random {r_.mean():.3f} diff {(g - r_).mean():+.3f} [{ci(g - r_)[0]:.3f};{ci(g - r_)[1]:.3f}] p={wilcoxon(g, r_).pvalue:.4f}")
print("attention entropy (ECG, EDA, Resp):", np.array([x[s]["ent"].mean(0) for s in xs]).mean(0).round(3))
for key in ["enr_att", "enr_ig"]:
    real, shift = np.array([x[s][key].mean() for s in xs]), np.array([x[s][key + "_shift"].mean() for s in xs])
    print(f"R-peak enrichment {key}: real {real.mean():.3f} shifted {shift.mean():.3f} p={wilcoxon(real, shift).pvalue:.4f}")
print("IG modality share (ECG, EDA, Resp):", np.array([x[s]["ig_share"].mean(0) for s in xs]).mean(0).round(3))

# ---------------------------------------------------------------- stability across seeds
S = load("xai_stability.pkl")
assert len(S) == 45, f"xai_stability.pkl has {len(S)} entries; the full run has 45 (15 subjects x 3 seeds)"
def rho(A, B):
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
rows = []
for s in sorted({k[0] for k in S}):
    for p, q in [(0, 1), (0, 2), (1, 2)]:
        A, B = S[(s, p)]["maps"], S[(s, q)]["maps"]; n = len(A); A, B = A.reshape(n, -1), B.reshape(n, -1)
        g = np.random.RandomState(s); dd = []
        for _ in range(5):
            pm = g.permutation(n); pm = np.where(pm == np.arange(n), (pm + 1) % n, pm); dd.append(rho(A, B[pm]).mean())
        rows.append(dict(sid=s, same=rho(A, B).mean(), diff=np.mean(dd), agree=(S[(s, p)]["pred"] == S[(s, q)]["pred"]).mean()))
per = pd.DataFrame(rows).groupby("sid").mean()
print("\n== Stability of IG maps across seeds (15 subjects)")
for c in ["same", "diff", "agree"]: print(f"{c}: {per[c].mean():.3f} [{ci(per[c])[0]:.3f};{ci(per[c])[1]:.3f}]")
print("same > diff in", int((per.same > per['diff']).sum()), "/ 15 subjects; Wilcoxon p = %.5f" % wilcoxon(per.same, per["diff"]).pvalue)
