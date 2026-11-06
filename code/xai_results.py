"""Robustness, attention and Integrated Gradients analysis on WESAD (source of xai_results.pkl).

For every held-out subject, Hybrid, CNN-only and RNN-only are trained once (seed 0) with the LOSO protocol, then:
 1) test windows are perturbed (noise, baseline wander, motion burst, channel removal) and macro-F1 is recorded;
 2) normalised attention entropy and R-peak enrichment (ECG branch) are computed on all test windows;
 3) Integrated Gradients (zero baseline, 8 midpoint steps, logit of the predicted class) is computed for 96 random
    test windows, with deletion/insertion curves against a random ranking.
Usage: python xai_results.py --data DATA_DIR --out OUT_DIR [--smoke]   (saved after every subject; re-run resumes)
"""
import argparse, math, os, pickle, time
import numpy as np, torch, torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score
import common
from common import CNNOnly, RNNOnly, State, set_data, run_fold, dev

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); SMOKE = a.smoke
dw = np.load(os.path.join(a.data, "wesad_windows.npz"))
set_data(dw["X"], dw["y"], dw["sid"], np.unique(dw["sid"]))
X, y, sid, SIDS = State.X, State.y, State.sid, State.SIDS
SIDS_RUN = SIDS[:1] if SMOKE else SIDS
OUT = os.path.join(a.out, "xai_results_smoke.pkl" if SMOKE else "xai_results.pkl")
FS_ = 3500 / 30.0
FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)
CONDS = ["clean", "gauss0.2", "gauss0.5", "gauss1.0", "wander", "motion", "drop_ecg", "drop_eda", "drop_resp"]


def train_model(name, s, seed=0):
    cls = {"hybrid": None, "cnn_only": CNNOnly, "rnn_only": RNNOnly}[name]
    return run_fold(s, seed=seed, verbose=False, model_fn=cls)["model"].eval()


@torch.no_grad()
def predict_x(model, xt, bs=64):
    model.eval(); out = []
    for b in range(0, len(xt), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.softmax(model(xt[b:b + bs]).float(), 1))
    return torch.cat(out)


def prob_t(model, xb, target):
    return predict_x(model, xb).gather(1, target[:, None]).squeeze(1)


def corrupt(x, kind, g):
    x = x.clone(); B, C, T = x.shape
    if kind.startswith("gauss"):                                  # additive white noise (z-scored units)
        x += torch.randn(x.shape, generator=g, device=x.device) * float(kind[5:])
    elif kind == "wander":                                        # 0.25 Hz, amplitude 0.5, ECG and respiration
        t = torch.arange(T, device=x.device) / FS_
        ph = torch.rand(B, 1, 1, generator=g, device=x.device) * 2 * math.pi
        w = 0.5 * torch.sin(2 * math.pi * 0.25 * t + ph)[:, 0]
        x[:, 0] += w; x[:, 2] += w
    elif kind == "motion":                                        # 3 s burst, N(0, 3^2) on all channels
        L = int(3 * FS_)
        st = torch.randint(0, T - L, (B,), generator=g, device=x.device)
        for b in range(B):
            a0 = int(st[b])
            x[b, :, a0:a0 + L] = torch.randn(C, L, generator=g, device=x.device) * 3.0
    elif kind.startswith("drop_"):                                # channel set to zero
        x[:, {"ecg": 0, "eda": 1, "resp": 2}[kind[5:]]] = 0
    return x


def integrated_gradients(model, xb, target, steps=8):
    total = torch.zeros_like(xb)
    with torch.backends.cudnn.flags(enabled=False):               # allows GRU backward in eval mode
        for k in range(steps):
            xi = (xb * ((k + 0.5) / steps)).clone().requires_grad_(True)
            gr, = torch.autograd.grad(model(xi).gather(1, target[:, None]).sum(), xi)
            total += gr
    return (xb * total / steps).detach()


@torch.no_grad()
def branch_attn(model, xb):
    outs = []
    for b, m in zip(model.br, model.mods):
        h = F.max_pool1d(F.relu(b.bn(b.conv(xb[:, m:m + 1]))), 2).transpose(1, 2).contiguous()
        h, _ = b.gru(h)
        outs.append(b.mha(h, h, h, need_weights=True, average_attn_weights=True)[1])
    return outs


def enrich(v, pk, rad, shift=0):
    """Mean map value within +-rad samples of the R peaks divided by the overall mean (NaN if < 3 peaks)."""
    if len(pk) < 3: return np.nan
    m = np.zeros(len(v), bool)
    for p in pk:
        c = (p + shift) % len(v)
        m[max(0, c - rad):c + rad + 1] = True
    return v[m].mean() / (v.mean() + 1e-12)


RES2 = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
T0 = time.time()
for s in SIDS_RUN:
    if s in RES2: continue
    te = torch.from_numpy(np.where(sid == s)[0]).to(dev)
    xt, yt = X[te], y[te].cpu().numpy()
    R = {"rob": {}}
    models = {n: train_model(n, s) for n in ("hybrid", "cnn_only", "rnn_only")}
    # 1) robustness (perturbations applied to test windows only)
    for n, m in models.items():
        R["rob"][n] = {}
        for c in CONDS:
            g = torch.Generator(device=dev); g.manual_seed(1000 + s)
            pr = predict_x(m, corrupt(xt, c, g) if c != "clean" else xt).argmax(1).cpu().numpy()
            R["rob"][n][c] = (accuracy_score(yt, pr), f1_score(yt, pr, average="macro"))   # (accuracy, macro-F1)
    hy = models["hybrid"]
    pred = predict_x(hy, xt).argmax(1)
    peaks = [find_peaks(e, height=2.0, distance=int(0.4 * FS_))[0] for e in xt[:, 0].cpu().numpy()]   # R peaks
    # 2) attention: normalised entropy and R-peak enrichment (ECG branch)
    ent, ea, eas = [], [], []
    for b in range(0, len(xt), 16):
        outs = branch_attn(hy, xt[b:b + 16])
        ent.append(torch.stack([-(w * torch.log(w + 1e-12)).sum(-1).mean(1) / math.log(w.shape[-1])
                                for w in outs], 1).cpu().numpy())
        r_ecg = outs[0].mean(1).cpu().numpy()                     # attention received, (B, 1750)
        for i in range(r_ecg.shape[0]):
            pk = peaks[b + i] // 2
            ea.append(enrich(r_ecg[i], pk, 6)); eas.append(enrich(r_ecg[i], pk, 6, shift=29))
        del outs
    R["ent"] = np.concatenate(ent); R["enr_att"] = np.array(ea); R["enr_att_shift"] = np.array(eas)
    R["y"] = yt; R["pred"] = pred.cpu().numpy()
    # 3) Integrated Gradients and deletion/insertion on 96 random windows
    gc = torch.Generator(); gc.manual_seed(s)
    sel = torch.randperm(len(xt), generator=gc)[:96].to(dev)
    xs, ts = xt[sel], pred[sel]
    ig = torch.cat([integrated_gradients(hy, xs[b:b + 32], ts[b:b + 32]) for b in range(0, len(xs), 32)])
    cells = ig.abs().reshape(-1, 3, 70, 50).sum(-1)               # (B, 3, 70) blocks of 50 samples
    R["ig_share"] = (cells.sum(-1) / cells.sum((1, 2))[:, None]).cpu().numpy()
    R["ig_y"] = yt[sel.cpu().numpy()]
    igp = ig.abs().cpu().numpy()
    R["enr_ig"] = np.array([enrich(igp[i, 0], peaks[int(sel[i])], 12) for i in range(len(xs))])
    R["enr_ig_shift"] = np.array([enrich(igp[i, 0], peaks[int(sel[i])], 12, shift=58) for i in range(len(xs))])
    n_c = 3 * 70; flat = cells.reshape(len(xs), n_c)
    order = flat.argsort(1, descending=True)
    g = torch.Generator(device=dev); g.manual_seed(7)
    rorder = torch.rand(len(xs), n_c, generator=g, device=dev).argsort(1)
    di = {"base": prob_t(hy, xs, ts).cpu().numpy(), "zero": prob_t(hy, torch.zeros_like(xs), ts).cpu().numpy()}
    for f in FRACS:
        k = int(round(f * n_c))
        for nm, od in (("ig", order), ("rand", rorder)):
            mk = torch.zeros(len(xs), n_c, device=dev); mk.scatter_(1, od[:, :k], 1.0)
            mk = mk.view(-1, 3, 70, 1).expand(-1, 3, 70, 50).reshape(-1, 3, 3500)
            di[(nm, "del", f)] = prob_t(hy, xs * (1 - mk), ts).cpu().numpy()
            di[(nm, "ins", f)] = prob_t(hy, xs * mk, ts).cpu().numpy()
    R["delins"] = di
    RES2[s] = R
    pickle.dump(RES2, open(OUT, "wb"))
    del models, hy; torch.cuda.empty_cache()
    print(f"fold S{s} done | hybrid clean F1 {R['rob']['hybrid']['clean'][1]:.3f} | {(time.time() - T0) / 60:.1f} min", flush=True)
print("subjects saved:", len(RES2), "(15 expected for the full run); summary statistics: analysis.py")
