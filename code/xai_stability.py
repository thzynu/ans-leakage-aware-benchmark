"""Stability of Integrated Gradients maps across independently trained models (WESAD).

15 subjects x 3 seeds = 45 models; 96 fixed windows per subject; 8-step midpoint IG with a zero baseline.
Usage: python xai_stability.py --data DATA_DIR --out OUT_DIR [--smoke]
Saved after every model; re-running resumes. The final file must hold 45 entries.
"""
import argparse, os, pickle, time
import numpy as np, torch, torch.nn as nn
import common
from common import State, set_data, run_fold, dev

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); SMOKE = a.smoke
dw = np.load(os.path.join(a.data, "wesad_windows.npz"))
set_data(dw["X"], dw["y"], dw["sid"], np.unique(dw["sid"]))
sid, SIDS = State.sid, State.SIDS
N_WIN, N_STEP, BLK = (16 if SMOKE else 96), 8, 50
SEEDS, EPOCHS = ((0,) if SMOKE else (0, 1, 2)), (1 if SMOKE else 30)
RESX = os.path.join(a.out, "xai_smoke.pkl" if SMOKE else "xai_stability.pkl")
resx = pickle.load(open(RESX, "rb")) if os.path.exists(RESX) else {}


def pick_windows(s):
    ids = np.where(sid == s)[0]
    rng = np.random.RandomState(1000 + s)                    # same windows for every seed
    return np.sort(rng.choice(ids, min(N_WIN, len(ids)), replace=False))


def ig_maps(model, ids_np, chunk=8):
    model.eval()
    for m in model.modules():                                # cuDNN GRU backward needs train mode (no GRU dropout)
        if isinstance(m, nn.GRU): m.train()
    for p in model.parameters(): p.requires_grad_(False)
    ids = torch.from_numpy(ids_np).to(dev)
    alphas = (torch.arange(N_STEP, device=dev).float() + 0.5) / N_STEP
    maps, preds = [], []
    for b in range(0, len(ids), chunk):
        xb = State.X[ids[b:b + chunk]].float(); n = xb.shape[0]
        with torch.no_grad():
            pr = model(xb).argmax(1)
        xi = (alphas.view(1, -1, 1, 1) * xb.unsqueeze(1)).reshape(n * N_STEP, 3, -1).requires_grad_(True)
        out = model(xi)
        out.gather(1, pr.repeat_interleave(N_STEP)[:, None]).sum().backward()
        g = xi.grad.view(n, N_STEP, 3, -1).mean(1)
        attr = (g * xb).abs().reshape(n, 3, -1, BLK).sum(-1)  # n x 3 x 70
        maps.append(attr.detach().cpu().numpy()); preds.append(pr.cpu().numpy())
    return np.concatenate(maps), np.concatenate(preds)


T0 = time.time()
for s in (SIDS[:1] if SMOKE else SIDS):
    ids = pick_windows(s)
    for sd in SEEDS:
        if (s, sd) in resx: continue
        r = run_fold(s, seed=sd, epochs=EPOCHS, verbose=False)
        maps, pred = ig_maps(r["model"], ids)
        resx[(s, sd)] = dict(ids=ids, maps=maps.astype(np.float32), pred=pred,
                             y=State.y[torch.from_numpy(ids).to(dev)].cpu().numpy(),
                             f1=r["f1"], epochs=r["epochs"], sec=r["sec"])
        pickle.dump(resx, open(RESX, "wb"))
        print(f"S{s} seed{sd}: F1 {r['f1']:.3f} ep{r['epochs']} maps {maps.shape} ({(time.time() - T0) / 60:.0f} min)", flush=True)
        del r; torch.cuda.empty_cache()
print("entries saved:", len(resx), "(45 expected for the full run)")
