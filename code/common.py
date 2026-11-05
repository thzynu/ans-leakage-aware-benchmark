"""Shared code: Hybrid network, augmentation, focal loss and the leave-one-subject-out fold runner.

Requires a CUDA GPU (mixed precision is used, as in the paper).
"""
import copy, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

dev = "cuda"


class State:
    """Data used by predict() and run_fold(); set with set_data()."""
    X = y = sid = SIDS = None


def set_data(X, y, sid, sids):
    State.X = torch.from_numpy(X).to(dev)
    State.y = torch.from_numpy(y).to(dev)
    State.sid = sid
    State.SIDS = [int(s) for s in sids]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Branch(nn.Module):
    def __init__(s, k, bi, attn="mha", h=16):
        super().__init__()
        s.conv = nn.Conv1d(1, 16, k, padding=k // 2)
        s.bn = nn.BatchNorm1d(16)
        s.gru = nn.GRU(16, h, batch_first=True, bidirectional=bi)
        s.out = h * (2 if bi else 1)
        s.attn = attn
        if attn == "mha":
            s.mha = nn.MultiheadAttention(s.out, 2, batch_first=True)
        s.drop = nn.Dropout(0.4)

    def forward(s, x):
        x = F.max_pool1d(F.relu(s.bn(s.conv(x))), 2).transpose(1, 2).contiguous()
        x, _ = s.gru(x)
        if s.attn == "mha":
            x = x + s.mha(x, x, x, need_weights=False)[0]
        elif s.attn == "uniform":
            x = x + x.mean(1, keepdim=True)
        return s.drop(x).mean(1)


class Hybrid(nn.Module):
    CFG = {0: (7, True), 1: (5, False), 2: (7, True)}   # ECG, EDA, Resp: (kernel, bidirectional)

    def __init__(s, mods=(0, 1, 2), attn="mha"):
        super().__init__()
        s.mods = list(mods)
        s.br = nn.ModuleList([Branch(*s.CFG[m], attn) for m in s.mods])
        s.fc = nn.Sequential(nn.Linear(sum(b.out for b in s.br), 16), nn.ReLU(),
                             nn.Dropout(0.3), nn.Linear(16, 3))

    def forward(s, x):
        return s.fc(torch.cat([b(x[:, m:m + 1]) for b, m in zip(s.br, s.mods)], 1))


def head(n):
    return nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 3))


class CNNOnly(nn.Module):
    def __init__(s):
        super().__init__()
        def blk():
            return nn.Sequential(
                nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 24, 7, stride=2, padding=3), nn.BatchNorm1d(24), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(24, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        s.br = nn.ModuleList([blk() for _ in range(3)])
        s.drop = nn.Dropout(0.4); s.fc = head(96)

    def forward(s, x):
        z = torch.cat([b(x[:, i:i + 1]).mean(2) for i, b in enumerate(s.br)], 1)
        return s.fc(s.drop(z))


class RNNOnly(nn.Module):
    BI = (True, False, True)

    def __init__(s, h=32):
        super().__init__()
        s.gru = nn.ModuleList([nn.GRU(1, h, batch_first=True, bidirectional=b) for b in s.BI])
        s.drop = nn.Dropout(0.4)
        s.fc = head(sum(h * (2 if b else 1) for b in s.BI))

    def forward(s, x):
        zs = []
        for i, g in enumerate(s.gru):
            z = F.avg_pool1d(x[:, i:i + 1], 8).transpose(1, 2).contiguous()
            zs.append(g(z)[0].mean(1))
        return s.fc(s.drop(torch.cat(zs, 1)))


def focal(logits, t, alpha, gamma=1.0):
    lp = F.log_softmax(logits, 1)
    lpt = lp.gather(1, t[:, None]).squeeze(1)
    return (-alpha[t] * (1 - lpt.exp()) ** gamma * lpt).mean()


SIG = torch.tensor([0.02, 0.05, 0.03], device=dev).view(1, 3, 1)


def augment(x):
    B, C, T = x.shape
    x = x + torch.randn_like(x) * SIG
    x = x + (torch.rand(B, C, 1, device=dev) - 0.5) * 0.4 * torch.linspace(-1, 1, T, device=dev)
    return x * (1 + (torch.rand(B, C, 1, device=dev) - 0.5) * 0.2)


@torch.no_grad()
def predict(model, ids, bs=128):
    model.eval(); out = []
    for b in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.softmax(model(State.X[ids[b:b + bs]]).float(), 1))
    return torch.cat(out).cpu().numpy()


def split_ids(test_sid):
    """Deterministic LOSO split: test subject, two validation subjects, the rest for training."""
    sids = State.SIDS
    rem = [s for s in sids if s != test_sid]; i = sids.index(test_sid)
    val_s = [rem[i % len(rem)], rem[(i + len(rem) // 2) % len(rem)]]
    tr_s = [s for s in rem if s not in val_s]
    idx = lambda ss: torch.from_numpy(np.where(np.isin(State.sid, ss))[0]).to(dev)
    return idx(tr_s), idx(val_s), idx([test_sid])


def train(model, tr, va, epochs=30, patience=6, bs=64, lr=1e-3, verbose=False):
    """Train with AdamW, focal loss, ReduceLROnPlateau and early stopping on validation macro-F1."""
    y = State.y
    cnt = torch.bincount(y[tr], minlength=3).float()
    alpha = cnt.sum() / cnt; alpha = alpha / alpha.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler()
    best, best_state, wait = -1, None, 0
    yv = y[va].cpu().numpy()
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        for b in range(0, len(perm), bs):
            ib = perm[b:b + bs]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(augment(State.X[ib]))
            loss = focal(logits.float(), y[ib], alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        f1v = f1_score(yv, predict(model, va).argmax(1), average="macro")
        sched.step(f1v)
        if verbose:
            print(f"ep{ep + 1:02d} loss {loss.item():.3f} val_F1 {f1v:.3f}")
        if f1v > best:
            best, best_state, wait = f1v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return ep + 1


def run_fold(test_sid, mods=(0, 1, 2), attn="mha", seed=0, epochs=30, patience=6,
             bs=64, lr=1e-3, verbose=True, model_fn=None):
    """One LOSO fold. model_fn() builds a different architecture (e.g. CNNOnly); default is Hybrid."""
    seed_all(seed)
    tr, va, te = split_ids(test_sid)
    model = (model_fn() if model_fn else Hybrid(mods, attn)).to(dev)
    t0 = time.time()
    n_ep = train(model, tr, va, epochs, patience, bs, lr, verbose)
    p = predict(model, te); yt = State.y[te].cpu().numpy()
    return dict(test=test_sid, acc=accuracy_score(yt, p.argmax(1)),
                f1=f1_score(yt, p.argmax(1), average="macro"),
                auc=roc_auc_score(yt, p, multi_class="ovr", average="macro"),
                probs=p, y=yt, epochs=n_ep, sec=time.time() - t0, model=model)
