"""Deep baselines and fusion variants used in the WESAD benchmark (taken from the Kaggle notebooks)."""
import torch, torch.nn as nn, torch.nn.functional as F
from common import Branch, Hybrid, head

CFG0 = dict(Hybrid.CFG)          # original branch configuration (ECG, EDA, Resp)


class TransformerMM(nn.Module):
    """Patch embedding (kernel = stride = 50, 70 tokens, width 32) per modality, two pre-norm encoder layers."""
    def __init__(s, d=32, L=2):
        super().__init__()
        s.emb = nn.ModuleList([nn.Conv1d(1, d, 50, stride=50) for _ in range(3)])
        s.pos = nn.Parameter(torch.randn(1, 70, d) * 0.02)
        s.mod = nn.Parameter(torch.zeros(3, 1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.2, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d); s.fc = head(d)

    def forward(s, x):
        t = torch.cat([e(x[:, i:i + 1]).transpose(1, 2) + s.pos + s.mod[i]
                       for i, e in enumerate(s.emb)], 1)
        return s.fc(s.norm(s.enc(t)).mean(1))


class CrossModal(nn.Module):
    """AttX-inspired: each modality's tokens query the tokens of the other two (2-head attention)."""
    def __init__(s, d=32, h=2):
        super().__init__()
        s.emb = nn.ModuleList([nn.Sequential(nn.Conv1d(1, d, 50, stride=50), nn.BatchNorm1d(d), nn.ReLU())
                               for _ in range(3)])
        s.pos = nn.Parameter(torch.randn(1, 70, d) * 0.02)
        s.mha = nn.ModuleList([nn.MultiheadAttention(d, h, batch_first=True) for _ in range(3)])
        s.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        s.drop = nn.Dropout(0.4); s.fc = head(3 * d)

    def forward(s, x):
        T = [e(x[:, i:i + 1]).transpose(1, 2) + s.pos for i, e in enumerate(s.emb)]
        zs = []
        for i in range(3):
            others = torch.cat([T[j] for j in range(3) if j != i], 1)
            a = s.mha[i](T[i], others, others, need_weights=False)[0]
            zs.append(s.ln[i](T[i] + a).mean(1))
        return s.fc(s.drop(torch.cat(zs, 1)))


class HybridFusion(nn.Module):
    """Hybrid branches with an alternative fusion rule: 'sum', 'gated' or 'xattn' (attention across branches)."""
    def __init__(s, kind, d=16):
        super().__init__()
        s.kind = kind
        s.br = nn.ModuleList([Branch(*CFG0[m], "mha") for m in (0, 1, 2)])
        s.proj = nn.ModuleList([nn.Linear(b.out, d) for b in s.br])
        if kind == "gated":
            s.gate = nn.Linear(3 * d, 3)
        if kind == "xattn":
            s.mha = nn.MultiheadAttention(d, 2, batch_first=True); s.ln = nn.LayerNorm(d)
        s.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, 3))

    def forward(s, x):
        z = torch.stack([p(b(x[:, m:m + 1])) for p, b, m in zip(s.proj, s.br, (0, 1, 2))], 1)  # (B, 3, d)
        if s.kind == "sum":
            h = z.sum(1)
        elif s.kind == "gated":
            w = F.softmax(s.gate(z.flatten(1)), 1).unsqueeze(-1)
            h = (w * z).sum(1)
        else:
            h = s.ln(z + s.mha(z, z, z, need_weights=False)[0]).mean(1)
        return s.fc(h)
