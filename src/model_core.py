"""
revision_lib.py — Pipelin�schlüssel für Major Revision (Track A/B đúng Table 1).
Thiết kế theo manuscript Table 1: 623 static dims (G1–G8) + 8 temporal (G9, Track A only = 631).
KHÔNG dùng SHA-256 prefix, KHÔNG dùng TLSH, KHÔNG dùng file_type hash (audit F1/F4).
Gating đúng Equation (5) bản thảo: p_joint,0=(1−p_mal)·p_comb,0; p_joint,c=p_mal·p_comb,c.
"""
import numpy as np
import torch
import torch.nn as nn

# ── Feature group schema (Table 1) ──────────────────────────────────────────
GROUP_DIMS = {
    'G1_ByteHistogram': 256,
    'G2_ByteEntropy': 256,
    'G3_StringsStats': 16,
    'G4_GeneralStats': 10,
    'G5_HeaderStats': 24,
    'G6_SectionStats': 25,
    'G7_ImportsExports': 20,
    'G8_DataDirectories': 16,
}
STATIC_DIM = sum(GROUP_DIMS.values())          # = 623
TEMPORAL_DIM = 8                                 # G9, Track A only
TRACK_A_DIM = STATIC_DIM + TEMPORAL_DIM          # = 631

FAMILY_NAMES = ['Benign', 'Ransomware', 'Keylogger', 'Trojan_RAT', 'Worm_Virus', 'Other_Rare', 'Dropper']
BEHAVIOR_NAMES = ['file_ops', 'network', 'registry', 'process_injection', 'persistence', 'encryption', 'keylogging', 'screenshot']
N_FAMILIES = len(FAMILY_NAMES)
N_BEHAVIORS = len(BEHAVIOR_NAMES)


def group_slices():
    """Trả về {tên nhóm: (start, end)} trong vector static 623 dims."""
    out, s = {}, 0
    for name, d in GROUP_DIMS.items():
        out[name] = (s, s + d)
        s += d
    assert s == STATIC_DIM
    return out


# ── Equation (5): hierarchical gating — BẢN SỬA ĐÚNG SPEC ──────────────────
def hierarchical_gating(p_mal, p_combined, theta=None, eps=1e-12):
    """
    Eq.(5) manuscript (đã sửa theo checklist §I.2 — công thức phân cấp chuẩn):
        P(benign) = 1 − p_mal
        P(family=c | benign) cho c=0; P(family=c | malware) cho c=1..6
        p_joint,0 = (1 − p_mal) · p̂_combined,0
        p_joint,c = p_mal · p̂_combined,c với p̂ = conditional renormalization
    Renormalization có điều kiện đảm bảo tổng = 1 MỌI p_mal (sửa lỗi số của Eq.5 gốc:
    bản gốc nhân trực tiếp p_combined chưa điều kiện → tổng = 1 - p_comb,0·(1-p_mal) - ... sai lệch).
    Trả về p_joint (không âm, tổng 1) và nhãn family theo θ-thresholding.
    """
    p_mal = np.clip(np.nan_to_num(np.asarray(p_mal, dtype=np.float64), nan=0.0), 0.0, 1.0)
    p_combined = np.asarray(p_combined, dtype=np.float64)
    p_combined = np.clip(np.nan_to_num(p_combined, nan=0.0), 0.0, None)

    # chuẩn hóa có điều kiện (conditional): phần benign mass và malware mass riêng
    benign_mass = np.clip(p_combined[..., 0:1], 0.0, 1.0)                       # khối benign
    mal_mass = np.clip(p_combined[..., 1:], 0.0, None)                          # khối malware
    zero_mal = (mal_mass.sum(-1, keepdims=True) <= eps)
    mal_cond = np.where(zero_mal, np.full_like(mal_mass, eps), mal_mass)
    mal_cond = mal_cond / (mal_cond.sum(-1, keepdims=True) + eps)               # P(c | malware)
    ben_cond = np.where(benign_mass > eps, 1.0, 0.0)                           # P(0 | benign) ≡ 1

    p_joint = np.empty_like(p_combined)
    p_joint[..., 0:1] = (1.0 - p_mal)[..., None] * ben_cond
    p_joint[..., 1:] = p_mal[..., None] * mal_cond

    theta = np.ones(p_combined.shape[-1]) if theta is None else np.asarray(theta, dtype=np.float64)
    scores = p_joint / (theta + eps)
    y_fam = scores.argmax(axis=-1)
    return p_joint, y_fam


def gated_family_from_binary(y_bin, p_joint, theta, eps=1e-12):
    """Step 4 bản thảo: nếu binary=benign → family=0; nếu binary=malware và argmax=0
    → chọn malware argmax c∈{1..6}; ngược lại argmax toàn bộ (chia θ)."""
    y_bin = np.asarray(y_bin)
    theta = np.asarray(theta, dtype=np.float64)
    scores = p_joint / (theta + eps)
    y_fam = scores.argmax(axis=-1)
    # fallback: binary=malware nhưng family argmax=benign → chọn malware tốt nhất
    need_fb = (y_bin == 1) & (y_fam == 0)
    if need_fb.any():
        alt = p_joint[need_fb, 1:] / (theta[1:] + eps)
        y_fam[need_fb] = 1 + alt.argmax(axis=-1)
    # binary=benign → ép về 0
    y_fam[y_bin == 0] = 0
    return y_fam


# ── GT-Transformer đúng spec manuscript §3.2 ────────────────────────────────
class GTTransformer(nn.Module):
    """9 group tokens (Track A) / 8 (Track B), d_model=256, 6 layers, 8 heads, ff=1024."""
    def __init__(self, n_groups=8, d_model=256, nhead=8, num_layers=6, dim_ff=1024, dropout=0.25,
                 static_only=True):
        super().__init__()
        self.static_only = static_only
        self.n_groups = n_groups if static_only else n_groups + 1  # +G9 token
        self.group_slices_ = group_slices()
        self.dims = list(GROUP_DIMS.values()) + ([TEMPORAL_DIM] if not static_only else [])
        # mỗi nhóm: 2-layer MLP project → 1 token d_model
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.GELU(),
            ) for d in self.dims
        ])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_groups + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        # heads: binary linear; family cosine-margin (7 lớp); behavior (8 sigmoid)
        self.bin_head = nn.Linear(d_model, 1)
        self.fam_head = nn.Linear(d_model, N_FAMILIES)          # dùng cùng cosine margin ở loss
        self.fam_scale = nn.Parameter(torch.tensor(16.0))
        self.beh_head = nn.Linear(d_model, N_BEHAVIORS)

    def forward(self, x):
        B = x.size(0)
        toks = []
        s = 0
        for i, d in enumerate(self.dims):
            toks.append(self.group_mlps[i](x[:, s:s + d]))
            s += d
        t = torch.stack(toks, dim=1)                            # B×G×d
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1) + self.pos_emb[:, :t.size(1) + 1]
        h = self.norm(self.encoder(t))[:, 0]                    # [CLS]
        return (self.bin_head(h).squeeze(-1),                   # l_b
                self.fam_head(h) * self.fam_scale,              # l_f (margin áp ở loss)
                self.beh_head(h))                               # l_beh (logits)


def focal_loss(logits, targets, gamma=2.0, alpha=0.5, label_smoothing=0.05):
    """Binary focal (Task 1) — γ1=2.0, smoothing 0.05 theo bản thảo."""
    p = torch.sigmoid(logits)
    targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()


def logit_adjusted_family_loss(lf, yf, class_prior, tau=1.2, gamma=2.5, ignore_index=-1, beta=0.9999):
    """Prior-based logit adjustment (τ=1.2) + focal (γ=2.5) + class-balanced weights (β=0.9999).
    yf == -1 (unknown family) bị loại bằng ignore mask như checklist §I.2."""
    mask = yf != ignore_index
    if mask.sum() == 0:
        return lf.sum() * 0.0
    lf_m, yf_m = lf[mask], yf_m = lf[mask], yf[mask]
    adj = torch.log(class_prior + 1e-12)
    la = lf_m + tau * adj
    n_c = torch.bincount(yf_m, minlength=lf.size(1)).float().clamp(min=1.0)
    w = (1 - beta) / (1 - beta ** n_c)
    w = w / w.sum() * lf.size(1)
    logp = torch.nn.functional.log_softmax(la, dim=1)
    p = logp.exp()
    y_onehot = torch.nn.functional.one_hot(yf_m, lf.size(1)).float()
    p_t = (p * y_onehot).sum(1).clamp(1e-8, 1 - 1e-8)
    focal = (1 - p_t) ** gamma * (- (y_onehot * logp).sum(1))
    return (w[yf_m] * focal).mean()


def soft_macro_f1_loss(lf, yf, ignore_index=-1, eps=1e-12):
    """Differentiable soft Macro-F1 (bản thảo §3.3) — 7 lớp, bỏ unknown."""
    mask = yf != ignore_index
    if mask.sum() == 0:
        return lf.sum() * 0.0
    p = torch.softmax(lf[mask], dim=1)
    C = lf.size(1)
    y_onehot = torch.nn.functional.one_hot(yf[mask], C).float()
    TP = (p * y_onehot).sum(0)
    FN = (y_onehot * (1 - p)).sum(0)
    FP = (p * (1 - y_onehot)).sum(0)
    f1 = 2 * TP / (2 * TP + FN + FP + eps)
    return 1.0 - f1.mean()
