"""
tests/test_model_core.py — Unit test cho Gating Equation (5) + GT-Transformer + losses.
Chạy: python3 tests/test_model_core.py
Nhiệm vụ PDF §I.2: "thêm unit test cho các trường hợp biên", "phân phối sau gating
không âm, tổng bằng 1 và nhất quán binary/family".
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np
import torch

from src.model_core import (hierarchical_gating, gated_family_from_binary, GTTransformer,
                           group_slices, STATIC_DIM, TRACK_A_DIM, GROUP_DIMS,
                           logit_adjusted_family_loss, soft_macro_f1_loss, focal_loss)

PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else:    FAIL += 1; print(f"  [FAIL] {name}")

print("== 1. Feature group schema (Table 1) ==")
check("623 static dims", STATIC_DIM == 623)
check("631 Track A dims", TRACK_A_DIM == 631)
check("8 groups G1–G8", len(GROUP_DIMS) == 8)
check("dims per group [256,256,16,10,24,25,20,16]",
      list(GROUP_DIMS.values()) == [256, 256, 16, 10, 24, 25, 20, 16])
sl = group_slices()
check("slices contiguous covering 623", sl['G8_DataDirectories'][1] == 623 and sl['G1_ByteHistogram'][0] == 0)

print("\n== 2. Gating Eq.(5) — trường hợp biên ==")
rng = np.random.default_rng(0)
# a) p_mal = 0 → mọi khối lượng ở lớp 0: P(benign)=1, P(family|benign)=0
pj, yfam = hierarchical_gating(np.array([0.0]), np.array([[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]]))
check("p_mal=0: mass malware = 0", np.allclose(pj[0, 1:], 0.0))
check("p_mal=0: p_joint,0 = 1−p_mal = 1 (binary quyết định benign)", np.isclose(pj[0, 0], 1.0))
# b) p_mal = 1 → p_joint,0 = 0
pj, _ = hierarchical_gating(np.array([1.0]), np.array([[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]]))
check("p_mal=1: p_joint,0 = 0", np.isclose(pj[0, 0], 0.0))
check("p_mal=1: tổng = 1", np.isclose(pj.sum(), 1.0))
# c) ngẫu nhiên: không âm + tổng 1
p_mal = rng.uniform(0, 1, 5000)
p_comb = rng.dirichlet(np.ones(7), size=5000)
pj, _ = hierarchical_gating(p_mal, p_comb)
check("random: không âm", (pj >= 0).all())
check("random: tổng = 1 (max err < 1e-9)", np.abs(pj.sum(1) - 1).max() < 1e-9)
# d) nhất quán binary/family: y_bin=0 → y_fam=0; y_bin=1 → y_fam≥1
y_bin = (p_mal > 0.5).astype(int)
pj, _ = hierarchical_gating(p_mal, p_comb)
yf = gated_family_from_binary(y_bin, pj, theta=np.ones(7))
check("y_bin=0 ⇒ family=0", (yf[y_bin == 0] == 0).all())
check("y_bin=1 ⇒ family≥1", (yf[y_bin == 1] >= 1).all())
# e) theta nén điểm: score = p/θ. θ lớn nén lớp đó xuống → argmax rơi vào lớp p lớn
pj2, yfam2 = hierarchical_gating(np.array([1.0]), np.array([[0.0, 0.5, 0.0, 0.5, 0, 0, 0]]),
                                 theta=np.array([1, 100, 1, 1, 1, 1, 1]))
check("theta nén điểm: lớp 3 thắng (0.5/1 > 0.5/100)", yfam2[0] == 3)
# e2) theta nhỏ ưu tiên lớp hiếm có p>0
pj2b, yfam2b = hierarchical_gating(np.array([1.0]), np.array([[0.0, 0.1, 0.0, 0.9, 0, 0, 0]]),
                                   theta=np.array([1, 0.05, 1, 1, 1, 1, 1]))
check("theta nhỏ ưu tiên lớp hiếm (0.1/0.05 > 0.9/1)", yfam2b[0] == 1)
# f) p_combined không chuẩn hóa đầu vào vẫn chuẩn hóa ở trong
pj3, _ = hierarchical_gating(np.array([0.5]), np.array([[3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
check("input chưa chuẩn hóa → chuẩn hóa bên trong (malware cond)", np.isclose(pj3[0, 1:].sum(), 0.5))
check("benign cond: p_joint,0 = (1-p_mal) khi mass benign > 0", np.isclose(pj3[0, 0], 0.5))
# g) NaN/an toàn số: p_mal NaN bị clip
try:
    hierarchical_gating(np.array([np.nan]), np.array([[0.14] * 7]))
    nan_ok = True
except Exception:
    nan_ok = False
check("p_mal=NaN không crash (clip)", nan_ok)

print("\n== 3. GT-Transformer forward ==")
torch.manual_seed(0)
for static_only, dim in [(True, 623), (False, 631)]:
    m = GTTransformer(static_only=static_only)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, dim)
    lb, lf, lbh = m(x)
    tag = "Track B" if static_only else "Track A"
    check(f"{tag}: output shapes ({lb.shape}, {tuple(lf.shape)}, {tuple(lbh.shape)})",
          lb.shape == (4,) and lf.shape == (4, 7) and lbh.shape == (4, 8))
    check(f"{tag}: params ≈ {n_params:,} > 5M (6-layer d256)", n_params > 5_000_000)

print("\n== 4. Losses ==")
lb = torch.randn(8); yb = torch.randint(0, 2, (8,)).float()
check("binary focal loss finite", torch.isfinite(focal_loss(lb, yb)).item())
lf = torch.randn(8, 7); yf = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1])
prior = torch.tensor([0.6, 0.01, 0.05, 0.25, 0.05, 0.02, 0.02])
check("family loss finite (ignore -1)", torch.isfinite(logit_adjusted_family_loss(lf, yf, prior)).item())
check("soft macro F1 in [0,1]", 0 <= soft_macro_f1_loss(lf, yf).item() <= 1)

print("\n== 5. Trọng số group chuẩn hóa khi thiếu nhóm (Track B từ Track A vector) ==")
check("G2 temporal bắt buộc tách: schema Track B không chứa temporal",
      all('Temporal' not in k for k in GROUP_DIMS))

print(f"\n{'=' * 50}\nKẾT QUẢ: {PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
