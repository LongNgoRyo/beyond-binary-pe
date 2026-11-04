"""
analysis/pilot_local.py — Pilot nhỏ trên local (CPU) để xác minh pipeline trước khi chạy lớn trên Kaggle.
Protocol: tuần 1-48 train, 49-52 val, 53-64 test (chuẩn EMBER2024 chronological).
Pilot dùng: 6 tuần train đầu × 4000 rows + 2 tuần val × 2000 + 2 tuần test × 4000 (RAM giới hạn 3.8GB).
Mọi chỉ số ở đây CHỈ để xác minh pipeline chạy đúng, KHÔNG dùng cho manuscript.
"""
import sys, os, json, glob, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # project root
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from collections import Counter

from src.feature_extraction import parse_row, FAMILY_NAMES

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pilot")
os.makedirs(OUT, exist_ok=True)

TRAIN_WEEKS = sorted(glob.glob(os.path.join(BASE, "Win64_train", "*.jsonl")))   # có 16 file tuần đầu
TEST_WEEKS = sorted(glob.glob(os.path.join(BASE, "Win64_test", "*.jsonl")))     # 12 file
print(f"train files: {len(TRAIN_WEEKS)}, test files: {len(TEST_WEEKS)}")

def load_subset(files, per_file, skip=0):
    X, yb, yf, ybeh, shas, wks = [], [], [], [], [], []
    t0 = time.time()
    for fp in files:
        cnt = 0
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i < skip: continue
                if cnt >= per_file: break
                try:
                    d = json.loads(line)
                    x623, xg9, y_bin, y_fam, y_beh, sha, wk = parse_row(d)
                    X.append(x623); yb.append(y_bin); yf.append(y_fam)
                    ybeh.append(y_beh); shas.append(sha); wks.append(wk)
                    cnt += 1
                except Exception:
                    continue
        print(f"  {os.path.basename(fp)}: +{cnt} rows ({time.time()-t0:.0f}s)")
    return (np.array(X, dtype=np.float32), np.array(yb), np.array(yf),
            np.array(ybeh, dtype=np.float32), shas, np.array(wks))

# ── Pilot data ──
print("Loading pilot TRAIN (weeks 1-6)...")
Xtr, ytr_b, ytr_f, ytr_h, _, _ = load_subset(TRAIN_WEEKS[:6], 4000)
print("Loading pilot VAL (weeks 7-8)...")
Xv, yv_b, yv_f, yv_h, _, _ = load_subset(TRAIN_WEEKS[6:8], 2000)
print("Loading pilot TEST (weeks 53-54)...")
Xt, yt_b, yt_f, yt_h, t_shas, t_wks = load_subset(TEST_WEEKS[:2], 4000)

print("\nPilot sizes:", Xtr.shape, Xv.shape, Xt.shape)
print("train binary:", dict(Counter(ytr_b)), "| test binary:", dict(Counter(yt_b)))
print("train fam:", dict(sorted(Counter(ytr_f.tolist()).items())))
print("test fam:", dict(sorted(Counter(yt_f.tolist()).items())))

# duplicate check (audit F6)
tr_shas = set()  # không lưu shas train để tiết kiệm — nhưng test hash uniqueness:
print("test unique sha256:", len(set(t_shas)), "/", len(t_shas))

# ── Model 1: LightGBM binary (standard) ──
print("\n[1] Standard LightGBM binary")
m1 = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
                'num_leaves': 63, 'max_depth': 8, 'verbose': -1, 'seed': 3407},
               lgb.Dataset(Xtr, label=ytr_b), num_boost_round=300,
               callbacks=[lgb.log_evaluation(0)])
p1 = m1.predict(Xt)
auc1 = roc_auc_score(yt_b, p1)
print(f"  ROC-AUC = {auc1:.4f}")

# ── Model 2: Cost-sensitive LightGBM 7-class family ──
print("\n[2] Cost-sensitive LightGBM family (7-class, ignore -1)")
mask_tr = ytr_f >= 0; mask_te = yt_f >= 0
cnt = Counter(ytr_f[mask_tr].tolist())
total = sum(cnt.values())
w = {c: total / (7.0 * max(1, cnt.get(c, 1))) for c in range(7)}
sw = np.array([w[c] for c in ytr_f[mask_tr]], dtype=np.float32)
m2 = lgb.train({'objective': 'multiclass', 'num_class': 7, 'metric': 'multi_logloss',
                'learning_rate': 0.05, 'num_leaves': 63, 'verbose': -1, 'seed': 3407},
               lgb.Dataset(Xtr[mask_tr], label=ytr_f[mask_tr], weight=sw),
               num_boost_round=250)
p2 = m2.predict(Xt[mask_te])
pred_fam = p2.argmax(1)
mf1_7 = f1_score(yt_f[mask_te], pred_fam, average='macro', zero_division=0)
mal = yt_f[mask_te] > 0
mf1_6 = f1_score(yt_f[mask_te][mal], pred_fam[mal], average='macro', zero_division=0)
print(f"  Macro-F1 (7-class incl benign) = {mf1_7:.4f}")
print(f"  Macro-F1 (malware-only 6-class) = {mf1_6:.4f}")
print("  per-class F1:", [round(float(x), 3) for x in
      f1_score(yt_f[mask_te], pred_fam, average=None, labels=list(range(7)), zero_division=0)])

# ── Model 3: ensemble binary = 0.5*p1 + 0.5*(1-p_lgb[:,0]) + hierarchical gating Eq.(5) ──
print("\n[3] Hierarchical gating Eq.(5) trên pilot")
from src.model_core import hierarchical_gating, gated_family_from_binary
p_mal = 0.5 * p1 + 0.5 * (1.0 - m2.predict(Xt)[:, 0])
p_comb = m2.predict(Xt)
auc_e = roc_auc_score(yt_b, p_mal)
print(f"  Ensemble ROC-AUC = {auc_e:.4f}")
pj, _ = hierarchical_gating(p_mal, p_comb)
y_bin = (p_mal >= 0.5).astype(int)
y_fam_gated = gated_family_from_binary(y_bin, pj, theta=np.ones(7))
mf1_gated = f1_score(yt_f[mask_te], y_fam_gated[mask_te], average='macro', zero_division=0)
mf1_gated6 = f1_score(yt_f[mask_te][mal], y_fam_gated[mask_te][mal], average='macro', zero_division=0)
print(f"  Gated Macro-F1 7 = {mf1_gated:.4f} | malware-only = {mf1_gated6:.4f}")
cm = confusion_matrix(yt_f[mask_te], y_fam_gated[mask_te], labels=list(range(7)))
print("  CM:\n", cm)

# ── Model 4: G2-only baseline (temporal shortcut check — nhiệm vụ A11) ──
print("\n[4] G2-only temporal baseline (shortcut check)")
def load_temporal(files, per_file):
    G, y = [], []
    for fp in files:
        cnt = 0
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if cnt >= per_file: break
                try:
                    d = json.loads(line)
                    _, xg9, y_bin, _, _, _, _ = parse_row(d)
                    G.append(xg9); y.append(y_bin); cnt += 1
                except Exception:
                    continue
    return np.array(G, dtype=np.float32), np.array(y)

Gtr, ytr_t = load_temporal(TRAIN_WEEKS[:6], 4000)
Gt, yt_t = load_temporal(TEST_WEEKS[:2], 4000)
m4 = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
                'num_leaves': 31, 'verbose': -1, 'seed': 3407},
               lgb.Dataset(Gtr, label=ytr_t), num_boost_round=200)
auc_g2 = roc_auc_score(yt_t, m4.predict(Gt))
print(f"  G2-only ROC-AUC = {auc_g2:.4f}  (nếu cao → temporal shortcut có thật)")

# ── lưu kết quả pilot ──
summary = {
    "pilot": True,
    "data": {"train": "weeks 1-6 (6 files, 4000/file)", "val": "weeks 7-8", "test": "weeks 53-54 (2 files, 4000/file)"},
    "n": {"train": int(len(Xtr)), "test": int(len(Xt))},
    "results": {
        "lgbm_binary_auc": float(auc1),
        "lgbm_fam_macro7": float(mf1_7), "lgbm_fam_macro6_malware_only": float(mf1_6),
        "ens_binary_auc": float(auc_e),
        "gated_macro7": float(mf1_gated), "gated_macro6_malware_only": float(mf1_gated6),
        "g2_only_auc": float(auc_g2),
    },
    "note": "PILOT CHỈ ĐỂ VERIFY PIPELINE — không dùng cho manuscript",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(OUT, "pilot_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
np.savez_compressed(os.path.join(OUT, "pilot_arrays.npz"),
                    Xtr=Xtr[:500], yt_b=yt_b[:500])  # mẫu nhỏ để debug
print("\nSaved:", os.path.join(OUT, "pilot_summary.json"))
print(json.dumps(summary["results"], indent=2))
