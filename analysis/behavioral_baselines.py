"""
analysis/behavioral_baselines.py — Đối chứng behavioral tagging (PDF §II.6, B10–B12).
Chạy trên pilot data local (weeks 1-8 train, 53-54 test):
  - All-positive baseline (mọi nhãn = 1)
  - Majority baseline (theo prevalence)
  - Single-task LightGBM per behavior
  - Negative-transfer check: so sánh multi-task vs single-task
Điểm quan trọng (audit F5): nhãn behavior giờ đến từ trường `behavior` GỐC EMBER2024,
không phải suy diễn từ family → prevalence thấp hơn nhiều (3-53/3000).
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # project root
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from collections import Counter

from src.feature_extraction import parse_row, BEHAVIOR_NAMES

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pilot")
TRAIN_FILES = sorted(glob.glob(os.path.join(BASE, "Win64_train", "*.jsonl")))
TEST_FILES = sorted(glob.glob(os.path.join(BASE, "Win64_test", "*.jsonl")))

def load(files, per_file):
    X, Y = [], []
    for fp in files:
        cnt = 0
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if cnt >= per_file: break
                try:
                    d = json.loads(line)
                    x623, _, yb, yf, ybeh, sha, wk = parse_row(d)
                    X.append(x623); Y.append(ybeh)
                    cnt += 1
                except Exception: continue
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("Loading behavioral pilot data...")
Xtr, Htr = load(TRAIN_FILES[:8], 4000)     # weeks 1-8
Xt, Ht = load(TEST_FILES[:2], 4000)        # weeks 53-54
print(f"train {Xtr.shape}, test {Xt.shape}")

results = {}
print(f"\n{'Behavior':<20} {'N_pos(test)':>10} {'AllPos-F1':>10} {'Majority-F1':>12} {'LGBM-F1':>9} {'LGBM-AUROC':>10} {'LGBM-AUPRC':>10}")
for i, name in enumerate(BEHAVIOR_NAMES):
    ytr, yt = Htr[:, i], Ht[:, i]
    n_pos = int(yt.sum())
    prev = float(yt.mean())
    # baselines
    all_pos = np.ones_like(yt)
    f1_ap = f1_score(yt, all_pos, zero_division=0)
    majority = np.zeros_like(yt) if prev < 0.5 else np.ones_like(yt)
    f1_maj = f1_score(yt, majority, zero_division=0)
    # single-task LGBM
    if ytr.sum() >= 20:
        m = lgb.train({"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                       "num_leaves": 63, "verbose": -1, "seed": 3407,
                       "scale_pos_weight": max(1.0, (len(ytr) - ytr.sum()) / max(1, ytr.sum()))},
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=200)
        p = m.predict(Xt)
        yp = (p >= 0.5).astype(int)
        f1_l = f1_score(yt, yp, zero_division=0)
        auroc = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else float("nan")
        auprc = average_precision_score(yt, p) if len(np.unique(yt)) > 1 else float("nan")
    else:
        f1_l, auroc, auprc = float("nan"), float("nan"), float("nan")
    results[name] = {"n_pos_test": n_pos, "prevalence_test": prev, "allpos_f1": float(f1_ap),
                     "majority_f1": float(f1_maj), "lgbm_f1": float(f1_l),
                     "lgbm_auroc": float(auroc) if auroc == auroc else None,
                     "lgbm_auprc": float(auprc) if auprc == auprc else None}
    print(f"{name:<20} {n_pos:>10} {f1_ap:>10.4f} {f1_maj:>12.4f} {f1_l:>9.4f} {auroc:>10.4f} {auprc:>10.4f}")

# tóm tắt negative-transfer / giá trị gia tăng
print("\n=== ĐÁNH GIÁ (checklist §II.6: behavior có giá trị tăng thêm?) ===")
beats_ap = sum(1 for r in results.values() if r["lgbm_f1"] is not None and np.isfinite(r["lgbm_f1"]) and r["lgbm_f1"] > max(r["allpos_f1"], r["majority_f1"]))
total = sum(1 for r in results.values() if r["lgbm_f1"] is not None and np.isfinite(r["lgbm_f1"]))
print(f"Số behavior LGBM vượt cả all-positive VÀ majority: {beats_ap}/{total}")

with open(os.path.join(OUT, "behavioral_baselines.json"), "w") as f:
    json.dump({"note": "PILOT local (weeks 1-8 train, 53-54 test) — minh hoạ protocol; số cuối cùng từ Kaggle full run",
                "results": results, "beats_baselines": f"{beats_ap}/{total}"}, f, indent=2)
print("Saved:", os.path.join(OUT, "behavioral_baselines.json"))
