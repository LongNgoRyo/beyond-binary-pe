"""
analysis/statistical_analysis.py — Phân tích thống kê (PDF §II.8, B15–B17).
Đầu vào: full_results.json + per-seed prediction files từ Kaggle (khi kernel xong).
Chạy: DeLong test (ensemble vs CS-LGBM binary), paired bootstrap 1000 lần,
per-class F1 95% CI, low-FPR operating points, TPR@FPR kèm CI.
CHỈ dùng số thật từ pipeline đã audit — không seed trước kết quả.
"""
import json, os, sys
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "kaggle")
FULL = os.path.join(OUT_DIR, "full_results.json")


# ── DeLong test cho 2 correlated ROC curves ──
def delong_roc_test(y_true, scores1, scores2):
    """DeLong et al. 1988 — so sánh AUC 2 mô hình trên CÙNG tập mẫu (paired)."""
    from scipy import stats
    y = np.asarray(y_true).astype(int)
    s1, s2 = np.asarray(scores1), np.asarray(scores2)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    def midrank(x):
        J = np.argsort(x)
        Z = x[J]
        i = 0; ranks = np.empty(len(x), dtype=float)
        while i < len(x):
            j = i
            while j + 1 < len(x) and Z[j + 1] == Z[i]:
                j += 1
            r = (i + j + 2) / 2.0
            ranks[J[i:j + 1]] = r
            i = j + 1
        return ranks
    # structural components (theo DeLong–DeLong–Clarke-Pearson)
    def structural_components(scores):
        phi = np.zeros((n1 + n0, 2))
        pos = scores[y == 1]; neg = scores[y == 0]
        ranks_pos = midrank(np.concatenate([pos, neg]))[:n1]
        ranks_neg = midrank(np.concatenate([pos, neg]))[n1:]
        phi[:n1, 0] = ranks_pos - np.arange(1, n1 + 1)
        phi[n1:, 1] = ranks_neg - np.arange(1, n0 + 1)
        return phi
    phi1 = structural_components(s1); phi2 = structural_components(s2)
    def auc_cov(phi):
        m = phi.mean(axis=0)
        cov = (phi - m).T @ (phi - m) / (len(phi) - 1)
        return m, cov
    m1, c1 = auc_cov(phi1); m2, c2 = auc_cov(phi2)
    # paired covariance
    m12 = (phi1 + phi2) / 2
    _, c12 = auc_cov(m12)
    auc1 = m1[0] / (n1 * n0) + 0.5 / n0 * 0  # V10 mean / n0*n1 — công thức chuẩn:
    auc1 = (phi1[:n1, 0].mean() - phi1[n1:, 1].mean() + n0 * 1.0) / 2.0 if False else None
    # đơn giản, an toàn: AUC bằng scipy
    from sklearn.metrics import roc_auc_score
    auc1 = roc_auc_score(y, s1); auc2 = roc_auc_score(y, s2)
    # phương sai paired qua structural components
    v1 = c1[0, 0] / (n1 * n0 ** 2) + c1[1, 1] / (n1 ** 2 * n0)
    v2 = c2[0, 0] / (n1 * n0 ** 2) + c2[1, 1] / (n1 ** 2 * n0)
    # cross-cov
    cs = (c12 * 2) - c1 - c2
    v12 = cs[0, 0] / (n1 * n0 ** 2) + cs[1, 1] / (n1 ** 2 * n0)
    var_diff = v1 + v2 - 2 * v12
    z = (auc1 - auc2) / np.sqrt(max(var_diff, 1e-15))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc1": float(auc1), "auc2": float(auc2), "z": float(z), "p": float(p)}


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_boot=1000, seed=0, average="macro"):
    """Paired bootstrap 1000 lần cho chênh lệch Macro-F1 (đúng §II.8: 1000 lần)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], pred_a[idx], average=average, zero_division=0)
        fb = f1_score(y_true[idx], pred_b[idx], average=average, zero_division=0)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "p_lt_0": float((diffs < 0).mean())}


def per_class_f1_ci(y_true, y_pred, n_boot=1000, seed=0, labels=range(7)):
    """95% bootstrap CI cho từng lớp (checklist §II.5: Ransomware/Dropper/Keylogger)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    out = {}
    for c in labels:
        base = f1_score(y_true == c, y_pred == c, zero_division=0)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(f1_score(y_true[idx] == c, y_pred[idx] == c, zero_division=0))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[int(c)] = {"f1": float(base), "ci95": [float(lo), float(hi)],
                       "support": int((y_true == c).sum())}
    return out


def low_fpr_operating_points(y_true, scores, fpr_targets=(0.01, 0.001, 0.0001)):
    """TPR@FPR + số false alerts thực tế (checklist §II.7) — threshold từ VALIDATION,
    ở đây tính trực tiếp trên ROC của tập được cho (dùng cho test sau khi tau lấy từ val)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, scores)
    out = {}
    n_neg = int((np.asarray(y_true) == 0).sum())
    for t in fpr_targets:
        i = np.searchsorted(fpr, t, side="right") - 1
        i = max(0, min(i, len(fpr) - 1))
        out[str(t)] = {"tpr": float(tpr[i]), "fpr_realized": float(fpr[i]),
                       "tau": float(thr[i]) if i < len(thr) else None,
                       "false_alerts_per_100k_neg": float(fpr[i] * 100000)}
    return out


def main():
    if not os.path.exists(FULL):
        print("full_results.json chưa có — chạy kernel Kaggle trước.")
        sys.exit(1)
    with open(FULL) as f:
        full = json.load(f)
    per_seed = full["per_seed"]
    seeds = sorted(per_seed.keys())
    agg = full.get("aggregate", {})

    # 1) seed variability (B15: tách variability giữa 5 training seeds)
    print("=" * 60)
    print("1) SEED VARIABILITY (5 seeds)")
    for k, v in agg.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # 2) low-FPR operating points per seed
    print("\n2) LOW-FPR OPERATING POINTS (per seed)")
    for s in seeds:
        r = per_seed[s]
        print(f"  seed {s}: TPR@1%={r.get('tpr_at_1pct'):.4f} (FPR={r.get('fpr_at_1pct'):.4f}) | "
              f"TPR@0.1%={r.get('tpr_at_01pct'):.4f} (FPR={r.get('fpr_at_01pct'):.4f})")

    # 3) per-class CI từ confusion matrices lưu sẵn
    print("\n3) PER-CLASS F1 (từ confusion matrix seed trung bình)")
    # cm = np.array(per_seed[seeds[0]]["confusion_gated"]) — chi tiết bootstrap cần predictions;
    # kernel lưu cm per seed, CI bootstrap chạy trên predictions npz nếu có (kaggle_results/*.npz)

    # 4) nếu có predictions npz (p_mal test + labels) → DeLong + paired bootstrap
    npz_candidates = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
    if npz_candidates:
        for f in npz_candidates:
            d = np.load(os.path.join(OUT_DIR, f), allow_pickle=True)
            if "y_bin_test" in d and "y_score_ensemble" in d and "y_score_cs_lgbm" in d:
                print(f"\n4) DELONG + PAIRED BOOTSTRAP trên {f}")
                dl = delong_roc_test(d["y_bin_test"], d["y_score_ensemble"], d["y_score_cs_lgbm"])
                print("  DeLong:", dl)
                if "y_pred_fam_gated" in d and "y_pred_fam_lgbm" in d and "y_fam_test" in d:
                    m = d["y_fam_test"] >= 0
                    pb = paired_bootstrap_f1(d["y_fam_test"][m], d["y_pred_fam_gated"][m],
                                             d["y_pred_fam_lgbm"][m])
                    print("  Paired bootstrap Macro-F1 diff:", pb)
                lf = low_fpr_operating_points(d["y_bin_test"], d["y_score_ensemble"])
                print("  Low-FPR points:", json.dumps(lf, indent=2))
    else:
        print("\n4) Chưa có predictions npz — cần kernel lưu thêm per-seed scores.")

    # 5) ghi bảng đối chiếu (B17)
    table = {
        "metric": ["roc_auc", "macro7_gated", "macro6_malware_only", "tpr_at_1pct", "tpr_at_01pct"],
        "mean": [agg.get(k, {}).get("mean") for k in
                 ["roc_auc", "macro7_gated", "macro6_malware_only", "tpr_at_1pct", "tpr_at_01pct"]],
        "std": [agg.get(k, {}).get("std") for k in
               ["roc_auc", "macro7_gated", "macro6_malware_only", "tpr_at_1pct", "tpr_at_01pct"]],
        "note": "Trạng thái: VERIFIED từ full_results.json — mọi số cũ trong manuscript UNVERIFIED",
    }
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("\nSaved:", os.path.join(OUT_DIR, "stats_summary.json"))


if __name__ == "__main__":
    main()
