# HolisticPE — Phân loại malware PE đa nhiệm vụ trên EMBER2024 (Win64)

Dự án nghiên cứu: phân loại PE malware **beyond binary** — không chỉ dừng ở benign/malware,
mà gộp 3 nhiệm vụ (binary + 7 họ family + 8 hành vi behavior) qua **hierarchical gating Eq.(5)**
và ensemble **GT-Transformer + LightGBM**. Data: EMBER2024 Win64, protocol chronological
(train tuần 1–48, val 49–52, test 53–64).

## Cấu trúc dự án

```
MAlwarePEHermes/
├── src/                       ← Module lõi (import chung cho local)
│   ├── feature_extraction.py  ← Extractor 623-dim static (G1–G8) + 8 temporal (G9)
│   └── model_core.py          ← Gating Eq.(5), GT-Transformer, các loss, schema Table 1
├── experiments/               ← 6 kernel Kaggle (self-contained, đẩy lên là chạy)
│   ├── k1_full_experiment/    ← Kernel 1: thí nghiệm chính 5 seeds
│   ├── k2_ablation_baselines/ ← Kernel 2: ablation + 20 baselines + disagreement
│   ├── k3_xai_behavior_drift/ ← Kernel 3: XAI + behavioral + temporal drift
│   ├── k4_openset_latency/    ← Kernel 4: packed/evasive + open-set + latency (CPU)
│   ├── k5_retraining_faithfulness/ ← Kernel 5 v1: retraining + XAI faithfulness
│   └── k5b_curriculum/        ← Kernel 5 v3 (bản đã push): curriculum 5 seeds
├── analysis/                  ← Tool chạy local (đọc data/, ghi results/)
│   ├── pilot_local.py         ← Pilot CPU: verify pipeline trước khi chạy lớn
│   ├── behavioral_baselines.py← Baseline 8 nhãn behavior (all-positive/majority/LGBM)
│   └── statistical_analysis.py← DeLong test + paired bootstrap + low-FPR points
├── tests/
│   └── test_model_core.py     ← 26 unit test (gating, transformer, loss, schema)
├── data/
│   ├── Win64_train/           ← 17/48 tuần train thật (xem "Trạng thái data")
│   └── Win64_test/            ← 12/12 tuần test (2024-09-22 → 2024-12-08)
├── results/
│   ├── kaggle/                ← Kết quả VERIFIED từ Kaggle (JSON + NPZ per seed)
│   └── pilot/                 ← Kết quả pilot local (chỉ để verify, không dùng paper)
├── docs/
│   ├── MODEL.md               ← Mô tả chi tiết mô hình HolisticPE
│   └── PROJECT_NOTES.md       ← Notes "kiểu tác giả": workflow, kiến trúc, kết quả, bài học
├── archive/                   ← Symlink → data/ (tương thích script cũ)
└── README.md
```

## Chi tiết từng file

### src/ — module lõi

| File | Vai trò |
|---|---|
| `feature_extraction.py` | Parse JSONL EMBER2024 → vector 623 dims (Track B: 8 nhóm đặc trưng static) + 8 dims temporal (G9, chỉ Track A). Mapping nhãn family 7 lớp (severity-priority rulebook) + nhãn behavior 8 lớp đọc trực tiếp từ field `behavior` gốc EMBER2024. **Không dùng** sha256/md5/tlsh/detection_ratio (audit F1/F4/F5). |
| `model_core.py` | Thư viện HolisticPE: `hierarchical_gating()` (Eq.5: p_joint,0=(1−p_mal)·p_comb,0; p_joint,c=p_mal·p_comb,c), `gated_family_from_binary()` (nhất quán binary→family), `GTTransformer` (6-layer Transformer d=256, ~5.4M params, 2 track A/B), các loss: binary focal, family logit-adjusted + focal γ=2.5, soft macro-F1 differentiable. |

### experiments/ — kernel Kaggle (mỗi folder 1 kernel)

Mỗi kernel là **single-file self-contained**: đã nhúng sẵn extractor + model_core bên trong,
tự discovery data bằng `glob("/kaggle/input/**/Win64_train")` — attach dataset
`weiweip/ember2024` là chạy. Nội dung code giữ nguyên từng byte so với bản ĐÃ CHẠY
trên Kaggle (kết quả trong `results/kaggle/` là output thật của chính các file này).

| Folder (file) | ID Kaggle | Nội dung |
|---|---|---|
| `k1_full_experiment/` | beyond-binary-revision-full-experiment | Thí nghiệm chính: 5 training seeds × (GT-Transformer + LightGBM + gating), baselines CS-LightGBM, low-FPR TPR@1%/0.1%, per-class F1 + CI. GPU. |
| `k2_ablation_baselines/` | beyond-binary-ablation-baselines | Fairness baseline: CS-CatBoost, Tree-Tree (LGBM+CatBoost cùng gating), GT standalone. Ablation B3 (bỏ focal/logit-adjust/soft-F1/behavior head/fusion). Disagreement analysis B4. Behavioral multi-task đầy đủ B10–B12. GPU. |
| `k3_xai_behavior_drift/` | beyond-binary-xai-behavior-drift | XAI permutation importance theo nhóm G1–G9, behavioral tagging chi tiết, temporal drift theo tuần (G2-only shortcut check). CPU. |
| `k4_openset_latency/` | beyond-binary-external-openset | Generalization ngoài phân phối (packed/evasive), open-set OOD (energy/max-prob), đo latency per-sample. CPU. |
| `k5_retraining_faithfulness/` | beyond-binary-retraining-curriculum-v1 | No-update vs periodic retraining (mục 10.3) + XAI faithfulness feature-dropping. GPU. |
| `k5b_curriculum/` | beyond-binary-retraining-curriculum | Bản v3 đã push cuối: 13.3 curriculum 5 seeds; 10.3/14.4 dùng số verified từ v1. GPU. |

Mỗi folder có `kernel-metadata.json` — push bằng:
`cd experiments/<folder> && kaggle kernels push`

### analysis/ — tool local

| File | Vai trò |
|---|---|
| `pilot_local.py` | Pilot nhỏ trên CPU (6 tuần train × 4000 + 2 val × 2000 + 2 test × 4000) xác minh pipeline: LightGBM binary → family 7-class cost-sensitive → ensemble + gating Eq.(5) → G2-only shortcut check. Output chỉ để verify, KHÔNG dùng cho manuscript. |
| `behavioral_baselines.py` | Đối chứng 8 nhãn behavior: all-positive baseline, majority baseline, single-task LightGBM per behavior. Trả lời "behavior head có giá trị tăng thêm?" (8/8 vượt baseline ở pilot). |
| `statistical_analysis.py` | Đọc `results/kaggle/`: seed variability (5 seeds), DeLong test ensemble vs CS-LGBM, paired bootstrap 1000 lần Macro-F1, per-class F1 95% CI, low-FPR operating points. |

### tests/

| File | Vai trò |
|---|---|
| `test_model_core.py` | 26 unit test: schema 623/631/8 nhóm đúng Table 1, gating Eq.(5) các trường hợp biên (p=0, p=1, NaN, theta nén điểm), nhất quán binary/family, GT-Transformer forward 2 track, 3 loss finite. Chạy: `python3 tests/test_model_core.py` |

### results/

| Thư mục | Vai trò |
|---|---|
| `kaggle/` | **VERIFIED** — output thật của 6 kernel đã chạy trên Kaggle: `full_results.json` (5 seeds, aggregate mean±std), `preds_seed*.npz` (predictions per seed cho DeLong/bootstrap), `ablation_baselines_results.json`, `behavioral_multitask_full.json`, `xai_permutation_importance.json`, `temporal_drift.json`, `external_openset_latency.json`, `kernel5_results.json`, `missing_metrics.json`. |
| `pilot/` | Output pilot local (pilot_summary.json, behavioral_baselines.json) — chỉ dùng cross-check, không đưa vào paper. |

## Chạy

```bash
# 1. Unit test (không cần data)
python3 tests/test_model_core.py

# 2. Pilot local (cần data 17 tuần + lightgbm)
python3 analysis/pilot_local.py
python3 analysis/behavioral_baselines.py

# 3. Thống kê từ kết quả Kaggle đã có
python3 analysis/statistical_analysis.py

# 4. Chạy thí nghiệm lớn → Kaggle
cd experiments/k1_full_experiment && kaggle kernels push
# (lặp tương tự cho k2…k5b; GPU T4/P100 cho k1/k2/k5*, k3/k4 CPU)
```

## Trạng thái data (EMBER2024 Win64, dataset `weiweip/ember2024`)

- **Win64_train: 17/48 tuần thật** = 2023-09-24→2023-12-03 (11 tuần) + 2024-08-11→2024-09-21 (6 tuần)
- **Thiếu:** 2023-12-10→2024-01-20 (6 tuần ~3.3GB) + 2024-01-21→2024-08-04 (16 tuần)
- **Win64_test: đủ 12/12** (2024-09-22→2024-12-08), đã verify
- Tải nốt (resumable): `bash /tmp/redl.sh` hoặc per-file:
  `kaggle datasets download -d weiweip/ember2024 --file "Win64_train/<tên_file>" -p data/Win64_train --unzip`
- **Lưu ý:** kết quả `results/kaggle/` là từ full 48 tuần đã chạy TRƯỚC khi data local bị mất —
  không cần train lại; số này chỉ cần khi tái hiện trên máy khác.

## Dataset gốc
https://www.kaggle.com/datasets/weiweip/ember2024
