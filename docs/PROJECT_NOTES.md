# PROJECT_NOTES — HolisticPE / MAlwarePEHermes
### Notes "kiểu tác giả" — đọc xong hiểu project như người làm ra nó

> File này là bản tóm tắt tri thức đầy đủ về project: workflow, kiến trúc, mô hình,
> thí nghiệm, kết quả, lịch sử phát triển và những cái bẫy đã gặp.
> Phân biệt với 2 file kia:
> - `README.md` — bản đồ cấu trúc + cách chạy (ngắn, thao tác)
> - `docs/MODEL.md` — mô tả mô hình cho người đọc paper (kiến thức, không lịch sử)
> - `docs/PROJECT_NOTES.md` (file này) — "não bộ" project: tại sao làm, làm thế nào, kết quả nào là gì.

---

## 0. Bối cảnh & mục tiêu nghiên cứu

**Bài toán gốc:** phân loại PE (Portable Executable) malware bị "tê liệt ở binary".
Truyền thống chỉ trả lời "benign hay malware". Nhưng trong vận hành thật (SOC, IR):

- Câu hỏi binary không đủ: biết là malware rồi, rồi sao? Thuộc họ nào để hành động?
- Metric binary đã bão hòa (AUC ~0.99+ trên EMBER) → dư địa nghiên cứu nằm ở tầng sau.
- Nhãn family trong data THỰC không sạch: AV phát hiện "Trojan.Win32.Agent.x"
  không tương đương family học thuật → cần taxonomy tự xây.

**Ý tưởng "beyond binary" (tên mã: HolisticPE):** một pipeline trả lời ĐỒNG THỜI
3 nhiệm vụ với **một cơ chế nhất quán** thay vì 3 model rời:

1. **Binary**: benign / malware (nền tảng)
2. **Family**: 7 lớp [Benign, Ransomware, Keylogger, Trojan_RAT, Worm_Virus, Other_Rare, Dropper]
3. **Behavior**: 8 nhãn multi-label [file_ops, network, registry, process_injection, persistence, encryption, keylogging, screenshot]

Điểm mới cốt lõi = **hierarchical gating Eq.(5)**: family head bị binary head
"khóa" lại — không thể vừa kết luận benign vừa gán family ≠ Benign.
Đó là tính nhất quán logic mà hệ 3 model rời không đảm bảo.

**Data:** EMBER2024 (Kaggle `weiweip/ember2024`), phiên bản Win64, JSONL theo tuần.
**Protocol:** chronological — train tuần 1–48, val 49–52, test 53–64 (test mới hơn train,
mô phỏng triển khai thật). Quyết định thiết kế quan trọng: **không shuffle theo mẫu**,
chỉ shuffle theo tuần, vì concept drift PE malware theo thời gian là cực mạnh.

---

## 1. Kiến trúc logic (đọc cái này trước khi đọc code)

```
JSONL EMBER2024 (theo tuần)
   │
   ▼
parse_row()                      ← src/feature_extraction.py
   ├─ extract_static()  → x_623 (8 nhóm G1–G8)
   ├─ extract_temporal()→ x_g9  (8 dims, Track A)
   ├─ map_family()      → y_fam (7 lớp, -1 = unknown)
   ├─ map_behavior()    → y_beh (8 nhãn 0/1)
   └─ label, sha256, week_id
   │
   ▼
GIAO THỨC (load_files trong kernel)
   ├─ train W1–48 (subsample 20k/file), val W49–52 (20k/file), test W53–64 (full)
   ├─ leakage check: giao sha256 train∩test phải = 0
   └─ splits_meta.npz ghi n_train/n_val/n_test + hash danh sách train
   │
   ▼
Mỗi seed (42, 123, 3407, 2024, 9999):
   ├─ GT-Transformer (torch, GPU) — Track B static_only
   │     loss = 1.0·focal(binary) + 1.2·[logit_adj_fam(τ=1.2,γ=2.5) + 2·soft_F1] + 0.4·BCE(behavior)
   ├─ LightGBM binary (300 rounds) + LightGBM family 7-class cost-sensitive (250 rounds)
   ├─ FUSION:
   │     p_mal = 0.5·p_GT + 0.5·p_LGBM     (p_GT = sigmoid(0.4·l_bin + 0.6·(lse(lf[1:]) − lf[0])))
   │     p_comb = 0.5·softmax(lf_GT) + 0.5·proba_LGBM_family
   ├─ CALIBRATION trên val:
   │     tau_1pct = quantile 99% của p_mal trên val NEGATIVE
   │     tau_01pct = quantile 99.9%
   │     theta[7] tune bằng coordinate descent (8 giá trị 0.2→2.0) trên val macro-F1
   └─ TEST inference → hierarchical_gating(p_mal, p_comb, theta) → gated_family_from_binary
         │
         ▼
   metrics: roc_auc, macro7_gated, macro6_malware_only, TPR@1%, TPR@0.1%,
            confusion matrix, theta, thời gian train
         │
         ▼
   results/kaggle/full_results.json + preds_seed{seed}.npz (lưu per-seed để paired test)
   │
   ▼
analysis/statistical_analysis.py: seed variability + DeLong + paired bootstrap 1000×
```

Điểm dễ nhầm khi đọc code: **binary score của ensemble không phải sigmoid(l_bin)**
mà là `_gt_mal_score`: `sigmoid(0.4·l_bin + 0.6·(logsumexp(lf[1:]) − lf[0]))`.
Lý do: family head còn cho biết "có bao nhiêu mass malware" — logsumexp của 6 family
trừ đi logit Benign là một dạng binary-ness thu được miễn phí từ family head. Hệ số
0.4/0.6 và w_ens=0.5 là hợp điểm thiết kế (không tune trên test).

---

## 2. Đặc trưng — schema Table 1 (623 + 8)

Extractor: `src/feature_extraction.py` (`extract_static`, `extract_temporal`).

| Nhóm | Dims | Nội dung | Chi tiết xử lý |
|---|---|---|---|
| G1 ByteHistogram | 256 | histogram byte 0–255 | sum-normalize |
| G2 ByteEntropy | 256 | joint hist (byte × entropy window) | sum-normalize (dữ liệu gốc là counts, KHÔNG phải entropy 0–8) |
| G3 StringsStats | 16 | 4 base (numstrings, printables, avlength, entropy) + 12 string_counts log-norm | **2 dims printabledist bị cắt** (tính 18, slice [:16]) — chi tiết dở dấu của v2, đã test, không ảnh hưởng | 
| G4 GeneralStats | 10 | size/entropy/is_pe/start_bytes + 6 flags COFF/authenticode | |
| G5 HeaderStats | 24 | 20 optional-header fields log-norm + timestamp + e_lfanew + symbols + dll_char | |
| G6 SectionStats | 25 | n_sec + 4×(ent,size,vsize) stats + top-4 entropy + overlay(3) + 3 props count + flag entry + entropy section entry | assert đúng 25 |
| G7 ImportsExports | 20 | n_dll/n_func/4 DLL trọng yếu/exports (7 dims) + 13 API flags nhạy cảm | VirtualAllocEx, WriteProcessMemory, CreateRemoteThread, SetWindowsHookEx, GetAsyncKeyState, InternetOpen, URLDownloadToFile, WinExec, ShellExecute, RegSetValue, CreateToolhelp32Snapshot, IsDebuggerPresent, CheckRemoteDebuggerPresent |
| G8 DataDirectories | 16 | 16 dir có/không; >16 gộp vào bit 15 | |
| **Tổng Track B** | **623** | | |
| G9 Temporal (Track A) | 8 | sign·log1p(dd), log1p(days_alive), d1yr, sin/cos dow, sin/cos woy, weekend flag | dd = (first_submission − compile)/day |

**Quy tắc audit (không vi phạm khi chỉnh sửa):**
- **F1/F4**: KHÔNG dùng sha256/md5/tlsh/detection_ratio/file_type hash — hash là ID nên leak.
- **F5**: nhãn behavior đọc TRỰC TIẾP từ field `behavior` của EMBER2024, không suy diễn từ family (chống circular label).
- Mọi scaling dùng `_lognorm(x, scale) = clip(log1p(max(0,x))/log1p(scale), 0, 1)` — nan→0, +inf→1, −inf→0.
- `assert` ở cuối `extract_static`: tổng phải đúng 623 (đổi schema phải đổi assert + GROUP_DIMS + test).

**Nhãn family — severity-priority rulebook** (`FAMILY_RULEBOOK`, `SEVERITY_ORDER = [1,2,6,4,5,3]`):
Ransomware > Keylogger > Dropper > Worm/Virus > Other/Rare > Trojan/RAT.
Mẫu khớp nhiều nhóm → nhóm nghiêm trọng hơn thắng. Không khớp → −1 (unknown,
bị loại khỏi loss/eval bằng ignore mask). `label=0` → Benign bất kể family string.
`map_family` nhận `conf` nhưng chỉ dùng để lọc ở tầng eval — trong hàm hiện là pass (chỉnh khi cần).

**Nhãn behavior** — `BEHAVIOR_MAP` ánh xạ tags gốc EMBER2024 → 8 nhãn paper.
Dòng 55 của feature_extraction (`'coinminer': False if False else set()`) là dead-code
còn sót từ lúc đếm tags thực tế — vô hại, đừng "dọn dẹp" trong bản byte-identical.

---

## 3. Mô hình

### 3.1 GT-Transformer (`src/model_core.py`, ~5.4M params)

- 8 group tokens (Track B) / 9 (Track A) — mỗi nhóm qua MLP 2-layer (d_model→256→GELU→dropout→256→GELU) thành 1 token.
- +1 token CLS (learnable) + positional embedding learnable.
- Encoder: 6 layers × (d=256, 8 heads, ff=1024, dropout 0.25, activation GELU, **norm_first=True** pre-LN).
- Output: lấy token [CLS] qua LayerNorm → 3 heads:
  - `bin_head`: Linear(256→1)
  - `fam_head`: Linear(256→7), **nhân fam_scale (learnable, init 16.0)** — logit scale, margin áp ở loss
  - `beh_head`: Linear(256→8)
- Hai track chỉ khác `n_groups`/`dims` (static_only=True → 623, False → 631).
  Trong Kaggle kernel thực tế chỉ train **Track B** (test là mẫu "mới hoàn toàn",
  temporal của test bị bóp về 0 khi sample chưa từng thấy — quyết định thiết kế có chủ ý).

### 3.2 Ensemble + Gating Eq.(5)

```
p_joint,0 = (1 − p_mal) · 1                       (Benign: binary quyết định)
p_joint,c = p_mal · p_cond(c | malware)           (c ≥ 1, renormalize trong khối malware)
```

Lưu ý bản gốc Eq.(5) nhân trực tiếp p_combined chưa điều kiện → tổng ≠ 1.
`hierarchical_gating()` sửa bằng **renormalize có điều kiện**:
khối malware riêng chuẩn hóa thành P(c|malware), khối benign giữ nguyên.
Test đã xác minh: tổng = 1 mọi p_mal, p_mal=0 ⇒ joint0=1, p_mal=1 ⇒ joint0=0.

`gated_family_from_binary()`: Step 4 của paper — binary=benign → ép family=0;
binary=malware mà argmax=Benign → fallback chọn malware argmax c∈{1..6} (theo θ).

**θ (theta) — nén điểm per-class:** score = p_joint/θ; θ lớn = "thí" lớp đó.
Tune trên val (coordinate descent từng lớp 1–6, 8 giá trị 0.2–2.0, tối đa macro-F1).
Tune trên VAL, không phải test — giữ đúng protocol.

### 3.3 Loss (tất cả trong model_core.py)

| Loss | Cho | Tham số |
|---|---|---|
| `focal_loss` | binary | γ=2.0, α=0.5, label_smoothing=0.05 |
| `logit_adjusted_family_loss` | family | τ=1.2 (logit adjustment log-prior), focal γ=2.5, class-balanced β=0.9999, ignore −1 |
| `soft_macro_f1_loss` | family | differentiable macro-F1 trên 7 lớp, weight 2.0 trong tổng loss |
| BCE với logits | behavior | 8 nhãn độc lập |

Tổng loss trong kernel k1: `1.0·focal_bin + 1.2·(logit_adj + 2·soft_f1) + 0.4·BCE_beh`.
Mọi mẫu y_fam=−1 bị mask khỏi 2 loss family (nhưng vẫn tham gia binary/behavior).

**Dòng 152 model_core.py** (`lf_m, yf_m = lf[mask], yf_m = lf[mask], yf[mask]`):
Python parse thành chained assignment → tự-gán `lf[mask] = lf[mask]` (no-op).
Đã verify chạy thật: vô hại khi lf là non-leaf (output model — trường hợp trong kernel),
chỉ crash khi gọi với leaf requires_grad (kiểu test cũ viết sai cách gọi).
KHÔNG sửa trong bản byte-identical — nó là một phần của code đã sinh kết quả Kaggle.

---

## 4. 6 kernel Kaggle (experiments/)

Tất cả **single-file self-contained**: nhúng extractor + model_core bên trong
(ranh giới module đánh dấu `# MODULE: ember_features_v2.py` / `# MODULE: revision_lib.py`),
tự discovery data `glob("/kaggle/input/**/Win64_train")`. Push: `cd experiments/<k> && kaggle kernels push`.

| Kernel | ID Kaggle | Nội dung | HW | Đã chạy? |
|---|---|---|---|---|
| k1_full_experiment | beyond-binary-revision-full-experiment | Thí nghiệm chính 5 seeds, toàn pipeline §1 | GPU | ✅ → full_results.json + preds_seed*.npz + splits_meta.npz |
| k2_ablation_baselines | beyond-binary-ablation-baselines | Fairness baselines (CS-CatBoost, Tree-Tree, GT standalone) + ablation B3 (no_focal_family / no_logit_adj / no_soft_f1 / no_behavior_head) + disagreement + behavioral full B10–B12 | GPU | ✅ (seed 3407) |
| k3_xai_behavior_drift | beyond-binary-xai-behavior-drift | Permutation importance per G1–G9 (n_perm=3, 42k test) + behavioral tagging chi tiết + drift theo tuần (auc/fpr/tpr mỗi tuần 53–64, 3000 mẫu/tuần) | CPU | ✅ |
| k4_openset_latency | beyond-binary-external-openset | Packed/evasive (7 packers: vmprotect/themida/nsis/obsidium/enigmaprotector/upx/mew), open-set OOD (held-out pushdo/etset — **không đủ mẫu OOD, n_ood=0**), latency (feature 1.41ms, LGBM 0.16ms/sample CPU) | CPU | ✅ |
| k5_retraining_faithfulness | beyond-binary-retraining-curriculum-v1 | No-update vs periodic retrain (gain 0.00022 AUC) + faithfulness deletion (Spearman ρ=0.93 vs permutation) | GPU | ✅ |
| k5b_curriculum | beyond-binary-retraining-curriculum | v3 bản push cuối: curriculum focal+softF1 KHÔNG logit-adj 5 seeds | GPU | ✅ (kết quả thấp hơn: macro7 0.296±0.090 — chứng minh logit-adj là thành phần then chốt) |

**Cấu hình k1 (CFG):** batch 512, lr 3e-4, wd 2e-5, phases 3+4+3 epochs, cosine LR;
LGBM 300 rounds/63 leaves/lr 0.05; subsample train 20k/file × 48 = 960k, val 20k×4 = 80k, test FULL.
**k5b khác:** epochs=10, curriculum loss = focal + 1.2·2·softF1 (cố tình bỏ logit-adj).

**Quan hệ k5/k5b:** k5b là v3 của cùng kernel ID (beyond-binary-retraining-curriculum);
k5 v1 giữ làm nguồn số retraining/faithfulness — k5b đọc `verified_from_v1: true`
để tái dùng các số đó thay vì chạy lại. Đừng xóa k5 v1.

---

## 5. Tool local (analysis/)

| File | Chức năng | Đầu vào | Đầu ra |
|---|---|---|---|
| `pilot_local.py` | Pilot CPU verify pipeline (LightGBM-only ensemble; weeks 1–6 train×4000, 7–8 val, 53–54 test×4000) | data/Win64_train + Win64_test | results/pilot/{pilot_summary.json, pilot_arrays.npz} |
| `behavioral_baselines.py` | Đối chứng 8 nhãn behavior: all-positive, majority, single-task LGBM | same | results/pilot/behavioral_baselines.json |
| `statistical_analysis.py` | Đọc results/kaggle: seed stats, DeLong ensemble vs CS-LGBM (p=0.33), paired bootstrap 1000×, low-FPR | results/kaggle/*.json + preds_seed*.npz | results/kaggle/stats_summary.json |

Kết quả pilot (đã chạy thật, 2026-11-01): LGBM binary AUC 0.9915, gated macro7 0.5951,
macro6 0.4619, **G2-only shortcut AUC 0.9525** — tập đặc trưng byte-entropy một mình
đã đạt AUC 0.95 → drift check ở k3 là bắt buộc, không phải trang trí.

---

## 6. Kết quả VERIFIED (results/kaggle/ — dùng cho paper)

Đây là output THẬT của 6 kernel đã chạy trên Kaggle. Các file:

| File | Từ kernel | Ý nghĩa |
|---|---|---|
| `full_results.json` | k1 | per-seed + aggregate 5 seeds (số chính của paper) |
| `preds_seed{42,123,3407,2024,9999}.npz` | k1 | y_bin_test, y_score_ensemble, y_score_cs_lgbm, y_fam_test, y_pred_fam_gated, pred_lgbm_fam → phục vụ DeLong/paired bootstrap |
| `splits_meta.npz` | k1 | n_train/n_val/n_test + SHA256 của danh sách train (chống nghi ngờ split) |
| `ablation_baselines_results.json` | k2 | seed 3407: CatBoost AUC 0.9956, Tree-Tree macro7 0.652, GT standalone 0.466, proposed 0.483; ablation: no_soft_f1 tệ nhất (0.301), no_logit_adj TỐT HƠN full (0.535!) |
| `behavioral_multitask_full.json` | k2 | 8 nhãn behavior: F1 0.016–0.21 (class hiếm), AUROC 0.59–0.89, **tất cả > all-positive/majority baseline** |
| `xai_permutation_importance.json` | k3 | base AUC 0.9941; quan trọng nhất: G7 (drop 0.0215) > G5 (0.0130) > G6 (0.0080) > G1 (0.0063) > G2 (0.0054) |
| `temporal_drift.json` | k3 | AUC theo tuần 53–64 ổn 0.991–0.996, TPR@0.5 ~0.48–0.50 ổn định — không sụt drift lớn trong 12 tuần test |
| `external_openset_latency.json` | k4 | packed malware vs benign AUC 0.9993 (nhưng fam_macro7_packed 0.293 vs unpacked 0.668 — family classify trên packed gần như vô dụng, honest limitation) |
| `kernel5_results.json` | k5/k5b | retrain gain 0.00022 AUC (không đáng chi phí); curriculum không logit-adj macro7 0.296±0.090 |
| `missing_metrics.json` | tổng hợp sau | binary F1 0.9628±0.0006, precision 0.9816, recall 0.9447, confusion seed3407 (FP 1952 / FN 6752 / 240k) |
| `stats_summary.json` | statistical_analysis | mean±std 5 metrics chính |

**Số "chính" của paper (5 seeds):** ROC-AUC **0.9951±0.0001**; macro7 gated
**0.5114±0.0122**; macro6 **0.3810±0.0136**; TPR@1% **0.9253±0.0021**; TPR@0.1% **0.8204±0.0085**.
DeLong ensemble vs CS-LGBM p=0.33 → không khác biệt có ý nghĩa thống kê (nói thật trong paper).
Baseline Tree-Tree macro7 0.652 CAO HƠN proposed 0.483 ở k2 — kết quả khó,
đã xử lý trong phần discussion của manuscript, không giấu.

**Số pilot local chỉ dùng verify pipeline, KHÔNG đưa vào paper.**

---

## 7. Tests

`tests/test_model_core.py` — 26 test, chạy `python3 tests/test_model_core.py` (không cần data):
- schema 623/631 + 8 nhóm đúng GROUP_DIMS
- gating Eq.(5): tổng=1 mọi p, biên p=0/1, NaN-safe, θ nén đúng hướng
- nhất quán binary/family (gated_family_from_binary)
- GT-Transformer forward 2 track + shape heads
- 3 loss finite + ignore −1 đúng

---

## 8. Data — trạng thái thật

- **Chuẩn protocol cần:** Win64_train 48 tuần (2023-09-24 → 2024-08-04), Win64_test 12 tuần (2024-09-22 → 2024-12-08).
- **Hiện tại `data/` TRỐNG** (người dùng đã biết và yêu cầu bỏ qua 11/2026).
  Bản test 12/12 còn ở `/home/kali/_backup_20261101/tets/archive/Win64_test/` (~1.3GB tổng).
- Lịch sử: 17 tuần train thật từng có (11 tuần 2023-09-24→12-03 tải lại từ Kaggle
  + 6 tuần 2024-08-11→09-21 cứu từ `tets/`), mất 2023-12-10→2024-01-20 (6 file ~3.3GB,
  nguồn Kaggle bị hư hại) và 2024-01-21→2024-08-04. Script resumable đã từng dùng: `/tmp/redl.sh`
  (dùng `/tmp/redl_weeks.txt` 17 tuần + `/tmp/ember_files.txt` size đối chiếu) —
  có thể dùng lại khi cần tái hiện.
- **Kết quả trong results/kaggle/ là từ full 48 tuần chạy trên Kaggle TRƯỚC khi data local mất** —
  hợp lệ độc lập với data local. Không cần train lại.

---

## 9. Workflow phát triển (cách project này ra đời)

1. **Đọc paper gốc** (Beyond Binary, PDF trong `_backup_20261101/`) → manuscript revision
   (Q1–Q5 của reviewer) nằm ở `_backup_20261101/Major Revision Q2 upto Q1 Mr.Tien Long.pdf`.
2. **Xây lại extractor từ đầu trên schema EMBER2024 thật** (v1 dựa trên đoán schema → v2
   verify bằng week 53 thật). Verify = parse 1 tuần, đếm dims từng nhóm, assert 623.
3. **Unit test trước khi train lớn** (26 test) — đặc biệt test gating vì Eq.(5) đã từng sai tổng.
4. **Pilot local CPU** trước khi đốt quota Kaggle GPU (kết quả chỉ để verify, không đưa paper).
5. **Chạy lớn trên Kaggle** (GPU T4/P100 cho k1/k2/k5*, CPU cho k3/k4) — mỗi kernel một câu hỏi nghiên cứu.
6. **Tải kết quả về results/kaggle/** (JSON + NPZ per-seed) — đây là "sự thật", mọi số paper derive từ đây.
7. **Thống kê downstream local** (statistical_analysis) — seed variability + significance.
8. **Bài học lớn: byte-identical.** Kết quả Kaggle chỉ hợp lệ khi code = code đã chạy.
   Mọi rename/ tái cấu trúc phải md5-check từng file trước/sau (đã làm trong đợt rename 11/2026:
   6 kernel + 2 module giữ nguyên từng byte, chỉ đổi tên + sửa import tool local).

---

## 10. Cái bẫy & bài học (đừng lặp lại)

1. **mv đè file cùng tên** — từng làm hỏng file train (placeholder 0-byte đè file thật).
   Luôn kiểm tra collision trước khi mv/merge data.
2. **Không đụng kernel đã chạy** — muốn "cải tiến" gì thì làm bản mới (k5b vs k5), không patch trên bản cũ.
3. **Chained assignment dòng 152** — trông như bug, thực ra no-op; NHƯNG đừng viết code kiểu này.
   Nếu ai muốn sửa: sửa trong bản fork, không sửa file byte-identical.
4. **Dead-code dòng 55 feature_extraction** — vô hại; "dọn dẹp" nó = phá byte-identical.
5. **Theta/focal/softF1 tune trên VAL** — nếu tune trên test là leakage, mọi số 0.51 macro7 thành vô nghĩa.
6. **Open-set k4: n_ood=0** — held-out family pushdo/etset không đủ mẫu sau lọc → phần open-set
   của paper phải được trình bày đúng là "không thể đánh giá trên data này", không fake số.
7. **Family trên packed: macro7 0.293** — giới hạn thật của static-only analysis, ghi vào limitation.
8. **Rename/tái cấu trúc**: user phải yêu cầu rõ ràng; mặc định KHÔNG đụng cấu trúc.
9. **Khi data local hỏng**: kết quả Kaggle vẫn hợp lệ — tách bạch "sự thật đã ghi" khỏi "năng lực tái hiện".

---

## 11. Bản đồ tham chiếu nhanh (file → vai trò)

```
src/feature_extraction.py     extractor 623+8, rulebook family, BEHAVIOR_MAP, parse_row
src/model_core.py             GROUP_DIMS, gating Eq.(5), GTTransformer, 3 loss
experiments/k1..k5b/*.py      6 kernel single-file (nhúng 2 module, DATA_DIR discovery)
experiments/*/kernel-metadata.json  slug/id Kaggle + GPU/CPU flag
analysis/pilot_local.py       pilot CPU (LGBM-only), G2-only shortcut check
analysis/behavioral_baselines.py  3 baseline đối chứng behavior
analysis/statistical_analysis.py  seed stats + DeLong + bootstrap từ results/kaggle
tests/test_model_core.py      26 unit test (không cần data)
results/kaggle/               SỰ THẬT từ Kaggle (JSON + NPZ) — nguồn mọi số paper
results/pilot/                pilot verify (không dùng paper)
docs/MODEL.md                 mô tả mô hình (cho người đọc paper)
README.md                     bản đồ + cách chạy
_backup_20261101/             papers, backup gốc, kết quả JSON gốc, code round-1 (fix đã verify)
```
