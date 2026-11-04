# Mô hình HolisticPE — Mô tả chi tiết

> Tài liệu tiếng Việt mô tả kiến trúc và cơ chế của pipeline HolisticPE
> ("Beyond Binary" PE malware analysis). Tham chiếu mã nguồn: `src/`.

## 1. Tổng quan bài toán

Phân loại PE (Portable Executable) malware truyền thống chỉ trả lời 1 câu hỏi:
benign hay malware (binary classification). HolisticPE trả lời đồng thời 3 câu:

1. **Binary**: mẫu này là malware hay benign?
2. **Family**: nếu là malware, thuộc họ nào trong 7 lớp?
3. **Behavior**: nó có 8 hành vi tĩnh nào? (file_ops, network, registry,
   process_injection, persistence, encryption, keylogging, screenshot)

Ba nhiệm vụ này **không độc lập**: family phải nhất quán với binary
(không thể family ≠ Benign khi binary = benign). Điểm mới cốt lõi là
**hierarchical gating Eq.(5)** — lớp family bị binary "khóa" lại.

## 2. Đặc trưng đầu vào (schema Table 1)

Nguồn: JSONL EMBER2024 Win64 — mỗi dòng là 1 mẫu PE có feature dict của EMBER.

### Track B — static 623 dims (8 nhóm G1–G8)

| Nhóm | Dims | Nội dung |
|---|---|---|
| G1 ByteHistogram | 256 | Histogram byte giá trị 0–255 |
| G2 ByteEntropy | 256 | Histogram entropy 2D (byte × entropy 256-byte window) |
| G3 StringsStats | 16 | Thống kê chuỗi (numstrings, avlength, entropy, printable-dist 2 dims aggregate, string_counts log-norm 12 dims…) |
| G4 GeneralStats | 10 | size, entropy, is_pe, start_bytes… |
| G5 HeaderStats | 24 | COFF/optional header (timestamp, machine, sections, subsystem, linker, dll_characteristics, authenticode…) |
| G6 SectionStats | 25 | Section name-hash/size/entropy/props + overlay |
| G7 ImportsExports | 20 | API imports theo DLL (kernel32/user32/ws2_32/advapi32…), exports, flags 12 API đặc trưng (VirtualAllocEx, SetWindowsHookExW, URLDownloadToFile…) |
| G8 DataDirectories | 16 | 16 data directory có/không + flag "thừa" dd>16 |
| **Tổng static** | **623** | |

### G9 — temporal 8 dims (chỉ Track A, tổng 631)

Ngày first_submission/last_analysis chuẩn hóa theo ngày đầu dataset (relative
days, log1p) — đặc trưng thời gian tương đối, **không** phải chữ ký tuần tuyệt đối.
Đây là đường "temporal shortcut" được giám sát riêng (G2-only check).

### Quy tắc audit (tránh data leakage)

- **F1/F4**: KHÔNG dùng sha256/md5/tlsh/detection_ratio/file_type hash —
  đây là chữ ký hash/telemetry sẽ leak nhãn.
- **F5**: nhãn behavior đọc TRỰC TIẾP từ field `behavior` gốc EMBER2024,
  KHÔNG suy diễn từ family (tránh circular label).

## 3. Nhãn

### Family 7 lớp (severity-priority rulebook)

| ID | Lớp | Ví dụ keyword |
|---|---|---|
| 0 | Benign | label=0 |
| 1 | Ransomware | lockbit, ryuk, revil, blackcat, wannacry… |
| 2 | Keylogger | agent.tesla, formbook, vidar, redline, spyware… |
| 3 | Trojan_RAT | trojan, .rat, backdoor, njrat, quasar, cobaltstrike… |
| 4 | Worm_Virus | mirai, conficker, sality, ramnit, autorun… |
| 5 | Other_Rare | rootkit, adware, miner/xmrig, hacktool, pua… |
| 6 | Dropper | dropper, downloader, loader, smokeloader, packed… |

Xử lý đặc biệt:
- **Severity-priority**: 1 mẫu khớp nhiều họ → gán lớp theo thứ tự ưu tiên
  Ransomware > Keylogger > Dropper > Worm/Virus > Other > Trojan/RAT
  (malware nghiêm trọng nhất thắng — tránh nhãn "kém đúng" cho mẫu đa họ).
- **Unknown → −1**: family không khớp rulebook ⇒ −1 (ignore trong loss/eval).
- family_confidence chỉ dùng để lọc ở tầng eval, không đổi nhãn.

### Behavior 8 nhãn (multi-label 0/1)

Map từ tags behavior gốc EMBER2024 qua `BEHAVIOR_MAP`; nếu field rỗng/thiếu
→ vector 0 (an toàn đầu vào).

## 4. Kiến trúc mô hình

### 4.1 GT-Transformer (~5.4M params, 2 track)

Transformer 6 layers, d_model=256, 8 heads, dim_ff=1024, dropout=0.25.
Đặc trưng 623/631 dims được **gom thành 8 group tokens** (G1…G8, +G9 cho Track A)
theo schema cố định — mỗi group 1 token + 1 learnable [CLS]-style token.

- **Track B (static_only=True)**: input 623 dims — dùng khi không có temporal
  (sample chưa thấy trước đó).
- **Track A (static_only=False)**: input 631 dims (623 + G9 temporal).

Mỗi track xuất 3 đầu ra:
- `logit_binary` (scalar) — nhiệm vụ binary
- `logits_family` (7) — nhiệm vụ family
- `logits_behavior` (8) — nhiệm vụ behavior (multi-label)

### 4.2 Ensemble với LightGBM

Mỗi seed train song song: (a) GT-Transformer (torch), (b) LightGBM 623-dim
(300 rounds, 63 leaves, lr 0.05). Điểm binary ensemble = trung bình 2 xác suất;
family dùng đầu ra 7-class của GT (cost-sensitive).

### 4.3 Hierarchical Gating — Equation (5) (lõi của "beyond binary")

Gọi `p_mal` = xác suất malware (ensemble binary), `p_comb` = phân phối
7-lớp từ family head. Phân phối joint:

```
p_joint,0 = (1 − p_mal) · p_comb,0        (benign: binary quyết định)
p_joint,c = p_mal · p_comb,c   (c ≥ 1)    (family: điều kiện vào malware)
```

Tính chất (đã có unit test xác minh):
- Không âm, tổng = 1
- p_mal = 0 ⇒ p_joint,0 = 1 (family mass = 0)
- p_mal = 1 ⇒ p_joint,0 = 0
- Nhất quán binary/family: predict family ≠ Benign ⇒ binary phải là malware

### 4.4 Prediction cuối

- Binary: `ŷ_bin = 1[p_mal ≥ τ]` (τ lấy từ validation, mặc định 0.5)
- Family: `ŷ_fam = argmax_c p_joint,c` — với **θ nén điểm** cho lớp hiếm
  (score = p/θ): θ lớn nén lớp phổ biến, θ < 1 ưu tiên lớp hiếm có p > 0.
- Behavior: threshold 0.5 per label.

## 5. Loss functions (multi-task, cost-sensitive)

| Loss | Dùng cho | Cơ chế |
|---|---|---|
| Focal binary | binary head | focal (γ) — tập trung mẫu khó |
| Logit-adjusted family | family head | trừ log prior^(1/τ) (τ=1.2) — hiệu chỉnh imbalance dài |
| Family focal γ=2.5 | family head | focal cho lớp hiếm |
| Soft macro-F1 | family head | differentiable macro-F1 — tối ưu trực tiếp metric báo cáo |
| BCE multi-label | behavior head | 8 nhãn độc lập |

Các nhiệm vụ được train **joint** (một backward pass, tổng loss có trọng số).
Mẫu family = −1 bị ignore trong family loss (mẫu benign/không rõ).

## 6. Giao thức đánh giá

- **Split chronological** (chuẩn EMBER2024): train tuần 1–48, val 49–52,
  test 53–64 — test là dữ liệu MỚI HƠN train (mô phỏng triển khai thật).
- **Quy mô đã chạy (VERIFIED, splits_meta.npz + full_results.json):** train 960,000
  (48×20k), val 80,000 (4×20k), test 240,000 (full 12 tuần); leakage train∩test sha256 = 0.
- **5 training seeds** (42, 123, 3407, 2024, 9999) → mean ± std cho mọi metric
  (tách rõ seed variability).
- **Binary**: ROC-AUC, TPR@FPR 1% và 0.1% (threshold từ validation).
- **Family**: Macro-F1 7 lớp (gồm benign) + macro-F1 6 lớp malware-only,
  per-class F1 kèm 95% bootstrap CI (1000 lần).
- **Thống kê so sánh**: DeLong test (ensemble vs CS-LGBM), paired bootstrap.

## 7. Các thí nghiệm (6 kernel trong experiments/)

| Kernel | Câu hỏi nghiên cứu |
|---|---|
| K1 full | Pipeline full 5 seeds + low-FPR + per-class CI |
| K2 ablation | Mỗi thành phần (focal, logit-adjust, soft-F1, behavior head, fusion) đóng góp bao nhiêu? Baseline công bằng: CS-CatBoost, Tree-Tree, GT standalone |
| K3 XAI + drift | Nhóm đặc trưng nào quan trọng (permutation importance)? Behavior head có giá trị? Temporal drift theo tuần (shortcut check G2-only) |
| K4 open-set + latency | Khả năng tổng quát trên packed/evasive, OOD detection (energy/max-prob), tốc độ inference |
| K5 retraining/faithfulness | No-update vs periodic retraining (concept drift); XAI faithfulness (feature-dropping) |
| K5b curriculum | Curriculum learning 5 seeds (bản đã push cuối) |

## 8. Kết quả chính (VERIFIED — results/kaggle/full_results.json, 5 seeds)

- Binary ROC-AUC: **0.9951 ± 0.0001** (CS-LGBM: 0.9975 — DeLong p=0.33, không khác biệt có ý nghĩa)
- Macro-F1 7-lớp (gated): **0.5114 ± 0.0122**
- Macro-F1 malware-only 6-lớp: **0.3810 ± 0.0136**
- TPR@FPR 1%: **0.9253 ± 0.0021**; TPR@FPR 0.1%: **0.8204 ± 0.0085**

Số pilot local (results/pilot/) chỉ có giá trị verify pipeline, không dùng cho paper.

## 9. Tái hiện

```bash
python3 tests/test_model_core.py     # kiểm tra mô hình/loss/gating
python3 analysis/pilot_local.py      # pilot CPU trên data local
cd experiments/k1_full_experiment && kaggle kernels push   # thí nghiệm lớn
```

Yêu cầu: python3.10+, numpy, lightgbm, torch, scikit-learn, scipy; data EMBER2024
Win64 (dataset Kaggle `weiweip/ember2024`) trong `data/`.
