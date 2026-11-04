# PROJECT.md — Mô tả dự án MAlwarePEHermes

Tài liệu này được tạo bởi Agent 1 sau khi đọc toàn bộ source code dự án tại `/home/kali/MAlwarePEHermes/`. Mục đích: ghi lại kiến trúc, cách hoạt động, từng module và luồng dữ liệu của dự án để phục vụ việc review bài báo và khắc phục nhược điểm.

---

## 1. Tổng quan dự án

Dự án **MAlwarePEHermes** là một pipeline phân tích mã độc PE (Portable Executable) tĩnh, dùng mô hình deep learning kết hợp ensemble để thực hiện ba nhiệm vụ đồng thời:
1. **Nhị phân** (binary): benign vs malware
2. **Nhóm mã độc** (family): 7 lớp (Benign, Ransomware, Keylogger, Trojan/RAT, Worm/Virus, Other/Rare, Dropper)
3. **Đặc tính hành vi** (behavioral): 8 nhãn nhị phân (file ops, network, registry, process injection, persistence, encryption, keylogging, screenshot)

Dự án được xây dựng trên dataset **EMBER2024**, sử dụng đặc trưng tĩnh 623 chiều (Track B, pure static) không phụ thuộc vào telemetry ingestion (vì đã bị phát hiện rò rỉ nhãn label leakage trong nhóm G9).

---

## 2. Cấu trúc thư mục

```
/home/kali/MAlwarePEHermes/
├── src/
│   ├── feature_extraction.py      # Trích xuất 623 đặc trưng PE từ 8 nhóm sémantics
│   └── model_core.py              # Mô hình GT-Transformer, gating, losses
├── analysis/
│   ├── pilot_local.py             # Chạy pilot local (không dùng Kaggle kernel)
│   ├── statistical_analysis.py    # DeLong test, paired bootstrap, per-class CI, low-FPR points
│   └── behavioral_baselines.py    # Baselines cho đặc tính hành vi
├── docs/
│   ├── MODEL.md                   # Tài liệu mô hình chi tiết
│   └── PROJECT_NOTES.md           # Ghi chú toàn diện của tác giả
├── experiments/
│   └── k1..k5b/                   # 6 kernel Kaggle (byte-identical bản đã chạy)
├── tests/
│   └── test_model_core.py         # Unit test gating equation (5) + GT-Transformer + losses
├── results/
│   └── kaggle/                    # Kết quả verified (JSON + NPZ)
├── README.md
└── data/                          # Thư mục data đang rỗng (bỏ qua theo yêu cầu user 11/2026)
```

---

## 3. Chi tiết từng module

### 3.1 `src/feature_extraction.py`

Trích xuất 623 đặc trưng tĩnh từ file PE, chia thành 8 nhóm sémantics (G1–G8):

| Nhóm | Tên | Kích thước | Mô tả |
|------|-----|-----------|-------|
| G1 | ByteHistogram | 256 | Phân phối tần số byte chuẩn hóa |
| G2 | ByteEntropy | 256 | Ma trận joint byte-value và entropy cục bộ |
| G3 | StringsStats | 16 | Số chuỗi in được, độ dài trung bình, entropy chuỗi |
| G4 | GeneralStats | 10 | Kích thước ảo, kích thước thô, số section, PE characteristics |
| G5 | HeaderStats | 24 | Loại máy, phiên bản linker, subsystem, compile headers |
| G6 | SectionStats | 25 | Kích thước raw/virtual per-section, Shannon entropy |
| G7 | ImportsExports | 20 | Chỉ số IAT, ordinal xuất khẩu |
| G8 | DataDirectories | 16 | RVA và size cho export, import, debug tables |

**Tiền xử lý:**
- Byte histogram và entropy array: sum-normalized
- Các chỉ số liên tục: signed-log transform `x̃ = sign(x)·log(1+|x|)` rồi robust z-score
- Imports: condense thành 20 chỉ số tóm tắt (IAT counts, kernel32/user32 linkage, export ordinals)
- Authenticode certificates: zeroed (bỏ qua vì chữ ký thường bị đánh cắp/lạm dụng)

**Không có nhóm G9** (RelativeTemporal, 8 chiều) trong bản pure static vì đã bị phát hiện label leakage từ telemetry ingestion metadata.

### 3.2 `src/model_core.py`

Xây dựng mô hình **GT-Transformer** (Group-Tokenized Transformer) và các thành phần liên quan:

**Kiến trúc:**
- Mỗi nhóm đặc trưng Gk được chiếu vào một token 256 chiều qua MLP 2 lớp:
  ```
  tok_k = W2 · GELU(LayerNorm(W1·xGk + b1)) + b2
  ```
- Một [CLS] token học được được thêm vào → 10 tokens (9 trong pure static mode)
- Truyền qua 6 lớp Pre-LayerNorm Transformer encoder (d_model=256, N_heads=8, d_ff=1024, dropout=0.25)
- [CLS] embedding `emb ∈ R^256` feeds ba head:
  1. Binary head: `l̂b = w_b^T · emb + b_b`
  2. Cosine-margin family head trên `z = MLP_fam(emb) ∈ R^128`:
     ```
     l̂f,c = s · (w_c^T z) / (||w_c|| · ||z||) − m · I(y=c),  s=20, m=0.35
     ```
  3. Behavioral head: 8 logits độc lập `l̂beh`

**Loss tổng hợp:**
```
L_total = λ_bin·L_FL,bin + λ_fam·L_fam + λ_beh·L_BCE,beh + λ_reg·||Θ||₂²
```
- Binary: Focal Loss (γ=2.0, label smoothing 0.05)
- Family: Logit adjustment (τ=1.2) + Focal cross-entropy (γ=2.5) + class-balanced weights (β=0.9999) + Soft Macro-F1 loss
- Behavioral: Multi-label BCE

**Four-phase curriculum:**
- Phase 1 (8 epochs): multi-task warm-up, Homoscedastic Uncertainty Weighting, AdamW lr=1e-3
- Phase 2 (12 epochs): minority re-balancing, oversample Nc = min(15·nc, 8000) + Gaussian jitter, Manifold Mixup
- Phase 3 (5 epochs): joint fine-tuning, discriminative rates, λ_bin=0.05, λ_fam=3.0, λ_beh=0.05
- Phase 4: ensemble với Cost-Sensitive LightGBM (300 cây, lr=0.05, num_leaves=63)

**Hierarchical Gated Inference (Eq. 5):**
- Bước 1: `s_bin = 0.4·l̂b + 0.6·logsumexp(c≥1)(l̂f,c) − l̂f,0`, `p_mal = w·p_GT,mal + (1−w)·p_LGB,mal`, w=0.5
- Bước 2: `p_combined = w·softmax(l̂f) + (1−w)·p_LGB`
- Bước 3: `p_joint,0 = (1−p_mal)·p_combined,0`, `p_joint,c = p_mal·p_combined,c` (c=1..6)
- Bước 4: `ŷ_fam = arg max_c p_joint,c/(θ_c+ε)`, với θ tối ưu trên validation; nếu ŷ_bin=0 → family=0

### 3.3 `analysis/pilot_local.py`

Chạy pipeline local (không cần Kaggle kernel). Đầu vào là dữ liệu đã trích xuất, đầu ra là kết quả dự đoán để đánh giá nhanh trước khi đẩy lên Kaggle.

### 3.4 `analysis/statistical_analysis.py`

Thực hiện kiểm kê thống kê cho bài báo:
- **DeLong test**: so sánh AUC 2 mô hình ROC trên cùng tập mẫu (paired)
- **Paired bootstrap 1000 lần**: cho chênh lệch Macro-F1
- **Per-class F1 95% CI**: bootstrap cho từng lớp (Ransomware, Keylogger, Dropper...)
- **Low-FPR operating points**: TPR@FPR (0.01, 0.001, 0.0001) + số false alerts per 100k negatives
- Xuất bảng đối chiếu (B17): roc_auc, macro7_gated, macro6_malware_only, tpr_at_1pct, tpr_at_01pct

### 3.5 `tests/test_model_core.py`

Unit test 26 cases cho:
- Feature group schema (Table 1): 623 static dims, 631 Track A dims, 8 groups, dims [256,256,16,10,24,25,20,16]
- Gating Eq.(5) trường hợp biên: p_mal=0, p_mal=1, ngẫu nhiên (không âm + tổng=1), nhất quán binary/family, theta nén điểm, NaN an toàn
- GT-Transformer forward: output shapes, params > 5M
- Losses: focal loss finite, family loss finite (ignore -1), soft macro F1 in [0,1]
- Trọng số group chuẩn hóa khi thiếu nhóm (Track B từ Track A vector)

---

## 4. Luồng dữ liệu tổng thể

```
PE file → feature_extraction.py (623 dims, 8 groups)
     → model_core.py (GT-Transformer + LightGBM)
     → hierarchical_gating() (Eq. 5, gated inference)
     → y_bin (benign/malware), y_fam (7-class), y_beh (8 attributes)
     → statistical_analysis.py (DeLong, bootstrap, CI, low-FPR)
     → kết quả lưu trong results/kaggle/ (full_results.json, .npz)
```

---

## 5. Điểm nổi bật kỹ thuật

- **Pure static representation**: 623 chiều, không dùng telemetry (tránh label leakage G9), phù hợp air-gapped
- **Group tokenization**: giảm chi phí attention từ ~624² xuống ~(624/9)² ≈ 4,800×
- **Hierarchical gating**: bảo toàn tính nhất quán phân loại (y_bin=0 ⇒ family=0)
- **Multi-task**: chia sẻ biểu diễn giữa binary, family, behavior
- **Latency**: feature extraction 1.41 ms + GT-Transformer 0.64 ms + LightGBM 0.16 ms + gating 0.02 ms ≈ 2.23 ms/binary (>400 binaries/s/core)

---

## 6. Ghi chú từ PROJECT_NOTES.md

- `model_core.py:152` có chained-assignment no-op vô hại khi train (không phải bug số học, verified 11/2026)
- `urlretrieve`/`urldownloadtofile` cũ cũng không phải bug
- data/Win64_train đang rỗng (user yêu cầu bỏ qua folder data 11/2026)
- experiments/k1..k5b là 6 kernel Kaggle BYTE-IDENTICAL đã chạy
- results/kaggle/ = KẾT QUẢ VERIFIED (JSON + NPZ)