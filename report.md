# REPORT.md — Tóm tắt bài báo "Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis"

Tài liệu này được tạo bởi Agent 2 sau khi đọc toàn bộ bài báo PDF (25 trang) tại `/home/kali/MAlwarePEHermes/Beyond_Binary__Multi_Task_Ensemble_for_PE_Malware_Analysis (2).pdf`. Mục đích: ghi lại thông tin quan trọng, cấu trúc dự án, phương pháp và kết quả để phục vụ việc review bài báo.

---

## 1. Thông tin bài báo

- **Tên bài báo**: Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis
- **Tác giả**: Tien Do, Tien Nguyen, Long Ngo, An Nguyen, Bao Phung* (trách nhiệm tương ứng)
- **Đơn vị**: Faculty of Information Technology, Ho Chi Minh City University of Industry and Trade, Việt Nam
- **Tạp chí mục tiêu**: Journal of Information Security and Applications (JISA)
- **Từ khóa**: PE malware triage, Multi-task learning, Group-Tokenized Transformer, LightGBM, EMBER2024, Explainable AI, Concept drift, Endpoint detection

---

## 2. Vấn đề nghiên cứu (Problem Statement)

Ba vấn đề cản trở hiệu quả của phân tích mã độc PE đa nhiệm vụ:

1. **Isolated representations**: Các hệ thống thường train riêng từng mô hình single-task cho binary và family classification, bỏ qua hiệu quả regularization từ nhãn binary dồi dào dành cho các family dữ dội.
2. **Extreme long-tailed skew**: Tỷ lệ Benign:Ransomware là 285.7:1 trong training (Weeks 1–48) và 740.7:1 trong testing (Weeks 53–64). Ransomware chỉ ≈0.82% labeled training malware, Droppers ≈0.83%, Trojan/RAT ≈78.65%. Dưới CE chuẩn, models collapse về majority classes (Trojan/RAT).
3. **Telemetry artifacts**: Submission intervals và analysis dates không phải thuộc tính nội tại của PE file và không khả dụng trên air-gapped hosts. Việc không isolate chúng có thể làm tăng laboratory accuracy.

---

## 3. Phương pháp (Methodology)

### 3.1 Bài toán phương trình

Học ánh xạ chung:
```
f : X → Ybin × Yfam × Ybeh
```
- `x ∈ R^623` (pure static) hoặc `R^631` (Track A với telemetry)
- `Ybin = {0, 1}` (benign/malware)
- `Yfam ∈ {0..6}`: 0 Benign, 1 Ransomware, 2 Keylogger, 3 Trojan/RAT, 4 Worm/Virus, 5 Other/Rare, 6 Dropper
- `Ybeh ∈ {0,1}^8`: 8 nhãn hành vi (file ops, network, registry, process injection, persistence, encryption, keylogging, screenshot)

### 3.2 Đặc trưng (Feature Representation)

**9 nhóm sémantics (631 chiều tổng, 623 pure static G1–G8):**

| Nhóm | Tên | Kích thước | Mô tả |
|------|-----|-----------|-------|
| G1 | ByteHistogram | 256 | Phân phối byte chuẩn hóa |
| G2 | ByteEntropy | 256 | Ma trận joint byte-value và entropy |
| G3 | StringsStats | 16 | Thống kê chuỗi in được |
| G4 | GeneralStats | 10 | Kích thước PE, số section, characteristics |
| G5 | HeaderStats | 24 | Loại máy, linker version, subsystem |
| G6 | SectionStats | 25 | Kích thước section, Shannon entropy |
| G7 | ImportsExports | 20 | IAT metrics, ordinal export |
| G8 | DataDirectories | 16 | RVA, size descriptors |
| G9 | RelativeTemporal | 8 | Telemetry ingestion metadata (bị loại, label leakage) |

**Tiền xử lý**: sum-normalized cho byte histogram/entropy; signed-log transform `x̃ = sign(x)·log(1+|x|)` + robust z-score; imports condense thành 20 chỉ số; Authenticode zeroed.

### 3.3 Taxonomy và phân bố family

7 lớp với severity priority: Ransomware ≻ Keylogger ≻ Dropper ≻ Worm/Virus ≻ Other/Rare ≻ Trojan/RAT

**Training (Weeks 1–48, N=960,000)**: 480,000 benign, 480,000 malware; 205,348 có high-confidence family tags
**Test (Weeks 53–64, N=240,000)**: 120,000 benign, 120,000 malware; 77,074 có confirmed tags

### 3.4 Kiến trúc GT-Transformer

Mỗi nhóm Gk chiếu vào token 256 chiều qua MLP 2 lớp:
```
tok_k = W2 · GELU(LayerNorm(W1·xGk + b1)) + b2
```

Thêm [CLS] token → 10 tokens (9 trong pure static mode). Giảm chi phí attention ≈ 4,800× so với scalar tokenization (624/9)² ≈ 4,807.

Truyền qua 6 lớp Pre-LayerNorm Transformer encoder (d_model=256, N_heads=8, d_ff=1024, dropout=0.25).

**[CLS] embedding feeds 3 heads:**
1. Binary head: `l̂b = w_b^T · emb + b_b`
2. Cosine-margin family head trên `z = MLP_fam(emb) ∈ R^128`:
   ```
   l̂f,c = s · (w_c^T z)/(||w_c||·||z||) − m·I(y=c),  s=20, m=0.35
   ```
3. Behavioral head: 8 logits độc lập `l̂beh`

### 3.5 Loss Function và Curriculum

**Loss tổng hợp:**
```
L_total = λ_bin·L_FL,bin + λ_fam·L_fam + λ_beh·L_BCE,beh + λ_reg·||Θ||₂²
```

- Binary: Focal Loss (γ=2.0, label smoothing 0.05)
- Family: Logit adjustment (τ=1.2) + Focal CE (γ=2.5) + class-balanced weights (β=0.9999) + Soft Macro-F1
- Behavioral: Multi-label BCE

**Four-phase curriculum:**
- Phase 1 (8 epochs): multi-task warm-up, Homoscedastic Uncertainty Weighting, AdamW lr=1e-3
- Phase 2 (12 epochs): minority re-balancing, oversample Nc = min(15·nc, 8000) + Gaussian jitter, Manifold Mixup (λ∼Beta(0.4,0.4))
- Phase 3 (5 epochs): joint fine-tuning, discriminative rates, λ_bin=0.05, λ_fam=3.0, λ_beh=0.05
- Phase 4: Cost-Sensitive LightGBM (300 cây, lr=0.05, num_leaves=63)

### 3.6 Hierarchical Gated Inference (Eq. 5)

**Bước 1** (binary evidence):
```
s_bin = 0.4·l̂b + 0.6·logsumexp(c≥1)(l̂f,c) − l̂f,0
p_mal = 0.5·p_GT,mal + 0.5·p_LGB,mal
```

**Bước 2** (family blending):
```
p_combined = 0.5·softmax(l̂f) + 0.5·p_LGB
```

**Bước 3** (hierarchical gating):
```
p_joint,0 = (1−p_mal)·p_combined,0
p_joint,c = p_mal·p_combined,c, c=1..6
```

**Bước 4** (thresholding):
```
ŷ_fam = arg max_c p_joint,c/(θ_c+ε), θ tối ưu trên validation
Nếu ŷ_bin=0 → family=0 (tính nhất quán phân loại)
```

---

## 4. Kết quả thực nghiệm (Results)

### 4.1 Binary Detection (Track B, 623 pure static features, 240,000 test samples)

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| CS-LightGBM | 0.9975±0.0000 | 97.91±0.10 | 97.62±0.11 | 0.9777±0.0002 |
| Proposed GT-ensemble (gated) | 0.9951±0.0001 | 98.16±0.06 | 94.47±0.09 | 0.9628±0.0006 |
| GT-Transformer standalone | 0.9925 | — | — | — |

- DeLong test: z = −0.89, p = 0.372 (statistically comparable to CS-LightGBM)
- Confusion matrix: TN=118,048, FP=1,952, FN=6,752, TP=113,248, accuracy ≈ 96.37%

### 4.2 Family Attribution (7-class, Macro-F1 on consensus subset N=197,074)

| Model | Family F1 (7-class) | Malware-only F1 (6-class) |
|-------|---------------------|---------------------------|
| CS-LightGBM family head | 0.6614±0.0066 | 0.6072±0.0076 |
| Tree–tree ensemble | 0.6522 | 0.5190 |
| Proposed GT-ensemble (ungated) | 0.4833 | 0.3479 |
| Proposed GT-ensemble (gated) | 0.5114±0.0122 | 0.3810±0.0136 |

Per-class F1 (seed 3407):
- Benign: 0.9839 [0.9835, 0.9844]
- Trojan/RAT: 0.7654 [0.763, 0.768]
- Worm/Virus: 0.6129 [0.607, 0.619]
- Dropper: 0.4398 [0.405, 0.474]
- Keylogger: 0.3708 [0.357, 0.386]
- Other/Rare: 0.3567 [0.344, 0.369]
- Ransomware: 0.1760 [0.132, 0.220] (high uncertainty, N=162)

### 4.3 Low-FPR and Base-Rate Evaluation

| Threshold | TPR@1% | TPR@0.1% | PPV@π=1% | PPV@π=0.1% |
|-----------|--------|----------|----------|------------|
| τ1.0% (FPR≈0.88%) | 92.53±0.21 | — | 51.61 | 9.56 |
| τ0.1% (FPR≈0.065%) | — | 82.04±0.85 | 92.76 | 55.94 |

### 4.4 Behavioral Attributes (Task 3, N=42,000)

Tất cả 8 attributes vượt baseline all-positive:
- file ops: F1=0.213, AUROC=0.881
- network: F1=0.208, AUROC=0.888
- registry: F1=0.016, AUROC=0.649
- process injection: F1=0.017, AUROC=0.595
- persistence: F1=0.229, AUROC=0.817
- encryption: F1=0.162, AUROC=0.473
- keylogging: F1=0.309, AUROC=0.917
- screenshot: F1=0.211, AUROC=0.917

### 4.5 Explainability (10 methods, 5 paradigms)

Consensus rank: G4 ≻ G7 ≻ G6 ≻ G5 ≻ G8 ≻ G3 ≻ G2 ≻ G1
- G4 (GeneralStats): density 0.245 (cao nhất)
- Permutation importance: G7 (∆AUC=0.0215), G5 (∆AUC=0.0130), G6 (∆AUC=0.0080), G8 (∆AUC=0.0006, negligible)
- Integrated Gradients và TreeSHAP đồng ý hoàn håll (τ=ρ=1.0)

### 4.6 Latency

| Thành phần | Latency (batch=1, CPU) |
|------------|----------------------|
| Feature extraction | 1.41±0.25 ms |
| GT-Transformer forward | 0.64±0.05 ms (0.18 ms GPU) |
| LightGBM inference | 0.16±0.02 ms |
| Gating | 0.02±0.005 ms |
| **Tổng** | **≈2.23 ms** (>400 binaries/s/core) |

### 4.7 Evasive (Packed) Evaluation

- 9,336 packed binaries trong test set (VMProtect, Themida, NSIS, Obsidium, Enigma, UPX, MEW)
- Packed malware binary detection: ROC-AUC 0.9993
- Packed malware family Macro-F1: 0.2930 vs 0.6676 unpacked (giảm 37.5 điểm)

### 4.8 Limitations (5 điểm)

1. Static surface: không quan sát được dynamic unpacking, memory-only injection, multi-stage downloaders, packing obfuscation
2. Ingestion telemetry dependency: Track A không re-run sau feature audit (label leakage G9)
3. Family label selection bias: chỉ 64.23% test malware có confirmed consensus
4. Heuristic label quality: EMBER2024 behavioral tags là regex-synthesized, không phải sandbox instrumentation
5. Scale and benchmark scope: train trên 960k subsample, test balanced 50:50, không so sánh trực tiếp với full-corpus leaderboard

---

## 5. Cấu trúc bài báo

1. Introduction — vấn đề, thách thức, đóng góp
2. Related Work — static PE detection, multi-task learning, deep tabular, long-tailed learning, explainability
3. Proposed Method — problem formulation, feature representation, taxonomy, GT-Transformer, gated inference, loss & curriculum
4. Experimental Results — setup, baseline comparison, ablation, low-FPR, behavioral, explainability, latency, limitations
5. Conclusion — tóm tắt đóng góp và hướng phát triển
6. References
7. Declarations (funding, conflict of interest, ethics, data/code availability, author contributions)

---

## 6. Đóng góp chính (Key Contributions)

1. Multi-task GT-Transformer–LightGBM ensemble với controlled component analysis (không claim kiến trúc mới mẻ)
2. Low-FPR và base-rate evaluation dưới π ≤ 1.0%
3. Explainability consensus qua 10 phương pháp từ 5 paradigms
4. Behavioral label analysis trên 8 operational attributes
5. Pure static 623-dim representation tránh label leakage G9, phù hợp air-gapped