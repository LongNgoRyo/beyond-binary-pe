# FIX.md — Những ý cần khắc phục trước khi gửi bài báo

Tài liệu này được tạo bởi Agent 3, sử dụng skill `academic-paper-reviewer` (5 reviewer personas: Journal-Fit, Methodology, Domain, Perspective, Devil's Advocate) để review bài báo "Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis" dựa trên:
- Nội dung bài báo (PDF, 25 trang) → report.md
- Source code thực tế tại /home/kali/MAlwarePEHermes/ → project.md

Mục đích: liệt kê các nhược điểm cần khắc phục trước khi nộp tạp chí Journal of Information Security and Applications (JISA).

---

## 1. JOURNAL-FIT REVIEWER (EIC) — Vị trí tạp chí và sự mới mẻ

### 1.1 Sự mới mẻ chưa đủ rõ rệt
**Vấn đề**: Bài báo tự nhận "không claim kiến trúc mới mẻ" (Key contribution #1: "We integrate established techniques... rather than claiming a novel architecture, and instead contribute a controlled component analysis"). Với tạp chí JISA (Journal of Information Security and Applications), một bài Review/Application cần có sự mới mẻ rõ ràng về phương pháp hoặc ứng dụng, không chỉ là "component analysis" của các kỹ thuật đã có.

**Khắc phục cần thiết**:
- Rõ ràng hơn về contribution: GT-Transformer với group tokenization (giảm attention cost 4,800×) có thể là điểm mới mẻ nếu được so sánh kỹ với FT-Transformer, TabNet trên dữ liệu PE.
- Nếu không có kiến trúc mới, cần positioning rõ ràng là "empirical study / controlled component analysis on EMBER2024" thay vì implying là một phương pháp mới.

### 1.2 Tiêu chí tạp chí JISA
JISA ưu tiên bài có:
- Ứng dụng thực tế trong bảo mật thông tin
- So sánh với state-of-the-art có cơ sở
- Thông báo rõ limitations và ethical considerations

**Khắc phục**: Thêm phần "Practical implications for SOC/EDR deployment" và "Ethical considerations for malware triage in air-gapped environments".

---

## 2. METHODOLOGY REVIEWER — Thiết kế nghiên cứu và tính hợp lệ thống kê

### 2.1 Mâu thuẫn về Logit Adjustment (QUAN TRỌNG — CRITICAL)
**Vấn đề**: Bài báo có mâu thuẫn nội bộ về prior-based logit adjustment:
- Single-seed ablation (Part II): REMOVING logit adjustment IMPROVES family F1 (0.4833 → 0.5345, +0.0512)
- Five-seed ablation (Section "Curriculum ablation across five seeds"): FULL curriculum WITH logit adjustment achieves 0.511±0.012, simplified (no logit adjustment) achieves 0.296±0.090 — đơn giản hóa KHÔNG được hỗ trợ
- Đoạn văn trong Results nói: "prior-based logit adjustment, recommended for long-tail problems, hurts family attribution here (removing it raises Macro-F1 from 0.483 to 0.535 in a single seed; however, a five-seed validation shows this estimate does not replicate)"

**Khắc phục**:
- Giải thích rõ tại sao single-seed và five-seed cho kết quả trái nhau (ví dụ: variance cao do ít mẫu Ransomware N=162)
- Không nên đưa kết luận mâu thuẫn trong Abstract và Conclusion mà không giải thích cơ chế
- Thêm bảng tóm tắt ablation 5 seeds cho từng component để độc giả dễ theo dõi

### 2.2 Behavioral head không cải thiện family attribution
**Vấn đề**: Removing behavioral head raises family Macro-F1 by +0.0088 (0.4833 → 0.4921, "within seed noise") nhưng ROC-AUC falls from 0.9951 to 0.9902. Bài báo vẫn giữ behavioral head nhưng không giải thích rõ trade-off.

**Khắc phục**:
- Giải thích lý do giữ behavioral head mặc dù không cải thiện family F1 (ví dụ: giá trị cho SOC analyst enrichment, không phải để cải thiện family classification)
- Hoặc loại bỏ behavioral head khỏi phần family contribution analysis

### 2.3 Family label selection bias không được giải quyết
**Vấn đề**: Chỉ 64.23% test malware (77,074/120,000) có confirmed ClarAVy consensus. 42,926 test malware bị excluded từ family Macro-F1. Bài báo nhận đây là limitation (#3) nhưng không có phương pháp giảm thiểu.

**Khắc phục**:
- Thêm phân tích: các sample bị exclude có đặc điểm khác gì so với sample có consensus? (ví dụ: packed, obfuscated, mới mẻ)
- Thử gán nhãn pseudo cho các sample không có consensus và đánh giá nhạy cảm
- Nêu rõ là family evaluation chỉ áp dụng cho subset có consensus, không thể generalize đến toàn bộ in-the-wild malware

### 2.4 Heuristic label quality
**Vấn đề**: EMBER2024 behavioral tags là regex-synthesized từ AV strings qua ClarAVy, không phải sandbox instrumentation. Taxonomy audit cũng dùng vendor labels, không phải independent ground truth.

**Khắc phục**:
- Thêm phần "Label quality assessment" với ví dụ cụ thể về discrepancy (12 discrepancies trong 210 manual audit)
- Nếu có thể, validate trên một tập nhỏ với sandbox run (Cuckoo, ANY.RUN) để xác nhận behavioral tags

### 2.5 Thiếu so sánh với SOTA gần đây trên EMBER2024
**Vấn đề**: Bài báo so sánh với Joyce et al (2025) (0.9981 ROC-AUC) nhưng dùng full 3.2M repository + 2,568 features. Không so sánh với các bài gần đây khác dùng cùng subsample 960k.

**Khắc phục**:
- Thêm so sánh với Horvath & Zsigmond (2026) (0.9979) và các bài khác trên cùng EMBER2024 subsample
- Giải thích rõ tại sao kết quả khác (training volume, tree capacity, parser differences)

---

## 3. DOMAIN REVIEWER — Domäne chuyên sâu PE malware

### 3.1 Packed malware performance degradation
**Vấn đề**: Packed malware family Macro-F1 giảm từ 0.6676 (unpacked) xuống 0.2930 (packed) — giảm 37.5 điểm. Đây là một vấn đề thực tế lớn vì hầu hết malware thực tế đều packed/obfuscated.

**Khắc phục**:
- Mở rộng Limitations về adversarial robustness
- Thêm phương án xử lý packed files (unpacking trước khi trích xuất đặc trưng, hoặc dùng dynamic features)
- Không chỉ báo cáo dưới dạng "benchmark-robustness evidence" mà cần có hướng khắc phục cụ thể

### 3.2 Ransomware detection rất yếu
**Vấn đề**: Ransomware F1 = 0.1760 (CI [0.132, 0.220]), chỉ N=162 test samples. Với Ransomware là mối đe dọa nghiêm trọng nhất, hiệu quả thấp như này là vấn đề lớn.

**Khắc phục**:
- Thêm phân tích tại sao Ransomware detection yếu (ít mẫu, đặc trưng tĩnh không catches encryption behavior)
- Đề xuất cách cải thiện: oversample Ransomware, thêm special-purpose detector cho ransomware patterns
- Nếu không cải thiện được, phải nêu rõ là "Ransomware requires dynamic/behavioral telemetry, not pure static"

### 3.3 Pure static limitations trong thực tế EDR
**Vấn đề**: Bài báo nhấn mạnh pure static cho air-gapped, nhưng trong thực tế EDR hiện đại kết hợp static + dynamic + telemetry.

**Khắc phục**:
- Thêm phần "Deployment architecture" mô tả cách static triage là stage 1, tiếp theo là dynamic sandbox enrichment
- Giải thích PPV thấp ở π=0.1% (9.56% và 55.94%) có nghĩa static không thể dùng làm standalone alert generator

---

## 4. PERSPECTIVE REVIEWER — Tác động thực tế và liên hệ fields khác

### 4.1 Khả năng triển khai thực tế (Deployment)
**Vấn đề**: Bài báo claim latency 2.23 ms/binary (>400 binaries/s/core) nhưng không nói về memory footprint trong production, chưa nói về việc integrate vào EDR agent thực tế.

**Khắc phục**:
- Thêm phần về memory usage (heap, RSS) trong production environment
- Thêm hướng dẫn deploy trên Windows endpoint (dự án viết bằng Python, cần xem xét chuyển sang C++/C# hoặc dùng ONNX Runtime)

### 4.2 Không có Code Availability rõ ràng
**Vấn đề**: Bài báo claim code available trên GitHub https://github.com/huit-sec/RMEP-EMBER2024-MTL nhưng không có trong source code lokal (/home/kali/MAlwarePEHermes/) có đầy đủ training pipeline hay không.

**Khắc phục**:
- Verify GitHub URL có tồn tại và có đầy đủ code replicate kết quả
- Nếu không, cần cung cấp reproducible artifacts (model weights, feature extraction scripts, inference code)

### 4.3 Data availability
**Vấn đề**: EMBER2024 có sẵn trên GitHub/HuggingFace nhưng extracted feature representations trên Zenodo (DOI: 10.5281/zenodo.14988210) cần verify.

**Khắc phục**:
- Verify DOI và link data availability
- Thêm hướng dẫn reproduce từ raw EMBER2024 JSON vectors

---

## 5. DEVIL'S ADVOCATE — Thách thức đối số cốt lõi

### 5.1 "Pure static là đủ cho air-gapped" — có thực sự đúng?
**Counter-argument**: Air-gapped endpoints vẫn có thể nhận file mới qua USB, CD, network intermittent. Nếu không có telemetry, model không thể biết file mới submit khi nào, nên các temporal features (G9) không dùng được. Nhưng vấn đề lớn hơn là: static-only không detects packed/obfuscated/memory-only malware, chiếm phần lớn real-world threats.

**Đánh giá**: Bài báo nhận đây là limitation (#1, #2) nhưng không đủ mạnh trong việc positioning static-only là "first-stage filter" thay vì "standalone solution".

**Khắc phục**: Rõ ràng hơn về vai trò của static-only trong tiered SOC pipeline.

### 5.2 Ensemble value proposition
**Counter-argument**: Tree–tree ensemble (CatBoost+LightGBM) đạt family Macro-F1 0.6522, cao hơn GT-ensemble gated (0.5114). Vậy GT-Transformer đóng góp gì? Bài báo nói "ensemble's value lies in joint binary–family–behavior prediction rather than in beating the tree on either task alone."

**Đánh giá**: Đây là một bằng chứng mạnh chống lại claim "proposed ensemble is better". Nếu tree tốt hơn trên cả hai task (binary comparable, family worse), tại sao vẫn dùng GT-Transformer?

**Khắc phục**: Giải thích rõ value proposition của GT-Transformer (behavioral outputs, explainability, group tokenization efficiency) và khi nào nên chọn GT-ensemble vs tree-only.

### 5.3 Five-seed variance cao cho rare classes
**Counter-argument**: Curriculum ablation 5 seeds cho thấy variance rất cao (0.416, 0.268, 0.349, 0.146, 0.300 cho simplified config). Với Ransomware chỉ 162 test samples, kết quả có thể không ổn định.

**Đánh giá**: Bài báo report mean ± std nhưng không có statistical test cho family F1 difference between full vs simplified curriculum across 5 seeds (chỉ báo cáo mean).

**Khắc phục**: Thêm paired bootstrap test cho family F1 across 5 seeds, không chỉ báo cáo mean ± std.

### 5.4 "Soft-F1 loss is single most critical component" — claims quá mạnh
**Counter-argument**: Removing soft-F1 drops family F1 từ 0.4833 xuống 0.3015 (−0.1818). Nhưng đây là single-seed ablation (Part II), không phải 5-seed. Có thể là fluctuation do seed.

**Đánh giá**: Claim "single most critical training component" cần được validate trên 5 seeds, không chỉ 1 seed.

**Khắc phục**: Thêm 5-seed ablation cho soft-F1 loss, hoặc soften claim thành "single-seed evidence suggests".

---

## 6. TỔ HỢP CÁC VẤN ĐỀ CẦN KHẮC PHỤC (THEO MỨC ĐỘ ƯU TIÊN)

### CRITICAL (phải khắc phục trước khi submit):

| # | Vấn đề | Vị trí | Khắc phục |
|---|--------|--------|-----------|
| C1 | Mâu thuẫn logit adjustment (1 seed vs 5 seeds) | Section 4.3, Abstract, Conclusion | Giải thích rõ, thêm 5-seed ablation table |
| C2 | Tree outperforms GT-ensemble trên family attribution | Section 4.2, 4.3 | Giải thích value proposition của GT-Transformer |
| C3 | Pure static limitations không đủ rõ | Section 5 Limitations | Mở rộng về packed/obfuscated/memory-only malware |
| C4 | Ransomware F1 quá thấp (0.176) | Section 4.5, Limitations | Phân tích causa và đề xuất cải thiện |

### MAJOR (nên khắc phục):

| # | Vấn đề | Vị trí | Khắc phục |
|---|--------|--------|-----------|
| M1 | Soft-F1 "single most critical" claim chỉ 1 seed | Section 4.3 | Thêm 5-seed validation hoặc soften claim |
| M2 | Behavioral head không cải thiện family F1 | Section 4.3, 4.7 | Giải thích trade-off rõ ràng |
| M3 | Family label selection bias | Section 3.3, 5 Limitations | Thêm phân tích sample bị exclude |
| M4 | Heuristic label quality | Section 3.3, 5 Limitations | Thêm label quality assessment |
| M5 | Packed malware degradation 37.5 điểm | Section "Evasive Evaluation" | Thêm hướng khắc phục cụ thể |

### MINOR (cải thiện chất lượng):

| # | Vấn đề | Vị trí | Khắc phục |
|---|--------|--------|-----------|
| N1 | Code availability link verify | Declarations | Verify GitHub URL và Zenodo DOI |
| N2 | Deployment guidance | Section 6 Conclusion | Thêm phần deployment architecture |
| N3 | Acronym check | Tất cả bài | Chạy script check_acronyms.py |
| N4 | Statistical test cho family F1 5 seeds | Section 4.3 | Thêm paired bootstrap test |

---

## 7. KẾT LUẬN VÀ ĐỀ XUẤT HÀNH ĐỘNG

Bài báo có giá trị nhưng cần khắc phục các vấn đề CRITICAL trước khi submit đến JISA:

1. **Giai đoạn 1**: Khắc phục C1–C4 (mâu thuẫn logit adjustment, tree vs GT-ensemble, static limitations, Ransomware F1)
2. **Giai đoạn 2**: Khắc phục M1–M5 (soft-F1 validation, behavioral head trade-off, label bias, label quality, packed malware)
3. **Giai đoạn 3**: Khắc phục N1–N4 (code availability, deployment, acronym, statistical tests)

Sau khi khắc phục, cần re-review để verify tất cả các vấn đề đã được xử lý.