# SYNTHESIS.md — Tổng hợp 4 file agent + Góp ý chỉnh sửa trước khi gửi.docx

Tài liệu này tổng hợp kết quả từ 4 agent và file góp ý gốc, đưa ra phương án khắc phục chi tiết cho bài báo "Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis" trước khi nộp tạp chí Journal of Information Security and Applications.

---

## 1. TỔ HỢP 4 FILE TỪ 4 AGENT

### 1.1 Project.md (Agent 1) — Cấu trúc dự án MAlwarePEHermes

Nội dung chính:
- Dự án phân tích mã độc PE tĩnh, multi-task (binary, family, behavioral)
- Đặc trưng 623 chiều pure static (G1–G8), không có G9 (telemetry, label leakage)
- Kiến trúc GT-Transformer: group tokenization (giảm attention cost ~4,800×), 6 lớp Transformer, 3 heads
- Loss: Focal Loss (binary), Logit Adjustment + Focal CE + Soft Macro-F1 (family), BCE (behavioral)
- Hierarchical Gated Inference (Eq. 5): gắn kết deep + LightGBM, đảm bảo nhất quán phân loại
- Latency: ~2.23 ms/binary (>400 binaries/s/core)
- 26 unit tests trong tests/test_model_core.py

### 1.2 Report.md (Agent 2) — Tóm tắt bài báo PDF

Nội dung chính:
- Bài báo: Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis
- Tác giả: Tien Do, Tien Nguyen, Long Ngo, An Nguyen, Bao Phung (HCMUT, Việt Nam)
- Kết quả chính: ROC-AUC 0.9951±0.0001, Binary F1 0.9628±0.0006, Gated Family Macro-F1 0.5114±0.0122
- 5 limitations được liệt kê (static surface, telemetry dependency, label selection bias, heuristic label quality, scale)
- Explainability: 10 phương pháp, consensus rank G4 ≻ G7 ≻ G6 ≻ G5 ≻ G8 ≻ G3 ≻ G2 ≻ G1

### 1.3 Fix.md (Agent 3) — Review bài báo (5 reviewer personas)

**CRITICAL (phải khắc phục trước khi submit):**
- C1: Mâu thuẫn logit adjustment (1 seed vs 5 seeds)
- C2: Tree outperforms GT-ensemble trên family attribution
- C3: Pure static limitations không đủ rõ
- C4: Ransomware F1 quá thấp (0.176)

**MAJOR (nên khắc phục):**
- M1: Soft-F1 "single most critical" claim chỉ 1 seed
- M2: Behavioral head không cải thiện family F1
- M3: Family label selection bias
- M4: Heuristic label quality
- M5: Packed malware degradation 37.5 điểm

**MINOR (cải thiện chất lượng):**
- N1: Code availability link verify
- N2: Deployment guidance
- N3: Acronym check
- N4: Statistical test cho family F1 5 seeds

### 1.4 Ai_detect.md (Agent 4) — Note ngôn từ cần sửa (Humanizer)

Các pattern AI cần kiểm tra/sửa:
- Pattern 1: Undue emphasis on significance ("foundational", "predominant threat vector", "addresses these operational gaps")
- Pattern 3: Superficial -ing phrases
- Pattern 7: Overused AI vocabulary (crucial, pivotal, key, landscape, showcase, underscore, highlight)
- Pattern 14: Em dash overuse
- Pattern 15: Overuse of boldface
- Pattern 16: Inline-header vertical lists
- Pattern 17: Title case in headings
- Pattern 23: Filler phrases (In order to, Due to the fact that, etc.)
- Pattern 24: Excessive hedging
- Pattern 25: Generic positive conclusions
- Pattern 26: Hyphenated word pair overuse
- Pattern 27: Persuasive authority tropes
- Pattern 28: Signposting and announcements
- Pattern 29: Fragmented headers

---

## 2. TỔ HỢP FILE "GÓP Ý CHỈNH SỬA TRƯỚC KHI GỬI.DOCX"

Nội dung file .docx (đọc từ /home/kali/Góp ý chỉnh sửa trước khi gửi.docx):

**Cần sửa trước khi nộp (nguy cơ desk reject):**

1. **Hình và chú thích không khớp:**
   - Fig. 1 ghi là kiến trúc nhưng hình là biểu đồ cột
   - Fig. 2 ghi là đường ROC/PR nhưng cũng là biểu đồ cột
   - Fig. 4 có hai hình trái và phải giống hệt nhau, thiếu hình TreeSHAP
   - Fig. 5 ghi (a)(b)(c) nhưng nội dung là hình khác

2. **Văn phong cần chỉnh sửa:**
   - Các cụm "superseding prior timing claims", "revised pipeline", "not re-run", "ablation kernel", "Kaggle CPU" cần bỏ
   - Phần cứng không thống nhất (i7-12700K ở chỗ này, Kaggle ở chỗ khác)

3. **Số liệu mâu thuẫn:**
   - LightGBM 2,568 chiều ghi 0.9942 ở mục 4.2 nhưng 0.9975 ở Table 7
   - Mục 4.2 nói tree–tree ensemble có gating đạt 0.6522, nhưng Table 3 ghi giá trị đó là ungated
   - Chi phí attention ghi "≈4,800×" ở một chỗ và "≈90×" ở chỗ khác
   - Đóng góp (ii) trong Introduction vẫn kể kết quả single-seed (0.535) mà mục 5 đã bác bỏ

4. **Dấu vết tạp chí cũ:**
   - "Gupta et al (this journal)" là bài của Journal of Cyber Security and Mobility; cần sửa

5. **Phần thừa:**
   - Bỏ mục OOD với n_ood = 0, hoặc thực hiện đánh giá OOD đúng nghĩa

6. **Link dữ liệu và mã nguồn:**
   - Cần kiểm tra Zenodo DOI và repo GitHub thực sự tồn tại và công khai

7. **Viết lại phần đóng góp:**
   - "đánh giá vận hành trung thực cho bài toán phân loại họ malware long-tail" kèm phát hiện rò rỉ telemetry, thay vì kiến trúc mới

8. **Tạp chí đề xuất:** Journal of Information Security and Applications

---

## 3. PHƯƠNG ÁN KHẮC PHỤC TỔNG HỢP

### 3.1 STT KHẮC PHỤC (THEO MỨC ĐỘ ƯU TIÊN)

#### PHA 1 — NGUY CỘ DESK REJECT (phải làm ngay)

| STT | Vấn đề | Nguồn | Hành động khắc phục |
|-----|--------|-------|---------------------|
| P1 | Fig. 1-5: hình và chú thích không khớp | .docx | Thay thế/hình lại các figura, đảm bảo chú thích khớp với nội dung hình |
| P2 | Số liệu mâu thuẫn (0.9942 vs 0.9975, gated vs ungated, 4,800× vs 90×) | .docx + fix.md | Kiểm tra lại từng con số, thống nhất giá trị, đánh dấu clearly single-seed vs 5-seed |
| P3 | Đóng góp (ii) vẫn kể kết quả single-seed (0.535) mà mục 5 đã bác bỏ | .docx + fix.md C1 | Xóa hoặc sửa thành 5-seed kết quả (0.511±0.012) |
| P4 | "Gupta et al (this journal)" sai tạp chí | .docx | Sửa thành "Journal of Cyber Security and Mobility" |
| P5 | Mục OOD n_ood = 0 | .docx | Bỏ mục hoặc thực hiện đánh giá OOD thực sự |
| P6 | Các cụm AI-isms cần bỏ ("superseding prior timing claims", "revised pipeline", "not re-run", "ablation kernel", "Kaggle CPU") | .docx + ai_detect.md | Viết lại các câu chứa các cụm này |

#### PHA 2 — VĂN PHONG VÀ NGÔN TỪ (Humanizer patterns)

| STT | Vấn đề | Nguồn | Hành động khắc phục |
|-----|--------|-------|---------------------|
| P7 | Overused AI vocabulary (crucial, pivotal, key, landscape, showcase) | ai_detect.md Pattern 7 | Thay bằng từ cụ thể hơn |
| P8 | Filler phrases (In order to, Due to the fact that, etc.) | ai_detect.md Pattern 23 | Rút ngắn |
| P9 | Excessive hedging | ai_detect.md Pattern 24 | Viết trực tiếp hơn |
| P10 | Signposting announcements (Let's dive in, etc.) | ai_detect.md Pattern 28 | Xóa bỏ |
| P11 | Em dash overuse | ai_detect.md Pattern 14 | Thay bằng dấu phẩy/chấm |
| P12 | Hyphenated word pair overuse | ai_detect.md Pattern 26 | Xóa dấu gạch nối không cần thiết |

#### PHA 3 — NỘI DUNG KỸ THUẬT (fix.md CRITICAL/MAJOR)

| STT | Vấn đề | Nguồn | Hành động khắc phục |
|-----|--------|-------|---------------------|
| P13 | Mâu thuẫn logit adjustment (1 seed vs 5 seeds) | fix.md C1 | Thêm bảng ablation 5 seeds, giải thích variance do ít mẫu Ransomware |
| P14 | Tree outperforms GT-ensemble trên family | fix.md C2 | Giải thích rõ value proposition của GT-Transformer (behavioral outputs, explainability) |
| P15 | Pure static limitations không đủ rõ | fix.md C3 | Mở rộng về packed/obfuscated/memory-only malware |
| P16 | Ransomware F1 quá thấp (0.176) | fix.md C4 | Phân tích causa, đề xuất cải thiện hoặc nêu rõ limitation |
| P17 | Soft-F1 claim chỉ 1 seed | fix.md M1 | Thêm 5-seed validation hoặc soften claim |
| P18 | Behavioral head không cải thiện family F1 | fix.md M2 | Giải thích trade-off rõ ràng |
| P19 | Family label selection bias | fix.md M3 | Thêm phân tích sample bị exclude |
| P20 | Heuristic label quality | fix.md M4 | Thêm label quality assessment |
| P21 | Packed malware degradation 37.5 điểm | fix.md M5 | Thêm hướng khắc phục cụ thể |

#### PHA 4 — META DATA VÀ DEPLOYMENT

| STT | Vấn đề | Nguồn | Hành động khắc phục |
|-----|--------|-------|---------------------|
| P22 | Verify Zenodo DOI và GitHub repo | .docx + fix.md N1 | Kiểm tra link thực sự tồn tại và công khai |
| P23 | Deployment guidance | fix.md N2 | Thêm phần deployment architecture |
| P24 | Acronym check | fix.md N3 | Chạy script check_acronyms.py |
| P25 | Statistical test cho family F1 5 seeds | fix.md N4 | Thêm paired bootstrap test |

---

## 4. CHI TIẾT CÁC HÀNH ĐỘNG KHẮC PHỤC

### 4.1 Sửa số liệu mâu thuẫn (P2)

**Vị trí trong bài báo:** Section 4.2, Table 3, Table 7, Abstract, Introduction

**Hành động:**
- Kiểm tra lại LightGBM 2,568 dims: giá trị đúng là 0.9942 (từ ablation kernel) hay 0.9975 (từ Table 7)? Nếu 0.9975 là main kernel, cần đánh dấu rõ.
- Tree–tree ensemble 0.6522: là ungated (Table 3) hay gated? Nếu ungated, cần sửa trong Section 4.2.
- Chi phí attention: "≈4,800×" (group tokenization vs scalar) và "≈90×" (FLOP count vs flat layout) — đây là hai con số khác nhau, cần giải thích rõ sự khác biệt.

### 4.2 Viết lại phần đóng góp (P3, P7 từ .docx)

**Trước (gần như AI-isms):**
> "We integrate established techniques... rather than claiming a novel architecture, and instead contribute a controlled component analysis with measurable findings"

**Sau:**
> "We provide an honest operational evaluation of multi-task PE malware triage on EMBER2024. Our key findings are: (i) soft Macro-F1 loss is critical for long-tailed family attribution; (ii) prior-based logit adjustment is a necessary stabilizer across 5 seeds (not just 1); (iii) a feature audit revealed label leakage in ingestion-telemetry features (G9), which must be excluded for air-gapped claims."

### 4.3 Xóa các cụm cần bỏ (P6)

| Cụm cần bỏ | Thay bằng |
|------------|-----------|
| "superseding prior timing claims" | "re-measured on the audited pipeline" |
| "revised pipeline" | "audited pipeline" or "final pipeline" |
| "not re-run" | "not evaluated in this work" |
| "ablation kernel" | "ablation experiment" |
| "Kaggle CPU" | "CPU" or "evaluation hardware" |

---

## 5. FILES ĐÃ TẠO (4 AGENT)

| File | Vị trí | Nội dung |
|------|--------|----------|
| project.md | /home/kali/MAlwarePEHermes/project.md | Cấu trúc dự án, kiến trúc, luồng dữ liệu |
| report.md | /home/kali/MAlwarePEHermes/report.md | Tóm tắt bài báo PDF, kết quả, limitations |
| fix.md | /home/kali/MAlwarePEHermes/fix.md | Review 5 personas, các vấn đề cần khắc phục |
| ai_detect.md | /home/kali/MAlwarePEHermes/ai_detect.md | Note ngôn từ cần sửa (Humanizer patterns) |

---

## 6. KẾT LUẬN VÀ BƯỚC TIẾP THEO

Sau khi tổng hợp 4 file agent và file .docx, có **25 hành động khắc phục** phân thành 4 pha. Các bước tiếp theo:

1. **Sửa file PDF:** Không thể sửa trực tiếp file PDF. Cần chỉnh sửa file LaTeX/Word nguồn, rồi xuất PDF mới.
2. **Sửa file .docx:** Nếu đây là file phản hồi của reviewer, cần triển khai các góp ý vào manuscript nguồn.
3. **Sửa các file .md:** project.md, report.md, fix.md, ai_detect.md đã được tạo và có thể được dùng làm tài liệu tham khảo khi sửa manuscript.
4. **Verify links:** Kiểm tra Zenodo DOI (10.5281/zenodo.14988210) và GitHub repo (https://github.com/huit-sec/RMEP-EMBER2024-MTL) có thực sự tồn tại và công khai.
5. **Re-review:** Sau khi khắcineux, chạy lại academic-paper-reviewer để verify tất cả vấn đề đã được xử lý.

---