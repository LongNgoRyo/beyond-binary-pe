# AI_DETECT.md — Note các chỗ cần sửa ngôn từ trong bài báo

Tài liệu này được tạo bởi Agent 4, sử dụng skill Humanizer để xác định và đánh dấu các dấu hiệu của văn phong AI (AI-isms) trong bài báo "Beyond Binary: A Multi-Task Ensemble for Holistic PE Malware Analysis". Mục đích: liệt kê các chỗ cần sửa ngôn từ trước khi gửi tạp chí.

---

## 1. TỔNG QUAN VĂN PHONG BÀI BÁO

Bài báo được viết bằng tiếng Anh học thuật, có nhiều đặc điểm thể loại học thuật tiêu chuẩn. Tuy nhiên, khi quét theo 29 patterns của Humanizer, có một số chỗ có nguy cơ nghe như viết bởi AI cần được điều chỉnh.

---

## 2. CÁC CHỞ CẦN SỬA NGÔN TỪ (THEO 29 PATTERNS CỦA HUMANIZER)

### Pattern 1: Undue Emphasis on Significance, Legacy, and Broader Trends

**Vị trí trong bài báo:**
- Abstract: "Static analysis of Windows Portable Executable (PE) files is foundational for Endpoint Detection and Response (EDR) platforms." — từ "foundational" có thể bị coi là overstatement.
- Introduction: "Malware targeting the Windows Portable Executable (PE) format remains the predominant threat vector in enterprise computing." — "predominant threat vector" là claims mạnh cần bằng chứng.
- Introduction: "Our contribution addresses these operational gaps." — "addresses these operational gaps" nghe như AI puffing up importance.

**Cần sửa:** Thay "foundational" bằng cụ thể hơn, thay "predominant threat vector" bằng bằng chứng dữ liệu, thay "addresses these operational gaps" bằng mô tả cụ thể contribution.

---

### Pattern 3: Superficial Analyses with -ing Endings

**Vị trí trong bài báo:**
- Có nhiều câu dùng present participle phrases như "addressing family skew", "reconciling deep and tree posteriors", "integrating established techniques".

**Cần sửa:** Chia thành các câu ngắn, trực tiếp hơn, tránh ghép -ing phrases để tạo độ sâu ảo.

---

### Pattern 7: Overused "AI Vocabulary" Words

**Từ cần kiểm tra trong bài báo:** crucial, pivotal, key, landscape, showcase, underscore, highlight, intricate, comprehensive, robust, comprehensive, additionally, furthermore, moreover, consequently, therefore, thus, hence, thereby, whereby, notwithstanding, wherein, whereby, nonetheless, nevertheless, furthermore, moreover, additionally, consequently, therefore, thus, hence, thereby, whereby.

**Vị trí có thể có:**
- "crucial role" — xuất hiện nhiều lần
- "key component" — xuất hiện nhiều lần
- "comprehensive analysis" — có thể có
- "robust" — có thể có

**Cần sửa:** Thay bằng từ cụ thể hơn, ít dùng các từ này quá mức.

---

### Pattern 14: Em Dash Overuse

**Vị trí cần kiểm tra:** Bài báo có sử dụng em dash (—) không? Cần quét toàn bộ文档.

**Cần suy nghĩ:** Nếu có em dash nhiều, thay bằng dấu phẩy hoặc chấm câu.

---

### Pattern 15: Overuse of Boldface

**Vị trí cần kiểm tra:** Bài báo PDF có dùng boldface không? Nếu có, cần xem xét việc bớt đi.

---

### Pattern 16: Inline-Header Vertical Lists

**Vị trí cần kiểm tra:** Các bảng trong bài báo có định dạng bold header + colon không?

**Cần sửa:** Nếu có, viết lại thành câu hoàn chỉnh thay vì list có bold header.

---

### Pattern 17: Title Case in Headings

**Vị trí cần kiểm tra:** Các heading trong bài báo có dùng Title Case không (ví dụ: "Experimental Results" vs "Experimental results")?

**Cần sửa:** Nếu dùng Title Case, thay bằng sentence case theo tiêu chuẩn học thuật.

---

### Pattern 20: Collaborative Communication Artifacts

**Vị trí cần kiểm tra:** Có câu như "I hope this helps", "Let me know if", "here is a overview" không? (Không có trong bài báo học thuật, nhưng cần kiểm tra.)

---

### Pattern 21: Knowledge-Cutoff Disclaimers

**Vị trí cần kiểm tra:** Có câu như "as of [date]", "based on available information" không?

**Cần sửa:** Nếu có, xóa đi hoặc thay bằng câu xác định.

---

### Pattern 23: Filler Phrases

**Cần kiểm tra:** "In order to", "Due to the fact that", "At this point in time", "It is important to note that", "The system has the ability to".

**Cần sửa:** Thay bằng forma ngắn gọn: "To", "Because", "Now", "Note that", "The system can".

---

### Pattern 24: Excessive Hedging

**Cần kiểm tra:** "could potentially possibly", "might have some effect", "it could be argued that".

**Vị trí trong bài báo:** Có thể có trong phần Limitations hoặc Discussion.

**Cần sửa:** Thay bằng forma trực tiếp hơn.

---

### Pattern 25: Generic Positive Conclusions

**Vị trí cần kiểm tra:** Kết luận phần Conclusion có vẻ như "The future looks bright", "Exciting times lie ahead" không?

**Cần sửa:** Thay bằng các cụ thể về hướng nghiên cứu tiếp theo.

---

### Pattern 26: Hyphenated Word Pair Overuse

**Từ cần kiểm tra:** third-party, cross-functional, client-facing, data-driven, decision-making, well-known, high-quality, real-time, long-term, end-to-end.

**Vị trí trong bài báo:** Có thể có "real-time", "end-to-end", "data-driven".

**Cần sửa:** Nếu không phải technical compound modifier, xóa dấu gạch nối.

---

### Pattern 27: Persuasive Authority Tropes

**Cần kiểm tra:** "The real question is", "at its core", "in reality", "what really matters", "fundamentally", "the deeper issue", "the heart of the matter".

**Vị trí trong bài báo:** Có thể có trong Discussion hoặc Conclusion.

**Cần sửa:** Xóa bỏ các từ này, viết trực tiếp.

---

### Pattern 28: Signposting and Announcements

**Cần kiểm tra:** "Let's dive in", "let's explore", "let's break this down", "here's what you need to know", "now let's look at", "without further ado".

**Vị trí trong bài báo:** Có thể có trong Introduction.

**Cần sửa:** Xóa bỏ, viết trực tiếp nội dung.

---

### Pattern 29: Fragmented Headers

**Cần kiểm tra:** Có heading sau đó là một dòng văn bản ngắn chỉ lặp lại heading không?

**Cần sửa:** Xóa dòng lặp lại, đi straight to content.

---

## 3. CÁC CÂU CỤ THỂ CẦN VVIẾT LẠI (DỰ BIỂU)

Dựa trên nội dung bài báo đã đọc, dưới đây là một số câu có thể cần viết lại:

**Trước (nghe như AI):**
> "Static analysis of Windows Portable Executable (PE) files is foundational for Endpoint Detection and Response (EDR) platforms."

**Sau (văn phong người viết thực):**
> "Static analysis of Windows PE files is one of the main techniques used in EDR platforms."

**Trước:**
> "Our contribution addresses these operational gaps."

**Sau:**
> "We focus on three operational gaps: isolated representations, extreme family imbalance, and telemetry label leakage."

**Trước:**
> "Additionally, a distinctive feature of our approach is the group-tokenized Transformer."

**Sau:**
> "Our approach uses a group-tokenized Transformer that groups 623 features into 8 semantic tokens."

---

## 4. KẾT LUẬN

Bài báo cần một lần quét toàn diện để loại bỏ các AI-isms nêu trên trước khi gửi tạp chí. Các thay đổi này không ảnh hưởng đến nội dung kỹ thuật nhưng giúp bài báo nghe thật hơn, ít giống văn phong tạo bởi mô hình ngôn ngữ lớn.

File này sẽ được kết hợp với project.md, report.md, fix.md và file "Góp ý chỉnh sửa trước khi gửi.docx" để tạo ra phương án khắc phục tổng thể.