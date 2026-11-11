# 6 bộ feature OULAD — chọn gì, vì sao, và liên hệ với nghiên cứu

Các bộ nằm trong `prediction/feature_sets/` và dùng chung `prediction/features_common.py` (cùng định nghĩa nhãn, cùng `base_frame`, cùng `leakage_check`). Mỗi dòng = một Student–Course; chỉ dùng thông tin có ngày ≤ cutoff.

Chọn bộ khi chạy dự đoán bằng `--feature-set`: `kuznetsov_ews`, `wu_jia`, `joive`, `mtapsp`, `xai` (mặc định), hoặc `trajectory`. Ví dụ:

```powershell
python prediction\predict_progress.py --data data_subset --feature-set wu_jia --progress 0 10 20 30 40 50 --no-shap --no-lime --out predict_wu_jia
```

## 1. Bảng tổng quan

| Bộ (file) | Nghiên cứu đối chiếu | Cutoff mặc định | # feature | Chọn gì | Vì sao |
|---|---|---|---|---|---|
| `feature_sets/features_kuznetsov_ews.py` | Shaikhanova, Kuznetsov et al., *Big Data Cogn. Comput.* 9(11):297 (2025) — DSS ngày 14 | 14 | 12 | 4 activity + 5 assessment (chỉ bài đầu tiên) + 3 demographic (mã hóa thứ tự) | Bài nhắm dự đoán "cực sớm" (ngày 14), AUC 0.789, precision 84% khi chỉ cảnh báo 15% sinh viên; cần feature giải thích được cho báo cáo giảng viên |
| `feature_sets/features_wu_jia.py` | Hướng "4 tuần đầu + kiểm soát rò rỉ + ít nhân khẩu học" (gần nhất tìm được: Wu, Duan & Luo, *PLOS ONE*; và LEAP, Lê et al. 2026) | 14, 28 | 28 | Volume VLE, click theo tuần 1–4, recency/xu hướng, tỉ trọng loại hoạt động, tóm tắt bài nộp; chỉ giữ `num_of_prev_attempts`, `studied_credits`, `date_registration` | Bỏ gender/region/imd/age/disability/education để giảm phụ thuộc nhân khẩu học; tuần chưa kết thúc = NaN thay vì 0 để không bịa dữ liệu |
| `feature_sets/features_joive.py` | JOIVE (2026): Inactivity Streak, Access Duration, Click frequency | 14, 28, 60, 90 | 15 | Chỉ log VLE: chuỗi ngày không hoạt động, số ngày/khoảng truy cập, tần suất click | Tách riêng "chuỗi hành vi LMS" thuần, không điểm, không nhân khẩu học → đo giá trị của hành vi đơn thuần |
| `feature_sets/features_mtapsp.py` | Appl. Sci. 14(6):2522 (2024), MTAPSP: chuỗi đa chiều + multi-head self-attention | 54, 108, 162 | 43 | 5 phân đoạn × (click, ngày hoạt động) + điểm TMA/CMA theo phân đoạn + 8 nhân khẩu học; thêm `build_sequences` cho LSTM/CNN | Bài gộp 3 nguồn (hành vi theo thời gian, điểm, nhân khẩu); bản phẳng cho RF/XGBoost, bản tensor cho mô hình chuỗi |
| `feature_sets/features_xai.py` | Gunasekara & Saarela, *Appl. Sci.* 15(3):1239 (2025) — DT vs ANN + LIME/SHAP trên OULAD | 28, 60, 90 | 30 | Số bài nộp/trễ/bỏ, click tổng và theo loại, sự kiện dịch được thành câu ("nộp trễ TMA1", "oucontent giảm N tuần") + nhân khẩu học (tắt được) | Mỗi feature phải kể được cho cố vấn; `demographics=False` + `sensitive_frame()` để kiểm tra công bằng bên ngoài mô hình |
| `feature_sets/features_trajectory.py` | Pecuchova & Drlik (2024) về phân cụm quỹ đạo; Gupta et al. (2022); nền Markov: Kuzilek et al. 2018 | 56, 84, 140, 160 | 19 | Đều đặn, đà tăng/giảm, bỏ cuộc ngầm, học dồn trước hạn, Markov 3 trạng thái tuần; kèm `weekly_profile` (cụm) và `survival_table` (Kaplan-Meier/Cox) | Coi học tập là quá trình theo thời gian, không phải ảnh chụp; chỉ hành vi trên các tuần đã kết thúc |

## 2. Tương quan giữa các bộ (loại tín hiệu nào có mặt ở đâu)

| Bộ | Volume VLE | Gần đây / streak | Hình dạng theo tuần | Loại hoạt động | Điểm/nộp bài | Nhân khẩu học |
|---|---|---|---|---|---|---|
| kuznetsov_ews | ✓ | ✓ (1 chỉ số) | – | – | chỉ bài đầu | ✓ thứ tự |
| wu_jia | ✓ | ✓ | ✓ (4 tuần) | ✓ | ✓ | tối thiểu |
| joive | ✓ | ✓✓ | – | – | – | – |
| mtapsp | ✓ (theo đoạn) | – | ✓ (5 đoạn) | – (chỉ ở tensor) | ✓ TMA/CMA | ✓ |
| xai | ✓ | – | một phần (oucontent) | ✓ | ✓ | ✓ (tắt được) |
| trajectory | một phần | ✓✓ | ✓✓ | – | – (chỉ ngày hạn) | – |

Cách đọc: `wu_jia` là bộ rộng nhất không dùng nhân khẩu học; `joive` và `trajectory` là hai bộ chỉ-hành-vi (khác nhau ở chỗ joive xét ngày, trajectory xét tuần); `mtapsp`, `xai`, `kuznetsov_ews` là ba bộ có nhân khẩu học nên phù hợp làm đối chứng khi so ablation.

Trùng lặp cố ý: `clicks_total`, `active_days`, `last_active_gap` xuất hiện ở nhiều bộ với cùng cách tính (đều từ `C.vle_core` / `C.streak_features`), nên chênh lệch kết quả giữa các bộ đến từ *tập feature*, không phải từ cách tính khác nhau.

## 3. Hai điều các nghiên cứu cho thấy mà nên giữ trong đầu

- **Rò rỉ thời gian làm "sớm" bị thổi phồng.** LEAP (Lê, Abel, Laforge 2026) chỉ ra rằng thông tin đánh giá sai thời điểm là nguồn thổi phồng chính, và đề xuất cắt dữ liệu theo cutoff *trước* khi join/aggregate. Đây chính là điều `leakage_check` làm: dựng feature trên dữ liệu đã cắt, phải ra kết quả y hệt. Nhóm assessment ở `wu_jia`, `xai`, `mtapsp` là nơi rủi ro cao nhất.
- **Nhân khẩu học: giảm hay giữ?** Một preprint (arXiv 2512.12493) báo cáo đặc trưng tĩnh/nhân khẩu học chiếm ~68% độ quan trọng ở tuần 2–4; ngược lại `wu_jia` cố tình bỏ chúng và da Silva (2026) coi rủi ro bỏ học là quá trình hành vi-thời gian hơn là hàm của nền tảng. So sánh `wu_jia` / `joive` / `trajectory` (không nhân khẩu) với `xai` / `mtapsp` (có) sẽ trả lời trực tiếp câu hỏi này trên dữ liệu của bạn.

## 4. Độ tin cậy của phần đối chiếu (đọc trước khi trích dẫn)

- **Đã xác minh tồn tại và nội dung chính (qua tóm tắt/abstract, chưa đọc toàn văn):** Kuznetsov/Shaikhanova 2025; MTAPSP 2024; Gunasekara & Saarela 2025; LEAP 2026; Kuzilek et al. 2018 (Markov, thời gian hấp thụ).
- **Chưa tìm thấy đúng bài:** "Wu & Jia (Frontiers in Education, 10/2026)" và "JOIVE (7/2026)" — tìm theo tên không ra. Với Wu & Jia, bài gần nhất là Wu, Duan & Luo (*PLOS ONE*, 4 tuần đầu, chia train/test theo presentation, kiểm tra công bằng). Danh sách feature của hai bộ này vẫn là lựa chọn của tôi theo mô tả bạn đưa.
- **Tên có thể lệch:** "Pecuchova & Drlik 2024" — kết quả gần nhất là Popovych & Drlik, *Appl. Sci.* 14(9):3615 (phân cụm nhóm sinh viên); "Gupta et al. 2022" không tìm thấy. Nên kiểm tra lại trích dẫn gốc trước khi đưa vào báo cáo.
- **XAI:** abstract không liệt kê "14 biến đầu vào", nên `xai` vẫn là bộ "ưu tiên giải thích được", không phải bản tái lập.
- Các giả định kỹ thuật (cutoff, ngưỡng `LOW_T = 30`, định nghĩa "bài đầu tiên", cách tính `avg_clicks_per_day`) nằm ở docstring từng file.

## 5. Nguồn

- Shaikhanova, Kuznetsov, Iklassova, Tokkuliyeva — https://www.mdpi.com/2504-2289/9/11/297
- MTAPSP — https://www.mdpi.com/2076-3417/14/6/2522
- Gunasekara & Saarela — https://www.mdpi.com/2076-3417/15/3/1239
- LEAP (Lê, Abel, Laforge) — https://arxiv.org/abs/2605.25794
- Wu, Duan, Luo (PLOS ONE) — https://econpapers.repec.org/RePEc:plo:pone00:0352867
- Preprint tĩnh vs động (tuần 2–20) — https://arxiv.org/abs/2512.12493
- Kuzilek et al. 2018, Markov drop-out — https://oro.open.ac.uk/58622
- da Silva 2026, survival benchmark — https://arxiv.org/abs/2604.08870
- Popovych & Drlik 2024 — Appl. Sci. 14(9):3615
- Chen & Cui 2020, chuỗi click + LSTM (nền cho nhóm click frequency) — ERIC EJ1273765
