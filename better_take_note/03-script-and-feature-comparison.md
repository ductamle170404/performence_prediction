# So sánh script và hai module feature

## 1. `rank_list.py` và `check_features.py` nên ở đâu?

Hiện tại, để hai file ở thư mục gốc là hợp lý nhất:

- `rank_list.py` là script chạy mô hình để xếp hạng sinh viên theo rủi ro và ghi kết quả vào `ranked/`. Nó dùng `features.py`, đồng thời import `make_xy` và `TEST_PRES` từ `train_cls.py`.
- `check_features.py` là công cụ kiểm tra các bộ feature trong `feature_sets/`; nó import `features_common.py` và nạp động các module feature.
- Cả hai đều là công cụ chạy quy trình, không phải module định nghĩa bộ feature. Vì vậy không nên chuyển chúng vào `feature_sets/`. `ranked/` là nơi chứa kết quả, không phải nơi chứa script.
- Repo hiện không có thư mục `scripts/` và các script đang import module ở thư mục gốc. Chuyển chúng vào `scripts/` ngay sẽ cần xử lý lại import/cách chạy. Nếu sau này muốn tổ chức riêng, có thể tạo `scripts/` sau khi chuẩn hóa entry point và các import dùng chung.

## 2. `train.py`, `train_cls.py` và `predict_progress.py`

| File | Mục đích | Đặc trưng và mốc | Mô hình/nhãn | Đầu ra, đánh giá |
|---|---|---|---|---|
| `train.py` | Thử phân loại và hồi quy tại các ngày cutoff cố định | Gọi `features.build(d, cutoff)`; dùng bộ feature cũ trong `features.py` | XGBoost và Random Forest; phân loại `at_risk` (Fail + Withdrawn), hồi quy `final_score` | `results/`; ngưỡng phân loại mặc định 0.5 hoặc `--threshold`; có MAE/RMSE/R² cho hồi quy |
| `train_cls.py` | Thử phân loại tại các ngày cutoff cố định | Cũng gọi `features.build(d, cutoff)` | XGBoost, Random Forest, Logistic Regression; mặc định `at_risk`, có thể chọn `fail_only` | `results_cls/`; thêm recall/precision top-K và AUC theo môn; ngưỡng mặc định 0.5 |
| `predict_progress.py` | Dự báo cảnh báo sớm theo phần trăm tiến độ môn | Mỗi lần chọn đúng một module trong `feature_sets/`; mặc định `xai`; mặc định các mốc 0, 10, 20, 30, 40, 50% | XGBoost, Random Forest, Logistic Regression; mặc định nhãn `fail_only`, có thể chọn `at_risk` | `predict/`; có OOF để chọn ngưỡng theo recall mục tiêu, các chỉ số top-K, importance và tùy chọn SHAP/LIME |

Ba file có điểm chung là chia train/test theo presentation mặc định: train trên `2013B`, `2013J`, `2014B`, test trên `2014J`. Tuy nhiên chúng **không tương đương**: hai file `train*` dùng cùng pipeline feature cũ, còn `predict_progress.py` dùng một trong sáu feature builder riêng; hai bên cũng khác cutoff, target mặc định và cách chọn ngưỡng.

### `predict_progress.py` có phải bản nâng cấp toàn diện không?

Không. Đây là pipeline khác, mở rộng tốt hơn cho một mục tiêu cụ thể: dự báo phân loại theo phần trăm tiến độ, dùng feature set lựa chọn được, chọn ngưỡng từ dự đoán out-of-fold trên train và có tùy chọn giải thích. Nó không thay thế toàn bộ khả năng của hai script kia:

- Không có hồi quy `final_score` như `train.py`.
- Không dùng đồng thời mọi feature set; mỗi lần chạy chỉ chọn một bộ (`--feature-set`).
- Định nghĩa feature khác với `features.py`, nên so sánh điểm số trực tiếp không phải phép so sánh ngang bằng nếu chưa đồng bộ feature, target và cutoff.
- Mặc định target khác: `predict_progress.py` là `fail_only`; `train_cls.py` là `at_risk`. `train.py` luôn đánh giá classification theo `at_risk`.
- Cách đánh giá/đầu ra cũng khác: `predict_progress.py` dùng cutoff theo phần trăm, OOF threshold và lưu dự đoán theo từng mốc; hai script còn lại dùng ngày cutoff và ngưỡng xác suất cố định.

Có thể xem `predict_progress.py` là pipeline chuyên biệt và nhiều chức năng hơn cho cảnh báo theo tiến độ, chứ không phải bản thay thế toàn diện.

## 3. `features.py` và `features_common.py` có giống nhau không?

Không. Chúng liên quan và dùng chung một phần định nghĩa, nhưng đảm nhiệm vai trò khác nhau:

| File | Vai trò | Điểm đáng chú ý |
|---|---|---|
| `features.py` | Bộ feature cũ hoàn chỉnh, đang được `train.py`, `train_cls.py` và `rank_list.py` sử dụng | Tự xây feature Student–Course; gồm tổng click, ngày hoạt động, click theo loại VLE, cửa sổ gần nhất 7/14 ngày, xu hướng 14 ngày và tóm tắt bài đánh giá |
| `features_common.py` | Bộ hàm tiện ích dùng chung cho các module trong `feature_sets/`; không phải bộ feature độc lập | Import `load`, `_targets`, `KEY3`, `KMP` từ `features.py`; bổ sung hàm nền/VLE/assessment, activity mix, streak, ma trận tuần và kiểm tra rò rỉ |

Các khác biệt cụ thể:

- Cách tạo feature khác: `features.py` dựng một bảng đặc trưng cố định; `features_common.py` cung cấp các helper để từng module chọn và ghép bộ đặc trưng riêng.
- Cửa sổ thời gian khác: bộ cũ có `clicks_last14`, `clicks_prev14`, `click_trend`; helper chung dùng `clicks_last7`, `clicks_prev7`, `click_trend_7d`, và có thể tạo ma trận theo tuần đã kết thúc.
- Nhóm hoạt động khác: `features.py` giữ cột theo `activity_type` gốc (`clk_<activity_type>`); helper chung ánh xạ chúng thành năm nhóm (`content`, `forum`, `quiz`, `homepage`, `other`).
- Một số hành vi lại được chia sẻ có chủ ý: loader/khóa và cách tính `final_score` lấy từ `features.py`; `features_common.py` cũng dùng chung nguyên tắc giữ sinh viên còn đăng ký tại cutoff và lọc dữ liệu theo ngày.

Vì vậy `features_common.py` không phải bản sao hay bản nâng cấp trực tiếp của `features.py`; nó là lớp tiện ích cho họ feature mới. Muốn biết mô hình nào thực sự dùng cột nào, cần xem module feature được chọn và bước `make_xy`.

## 4. File liên quan

- [features.py](../features.py)
- [features_common.py](../features_common.py)
- [train.py](../train.py)
- [train_cls.py](../train_cls.py)
- [rank_list.py](../rank_list.py)
- [check_features.py](../check_features.py)
- [predict_progress.py](../predict_progress.py)
- [feature_sets_overview.md](../feature_sets_overview.md)
