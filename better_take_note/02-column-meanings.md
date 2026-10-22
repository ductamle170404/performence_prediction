# Ý nghĩa các cột và con số trong pipeline

## Quy ước chung

- `cutoff`: mốc ngày cảnh báo tính từ ngày bắt đầu môn.
- `n_*`: số lượng.
- `*_total`: tổng cộng dồn.
- `*_last7`, `*_last14`: tổng trong cửa sổ 7 hoặc 14 ngày gần cutoff.
- `*_prev14`: tổng trong cửa sổ 14 ngày trước đó, từ ngày `cutoff - 28` đến
  `cutoff - 14`.
- `*_rate`: tỷ lệ trong khoảng 0 đến 1, không phải phần trăm.
- `*_score`: điểm, thường theo thang điểm của OULAD (0 đến 100).
- `*_late` hoặc `late_days`: số ngày trễ, giá trị âm được chặn về 0.
- `auc`, `pr_auc`, `f1`, `recall`, `precision`, `accuracy`, `r2`: tỷ lệ/điểm
  trong khoảng 0 đến 1.
- `mae`, `rmse`: sai số điểm; càng thấp càng tốt.

## Cột được tạo trong `features.py`

| Cột | Ý nghĩa con số |
|---|---|
| `date_registration` | Số ngày đăng ký so với ngày bắt đầu môn. |
| `clicks_total` | Tổng số click VLE đến `cutoff`. |
| `active_days` | Số ngày có ít nhất một hoạt động VLE. |
| `n_sites` | Số site VLE khác nhau đã truy cập. |
| `clicks_pre_start` | Click xảy ra trước ngày bắt đầu môn. |
| `clicks_last7` | Click trong 7 ngày cuối trước/đến cutoff. |
| `clicks_last14` | Click trong 14 ngày cuối trước/đến cutoff. |
| `clicks_prev14` | Click trong cửa sổ 14 ngày liền trước cửa sổ gần nhất. |
| `last_active_gap` | Khoảng cách ngày từ lần hoạt động cuối đến cutoff. |
| `click_trend` | Click 14 ngày gần nhất trừ click 14 ngày trước đó. |
| `clicks_per_active_day` | Click trung bình trên mỗi ngày có hoạt động. |
| `clk_<activity_type>` | Tổng click của một loại hoạt động VLE cụ thể. |
| `n_due` | Số bài đánh giá không phải Exam đã đến hạn. |
| `weight_due` | Tổng trọng số của các bài đã đến hạn. |
| `n_submitted` | Số bài đã nộp trước hoặc đúng cutoff. |
| `mean_score` | Điểm trung bình các bài đã nộp. |
| `min_score` | Điểm thấp nhất các bài đã nộp. |
| `last_score` | Điểm của bài nộp cuối cùng. |
| `mean_late` | Số ngày nộp trễ trung bình. |
| `submit_rate` | `n_submitted / n_due`; thiếu nếu `n_due = 0`. |
| `n_missed` | `n_due - n_submitted`, không nhỏ hơn 0. |
| `final_score` | Proxy điểm cuối, trung bình có trọng số từ các bài đã chấm; nếu tổng trọng số bằng 0 thì dùng trung bình thường. Đây là target regression, không phải feature. |

## Nhãn mục tiêu

| Cột | Ý nghĩa con số |
|---|---|
| `at_risk` | `1` nếu `final_result` là `Fail` hoặc `Withdrawn`, ngược lại `0`. |
| `fail_only` | `1` nếu `final_result` là `Fail`, ngược lại `0`; `Withdrawn` được xem là `0`. |
| `final_result` | Nhãn gốc: kết quả cuối như `Pass`, `Fail`, `Withdrawn`, `Distinction`. Không đưa vào feature. |

## Cột kết quả classification trong `train_cls.py`

| Cột | Ý nghĩa con số |
|---|---|
| `cutoff` | Mốc ngày tạo feature và phát cảnh báo. |
| `model` | Tên mô hình: XGBoost, RandomForest hoặc LogReg(baseline). |
| `auc` | Khả năng xếp hạng sinh viên rủi ro; 0.5 gần ngẫu nhiên, 1.0 tốt nhất. |
| `pr_auc` | Diện tích đường Precision-Recall, hữu ích khi lớp rủi ro ít. |
| `f1` | Trung bình điều hòa của precision và recall tại threshold. |
| `recall` | Tỷ lệ sinh viên dương tính thật được cảnh báo đúng. |
| `precision` | Tỷ lệ cảnh báo dương tính là đúng. |
| `accuracy` | Tỷ lệ dự đoán đúng trên toàn bộ sinh viên. |
| `recall_top10` | Tỷ lệ sinh viên rủi ro thật bắt được khi chỉ liên hệ 10% có xác suất cao nhất. |
| `prec_top10` | Tỷ lệ rủi ro thật trong nhóm 10% được chọn. |
| `recall_top30` | Tương tự `recall_top10` nhưng chọn 30%. |
| `prec_top30` | Tỷ lệ rủi ro thật trong nhóm 30% được chọn. |

## Cột kết quả classification trong `train.py`

`train.py` dùng các cột `cutoff`, `model`, `auc`, `pr_auc`, `f1`, `recall`,
`precision`, `accuracy` với cùng ý nghĩa như bảng trên, nhưng chỉ có XGBoost và
RandomForest và target cố định là `at_risk`.

## Cột kết quả regression trong `train.py`

| Cột | Ý nghĩa con số |
|---|---|
| `cutoff` | Mốc ngày cảnh báo. |
| `model` | `Baseline(mean)`, XGBoost hoặc RandomForest. |
| `mae` | Mean Absolute Error: sai lệch tuyệt đối trung bình so với `final_score`. |
| `rmse` | Root Mean Squared Error: phạt mạnh các sai số lớn hơn MAE. |
| `r2` | Mức giải thích biến thiên so với baseline; `1.0` tốt nhất, `0.0` tương đương baseline. |

## Cột theo môn và feature importance

| Cột | Ý nghĩa con số |
|---|---|
| `module` | Mã môn học, ví dụ `AAA`, `BBB`. |
| `n` | Số sinh viên test trong môn đó. |
| `pos_rate` | Tỷ lệ sinh viên dương tính/rủi ro trong môn. |
| `importance` | Mức đóng góp tương đối của feature trong XGBoost hoặc Random Forest; dùng để xếp hạng, không phải xác suất và không tự nó biểu thị tác động tăng/giảm. |
