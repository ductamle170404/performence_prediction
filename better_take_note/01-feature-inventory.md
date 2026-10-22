# Danh sách feature được sử dụng

## Phạm vi

Hai chương trình [train.py](../train.py) và [train_cls.py](../train_cls.py) đều
gọi `F.build(d, cutoff)` trong [features.py](../features.py), sau đó gọi
`make_xy(df)`. Vì vậy, phần lớn feature đầu vào của hai chương trình là giống
nhau. Khác biệt chính là:

- `train.py`: classification `at_risk` và regression `final_score`.
- `train_cls.py`: classification với target `at_risk` hoặc `fail_only`, dùng
  thêm Logistic Regression làm baseline.

Các feature chỉ dùng dữ liệu có ngày không vượt quá `cutoff`. Tập test là kỳ
`2014J`; các kỳ `2013B`, `2013J`, `2014B` dùng để train.

## Cột đầu vào trước `make_xy`

### Thông tin sinh viên và đăng ký

- `gender`: giới tính.
- `region`: khu vực.
- `highest_education`: trình độ học vấn cao nhất.
- `imd_band`: nhóm chỉ số thiếu thốn/khu vực; giá trị thiếu được đổi thành
  `Unknown`.
- `age_band`: nhóm tuổi.
- `disability`: trạng thái khuyết tật.
- `num_of_prev_attempts`: số lần học lại môn trước đó.
- `studied_credits`: số tín chỉ đã đăng ký/học.
- `date_registration`: số ngày đăng ký so với ngày bắt đầu môn.

### Tương tác với VLE

- `clicks_total`: tổng số lượt click đến cutoff.
- `active_days`: số ngày có hoạt động.
- `n_sites`: số site VLE khác nhau đã truy cập.
- `clicks_pre_start`: tổng click trước ngày bắt đầu môn.
- `clicks_last7`: tổng click trong 7 ngày gần cutoff.
- `clicks_last14`: tổng click trong 14 ngày gần cutoff.
- `clicks_prev14`: tổng click trong khoảng từ 28 đến 14 ngày trước cutoff.
- `last_active_gap`: số ngày từ lần hoạt động cuối đến cutoff; sinh viên không
  có log nhận giá trị `cutoff + 1`.
- `click_trend`: `clicks_last14 - clicks_prev14`; dương nghĩa là hoạt động gần
  đây tăng.
- `clicks_per_active_day`: `clicks_total / active_days`; không có ngày hoạt
  động thì để thiếu để pipeline xử lý.

Ngoài ra, mỗi `activity_type` trong dữ liệu VLE tạo một cột dạng:

- `clk_<activity_type>`: tổng click của loại hoạt động đó đến cutoff.

Các loại cụ thể phụ thuộc file `vle.csv` và có thể gồm `resource`, `oucontent`,
`subpage`, `homepage`, `forumng`, `quiz`, `url`, `page`, `ouwiki`,
`questionnaire`, `externalquiz`, `dataplus`, `sharedsubpage`, `glossary`,
`folder`, `dualpane`, `htmlactivity`, `repeatactivity`, ` oucollaborate` hoặc
các loại khác có trong dữ liệu. Không nên hard-code số lượng các cột này.

### Bài đánh giá

Chỉ bài không phải `Exam` và có ngày đến hạn `<= cutoff` được sử dụng:

- `n_due`: số bài đã đến hạn.
- `weight_due`: tổng trọng số của các bài đã đến hạn.
- `n_submitted`: số bài đã nộp trước hoặc đúng cutoff.
- `mean_score`: điểm trung bình các bài đã nộp.
- `min_score`: điểm thấp nhất trong các bài đã nộp.
- `last_score`: điểm của bài nộp cuối cùng theo thứ tự dữ liệu.
- `mean_late`: số ngày nộp trễ trung bình; trễ trước ngày đến hạn được tính là
  `0`.
- `submit_rate`: `n_submitted / n_due`; thiếu khi chưa có bài đến hạn.
- `n_missed`: số bài đến hạn nhưng chưa nộp.

## Cột bị loại trước khi huấn luyện

`make_xy` loại các cột sau, nên chúng không phải feature đầu vào:

`id_student`, `code_presentation`, `final_result`, `at_risk`, `fail_only`,
`final_score`, `date_unregistration`, `module_presentation_length`.

## Cột sau `make_xy`

`pd.get_dummies` biến 7 cột phân loại thành cột nhị phân dạng `0.0/1.0`:

`code_module`, `gender`, `region`, `highest_education`, `imd_band`,
`age_band`, `disability`.

Các cột số giữ nguyên. Sau đó tên cột được chuẩn hóa để XGBoost chấp nhận:

- `<=` đổi thành `le_`.
- `<` đổi thành `lt_`.
- `>` đổi thành `gt_`.
- `[` và `]` đổi thành `(` và `)`.
- khoảng trắng đổi thành `_`.

Ví dụ `age_band_55<=` trở thành `age_band_55le_`. Vì danh sách dummy phụ thuộc
giá trị thực tế của dữ liệu, số lượng feature có thể thay đổi giữa các dataset.

## Feature importance

Các file `importance_*.csv` chỉ lưu 20 feature quan trọng nhất của XGBoost và
Random Forest ở từng cutoff. Tên trong file là **tên sau `make_xy`**, không phải
tên cột thô trước one-hot encoding.
