# Luồng feature và cutoff trong `predict_progress.py`

## Trả lời ngắn

Đúng theo ý chính: mỗi `--feature-set` được chọn sẽ được dựng lại độc lập tại **từng mốc `--progress`**. Với XGBoost, mỗi mốc có ma trận feature riêng, mô hình được huấn luyện lại trên dữ liệu train của mốc đó rồi mới dự đoán test cùng mốc. Mô hình không nhận feature ở các mốc trước làm dữ liệu nối thêm vào mốc sau.

Tuy vậy, “dữ liệu tới cutoff” không có nghĩa mọi cột đều cộng dồn ngày 0 đến cutoff theo cùng một cách. Có bộ tính cả log trước ngày khai giảng; feature theo tuần thường chỉ lấy **tuần đã kết thúc**; bộ MTAPSP chia thời gian thành đoạn 54 ngày. Chi tiết phụ thuộc module feature.

## Luồng chạy từ đầu xuống dưới

1. **Chọn đúng một bộ feature.**  
   `--feature-set` chọn một trong `kuznetsov_ews`, `wu_jia`, `joive`, `mtapsp`, `xai`, `trajectory`; mặc định là `xai`. Một lần chạy không tự động lần lượt so sánh cả sáu bộ. Muốn so sánh, chạy riêng từng bộ.

2. **Đọc dữ liệu nguồn.**  
   Module đã chọn gọi `load()` (được dùng lại từ `features.py`) để đọc các bảng. `predict_progress.py` nạp dữ liệu một lần; trong từng bước dựng feature, nó thu hẹp bảng về đúng môn và presentation.

3. **Lặp qua từng phần trăm tiến độ.**  
   Mặc định là `0, 10, 20, 30, 40, 50`, có thể thay bằng `--progress`. Với mỗi môn/presentation, chương trình tính:

   `cutoff_day = floor(module_presentation_length × progress_pct / 100)`

   Do độ dài presentation có thể khác nhau, cùng một phần trăm có thể ứng với số ngày khác nhau.

4. **Tách dữ liệu theo presentation.**  
   `subset_presentation()` giữ student, registration, course, assessment và log của đúng presentation. Sau đó `build_all_presentations()` gọi builder của feature set với `cutoff_day` vừa tính.

5. **Builder dựng ảnh chụp feature tại cutoff.**  
   Các builder gọi helper chung như `base_frame()` và `vle_until()`, nhưng tự chọn cột, quy tắc tuần/đoạn và quy tắc lọc riêng. Sinh viên đã rút khỏi môn tại hoặc trước cutoff bị loại; điều kiện giữ lại là `date_unregistration` bị thiếu hoặc lớn hơn cutoff.

   Các hàm `build()` có tham số cutoff mặc định, và module có `DEFAULT_CUTOFFS`, nhưng **giá trị mặc định đó không điều khiển `predict_progress.py`**: lời gọi từ pipeline truyền `cutoff_day` một cách tường minh. `DEFAULT_CUTOFFS` chủ yếu được dùng bởi `check_features.py` khi người chạy không đưa `--cutoffs`.

6. **Tách train/test tại chính mốc đó.**  
   Mặc định train là `2013B`, `2013J`, `2014B`, test là `2014J`. Tập train và test đều là feature snapshot của cùng `progress_pct`, không phải một bộ lấy ở thời điểm khác.

7. **Tạo ma trận đầu vào.**  
   `make_xy()` bỏ ID sinh viên, mã presentation, nhãn/kết quả cuối (`final_result`, `at_risk`, `fail_only`, `final_score`) và metadata cutoff/độ dài. `code_module` không bị loại: nó được one-hot encode cùng các trường phân loại khác. Các cột số giữ nguyên; giá trị vô hạn được đổi thành thiếu. Cột test được căn theo cột train; cột thiếu ở test được điền 0. Nhãn `at_risk` hoặc `fail_only` được lấy riêng làm `y`, không đưa vào `X`.

8. **Chọn ngưỡng bằng train, không dùng nhãn test.**  
   Ở mỗi mốc, chương trình tạo dự đoán out-of-fold trên các presentation train theo cách lần lượt giữ lại một presentation. Với mỗi mô hình, nó chọn ngưỡng cao nhất đạt recall mục tiêu (`--target-recall`, mặc định 0.85) trên OOF. Nhãn test không dùng để chọn ngưỡng.

9. **Huấn luyện và dự đoán XGBoost ở mốc hiện tại.**  
   XGBoost được fit trên toàn bộ `X_train`/`y_train` của cutoff hiện tại rồi dự đoán xác suất trên `X_test` của cutoff đó. Sang cutoff tiếp theo, feature được dựng lại và XGBoost được fit lại. XGBoost nhận các giá trị thiếu của feature trực tiếp; `predict_progress.py` không áp dụng bước điền median cho XGBoost (median imputation nằm trong pipeline Random Forest/Logistic Regression).

10. **Ghi kết quả riêng theo mốc.**  
    Dự đoán lưu metadata (sinh viên, môn, presentation, tiến độ/ngày cutoff, kết quả thật để đối chiếu), xác suất, ngưỡng và cờ dự đoán. Chương trình ghi file dự đoán, đánh giá, đánh giá theo môn cho XGBoost, importance; SHAP/LIME là tùy chọn. Nếu dùng kết quả để vận hành, cần lưu ý cột `actual_fail`/`final_result` trong file dự đoán là thông tin đánh giá hậu nghiệm, không phải đầu vào khi chấm điểm.

## Mỗi bộ lấy feature theo cửa sổ nào?

`vle_until(d, cutoff)` giữ log VLE có `date <= cutoff`, tức về nguyên tắc có cả log ngày âm trước khi bắt đầu khóa. Một số builder lọc bỏ ngày âm, một số giữ chúng cho feature tổng hợp. Vì thế “từ đầu đến cutoff” phải hiểu theo định nghĩa từng cột, không phải một luật duy nhất cho cả bộ.

| Bộ | Cách tính ở cutoff |
|---|---|
| `kuznetsov_ews` | Lọc VLE thêm `date >= 0`, vì vậy feature hoạt động lấy từ ngày 0 đến cutoff (bao gồm ngày cutoff), không lấy log trước khóa. Tính tổng click, số ngày hoạt động, tỉ lệ ngày hoạt động, khoảng cách từ lần cuối và trung bình click/ngày hoạt động. Chọn bài không phải Exam có hạn sớm nhất; chỉ kết quả nộp có `date_submitted <= cutoff` mới được đưa vào. Có cờ bài đầu đã đến hạn hay chưa và feature nhân khẩu học mã hóa thứ tự. |
| `wu_jia` | Feature VLE tổng hợp, recency và activity mix dùng log `date <= cutoff`; do không lọc ngày âm, các tổng này có thể gồm log trước khóa (`clicks_pre_start` cũng được tách riêng). Click theo tuần chỉ tạo cho tuần 1–4 đã kết thúc: một tuần đủ 7 ngày từ ngày 0; tuần chưa đủ thì `NaN`, không phải 0. Tóm tắt assessment chỉ xét bài không phải Exam đã đến hạn và đã nộp trước/đúng cutoff. |
| `joive` | Lọc log VLE về `0 <= date <= cutoff`. Từ những ngày có click đó tính tổng click, số ngày hoạt động, các chuỗi/khoảng nghỉ, khoảng cách từ hoạt động cuối, số click/ngày và độ biến thiên click theo ngày. Không dùng điểm hay demographic. |
| `mtapsp` | Chia ngày khóa thành 5 đoạn cố định, mỗi đoạn 54 ngày (0–53, 54–107, ...). Log được giới hạn `date <= cutoff`, rồi cộng click/số ngày hoạt động theo đoạn. Đoạn đã bắt đầu nhưng mới quan sát một phần chỉ chứa dữ liệu phần đã thấy; đoạn chưa bắt đầu là `NaN`; đoạn đã bắt đầu nhưng chưa có hoạt động là 0. Log trước khóa được tính riêng ở `clicks_pre_start`. TMA/CMA chỉ lấy bài đủ điều kiện theo cutoff; gom theo ngày nộp đã cắt về không âm. Có thêm điểm tổng hợp và demographic. |
| `xai` | Feature VLE tổng hợp và activity mix lấy log `date <= cutoff`, nên có thể bao gồm log trước khóa; `clicks_pre_start` nằm riêng trong nhóm VLE cơ bản. `clicks_last14` được ghép từ hai cửa sổ 7 ngày gần nhất. Feature xu hướng `oucontent` dùng ma trận tuần đã kết thúc, không dùng phần tuần đang dở. Assessment lọc theo cutoff. TMA đầu tiên được chọn theo hạn sớm nhất; điểm/ngày nộp chỉ có nếu đã nộp tới cutoff, và cờ chưa nộp phân biệt chưa đến hạn với đã đến hạn. |
| `trajectory` | Feature thống kê theo tuần lấy ma trận các tuần **đã kết thúc**; khi chưa đủ 3 tuần thì nhóm thống kê tuần là `NaN`. Các feature streak/ngày hoạt động khác tính từ ngày 0 đến cutoff, có thể gồm phần tuần hiện tại đã quan sát. Feature học dồn dùng log tới cutoff và lịch hạn của assessment đã đến hạn; tỉ lệ click trước khóa được giữ riêng. Bộ này không dùng điểm bài đánh giá. |

### Cách hiểu đúng về VLE và tiến trình

- Với các feature cộng dồn, mỗi cutoff được tính lại trên toàn bộ log hợp lệ trong cửa sổ của feature từ dữ liệu ban đầu đến cutoff; không chỉ lấy log đúng ngày cutoff.
- Các cột như `clicks_last7`/`clicks_last14` là cửa sổ gần đây, không phải tổng từ đầu khóa.
- `weekly_matrix()` chỉ dựng các tuần trọn vẹn từ ngày 0. Vì thế `xai`/`trajectory` có thể bỏ qua những ngày của tuần hiện tại đối với feature theo tuần, trong khi các feature VLE tổng hợp hoặc streak vẫn dùng log đến cutoff.
- MTAPSP là ngoại lệ có biểu diễn theo 5 đoạn ngày cố định; đoạn đang diễn ra có thể chỉ là dữ liệu một phần.
- Không phải feature nào cũng là hành vi. Assessment, demographic, trạng thái đăng ký có luật lọc riêng; đặc biệt trạng thái đăng ký làm thay đổi cả số lượng và danh tính sinh viên được chấm ở các cutoff.

## Kết luận

Với XGBoost, mỗi bộ feature được chọn là một cấu hình đầu vào riêng; mỗi `progress` tạo một snapshot riêng và một lần fit/predict riêng. Cửa sổ log VLE thường là tích lũy tới ngày cutoff, nhưng cần đọc định nghĩa từng feature để biết có lọc ngày âm, chỉ lấy tuần đủ, hay chia đoạn thời gian. Không nên giả định mọi feature đều chứa đầy đủ từng ngày từ 0 đến cutoff.

## File liên quan

- [predict_progress.py](../prediction/predict_progress.py)
- [features_common.py](../prediction/features_common.py)
- [features.py](../prediction/features.py)
- [features_kuznetsov_ews.py](../prediction/feature_sets/features_kuznetsov_ews.py)
- [features_wu_jia.py](../prediction/feature_sets/features_wu_jia.py)
- [features_joive.py](../prediction/feature_sets/features_joive.py)
- [features_mtapsp.py](../prediction/feature_sets/features_mtapsp.py)
- [features_xai.py](../prediction/feature_sets/features_xai.py)
- [features_trajectory.py](../prediction/feature_sets/features_trajectory.py)
