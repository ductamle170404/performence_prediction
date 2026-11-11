# Tools

Các script tiện ích được nhóm theo mục đích:

| Thư mục | Nội dung |
|---|---|
| `data/` | `assign_classes.py` tạo phân lớp/giảng viên; `make_subset.py` tạo subset để chạy thử |
| `analysis/` | `check_features.py` kiểm tra feature; `audit_xgb_cutoff.py` kiểm tra mô hình ở các cutoff |
| `ranking/` | `rank_list.py` tạo danh sách xếp hạng nguy cơ và các bảng đánh giá |
| `dashboard/` | Dashboard để xem CSV dự đoán; xem [hướng dẫn](dashboard/README.md) |
| `data_checks/` | Các script kiểm tra/khám phá dữ liệu |

Chạy các lệnh từ thư mục gốc dự án để các đường dẫn dữ liệu tương đối như `data_using` và `data_subset` được phân giải đúng. Các script train cũ nằm riêng trong `archive/training/`.

Ví dụ:

```powershell
python tools\data\make_subset.py --src data_pure --out data_subset --frac 0.2
python tools\analysis\check_features.py --data data_subset --cutoffs 28 60
python tools\analysis\audit_xgb_cutoff.py --data data_subset --progress 0 10 20
python tools\ranking\rank_list.py --data data_using --cutoff 60
```
