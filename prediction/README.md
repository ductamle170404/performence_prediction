# Prediction pipeline

`predict_progress.py` là entry point dự đoán đang dùng. Chạy lệnh từ thư mục gốc repo:

```powershell
python prediction\predict_progress.py --data data_subset --no-shap --no-lime
```

Nếu bỏ `--out`, các file của lần chạy mới được ghi vào `database/predictions/`. Có thể chọn một thư mục khác bằng `--out`.

`features.py` cung cấp cách nạp dữ liệu, nhãn và feature nền. `features_common.py` và các module trong `feature_sets/` xây dựng feature theo cutoff; chúng cùng thuộc pipeline đang chạy. Dữ liệu mặc định của pipeline là `data_using/`, hoặc truyền bộ dữ liệu khác qua `--data`.

Các kết quả cũ không được tự di chuyển: `predict/` và `predict_xai/` vẫn được giữ ở vị trí hiện tại.
