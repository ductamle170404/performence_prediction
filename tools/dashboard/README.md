# Xem kết quả dự đoán

Dashboard đọc các file do `prediction\predict_progress.py` sinh ra. Từ thư mục gốc dự án, chạy:

```powershell
.\venv\Scripts\python.exe tools\dashboard\dashboard.py
```

Mặc định dashboard đọc `database/predictions/` và mở tại `http://127.0.0.1:8765/`. Nếu muốn xem bộ kết quả cũ đang giữ ở `predict/`, chạy:

```powershell
.\venv\Scripts\python.exe tools\dashboard\dashboard.py --data-dir predict
```

Để đóng gói dữ liệu hiện chọn vào file HTML độc lập, dùng `--export`:

```powershell
.\venv\Scripts\python.exe tools\dashboard\dashboard.py --export
```

Lệnh export ghi dữ liệu vào `tools/dashboard/dashboard.html`. Chỉ chạy khi muốn cập nhật bản HTML độc lập; để mở dashboard thông thường không cần export.
