"""Nhánh 1 — Wu & Jia (Frontiers in Education, 10/2026): dự báo rủi ro trong 4 tuần đầu, kiểm soát rò rỉ thời gian,
giảm phụ thuộc nhân khẩu học.

LƯU Ý: tôi chưa đọc bài gốc; bộ feature dựng theo mô tả trong file nhánh bạn gửi (chỉ có 3 ý: 4 tuần đầu,
leakage-controlled, ít nhân khẩu học). Danh sách feature cụ thể là lựa chọn của tôi theo 3 ý đó.

Thiết kế:
  - cutoff mặc định 28 (4 tuần); cũng chạy được 14 (2 tuần). Tuần chưa kết thúc -> NaN, không điền 0.
  - KHÔNG dùng: gender, region, imd_band, age_band, disability, highest_education
  - chỉ giữ 2 biến "tĩnh" không nhạy cảm: num_of_prev_attempts, studied_credits, + date_registration (hành vi đăng ký)
  - audit(d, cutoff) chạy leakage_check: dựng feature trên dữ liệu đã bị cắt sau cutoff, phải ra kết quả y hệt

Lưu ý với make_xy của train_cls: cột code_module luôn được đưa vào mô hình (nó nằm trong KEY3).

Dùng:  from feature_sets import features_wu_jia as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

DEFAULT_CUTOFFS = [14, 28]
N_WEEKS = 4

FEATURE_GROUPS = {
    "vle_volume": ["clicks_total", "active_days", "clicks_per_active_day", "clicks_pre_start", "n_sites"],
    "vle_weekly": [f"wk{w}_clicks" for w in range(1, N_WEEKS + 1)],
    "vle_recency_trend": ["clicks_last7", "clicks_prev7", "click_trend_7d", "last_active_gap"],
    "vle_activity_mix": [f"share_{g}" for g in C.ACT_GROUPS],
    "assessment": ["n_due", "n_submitted", "n_missed", "submit_rate", "mean_score", "min_score", "mean_late"],
    "static_minimal": ["num_of_prev_attempts", "studied_credits", "date_registration"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def build(d, cutoff=28):
    df = C.base_frame(d, cutoff)
    s = C.vle_until(d, cutoff, df)
    df = C.vle_core(df, s, cutoff)
    df = C.activity_mix(df, s)

    # clicks theo từng tuần đã kết thúc (tối đa N_WEEKS); tuần chưa đủ -> NaN
    W = min((cutoff + 1) // 7, N_WEEKS)
    M = C.weekly_matrix(s, df, cutoff, n_steps=W) if W > 0 else np.zeros((len(df), 0))
    for w in range(1, N_WEEKS + 1):
        df[f"wk{w}_clicks"] = M[:, w - 1] if w <= W else np.nan

    df = C.assess_summary(df, d, cutoff)
    return C.finalize(df, d, FEATURES)


def audit(d, cutoff=28):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
