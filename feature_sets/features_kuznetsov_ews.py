"""Bộ feature tái dựng theo repo KuznetsovKarazin/oulad-early-warning (Interpretable DSS, 14 ngày đầu).

LƯU Ý: tôi chỉ đọc README của repo, CHƯA đọc mã nguồn. Tên feature lấy từ README; những chỗ README không
định nghĩa thì tôi chọn một cách tính hợp lý và ghi chú "(giả định)" bên dưới.

3 nhóm như repo (activity / assessment / demographic):
  activity     total_clicks, active_days_ratio, days_since_last_activity, avg_clicks_per_day
  assessment   completed_first_assessment, first_assessment_score(_norm), first_assessment_on_time
  demographic  education_level (0-4), age_group (0-2), num_of_prev_attempts

Giả định:
  - cửa sổ = ngày 0..cutoff (repo dùng cutoff = 14; ở đây cutoff tham số hóa để chạy các mốc khác)
  - avg_clicks_per_day = tổng click / số ngày có hoạt động (README không nói là theo ngày lịch hay ngày hoạt động)
  - "bài đánh giá đầu tiên" = bài TMA/CMA (không phải Exam) có hạn nộp sớm nhất của module-đợt đó
Mở rộng (không có trong README): first_assessment_due_by_cutoff, để phân biệt "chưa đến hạn" với "không nộp".

Dùng:  from feature_sets import features_kuznetsov_ews as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

DEFAULT_CUTOFFS = [14]

FEATURE_GROUPS = {
    "activity": ["total_clicks", "active_days_ratio", "days_since_last_activity", "avg_clicks_per_day"],
    "assessment": ["completed_first_assessment", "first_assessment_score", "first_assessment_score_norm",
                   "first_assessment_on_time", "first_assessment_due_by_cutoff"],
    "demographic": ["education_level", "age_group", "num_of_prev_attempts"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def build(d, cutoff=14):
    df = C.base_frame(d, cutoff)

    # --- activity: chỉ tính từ ngày bắt đầu khóa (date >= 0) ---
    s = C.vle_until(d, cutoff, df)
    s = s[s["date"] >= 0]
    g = s.groupby(KEY3)
    f = g["sum_click"].sum().rename("total_clicks").to_frame()
    f["active_days"] = g["date"].nunique()
    f["last_day"] = g["date"].max()
    df = df.merge(f.reset_index(), on=KEY3, how="left")
    df[["total_clicks", "active_days"]] = df[["total_clicks", "active_days"]].fillna(0)
    df["active_days_ratio"] = df["active_days"] / (cutoff + 1)
    df["days_since_last_activity"] = (cutoff - df["last_day"]).fillna(cutoff + 1)
    df["avg_clicks_per_day"] = np.where(df["active_days"] > 0,
                                        df["total_clicks"] / df["active_days"].where(df["active_days"] > 0), 0.0)

    # --- assessment: bài đầu tiên (hạn sớm nhất, không phải Exam) của từng module-đợt ---
    a, sa = d["assessments"], d["studentAssessment"]
    fa = (a[(a["assessment_type"] != "Exam") & a["date"].notna()]
          .sort_values(KMP + ["date", "id_assessment"]).groupby(KMP).head(1)
          [["id_assessment", "date"] + KMP].rename(columns={"id_assessment": "fa_id", "date": "fa_due"}))
    sub = sa.merge(fa, left_on="id_assessment", right_on="fa_id")
    sub = sub[sub["date_submitted"] <= cutoff][KEY3 + ["date_submitted", "score", "fa_due"]]
    sub = sub.rename(columns={"date_submitted": "fa_sub", "score": "first_assessment_score"})
    df = df.merge(fa[KMP + ["fa_due"]], on=KMP, how="left").merge(sub.drop(columns="fa_due"), on=KEY3, how="left")
    df["completed_first_assessment"] = df["fa_sub"].notna().astype(int)
    df["first_assessment_score_norm"] = df["first_assessment_score"] / 100.0
    df["first_assessment_on_time"] = np.where(df["fa_sub"].notna(), (df["fa_sub"] <= df["fa_due"]).astype(float), np.nan)
    df["first_assessment_due_by_cutoff"] = (df["fa_due"] <= cutoff).astype(int)

    # --- demographic: mã hóa thứ tự (ordinal) như repo ---
    df["education_level"] = df["highest_education"].map(C.EDU_ORDER).astype(float)
    df["age_group"] = df["age_band"].map(C.AGE_ORDER).astype(float)

    return C.finalize(df, d, FEATURES)


def audit(d, cutoff=14):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
