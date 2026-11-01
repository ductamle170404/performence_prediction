"""Nhánh 3 — XAI & công bằng (MDPI Applied Sciences, 1/2025, "Explainable AI in Education").

LƯU Ý: file nhánh bạn gửi nói bài dùng "14 biến đầu vào của OULAD" nhưng KHÔNG liệt kê 14 biến đó, nên đây
KHÔNG phải bản tái lập. Đây là bộ "ưu tiên giải thích được": mỗi feature đều dịch được ra một câu cho cố vấn
(xem EXPLAIN, describe), gồm đúng loại ví dụ trong mô tả nhánh: "nộp muộn TMA số 1", "click oucontent giảm nhiều tuần liên tiếp".
Hai biến mô tả nói là quan trọng nhất (số bài đã nộp, tương tác VLE) có mặt: n_submitted, clicks_total.

Nhóm:
  assessment_core    n_submitted (assessment_count), n_due, n_missed, submit_rate, mean_score, n_late, late_ratio
  vle_core           clicks_total, active_days, clicks_last14
  vle_by_type        clk_content, clk_forum, clk_quiz, clk_homepage, clk_other, oucontent_clicks
  explainable_events tma1_submitted, tma1_late_days, tma1_score, oucontent_decline_weeks, oucontent_last_week_change,
                     oucontent_zero_weeks_last3
  demographic        gender, region, highest_education, imd_band, age_band, disability, num_of_prev_attempts, studied_credits

build(d, cutoff, demographics=True): đặt demographics=False để bỏ nhóm nhân khẩu học khỏi mô hình mà vẫn kiểm tra công bằng
bằng sensitive_frame() (cùng khóa KEY3) — nối với dự đoán để tính Demographic Parity / Equalized Odds ở bên ngoài.

Dùng:  import features_xai as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

DEFAULT_CUTOFFS = [28, 60, 90]
SENSITIVE = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]

FEATURE_GROUPS = {
    "assessment_core": ["n_submitted", "n_due", "n_missed", "submit_rate", "mean_score", "n_late", "late_ratio"],
    "vle_core": ["clicks_total", "active_days", "clicks_last14"],
    "vle_by_type": [f"clk_{g}" for g in C.ACT_GROUPS] + ["oucontent_clicks"],
    "explainable_events": ["tma1_submitted", "tma1_late_days", "tma1_score", "oucontent_decline_weeks",
                           "oucontent_last_week_change", "oucontent_zero_weeks_last3"],
    "demographic": SENSITIVE + ["num_of_prev_attempts", "studied_credits"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

EXPLAIN = {
    "n_submitted": "đã nộp {v:.0f} bài đánh giá",
    "n_missed": "chưa nộp {v:.0f} bài đã đến hạn",
    "submit_rate": "tỉ lệ nộp bài đến hạn {v:.0%}",
    "mean_score": "điểm trung bình các bài đã nộp {v:.0f}",
    "n_late": "nộp trễ {v:.0f} bài",
    "late_ratio": "{v:.0%} bài đã nộp là nộp trễ",
    "clicks_total": "tổng {v:.0f} lượt click trên VLE",
    "active_days": "hoạt động {v:.0f} ngày",
    "clicks_last14": "{v:.0f} lượt click trong 14 ngày gần nhất",
    "tma1_submitted": "TMA số 1: {v:.0f} (1 = đã nộp, 0 = chưa nộp dù đã đến hạn)",
    "tma1_late_days": "nộp TMA số 1 trễ {v:.0f} ngày",
    "tma1_score": "điểm TMA số 1: {v:.0f}",
    "oucontent_decline_weeks": "click oucontent giảm {v:.0f} tuần liên tiếp",
    "oucontent_last_week_change": "click oucontent tuần gần nhất thay đổi {v:+.0%} so với tuần trước",
    "oucontent_zero_weeks_last3": "{v:.0f}/3 tuần gần nhất không click oucontent",
}


def describe(feature, value):
    """Câu mô tả tiếng Việt cho (feature, giá trị) — dùng để hiển thị cùng SHAP/LIME."""
    if feature in EXPLAIN and value == value:      # value == value: loại NaN
        return EXPLAIN[feature].format(v=value)
    return f"{feature} = {value}"


def sensitive_frame(d, cutoff):
    """KEY3 + thuộc tính nhạy cảm của đúng nhóm sinh viên mà build() trả về (để kiểm tra công bằng)."""
    return C.base_frame(d, cutoff)[KEY3 + SENSITIVE + ["num_of_prev_attempts"]]


def _decline_features(M):
    """Từ ma trận click theo tuần (n, W): số tuần giảm liên tiếp tính đến tuần cuối, % đổi tuần cuối, số tuần = 0 trong 3 tuần cuối."""
    n, W = M.shape
    nan = np.full(n, np.nan)
    if W < 2:
        return nan, nan.copy(), nan.copy()
    dec = M[:, 1:] < M[:, :-1]
    alive, run = np.ones(n, dtype=bool), np.zeros(n)
    for j in range(dec.shape[1] - 1, -1, -1):
        alive &= dec[:, j]
        run += alive
    change = (M[:, -1] - M[:, -2]) / np.maximum(M[:, -2], 1.0)
    zero3 = (M[:, -3:] == 0).sum(axis=1).astype(float)
    return run, change, zero3


def build(d, cutoff=60, demographics=True):
    df = C.base_frame(d, cutoff)
    s = C.vle_until(d, cutoff, df)
    df = C.vle_core(df, s, cutoff)
    df["clicks_last14"] = df["clicks_last7"] + df["clicks_prev7"]
    df = C.activity_mix(df, s)

    s_ou = s[s["activity_type"] == "oucontent"]
    ou = s_ou.groupby(KEY3)["sum_click"].sum().rename("oucontent_clicks").reset_index()
    df = df.merge(ou, on=KEY3, how="left")
    df["oucontent_clicks"] = df["oucontent_clicks"].fillna(0)
    run, change, zero3 = _decline_features(C.weekly_matrix(s_ou, df, cutoff))
    df["oucontent_decline_weeks"], df["oucontent_last_week_change"], df["oucontent_zero_weeks_last3"] = run, change, zero3

    df = C.assess_summary(df, d, cutoff)
    df["late_ratio"] = np.where(df["n_submitted"] > 0, df["n_late"] / df["n_submitted"].where(df["n_submitted"] > 0), np.nan)

    # TMA số 1 = TMA có hạn sớm nhất của module-đợt
    a, sa = d["assessments"], d["studentAssessment"]
    t1 = (a[(a["assessment_type"] == "TMA") & a["date"].notna()]
          .sort_values(KMP + ["date", "id_assessment"]).groupby(KMP).head(1)
          [["id_assessment", "date"] + KMP].rename(columns={"id_assessment": "t1_id", "date": "t1_due"}))
    sub = sa.merge(t1, left_on="id_assessment", right_on="t1_id")
    sub = sub[sub["date_submitted"] <= cutoff][KEY3 + ["date_submitted", "score"]]
    sub = sub.rename(columns={"date_submitted": "t1_sub", "score": "tma1_score"})
    df = df.merge(t1[KMP + ["t1_due"]], on=KMP, how="left").merge(sub, on=KEY3, how="left")
    df["tma1_submitted"] = np.where(df["t1_sub"].notna(), 1.0, np.where(df["t1_due"] <= cutoff, 0.0, np.nan))
    df["tma1_late_days"] = (df["t1_sub"] - df["t1_due"]).clip(lower=0)

    feats = [f for g, fs in FEATURE_GROUPS.items() if demographics or g != "demographic" for f in fs]
    return C.finalize(df, d, feats)


def audit(d, cutoff=60):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
