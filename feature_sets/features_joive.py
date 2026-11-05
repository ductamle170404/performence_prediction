"""Nhánh 1 — nghiên cứu JOIVE (7/2026): đặc trưng chuỗi hành vi LMS = Inactivity Streak, Access Duration, Click frequency.

LƯU Ý: tôi chưa đọc bài gốc. Mô tả bạn gửi chỉ nêu tên 3 nhóm này nên cách tính là của tôi.
OULAD KHÔNG có thời lượng phiên (session) nên "Access Duration" chỉ là xấp xỉ qua số ngày truy cập và
khoảng cách từ lần truy cập đầu đến cuối.

  inactivity_streak : max_inactive_streak, current_inactive_streak, n_streaks_ge7, n_inactive_days, inactive_ratio
  access_duration   : n_active_days, access_span, first_active_day, last_active_gap
  click_frequency   : clicks_total, clicks_per_active_day, clicks_per_day, daily_clicks_std, daily_clicks_max, clicks_cv

Chỉ dùng log VLE (date >= 0, tới cutoff) — đúng tinh thần "chuỗi hành vi LMS", không có nhân khẩu học, không có điểm.
Sinh viên chưa từng truy cập: streak = cutoff+1, các thống kê click = 0 hoặc NaN (access_span, first_active_day).

Dùng:  from feature_sets import features_joive as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

DEFAULT_CUTOFFS = [14, 28, 60, 90]

FEATURE_GROUPS = {
    "inactivity_streak": ["max_inactive_streak", "current_inactive_streak", "n_streaks_ge7",
                          "n_inactive_days", "inactive_ratio"],
    "access_duration": ["n_active_days", "access_span", "first_active_day", "last_active_gap"],
    "click_frequency": ["clicks_total", "clicks_per_active_day", "clicks_per_day",
                        "daily_clicks_std", "daily_clicks_max", "clicks_cv"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def build(d, cutoff=28):
    df = C.base_frame(d, cutoff)
    s = C.vle_until(d, cutoff, df)
    s = s[s["date"] >= 0]
    df = C.streak_features(df, s, cutoff)

    tot = s.groupby(KEY3)["sum_click"].sum().rename("clicks_total").reset_index()
    df = df.merge(tot, on=KEY3, how="left")
    df["clicks_total"] = df["clicks_total"].fillna(0)
    df["clicks_per_active_day"] = np.where(df["n_active_days"] > 0,
                                           df["clicks_total"] / df["n_active_days"].where(df["n_active_days"] > 0), 0.0)
    df["clicks_per_day"] = df["clicks_total"] / (cutoff + 1)
    df["last_active_gap"] = (cutoff - df["last_active_day"]).fillna(cutoff + 1)
    df["daily_clicks_std"] = df["daily_clicks_std"].fillna(0)      # 0 hoặc 1 ngày hoạt động -> không có độ lệch
    df["daily_clicks_max"] = df["daily_clicks_max"].fillna(0)
    df["clicks_cv"] = np.where(df["daily_clicks_mean"] > 0,
                               df["daily_clicks_std"] / df["daily_clicks_mean"].where(df["daily_clicks_mean"] > 0), 0.0)
    return C.finalize(df, d, FEATURES)


def audit(d, cutoff=28):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
