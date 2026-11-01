"""Nhánh 4 — quỹ đạo học tập (Pecuchova & Drlik 2024; Gupta et al. 2022): phân cụm, Markov, phân tích sống còn.

LƯU Ý: tôi chưa đọc hai bài này; bộ feature dựng theo mô tả trong file nhánh bạn gửi (nhóm "học dồn trước kỳ thi",
"học đều đặn", "bỏ cuộc ngầm"; chuỗi Markov; tuần rời khóa). Danh sách cụ thể là lựa chọn của tôi.

Chỉ dùng HÀNH VI (không nhân khẩu học, không điểm) trên các tuần ĐÃ KẾT THÚC tới cutoff; ít hơn MIN_WEEKS tuần -> NaN.
Cutoff mặc định 56/84/140/160 ngày = 8/12/20/23 tuần (140–160 ≈ 60% khóa).

  regularity     active_week_ratio, zero_week_ratio, weekly_cv, weekly_entropy, burstiness
  momentum       trend_slope_norm, late_early_ratio, peak_week_pos, recent_vs_mean
  silent_dropout trailing_zero_weeks, longest_zero_run, n_zero_runs, last_active_gap
  cramming       pre_deadline_click_share, pre_start_click_share, first_active_day
  markov         p_stay_zero, p_drop_to_zero, state_change_rate
                 (trạng thái tuần: 0 = không click, 1 = 1..LOW_T click, 2 = hơn LOW_T; LOW_T là ngưỡng tùy ý, chỉnh được)

Hàm phụ trợ:
  weekly_profile(d, df, cutoff)  tỉ trọng click mỗi tuần (n × W) để chạy K-Means / DBSCAN / Hierarchical
  survival_table(d)              (duration, event) cho Kaplan-Meier / Cox: event = Withdrawn có ngày hủy đăng ký

Dùng:  import features_trajectory as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

DEFAULT_CUTOFFS = [56, 84, 140, 160]
MIN_WEEKS = 3
LOW_T = 30

FEATURE_GROUPS = {
    "regularity": ["active_week_ratio", "zero_week_ratio", "weekly_cv", "weekly_entropy", "burstiness"],
    "momentum": ["trend_slope_norm", "late_early_ratio", "peak_week_pos", "recent_vs_mean"],
    "silent_dropout": ["trailing_zero_weeks", "longest_zero_run", "n_zero_runs", "last_active_gap"],
    "cramming": ["pre_deadline_click_share", "pre_start_click_share", "first_active_day"],
    "markov": ["p_stay_zero", "p_drop_to_zero", "state_change_rate"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _div(a, b):
    """a / b, NaN khi b == 0."""
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _weekly_stats(M):
    n, W = M.shape
    out = {}
    tot, mean = M.sum(1), M.mean(1)
    std = M.std(1)
    z = M == 0
    out["active_week_ratio"] = (~z).mean(1)
    out["zero_week_ratio"] = z.mean(1)
    out["weekly_cv"] = _div(std, mean)
    p = M / np.where(tot > 0, tot, 1.0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(W)
    out["weekly_entropy"] = np.where(tot > 0, ent, np.nan)
    out["burstiness"] = _div(std - mean, std + mean)
    x = np.arange(W) - (W - 1) / 2
    out["trend_slope_norm"] = _div((M * x).sum(1) / (x ** 2).sum(), mean)
    h = W // 2
    out["late_early_ratio"] = (M[:, W - h:].sum(1) + 1) / (M[:, :h].sum(1) + 1)
    out["peak_week_pos"] = np.where(tot > 0, M.argmax(1) / max(W - 1, 1), np.nan)
    out["recent_vs_mean"] = _div(M[:, -2:].mean(1), mean)

    alive, tz = np.ones(n, dtype=bool), np.zeros(n)
    cur, longest, runs, prev_z = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    for j in range(W - 1, -1, -1):
        alive &= z[:, j]
        tz += alive
    for j in range(W):
        cur = (cur + 1) * z[:, j]
        longest = np.maximum(longest, cur)
        runs += z[:, j] & ~prev_z
        prev_z = z[:, j]
    out["trailing_zero_weeks"], out["longest_zero_run"], out["n_zero_runs"] = tz, longest, runs

    S = np.where(z, 0, np.where(M <= LOW_T, 1, 2))
    a, b = S[:, :-1], S[:, 1:]
    out["p_stay_zero"] = _div(((a == 0) & (b == 0)).sum(1), (a == 0).sum(1))
    out["p_drop_to_zero"] = _div(((a > 0) & (b == 0)).sum(1), (a > 0).sum(1))
    out["state_change_rate"] = (a != b).mean(1)
    return out


def build(d, cutoff=84):
    df = C.base_frame(d, cutoff)
    s = C.vle_until(d, cutoff, df)
    M = C.weekly_matrix(s, df, cutoff)
    W = M.shape[1]
    if W >= MIN_WEEKS:
        for k, v in _weekly_stats(M).items():
            df[k] = v
    else:
        for k in ["active_week_ratio", "zero_week_ratio", "weekly_cv", "weekly_entropy", "burstiness",
                  "trend_slope_norm", "late_early_ratio", "peak_week_pos", "recent_vs_mean", "trailing_zero_weeks",
                  "longest_zero_run", "n_zero_runs", "p_stay_zero", "p_drop_to_zero", "state_change_rate"]:
            df[k] = np.nan

    # khoảng cách từ lần hoạt động cuối + ngày hoạt động đầu tiên (cửa sổ 0..cutoff)
    st = C.streak_features(df[KEY3], s[s["date"] >= 0], cutoff)[KEY3 + ["first_active_day", "current_inactive_streak"]]
    df = df.merge(st, on=KEY3, how="left").rename(columns={"current_inactive_streak": "last_active_gap"})

    # học dồn: tỉ trọng click trong 7 ngày trước mỗi hạn nộp (chỉ các hạn <= cutoff)
    cal = C.deadline_calendar(d, cutoff, k=7)
    sc = s.merge(cal, on=KMP + ["date"], how="left")
    sc["pre_deadline"] = sc["pre_deadline"].fillna(0)
    sc["pre_clicks"] = sc["sum_click"] * sc["pre_deadline"]
    sc["pre_start_clicks"] = sc["sum_click"] * (sc["date"] < 0)
    g = sc.groupby(KEY3).agg(tot=("sum_click", "sum"), pre=("pre_clicks", "sum"), pre_start=("pre_start_clicks", "sum")).reset_index()
    df = df.merge(g, on=KEY3, how="left")
    df["pre_deadline_click_share"] = _div(df["pre"].fillna(0).to_numpy(), df["tot"].fillna(0).to_numpy())
    df["pre_start_click_share"] = _div(df["pre_start"].fillna(0).to_numpy(), df["tot"].fillna(0).to_numpy())
    return C.finalize(df, d, FEATURES)


def weekly_profile(d, df, cutoff):
    """Tỉ trọng click mỗi tuần (n × W), hàng khớp df (kết quả build()); sinh viên không click -> toàn 0."""
    s = C.vle_until(d, cutoff, df)
    M = C.weekly_matrix(s, df, cutoff)
    tot = M.sum(1, keepdims=True)
    P = np.where(tot > 0, M / np.where(tot > 0, tot, 1.0), 0.0)
    return pd.DataFrame(P, columns=[f"prof_w{w + 1:02d}" for w in range(M.shape[1])], index=df.index)


def survival_table(d):
    """KEY3 + duration (ngày kể từ ngày bắt đầu khóa) + event (1 = Withdrawn có ngày hủy, 0 = bị kiểm duyệt).
    Withdrawn không có ngày hủy đăng ký -> duration = NaN (nên loại). Dùng cùng với feature tại một cutoff
    chỉ cho sinh viên còn đang học tại cutoff, nếu không sẽ rò rỉ (duration > cutoff)."""
    t = (d["studentInfo"][KEY3 + ["final_result"]]
         .merge(d["studentRegistration"][KEY3 + ["date_unregistration"]], on=KEY3, how="left")
         .merge(d["courses"], on=KMP, how="left"))
    t["date_unregistration"] = pd.to_numeric(t["date_unregistration"], errors="coerce")
    wd = t["final_result"] == "Withdrawn"
    t["event"] = wd.astype(int)
    t["duration"] = np.where(wd, t["date_unregistration"], t["module_presentation_length"])
    return t[KEY3 + ["duration", "event"]]


def audit(d, cutoff=84):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
