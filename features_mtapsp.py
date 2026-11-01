"""Nhánh 2 — MTAPSP (MDPI Applied Sciences, 2024): chuỗi clickstream chia 5 phân đoạn (20%..100% của 270 ngày) +
điểm TMA/CMA + nhân khẩu học.

LƯU Ý: tôi chưa đọc bài gốc; dựng theo mô tả trong file nhánh bạn gửi. Bài gốc dùng LSTM + attention trên chuỗi;
ở đây có hai đầu ra:
  1) build(d, cutoff)           bảng phẳng (tabular) cho RF/XGBoost: mỗi phân đoạn -> vài số tóm tắt
  2) build_sequences(d, df, ..) tensor (n, T, C) clicks theo tuần × nhóm hoạt động, cho LSTM/CNN sau này

Phân đoạn: 5 đoạn × 54 ngày (= 20% của 270 ngày). Không co giãn độ dài khóa thật (234–269 ngày) về 270 — chênh lệch nhỏ.
Phân đoạn chưa bắt đầu tại cutoff -> NaN (không điền 0). Phân đoạn đang dở -> chỉ phần đã quan sát được tính
(n_ và clicks là số cộng dồn tới cutoff, vì vậy ở cutoff giữa đoạn, đoạn cuối có giá trị nhỏ hơn đoạn đầy đủ).

Điểm TMA/CMA theo đoạn: chỉ tính bài đến hạn <= cutoff và đã nộp <= cutoff, phân đoạn theo ngày nộp.

Cutoff khớp mốc 20/40/60% của bài: 54, 108, 162.

Dùng:  import features_mtapsp as F
"""
import numpy as np
import pandas as pd

import features_common as C
from features_common import KEY3, KMP, load  # noqa: F401

COURSE_LEN, N_SEG = 270, 5
SEG_LEN = COURSE_LEN // N_SEG          # 54 ngày
DEFAULT_CUTOFFS = [54, 108, 162]
DEMOGRAPHIC = ["gender", "region", "highest_education", "imd_band", "age_band", "disability",
               "num_of_prev_attempts", "studied_credits"]

FEATURE_GROUPS = {
    "clickstream_segments": [f"seg{s}_{k}" for s in range(1, N_SEG + 1) for k in ("clicks", "active_days")]
                            + ["clicks_pre_start"],
    "tma_cma_segments": [f"seg{s}_{t}_{k}" for s in range(1, N_SEG + 1) for t in ("tma", "cma") for k in ("n", "mean")],
    "tma_cma_overall": ["tma_n", "tma_mean", "cma_n", "cma_mean"],
    "demographic": DEMOGRAPHIC,
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _seg(date):
    return (np.clip(date // SEG_LEN, 0, N_SEG - 1) + 1).astype(int)


def build(d, cutoff=108):
    df = C.base_frame(d, cutoff)
    started = {s: (s - 1) * SEG_LEN <= cutoff for s in range(1, N_SEG + 1)}

    # --- clickstream theo phân đoạn ---
    s = C.vle_until(d, cutoff, df)
    pre = s[s["date"] < 0].groupby(KEY3)["sum_click"].sum().rename("clicks_pre_start").reset_index()
    df = df.merge(pre, on=KEY3, how="left")
    df["clicks_pre_start"] = df["clicks_pre_start"].fillna(0)
    s0 = s[s["date"] >= 0].copy()
    s0["seg"] = _seg(s0["date"])
    clk = s0.pivot_table(index=KEY3, columns="seg", values="sum_click", aggfunc="sum")
    act = s0.groupby(KEY3 + ["seg"])["date"].nunique().unstack("seg")
    for s_id in range(1, N_SEG + 1):
        for name, tab in (("clicks", clk), ("active_days", act)):
            col = f"seg{s_id}_{name}"
            ser = tab[s_id].rename(col).reset_index() if s_id in tab.columns else None
            if ser is not None:
                df = df.merge(ser, on=KEY3, how="left")
            else:
                df[col] = np.nan
            df[col] = df[col].fillna(0.0) if started[s_id] else np.nan

    # --- điểm TMA / CMA theo phân đoạn (ngày nộp) ---
    _, _, m = C.assessments_until(d, cutoff)
    m = m[m["assessment_type"].isin(["TMA", "CMA"])].copy()
    m["seg"] = _seg(m["date_submitted"].clip(lower=0))
    agg = m.groupby(KEY3 + ["assessment_type", "seg"]).agg(n=("id_assessment", "size"), mean=("score", "mean")).reset_index()
    for t in ("TMA", "CMA"):
        low = t.lower()
        for s_id in range(1, N_SEG + 1):
            sub = agg[(agg["assessment_type"] == t) & (agg["seg"] == s_id)][KEY3 + ["n", "mean"]]
            sub = sub.rename(columns={"n": f"seg{s_id}_{low}_n", "mean": f"seg{s_id}_{low}_mean"})
            df = df.merge(sub, on=KEY3, how="left")
            df[f"seg{s_id}_{low}_n"] = df[f"seg{s_id}_{low}_n"].fillna(0.0) if started[s_id] else np.nan
            if not started[s_id]:
                df[f"seg{s_id}_{low}_mean"] = np.nan
        tot = m[m["assessment_type"] == t].groupby(KEY3).agg(**{f"{low}_n": ("id_assessment", "size"),
                                                                f"{low}_mean": ("score", "mean")}).reset_index()
        df = df.merge(tot, on=KEY3, how="left")
        df[f"{low}_n"] = df[f"{low}_n"].fillna(0.0)

    return C.finalize(df, d, FEATURES)


def build_sequences(d, df, cutoff, step=7, n_steps=None):
    """Tensor cho mô hình chuỗi (LSTM/CNN/Transformer).
    Trả về (X, mask, channels): X shape (len(df), T, C) float32 = clicks mỗi tuần theo 5 nhóm hoạt động
    (C.ACT_GROUPS); mask shape (T,) = True ở những bước thời gian đã quan sát tới cutoff (bước cuối có thể dở dang);
    các bước sau cutoff = 0 và mask=False. Hàng của X khớp thứ tự df (df = kết quả build()).
    Ngày < 0 (trước khi bắt đầu khóa) không nằm trong tensor (xem feature clicks_pre_start)."""
    T = n_steps or int(np.ceil(COURSE_LEN / step))
    channels = C.ACT_GROUPS
    n = len(df)
    s = C.vle_until(d, cutoff, df)
    s = s[(s["date"] >= 0) & (s["date"] // step < T)]
    s = s.merge(df[KEY3].assign(_row=np.arange(n)), on=KEY3, how="inner")
    ch = s["act_group"].map({g: i for i, g in enumerate(channels)}).to_numpy()
    flat = (s["_row"].to_numpy() * T + (s["date"].to_numpy() // step).astype(int)) * len(channels) + ch
    X = np.bincount(flat, weights=s["sum_click"].to_numpy(dtype=float), minlength=n * T * len(channels))
    X = X.reshape(n, T, len(channels)).astype(np.float32)
    mask = (np.arange(T) * step) <= cutoff
    return X, mask, channels


def audit(d, cutoff=108):
    """Danh sách cột bị rò rỉ thời gian (rỗng = sạch)."""
    return C.leakage_check(build, d, cutoff)
