"""Hàm dùng chung cho các file features_<tên>.py (đây KHÔNG phải một bộ feature riêng).

Quy ước giống features.py của bạn:
  - build(d, cutoff) -> DataFrame, mỗi dòng một Student–Course
    cột = KEY3 + nhãn (final_result, at_risk, fail_only, final_score) + feature của bộ đó
  - chỉ dùng thông tin có ngày <= cutoff (kiểm tra bằng leakage_check)
  - nhãn và hàm load() được import thẳng từ features.py -> mọi bộ feature dùng đúng cùng một định nghĩa nhãn

Cần có features.py trong thư mục prediction/; các bộ feature nằm trong prediction/feature_sets/.
"""
import numpy as np
import pandas as pd

from features import load, _targets, KEY3, KMP  # noqa: F401  (load được re-export cho các file con)

LABELS = ["final_result", "at_risk", "fail_only", "final_score"]

# Gom ~20 loại activity_type của OULAD thành 5 nhóm (loại lạ -> "other")
ACT_GROUPS = ["content", "forum", "quiz", "homepage", "other"]
ACTIVITY_GROUP = {
    "oucontent": "content", "resource": "content", "url": "content", "subpage": "content",
    "glossary": "content", "page": "content", "folder": "content", "htmlactivity": "content",
    "dualpane": "content", "sharedsubpage": "content", "repeatactivity": "content",
    "dataplus": "content", "ouwiki": "content",
    "forumng": "forum", "oucollaborate": "forum", "ouelluminate": "forum",
    "quiz": "quiz", "externalquiz": "quiz", "questionnaire": "quiz",
    "homepage": "homepage",
}
EDU_ORDER = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
             "HE Qualification": 3, "Post Graduate Qualification": 4}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}


# --------------------------------------------------------------------------------------
# Bảng nền, VLE, bài đánh giá
# --------------------------------------------------------------------------------------
def base_frame(d, cutoff):
    """studentInfo + đăng ký + courses; chỉ giữ sinh viên còn đang học tại cutoff (giống features.build)."""
    df = (d["studentInfo"].merge(d["studentRegistration"], on=KEY3, how="left")
          .merge(d["courses"], on=KMP, how="left"))
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_unregistration"] = pd.to_numeric(df["date_unregistration"], errors="coerce")
    still = df["date_unregistration"].isna() | (df["date_unregistration"] > cutoff)
    df = df[still].copy()
    df["imd_band"] = df["imd_band"].fillna("Unknown")
    return df.reset_index(drop=True)


def vle_until(d, cutoff, df=None):
    """Log VLE có date <= cutoff (+ activity_type, act_group). Truyền df để chỉ giữ sinh viên đang xét."""
    s = d["studentVle"]
    s = s[s["date"] <= cutoff]
    if df is not None:
        s = s.merge(df[KEY3], on=KEY3, how="inner")
    s = s.merge(d["vle"][["id_site", "activity_type"]], on="id_site", how="left")
    s["act_group"] = s["activity_type"].map(ACTIVITY_GROUP).fillna("other")
    return s


def vle_core(df, s, cutoff):
    """Nhóm VLE cơ bản (cùng cách tính với features.py): tổng click, ngày hoạt động, gần đây, xu hướng 7 ngày."""
    g = s.groupby(KEY3)
    f = g["sum_click"].sum().rename("clicks_total").to_frame()
    f["active_days"] = g["date"].nunique()
    f["n_sites"] = g["id_site"].nunique()
    f["clicks_pre_start"] = s[s["date"] < 0].groupby(KEY3)["sum_click"].sum()
    f["clicks_last7"] = s[s["date"] > cutoff - 7].groupby(KEY3)["sum_click"].sum()
    f["clicks_prev7"] = s[(s["date"] > cutoff - 14) & (s["date"] <= cutoff - 7)].groupby(KEY3)["sum_click"].sum()
    f["last_active_gap"] = cutoff - g["date"].max()
    out = df.merge(f.reset_index(), on=KEY3, how="left")
    zero = ["clicks_total", "active_days", "n_sites", "clicks_pre_start", "clicks_last7", "clicks_prev7"]
    out[zero] = out[zero].fillna(0)
    out["last_active_gap"] = out["last_active_gap"].fillna(cutoff + 1)
    out["click_trend_7d"] = out["clicks_last7"] - out["clicks_prev7"]
    out["clicks_per_active_day"] = np.where(out["active_days"] > 0,
                                            out["clicks_total"] / out["active_days"].where(out["active_days"] > 0), 0.0)
    return out


def activity_mix(df, s):
    """Tổng click theo 5 nhóm hoạt động (clk_content, clk_forum, ...) và tỉ trọng share_<nhóm>."""
    p = s.pivot_table(index=KEY3, columns="act_group", values="sum_click", aggfunc="sum")
    p = p.reindex(columns=ACT_GROUPS)
    p.columns = [f"clk_{c}" for c in p.columns]
    out = df.merge(p.reset_index(), on=KEY3, how="left")
    clk = [f"clk_{c}" for c in ACT_GROUPS]
    out[clk] = out[clk].fillna(0)
    tot = out[clk].sum(axis=1)
    for c in ACT_GROUPS:
        out[f"share_{c}"] = np.where(tot > 0, out[f"clk_{c}"] / tot.where(tot > 0), 0.0)
    return out


def daily_table(s):
    """Mỗi dòng = (sinh viên, ngày có hoạt động >= 0) với tổng click trong ngày; đã sắp xếp."""
    u = s[(s["date"] >= 0) & (s["sum_click"] > 0)]
    u = u.groupby(KEY3 + ["date"], as_index=False)["sum_click"].sum()
    return u.sort_values(KEY3 + ["date"]).reset_index(drop=True)


def weekly_matrix(s, df, cutoff, step=7, n_steps=None):
    """Ma trận click theo tuần, shape (len(df), W). Mặc định chỉ lấy các tuần ĐÃ KẾT THÚC (ngày 0..7W-1 <= cutoff).
    Hàng khớp thứ tự df. Ngày < 0 (trước khi bắt đầu khóa) không nằm trong ma trận."""
    W = (cutoff + 1) // step if n_steps is None else n_steps
    n = len(df)
    if W == 0 or n == 0:
        return np.zeros((n, W))
    x = s[(s["date"] >= 0) & (s["date"] < W * step)]
    x = x.merge(df[KEY3].assign(_row=np.arange(n)), on=KEY3, how="inner")
    flat = x["_row"].to_numpy() * W + (x["date"].to_numpy() // step).astype(int)
    M = np.bincount(flat, weights=x["sum_click"].to_numpy(dtype=float), minlength=n * W)
    return M.reshape(n, W)


def streak_features(df, s, cutoff):
    """Chuỗi ngày không hoạt động + thời lượng truy cập. Cửa sổ ngày 0..cutoff."""
    u = daily_table(s)
    cols = ["max_gap", "n_g7", "first_active_day", "last_active_day", "n_active_days",
            "daily_clicks_std", "daily_clicks_max", "daily_clicks_mean"]
    if len(u):
        u["prev"] = u.groupby(KEY3)["date"].shift(1).fillna(-1)
        u["gap"] = u["date"] - u["prev"] - 1
        u["g7"] = (u["gap"] >= 7).astype(int)
        f = u.groupby(KEY3).agg(max_gap=("gap", "max"), n_g7=("g7", "sum"),
                                first_active_day=("date", "min"), last_active_day=("date", "max"),
                                n_active_days=("date", "size"), daily_clicks_std=("sum_click", "std"),
                                daily_clicks_max=("sum_click", "max"),
                                daily_clicks_mean=("sum_click", "mean")).reset_index()
        out = df.merge(f, on=KEY3, how="left")
    else:
        out = df.copy()
        for c in cols:
            out[c] = np.nan
    has = out["last_active_day"].notna()
    trailing = cutoff - out["last_active_day"]
    out["max_inactive_streak"] = np.where(has, np.maximum(out["max_gap"], trailing), cutoff + 1)
    out["current_inactive_streak"] = np.where(has, trailing, cutoff + 1)
    out["n_streaks_ge7"] = np.where(has, out["n_g7"] + (trailing >= 7), int(cutoff + 1 >= 7))
    out["n_active_days"] = out["n_active_days"].fillna(0)
    out["n_inactive_days"] = (cutoff + 1) - out["n_active_days"]
    out["inactive_ratio"] = out["n_inactive_days"] / (cutoff + 1)
    out["access_span"] = out["last_active_day"] - out["first_active_day"]       # NaN nếu chưa từng truy cập
    return out.drop(columns=["max_gap", "n_g7"])


def assessments_until(d, cutoff):
    """(bài đến hạn <= cutoff [bỏ Exam], bảng due theo module-đợt, bài đã nộp <= cutoff). Giống features.build."""
    a, sa = d["assessments"], d["studentAssessment"]
    ad = a[(a["assessment_type"] != "Exam") & (a["date"] <= cutoff)]
    due = ad.groupby(KMP).agg(n_due=("id_assessment", "count"), weight_due=("weight", "sum")).reset_index()
    m = sa.merge(ad[["id_assessment", "assessment_type", "date", "weight"] + KMP], on="id_assessment")
    m["late_days"] = (m["date_submitted"] - m["date"]).clip(lower=0)
    m = m[m["date_submitted"] <= cutoff].copy()
    return ad, due, m


def assess_summary(df, d, cutoff):
    """n_due, weight_due, n_submitted, n_missed, submit_rate, mean/min/last_score, mean_late, n_late."""
    ad, due, m = assessments_until(d, cutoff)
    m = m.sort_values("date_submitted")          # để last_score đúng là điểm bài nộp gần nhất
    m["is_late"] = (m["late_days"] > 0).astype(int)
    af = m.groupby(KEY3).agg(n_submitted=("id_assessment", "count"), mean_score=("score", "mean"),
                             min_score=("score", "min"), last_score=("score", "last"),
                             mean_late=("late_days", "mean"), n_late=("is_late", "sum")).reset_index()
    out = df.merge(af, on=KEY3, how="left").merge(due, on=KMP, how="left")
    out[["n_due", "weight_due", "n_submitted", "n_late"]] = out[["n_due", "weight_due", "n_submitted", "n_late"]].fillna(0)
    out["submit_rate"] = np.where(out["n_due"] > 0, out["n_submitted"] / out["n_due"].where(out["n_due"] > 0), np.nan)
    out["n_missed"] = (out["n_due"] - out["n_submitted"]).clip(lower=0)
    return out


def deadline_calendar(d, cutoff, k=7):
    """Bảng (module, đợt, ngày) nằm trong k ngày kết thúc tại hạn nộp của một bài đã đến hạn <= cutoff."""
    a = d["assessments"]
    ad = a[(a["assessment_type"] != "Exam") & (a["date"] <= cutoff)].dropna(subset=["date"])
    parts = []
    for off in range(k):
        t = ad[KMP].copy()
        t["date"] = ad["date"] - off
        parts.append(t)
    cal = pd.concat(parts, ignore_index=True).drop_duplicates()
    cal["pre_deadline"] = 1
    return cal


# --------------------------------------------------------------------------------------
# Hoàn tất, nhóm feature, kiểm tra rò rỉ
# --------------------------------------------------------------------------------------
def finalize(df, d, features):
    """Gắn nhãn (y hệt features.build) và chỉ giữ KEY3 + nhãn + feature."""
    df = df.copy()
    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    df["fail_only"] = (df["final_result"] == "Fail").astype(int)
    if "_targets_cache" not in d:                       # final_score tính một lần cho mỗi bộ dữ liệu
        d["_targets_cache"] = _targets(d)
    df = df.merge(d["_targets_cache"], on=KEY3, how="left")
    cols = KEY3 + LABELS + [c for c in features if c not in KEY3 + LABELS]
    return df[cols].reset_index(drop=True)


def column_groups(columns, feature_groups):
    """Ánh xạ tên cột (kể cả cột one-hot như gender_M) -> tên nhóm; không khớp -> 'other'."""
    out = {}
    for c in columns:
        out[c] = "other"
        for g, names in feature_groups.items():
            if any(c == n or c.startswith(n + "_") for n in names):
                out[c] = g
                break
    return out


def select_groups(X, feature_groups, only=None, drop=None):
    """Lọc cột của X (sau make_xy) theo nhóm: only=[...] chỉ giữ, drop=[...] bỏ. Dùng cho ablation."""
    cg = column_groups(X.columns, feature_groups)
    keep = [c for c in X.columns
            if (only is None or cg[c] in only) and (drop is None or cg[c] not in drop)]
    return X[keep]


def leakage_check(build_fn, d, cutoff, rtol=1e-6):
    """Chạy build trên dữ liệu đầy đủ và trên dữ liệu đã CẮT hết thông tin sau cutoff; mọi feature phải giống hệt.
    Trả về danh sách cột bị lệch (rỗng = không phát hiện rò rỉ thời gian)."""
    if "_targets_cache" not in d:
        d["_targets_cache"] = _targets(d)
    d2 = {k: v for k, v in d.items() if not k.startswith("_")}
    d2["_targets_cache"] = d["_targets_cache"]          # nhãn không so sánh; tránh tính lại trên bảng đã cắt (có thể rỗng)
    sv, sa = d["studentVle"], d["studentAssessment"]
    d2["studentVle"] = sv[sv["date"] <= cutoff]
    d2["studentAssessment"] = sa[sa["date_submitted"] <= cutoff]
    sr = d["studentRegistration"].copy()
    un = pd.to_numeric(sr["date_unregistration"], errors="coerce")
    sr.loc[un > cutoff, "date_unregistration"] = np.nan
    d2["studentRegistration"] = sr
    a = build_fn(d, cutoff).sort_values(KEY3).reset_index(drop=True)
    b = build_fn(d2, cutoff).sort_values(KEY3).reset_index(drop=True)
    if len(a) != len(b):
        return ["<số dòng khác nhau>"]
    bad = []
    for c in [c for c in a.columns if c not in KEY3 + LABELS]:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            same = np.isclose(x.astype(float), y.astype(float), rtol=rtol, equal_nan=True).all()
        else:
            same = (x.fillna("~") == y.fillna("~")).all()
        if not same:
            bad.append(c)
    return bad
