"""Xây bảng đặc trưng Student–Course tại một mốc cảnh báo (cutoff, tính theo ngày so với ngày bắt đầu môn).

Nguyên tắc chống rò rỉ dữ liệu: đặc trưng CHỈ dùng thông tin có ngày <= cutoff.
Nhãn (final_result, điểm cuối) lấy từ toàn bộ kỳ học.
"""
import numpy as np
import pandas as pd

KEY3 = ["code_module", "code_presentation", "id_student"]
KMP = ["code_module", "code_presentation"]
NAMES = ["courses", "assessments", "vle", "studentInfo",
         "studentRegistration", "studentAssessment", "studentVle"]


def load(data_dir="data_using"):
    d = {n: pd.read_csv(f"{data_dir}/{n}.csv", na_values=["?"]) for n in NAMES}
    sv = d["studentVle"]
    # Khóa (student, site, date) bị trùng với sum_click khác nhau -> cộng gộp lại
    d["studentVle"] = sv.groupby(KEY3 + ["id_site", "date"], as_index=False)["sum_click"].sum()
    return d


def _targets(d):
    """Điểm cuối ước lượng = trung bình có trọng số của điểm các bài đã nộp (weight>0).
    Lưu ý: OULAD không có điểm cuối chính thức, đây là proxy. GGG có weight=0 ở TMA/CMA -> dùng trung bình thường."""
    a, sa = d["assessments"], d["studentAssessment"]
    m = sa.merge(a[["id_assessment", "weight"] + KMP], on="id_assessment")
    m = m.dropna(subset=["score"])

    def agg(g):
        w = g["weight"].sum()
        return (g["score"] * g["weight"]).sum() / w if w > 0 else g["score"].mean()

    t = m.groupby(KEY3).apply(agg, include_groups=False).rename("final_score").reset_index()
    return t


def build(d, cutoff):
    si, sr, sv, v = d["studentInfo"], d["studentRegistration"], d["studentVle"], d["vle"]
    a, sa, c = d["assessments"], d["studentAssessment"], d["courses"]

    # --- 1. Nền: studentInfo + đăng ký ---
    df = si.merge(sr, on=KEY3, how="left").merge(c, on=KMP, how="left")
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_unregistration"] = pd.to_numeric(df["date_unregistration"], errors="coerce")

    # Chỉ dự báo cho sinh viên còn đang học tại thời điểm cutoff
    still = df["date_unregistration"].isna() | (df["date_unregistration"] > cutoff)
    df = df[still].copy()
    df["imd_band"] = df["imd_band"].fillna("Unknown")

    # --- 2. VLE tới cutoff ---
    s = sv[sv["date"] <= cutoff].merge(v[["id_site", "activity_type"]], on="id_site", how="left")
    g = s.groupby(KEY3)
    f = g["sum_click"].sum().rename("clicks_total").to_frame()
    f["active_days"] = g["date"].nunique()
    f["n_sites"] = g["id_site"].nunique()
    f["clicks_pre_start"] = s[s["date"] < 0].groupby(KEY3)["sum_click"].sum()
    for k in (7, 14):
        f[f"clicks_last{k}"] = s[s["date"] > cutoff - k].groupby(KEY3)["sum_click"].sum()
    f["clicks_prev14"] = s[(s["date"] > cutoff - 28) & (s["date"] <= cutoff - 14)].groupby(KEY3)["sum_click"].sum()
    f["last_active_gap"] = cutoff - g["date"].max()
    pv = s.pivot_table(index=KEY3, columns="activity_type", values="sum_click", aggfunc="sum")
    pv.columns = [f"clk_{x}" for x in pv.columns]
    f = f.join(pv).reset_index()
    df = df.merge(f, on=KEY3, how="left")
    clk_cols = [x for x in df.columns if x.startswith("clk")] + ["clicks_total", "active_days", "n_sites",
                                                                 "clicks_pre_start", "clicks_last7", "clicks_last14",
                                                                 "clicks_prev14"]
    df[clk_cols] = df[clk_cols].fillna(0)          # không có log = 0 click
    df["last_active_gap"] = df["last_active_gap"].fillna(cutoff + 1)
    df["click_trend"] = df["clicks_last14"] - df["clicks_prev14"]
    df["clicks_per_active_day"] = df["clicks_total"] / df["active_days"].replace(0, np.nan)

    # --- 3. Bài đánh giá đến hạn tới cutoff (bỏ Exam: thường sau cutoff / thiếu ngày) ---
    ad = a[(a["assessment_type"] != "Exam") & (a["date"] <= cutoff)]
    due = ad.groupby(KMP).agg(n_due=("id_assessment", "count"), weight_due=("weight", "sum")).reset_index()
    m = sa.merge(ad[["id_assessment", "date", "weight"] + KMP], on="id_assessment")
    m["late_days"] = (m["date_submitted"] - m["date"]).clip(lower=0)
    m = m[m["date_submitted"] <= cutoff]
    g = m.groupby(KEY3)
    af = g.agg(n_submitted=("id_assessment", "count"),
               mean_score=("score", "mean"),
               min_score=("score", "min"),
               last_score=("score", "last"),
               mean_late=("late_days", "mean")).reset_index()
    df = df.merge(af, on=KEY3, how="left").merge(due, on=KMP, how="left")
    df[["n_due", "weight_due"]] = df[["n_due", "weight_due"]].fillna(0)
    df["n_submitted"] = df["n_submitted"].fillna(0)
    df["submit_rate"] = np.where(df["n_due"] > 0, df["n_submitted"] / df["n_due"], np.nan)
    df["n_missed"] = (df["n_due"] - df["n_submitted"]).clip(lower=0)

    # --- 4. Nhãn ---
    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)   # phân loại
    df["fail_only"] = (df["final_result"] == "Fail").astype(int)
    df = df.merge(_targets(d), on=KEY3, how="left")                              # hồi quy
    return df
