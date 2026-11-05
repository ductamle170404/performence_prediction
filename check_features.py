"""Kiểm tra nhanh các bộ feature theo nhánh trên dữ liệu thật hoặc subset.

    python check_features.py --data data_subset --cutoffs 28 60
    python check_features.py --data data --only features_wu_jia features_joive --leak

Với mỗi (bộ feature, cutoff) in: số dòng, số feature, khóa có duy nhất không, % NaN, cột hằng số/toàn NaN,
số cột sau make_xy (get_dummies) và (nếu --leak) kết quả kiểm tra rò rỉ thời gian.
"""
import argparse, importlib, time, warnings
import numpy as np
import pandas as pd
import features_common as C

warnings.filterwarnings("ignore")
MODULES = ["features_kuznetsov_ews", "features_wu_jia", "features_joive",
           "features_mtapsp", "features_xai", "features_trajectory"]
CAT = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
DROP = ["id_student", "code_presentation", "final_result", "at_risk", "fail_only", "final_score",
        "date_unregistration", "module_presentation_length"]

try:                                    # dùng đúng make_xy của bạn nếu train_cls.py import được
    from train_cls import make_xy
except Exception:                       # (thiếu xgboost, ...) -> bản sao cùng logic
    def make_xy(df):
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        return pd.get_dummies(X, columns=[c for c in CAT if c in X.columns], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_subset")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=None, help="mặc định: DEFAULT_CUTOFFS của từng bộ")
    ap.add_argument("--only", nargs="+", default=MODULES)
    ap.add_argument("--leak", action="store_true", help="chạy leakage_check (chậm gấp đôi)")
    a = ap.parse_args()

    d = C.load(a.data)
    for name in a.only:
        F = importlib.import_module(f"feature_sets.{name}")
        print(f"\n######## {name}  ({len(F.FEATURES)} feature, nhóm: {', '.join(F.FEATURE_GROUPS)})")
        for cut in (a.cutoffs or F.DEFAULT_CUTOFFS):
            t0 = time.time()
            df = F.build(d, cut)
            feat = [c for c in df.columns if c not in C.KEY3 + C.LABELS]
            nan = df[feat].isna().mean()
            const = [c for c in feat if df[c].nunique(dropna=True) <= 1]
            X = make_xy(df)
            non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            line = (f"  cutoff {cut:>3}: {len(df):>6,} dòng | {len(feat)} feature -> {X.shape[1]} cột sau make_xy | "
                    f"key duy nhất: {not df.duplicated(C.KEY3).any()} | at_risk {df['at_risk'].mean():.2f} | {time.time()-t0:.1f}s")
            print(line)
            print(f"      NaN trung bình {nan.mean():.1%}; cột NaN > 50%: {list(nan[nan > .5].index)[:8]}")
            if const:
                print(f"      cột hằng số / toàn NaN: {const[:10]}")
            if non_num:
                print(f"      !! cột không phải số sau make_xy: {non_num}")
            if a.leak:
                bad = F.audit(d, cut)
                print("      rò rỉ thời gian:", "KHÔNG phát hiện" if not bad else f"!! {bad}")


if __name__ == "__main__":
    main()
