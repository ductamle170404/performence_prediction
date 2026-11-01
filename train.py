"""Chạy thử XGBoost (chính) + Random Forest (phụ), cả Classification và Regression, ở nhiều mốc cảnh báo.

Ví dụ:
    python train.py --data data_subset --cutoffs 0 30 60 90          # chạy thử nhanh trên subset
    python train.py --data data --cutoffs 30 60 90 --seed 1           # chạy trên toàn bộ dữ liệu

Chia tập: TRAIN = 2013B, 2013J, 2014B ; TEST = 2014J (chia theo thời gian, tránh rò rỉ).
Kết quả lưu ở thư mục results/.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
import features as F

warnings.filterwarnings("ignore")
TRAIN_PRES = ["2013B", "2013J", "2014B"]
TEST_PRES = "2014J"
CAT = ["code_module", "gender", "region", "highest_education", "imd_band",
       "age_band", "disability"]
DROP = ["id_student", "code_presentation", "final_result", "at_risk", "fail_only", "final_score",
        "date_unregistration", "module_presentation_length"]


def make_xy(df):
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    X = pd.get_dummies(X, columns=[c for c in CAT if c in X.columns], dtype=float)
    # XGBoost không cho phép [ ] < trong tên cột (vd. age_band "55<=")
    X.columns = (X.columns.astype(str)
                 .str.replace("<=", "le_", regex=False).str.replace("<", "lt_", regex=False)
                 .str.replace(">", "gt_", regex=False).str.replace("[", "(", regex=False)
                 .str.replace("]", ")", regex=False).str.replace(r"\s+", "_", regex=True))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[0, 30, 60, 90])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="ngưỡng cảnh báo cho classification")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = F.load(a.data)
    rows_cls, rows_reg = [], []

    for cut in a.cutoffs:
        df = F.build(d, cut)
        is_test = df["code_presentation"] == TEST_PRES
        is_train = df["code_presentation"].isin(TRAIN_PRES)
        X = make_xy(df)
        Xtr, Xte = X[is_train], X[is_test]
        print(f"\n=== cutoff = ngày {cut} | train {len(Xtr):,} | test {len(Xte):,} | "
              f"{X.shape[1]} đặc trưng | tỉ lệ at-risk train {df.loc[is_train,'at_risk'].mean():.2f}")

        # ---------- Classification: at_risk (Fail + Withdrawn) ----------
        ytr, yte = df.loc[is_train, "at_risk"], df.loc[is_test, "at_risk"]
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        clfs = {
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=a.seed, n_jobs=-1),
            "RandomForest": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                          RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                 class_weight="balanced", random_state=a.seed,
                                                                 n_jobs=-1)),
        }
        for name, m in clfs.items():
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            pred = (p >= a.threshold).astype(int)
            rows_cls.append(dict(cutoff=cut, model=name, auc=roc_auc_score(yte, p),
                                 pr_auc=average_precision_score(yte, p), f1=f1_score(yte, pred),
                                 recall=recall_score(yte, pred), precision=precision_score(yte, pred),
                                 accuracy=accuracy_score(yte, pred)))
            if name == "XGBoost":
                imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
                imp.head(20).to_csv(f"{a.out}/importance_xgb_cutoff{cut}.csv", header=["importance"])

        # ---------- Regression: final_score (proxy) ----------
        has = df["final_score"].notna()
        tr, te = ~is_test & has, is_test & has
        regs = {
            "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, random_state=a.seed, n_jobs=-1),
            "RandomForest": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                          RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                random_state=a.seed, n_jobs=-1)),
        }
        base = np.full(te.sum(), df.loc[tr, "final_score"].mean())
        rows_reg.append(dict(cutoff=cut, model="Baseline(mean)",
                             mae=mean_absolute_error(df.loc[te, "final_score"], base),
                             rmse=mean_squared_error(df.loc[te, "final_score"], base) ** 0.5, r2=0.0))
        for name, m in regs.items():
            m.fit(X[tr], df.loc[tr, "final_score"])
            p = m.predict(X[te])
            yt = df.loc[te, "final_score"]
            rows_reg.append(dict(cutoff=cut, model=name, mae=mean_absolute_error(yt, p),
                                 rmse=mean_squared_error(yt, p) ** 0.5, r2=r2_score(yt, p)))

    cls, reg = pd.DataFrame(rows_cls), pd.DataFrame(rows_reg)
    cls.to_csv(f"{a.out}/classification.csv", index=False)
    reg.to_csv(f"{a.out}/regression.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n##### CLASSIFICATION (at_risk = Fail + Withdrawn) #####")
    print(cls.round(3).to_string(index=False))
    print("\n##### REGRESSION (final_score proxy) #####")
    print(reg.round(3).to_string(index=False))
    print(f"\nĐã lưu vào {a.out}/")


if __name__ == "__main__":
    main()