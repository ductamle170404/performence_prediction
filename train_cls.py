"""Classification cảnh báo sớm: XGBoost (chính) + Random Forest (phụ) + Logistic Regression (baseline).

Ví dụ:
    python train_cls.py --data data_subset --cutoffs 0 30 60 90
    python train_cls.py --data data --cutoffs 30 60 90 --target fail_only --seed 1

--target at_risk   : Fail + Withdrawn = 1  (mặc định)
--target fail_only : chỉ Fail = 1 (Withdrawn coi là 0)
Chia tập theo thời gian: train = 2013B, 2013J, 2014B ; test = 2014J.
Ngoài AUC/F1, in thêm recall tại ngưỡng cố định và "top-K%" (nếu cố vấn chỉ liên hệ K% sinh viên
có rủi ro cao nhất thì bắt được bao nhiêu % sinh viên thực sự rớt) và AUC theo từng môn.
Kết quả lưu ở results_cls/.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, accuracy_score)
from xgboost import XGBClassifier
import features as F

warnings.filterwarnings("ignore")
TEST_PRES = "2014J"
CAT = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
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


def topk_recall(y, p, k):
    """Recall nếu chỉ cảnh báo k% sinh viên có xác suất cao nhất."""
    n = max(int(len(y) * k), 1)
    idx = np.argsort(-p)[:n]
    return y.values[idx].sum() / max(y.sum(), 1), y.values[idx].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[0, 30, 60, 90])
    ap.add_argument("--target", choices=["at_risk", "fail_only"], default="at_risk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results_cls")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = F.load(a.data)
    rows, rows_mod = [], []

    for cut in a.cutoffs:
        df = F.build(d, cut)
        is_test = df["code_presentation"] == TEST_PRES
        X = make_xy(df)
        Xtr, Xte = X[~is_test], X[is_test]
        ytr, yte = df.loc[~is_test, a.target], df.loc[is_test, a.target]
        mods_te = df.loc[is_test, "code_module"].values
        print(f"\n=== cutoff = ngày {cut} | train {len(Xtr):,} | test {len(Xte):,} | "
              f"{X.shape[1]} đặc trưng | tỉ lệ dương train {ytr.mean():.2f}, test {yte.mean():.2f}")

        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models = {
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                     random_state=a.seed, n_jobs=-1),
            "RandomForest": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                          RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                 class_weight="balanced",
                                                                 random_state=a.seed, n_jobs=-1)),
            "LogReg(baseline)": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                              LogisticRegression(max_iter=1000, class_weight="balanced")),
        }
        for name, m in models.items():
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            pred = (p >= a.threshold).astype(int)
            r10, pr10 = topk_recall(yte, p, 0.10)
            r30, pr30 = topk_recall(yte, p, 0.30)
            rows.append(dict(cutoff=cut, model=name, auc=roc_auc_score(yte, p),
                             pr_auc=average_precision_score(yte, p), f1=f1_score(yte, pred),
                             recall=recall_score(yte, pred), precision=precision_score(yte, pred),
                             accuracy=accuracy_score(yte, pred),
                             recall_top10=r10, prec_top10=pr10, recall_top30=r30, prec_top30=pr30))
            if name == "XGBoost":
                for mod in np.unique(mods_te):
                    sel = mods_te == mod
                    if yte.values[sel].min() != yte.values[sel].max():
                        rows_mod.append(dict(cutoff=cut, module=mod, n=int(sel.sum()),
                                             pos_rate=yte.values[sel].mean(),
                                             auc=roc_auc_score(yte.values[sel], p[sel])))
            if name in ("XGBoost", "RandomForest"):
                est = m if name == "XGBoost" else m[-1]
                imp = pd.Series(est.feature_importances_, index=X.columns).sort_values(ascending=False)
                imp.head(20).to_csv(f"{a.out}/importance_{name}_cutoff{cut}.csv", header=["importance"])

    res, mod = pd.DataFrame(rows), pd.DataFrame(rows_mod)
    res.to_csv(f"{a.out}/classification_{a.target}.csv", index=False)
    mod.to_csv(f"{a.out}/by_module_{a.target}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n##### CLASSIFICATION — target = {a.target} #####")
    print(res.round(3).to_string(index=False))
    print("\n##### AUC của XGBoost theo từng môn (tập test 2014J) #####")
    print(mod.round(3).to_string(index=False))
    print(f"\nĐã lưu vào {a.out}/")


if __name__ == "__main__":
    main()