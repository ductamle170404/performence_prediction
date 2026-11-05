"""Danh sách xếp hạng rủi ro cho cố vấn học tập + đánh giá độ chính xác khi đi xuống danh sách.

Ví dụ:
    python rank_list.py --data data_using --cutoff 60
    python rank_list.py --data data_using --cutoff 30 --model rf --target fail_only

Giả lập triển khai thật: huấn luyện trên các đợt đã kết thúc (2013B, 2013J, 2014B),
chấm điểm đợt "hiện tại" (2014J) tại ngày --cutoff. Nhãn thật của 2014J CHỈ dùng để đánh giá.

Kết quả (thư mục ranked/):
  ranked_list_day{cutoff}.csv         danh sách đưa cho cố vấn (KHÔNG chứa nhãn thật)
  evaluation_top_k_day{cutoff}.csv    nếu chỉ liên hệ top K% thì đúng/sai bao nhiêu
  evaluation_deciles_day{cutoff}.csv  từng 10% một: tỉ lệ thật sự rủi ro, điểm dự đoán trung bình
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import features as F
from train_cls import make_xy, TEST_PRES

warnings.filterwarnings("ignore")
CONTEXT = ["submit_rate", "n_missed", "n_submitted", "mean_score", "min_score",
           "last_active_gap", "clicks_last7", "clicks_last14"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cutoff", type=int, default=60)
    ap.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    ap.add_argument("--target", choices=["at_risk", "fail_only"], default="at_risk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ranked")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = F.load(a.data)
    df = F.build(d, a.cutoff)
    is_test = df["code_presentation"] == TEST_PRES
    X = make_xy(df)
    ytr = df.loc[~is_test, a.target]

    if a.model == "xgb":
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                          random_state=a.seed, n_jobs=-1)
    else:
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                 class_weight="balanced", random_state=a.seed, n_jobs=-1))
    m.fit(X[~is_test], ytr)

    te = df.loc[is_test].copy()
    te["risk_score"] = m.predict_proba(X[is_test])[:, 1]
    te["actual"] = te[a.target].values
    te = te.sort_values("risk_score", ascending=False).reset_index(drop=True)
    te["rank"] = te.index + 1
    n = len(te)
    te["top_percent"] = (te["rank"] / n * 100).round(1)

    # ---- 1. Danh sách cho cố vấn (không có nhãn thật) ----
    cols = ["rank", "top_percent", "id_student", "code_module", "code_presentation", "risk_score"]
    cols += [c for c in CONTEXT if c in te.columns]
    lst = te[cols].copy()
    lst["risk_score"] = lst["risk_score"].round(3)
    lst.to_csv(f"{a.out}/ranked_list_day{a.cutoff}.csv", index=False, encoding="utf-8-sig")

    # ---- 2. Đánh giá: lấy top K% thì đúng/sai bao nhiêu ----
    base, total_pos = te["actual"].mean(), te["actual"].sum()
    rows = []
    for k in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        mk = max(int(n * k / 100), 1)
        top = te.iloc[:mk]
        prec = top["actual"].mean()
        rows.append(dict(top_percent=k, n_flagged=mk, true_at_risk=int(top["actual"].sum()),
                         precision=prec, false_alarm_rate=1 - prec,
                         recall=top["actual"].sum() / total_pos, lift=prec / base))
    topk = pd.DataFrame(rows)
    topk.to_csv(f"{a.out}/evaluation_top_k_day{a.cutoff}.csv", index=False)

    # ---- 3. Từng 10% một ----
    te["decile"] = np.minimum(te.index * 10 // n, 9) + 1      # 1 = rủi ro cao nhất
    dec = te.groupby("decile").agg(n=("actual", "size"), actual_rate=("actual", "mean"),
                                   mean_risk_score=("risk_score", "mean")).reset_index()
    dec.to_csv(f"{a.out}/evaluation_deciles_day{a.cutoff}.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"Ngày {a.cutoff} | mô hình {a.model} | nhãn {a.target} | {n:,} sinh viên đợt {TEST_PRES} "
          f"| tỉ lệ rủi ro thật (mức nền) {base:.3f}")
    print("\n## Chỉ liên hệ top K% sinh viên có điểm rủi ro cao nhất")
    print(topk.round(3).to_string(index=False))
    print("\n## Từng 10% một (decile 1 = rủi ro cao nhất)")
    print(dec.round(3).to_string(index=False))
    print(f"\nĐã lưu vào {a.out}/")


if __name__ == "__main__":
    main()
