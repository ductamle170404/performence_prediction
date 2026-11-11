"""Inspect XAI features at each course-progress cutoff and train XGBoost only.

Run from project root:
  python tools\\analysis\\audit_xgb_cutoff.py --data data_subset --progress 0 10 20 30 40 50
  python tools\\analysis\\audit_xgb_cutoff.py --data data --progress 30 --no-shap

Expected imports: prediction/features.py and prediction/feature_sets/features_xai.py
"""
import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "prediction"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
TEST_PRESENTATION = "2014J"
CAT = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
DROP = ["id_student", "code_presentation", "final_result", "at_risk", "fail_only", "final_score",
        "date_unregistration", "module_presentation_length", "progress_pct", "cutoff_day", "presentation_length"]


def safe_columns(columns):
    x = pd.Index(columns).astype(str)
    for old, new in [("<=", "le_"), ("<", "lt_"), (">", "gt_"), ("[", "("), ("]", ")")]:
        x = x.str.replace(old, new, regex=False)
    return x.str.replace(r"\s+", "_", regex=True)


def subset_presentation(data, module, presentation, fm):
    out, keys = {}, fm.KEY3
    mask = ((data["studentInfo"]["code_module"] == module) &
            (data["studentInfo"]["code_presentation"] == presentation))
    student_keys = data["studentInfo"].loc[mask, keys]
    for name, frame in data.items():
        if name == "studentInfo":
            out[name] = frame.loc[mask].copy()
        elif name == "studentRegistration":
            out[name] = frame.merge(student_keys, on=keys, how="inner").copy()
        elif name in ("courses", "assessments"):
            out[name] = frame[(frame.code_module == module) &
                              (frame.code_presentation == presentation)].copy()
        elif name == "studentVle":
            out[name] = frame.merge(student_keys, on=keys, how="inner").copy()
        elif name == "studentAssessment":
            ids = data["assessments"].loc[
                (data["assessments"].code_module == module) &
                (data["assessments"].code_presentation == presentation), "id_assessment"].unique()
            out[name] = frame[frame.id_assessment.isin(ids)].copy()
        elif name == "vle":
            sv = data["studentVle"].merge(student_keys, on=keys, how="inner")
            out[name] = frame[frame.id_site.isin(sv.id_site.unique())].copy()
        else:
            out[name] = frame.copy()
    return out


def build_snapshot(data, progress, fm, train_presentations, test_presentation, verbose=False):
    pieces, eligibility = [], []
    for _, course in data["courses"].iterrows():
        module, pres = course["code_module"], course["code_presentation"]
        length = pd.to_numeric(pd.Series([course["module_presentation_length"]]), errors="coerce").iloc[0]
        if pd.isna(length) or length <= 0:
            continue
        day = int(np.floor(float(length) * float(progress) / 100))
        d = subset_presentation(data, module, pres, fm)
        if d["studentInfo"].empty:
            continue
        # Mirror features_common.base_frame() enrollment filter for an auditable roster.
        cand = (d["studentInfo"].merge(d["studentRegistration"], on=fm.KEY3, how="left")
                .merge(d["courses"], on=fm.KMP, how="left"))
        cand["date_unregistration"] = pd.to_numeric(cand["date_unregistration"], errors="coerce")
        cand["eligible_at_cutoff"] = cand.date_unregistration.isna() | (cand.date_unregistration > day)
        cand["eligibility_reason"] = np.where(cand.eligible_at_cutoff,
            "included: no unregistration by cutoff", "excluded: unregistered on/before cutoff")
        cand["progress_pct"], cand["cutoff_day"], cand["presentation_length"] = float(progress), day, int(length)
        eligibility.append(cand[fm.KEY3 + ["date_registration", "date_unregistration", "eligible_at_cutoff",
                                           "eligibility_reason", "progress_pct", "cutoff_day", "presentation_length"]])
        frame = fm.build(d, day)
        frame["progress_pct"], frame["cutoff_day"], frame["presentation_length"] = float(progress), day, int(length)
        pieces.append(frame)
        if verbose:
            print(f"{module}/{pres}: day {day}/{int(length)}; {len(frame):,} eligible rows")
    snap = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    elig = pd.concat(eligibility, ignore_index=True) if eligibility else pd.DataFrame()
    allowed = list(train_presentations) + [test_presentation]
    if not snap.empty:
        snap = snap[snap.code_presentation.isin(allowed)].copy()
    if not elig.empty:
        elig = elig[elig.code_presentation.isin(allowed)].copy()
    return snap, elig


def make_xy(df, columns=None):
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    cats = [c for c in CAT if c in X.columns]
    X = pd.get_dummies(X, columns=cats, dtype=float)
    X.columns = safe_columns(X.columns)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.replace([np.inf, -np.inf], np.nan)


def target(df, name):
    return df["fail_only"].astype(int) if name == "fail_only" else df["at_risk"].astype(int)


def group_of(col, groups):
    for group, names in groups.items():
        if any(col == n or col.startswith(n + "_") for n in names):
            return group
    return "other"


def export_audit(snapshot, train_df, test_df, X_train, X_test, fm, out, pct):
    tag = f"p{pct:g}"
    snapshot.to_csv(out / f"snapshot_{tag}.csv", index=False)
    raw = [c for c in snapshot.columns if c not in fm.KEY3 + ["final_result", "at_risk", "fail_only",
           "final_score", "progress_pct", "cutoff_day", "presentation_length"]]
    stats = []
    for c in raw:
        s = snapshot[c]
        r = {"feature": c, "group": group_of(c, fm.FEATURE_GROUPS), "dtype": str(s.dtype),
             "missing_n": int(s.isna().sum()), "missing_pct": float(s.isna().mean()),
             "n_unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            r.update(zero_n=int((s == 0).sum()), min=s.min(), mean=s.mean(), median=s.median(), max=s.max())
        stats.append(r)
    pd.DataFrame(stats).to_csv(out / f"feature_stats_{tag}.csv", index=False)
    pd.DataFrame({"feature": raw, "group": [group_of(c, fm.FEATURE_GROUPS) for c in raw],
                  "description": [fm.EXPLAIN.get(c, "") for c in raw]}).to_csv(
        out / f"feature_catalog_{tag}.csv", index=False)
    meta = [c for c in ["id_student", "code_module", "code_presentation", "progress_pct", "cutoff_day", "final_result"]
            if c in train_df.columns]
    pd.concat([train_df[meta].reset_index(drop=True), X_train.reset_index(drop=True)], axis=1).to_csv(
        out / f"X_train_{tag}.csv", index=False)
    pd.concat([test_df[meta].reset_index(drop=True), X_test.reset_index(drop=True)], axis=1).to_csv(
        out / f"X_test_{tag}.csv", index=False)
    pd.DataFrame({"model_feature": X_train.columns,
                  "group": [group_of(c, fm.FEATURE_GROUPS) for c in X_train.columns]}).to_csv(
        out / f"model_feature_catalog_{tag}.csv", index=False)


def run_shap(model, X_train, X_test, test_meta, out, pct, seed, sample=200):
    try:
        shap = importlib.import_module("shap")
    except Exception as e:
        print(f"SHAP skipped: {e}")
        return
    if X_train.empty or X_test.empty:
        return
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(X_test), min(sample, len(X_test)), replace=False)
    explain = X_test.iloc[ix]
    try:
        vals = shap.TreeExplainer(model).shap_values(explain)
        if isinstance(vals, list):
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        pd.DataFrame({"feature": X_train.columns, "mean_abs_shap": np.abs(vals).mean(axis=0)}).sort_values(
            "mean_abs_shap", ascending=False).to_csv(out / f"shap_importance_p{pct:g}.csv", index=False)
        pd.concat([test_meta.iloc[ix].reset_index(drop=True),
                   pd.DataFrame(vals, columns=X_train.columns)], axis=1).to_csv(
            out / f"shap_local_p{pct:g}.csv", index=False)
    except Exception as e:
        print(f"SHAP failed at {pct:g}%: {e}")


def main():
    ap = argparse.ArgumentParser(description="Audit XAI features by cutoff and fit XGBoost only.")
    ap.add_argument("--data", default="data_using")
    ap.add_argument("--feature-module", default="feature_sets.features_xai")
    ap.add_argument("--progress", nargs="+", type=float, default=[0, 10, 20, 30, 40, 50])
    ap.add_argument("--target", choices=["fail_only", "at_risk"], default="fail_only")
    ap.add_argument("--train-presentations", nargs="+", default=TRAIN_PRESENTATIONS)
    ap.add_argument("--test-presentation", default=TEST_PRESENTATION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="audit_xgb")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--check-leakage", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    fm = importlib.import_module(args.feature_module)
    data = fm.load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_all = []

    for pct in args.progress:
        print(f"\n=== Cutoff {pct:g}% ===")
        snapshot, eligibility = build_snapshot(data, pct, fm, args.train_presentations,
                                               args.test_presentation, args.verbose)
        if snapshot.empty:
            print("No rows; skipped")
            continue
        tag = f"p{pct:g}"
        eligibility.to_csv(out / f"eligibility_{tag}.csv", index=False)
        tr = snapshot[snapshot.code_presentation.isin(args.train_presentations)].copy().reset_index(drop=True)
        te = snapshot[snapshot.code_presentation.eq(args.test_presentation)].copy().reset_index(drop=True)
        if tr.empty or te.empty:
            print(f"Missing train/test: {len(tr)}/{len(te)}")
            continue
        ytr, yte = target(tr, args.target), target(te, args.target)
        Xtr = make_xy(tr)
        Xte = make_xy(te, columns=Xtr.columns)
        Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
        ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)
        export_audit(pd.concat([tr, te], ignore_index=True), tr, te, Xtr, Xte, fm, out, pct)
        print(f"Train={len(tr):,}; test={len(te):,}; features={Xtr.shape[1]:,}; "
              f"positive train={ytr.mean():.3f}; positive test={yte.mean():.3f}")

        pos, neg = max(int(ytr.sum()), 1), max(int((ytr == 0).sum()), 1)
        model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos,
            eval_metric="logloss", random_state=args.seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        meta_cols = [c for c in ["id_student", "code_module", "code_presentation", "progress_pct",
                                 "cutoff_day", "presentation_length", "final_result"] if c in te.columns]
        meta = te[meta_cols].copy().reset_index(drop=True)
        meta["actual_target"] = yte.to_numpy()
        meta["probability_positive"] = prob
        meta["risk_rank"] = pd.Series(prob).rank(ascending=False, method="first").astype(int)
        meta.to_csv(out / f"predictions_XGBoost_{tag}.csv", index=False)
        imp = pd.DataFrame({"feature": Xtr.columns, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False)
        imp.to_csv(out / f"xgb_importance_{tag}.csv", index=False)

        auc = roc_auc_score(yte, prob) if yte.nunique() > 1 else np.nan
        pr = average_precision_score(yte, prob) if yte.sum() > 0 else np.nan
        metrics = {"progress_pct": pct, "cutoff_day_min": int(snapshot.cutoff_day.min()),
                   "cutoff_day_max": int(snapshot.cutoff_day.max()), "train_n": len(tr), "test_n": len(te),
                   "n_features": Xtr.shape[1], "positive_rate_test": float(yte.mean()),
                   "roc_auc": auc, "pr_auc": pr, "brier": brier_score_loss(yte, prob)}
        for k in (10,  30):
            n = max(int(np.ceil(len(prob) * k / 100)), 1)
            idx = np.argsort(-prob)[:n]
            metrics[f"precision_top{k}"] = float(yte.to_numpy()[idx].mean())
            metrics[f"recall_top{k}"] = float(yte.to_numpy()[idx].sum() / max(int(yte.sum()), 1))
        metrics_all.append(metrics)
        if not args.no_shap:
            run_shap(model, Xtr, Xte, meta, out, pct, args.seed)

        if args.check_leakage and hasattr(fm, "audit"):
            # Audit each distinct course-day represented at this progress point.
            checks = []
            for day in sorted(snapshot.cutoff_day.unique()):
                bad = fm.audit(data, int(day))
                checks.append({"cutoff_day": int(day), "mismatched_features": ";".join(bad),
                               "passed": len(bad) == 0})
            pd.DataFrame(checks).to_csv(out / f"leakage_check_{tag}.csv", index=False)

    if metrics_all:
        pd.DataFrame(metrics_all).to_csv(out / "xgboost_metrics_by_cutoff.csv", index=False)
    print(f"\nOutputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
