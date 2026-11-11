"""
Early-warning probability prediction by course progress.

Three models:
  - XGBoost
  - Random Forest
  - Logistic Regression (baseline)

Each row in prediction output represents a student-course-presentation at a
progress cutoff (default: 0, 10, 20, 30, 40, 50 percent).

Important:
  * The feature builder is called with a presentation-specific day cutoff.
  * Only students still enrolled at that cutoff are scored.
  * Final outcome is used only as the target/evaluation label, never as input.
  * The default temporal split trains on 2013B/2013J/2014B and tests on 2014J.

Example:
    python prediction\\predict_progress.py --data data_pure --progress 0 10 20 30 40 50
    python prediction\\predict_progress.py --data data_subset --progress 0 10 20 30 40 50
    python prediction\\predict_progress.py --data data_subset --feature-set wu_jia --out predict_wu_jia
    python prediction\\predict_progress.py --data data_subset --feature-set kuznetsov_ews --out predict_kuznetsov
    python prediction\\predict_progress.py --data data_subset --feature-set joive --out predict_joive
    python prediction\\predict_progress.py --data data_subset --feature-set mtapsp --out predict_mtapsp
    python prediction\\predict_progress.py --data data_subset --feature-set trajectory --out predict_trajectory
    python prediction\\predict_progress.py --data data_subset --feature-set xai --out predict_xai
    python prediction\\predict_progress.py --data data --target fail_only --seed 1
    python prediction\\predict_progress.py --data data --no-shap --no-lime
    python prediction\\predict_progress.py --data data --out predict_custom
    python prediction\\predict_progress.py --data data_subset --progress 0 10 20 30 40 50 --no-shap --no-lime --target-recall 0.8

Outputs are written under database/predictions/ by default.
"""
import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
import os
import re
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "database" / "predictions"

DEFAULT_TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
DEFAULT_TEST_PRESENTATION = "2014J"
FEATURE_SET_MODULES = {
    "kuznetsov_ews": "feature_sets.features_kuznetsov_ews",
    "wu_jia": "feature_sets.features_wu_jia",
    "joive": "feature_sets.features_joive",
    "mtapsp": "feature_sets.features_mtapsp",
    "xai": "feature_sets.features_xai",
    "trajectory": "feature_sets.features_trajectory",
}

# Mức recall mục tiêu để chọn ngưỡng quyết định (nằm trong khoảng 0.80-0.90).
# 0.85: bỏ sót tối đa ~15% sinh viên sẽ rớt, trong khi vẫn giữ precision
# không sụp ở các mốc tiến độ sớm (0.90 thường làm flag gần hết lớp).
DEFAULT_TARGET_RECALL = 0.85
TOP_K_FRACTIONS = (0.05, 0.10, 0.20, 0.30)

CAT = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability",
]
DROP = [
    "id_student", "code_presentation", "final_result", "at_risk",
    "fail_only", "final_score", "date_unregistration",
    "module_presentation_length", "progress_pct", "cutoff_day",
    "presentation_length",
]


def safe_columns(columns):
    """Make column names safe for XGBoost and consistent across snapshots."""
    out = pd.Index(columns).astype(str)
    out = out.str.replace("<=", "le_", regex=False)
    out = out.str.replace("<", "lt_", regex=False)
    out = out.str.replace(">", "gt_", regex=False)
    out = out.str.replace("[", "(", regex=False)
    out = out.str.replace("]", ")", regex=False)
    out = out.str.replace(r"\s+", "_", regex=True)
    return out


def subset_presentation(data, module, presentation, feature_module):
    """
    Restrict source tables to one presentation before calling the feature builder.

    This makes the cutoff specific to the presentation length and avoids
    accidentally combining records from other presentations.
    """
    out = {}
    kmask = (
        (data["studentInfo"]["code_module"] == module)
        & (data["studentInfo"]["code_presentation"] == presentation)
    )
    student_keys = data["studentInfo"].loc[kmask, feature_module.KEY3]

    for name, frame in data.items():
        if name == "studentInfo":
            out[name] = frame.loc[kmask].copy()
        elif name == "studentRegistration":
            out[name] = frame.merge(
                student_keys, on=feature_module.KEY3, how="inner"
            ).copy()
        elif name == "courses":
            out[name] = frame[
                (frame["code_module"] == module)
                & (frame["code_presentation"] == presentation)
            ].copy()
        elif name == "assessments":
            out[name] = frame[
                (frame["code_module"] == module)
                & (frame["code_presentation"] == presentation)
            ].copy()
        elif name == "vle":
            # VLE resources are linked to this presentation through its assessment/course data.
            # Retain the presentation's sites from studentVle.
            sv = data["studentVle"].merge(
                student_keys, on=feature_module.KEY3, how="inner"
            )
            site_ids = sv["id_site"].unique()
            out[name] = frame[frame["id_site"].isin(site_ids)].copy()
        elif name == "studentAssessment":
            ids = out.get("assessments", data["assessments"][
                (data["assessments"]["code_module"] == module)
                & (data["assessments"]["code_presentation"] == presentation)
            ])["id_assessment"].unique()
            out[name] = frame[frame["id_assessment"].isin(ids)].copy()
        elif name == "studentVle":
            out[name] = frame.merge(
                student_keys, on=feature_module.KEY3, how="inner"
            ).copy()
        else:
            out[name] = frame.copy()
    return out


def build_all_presentations(data, progress_pct, feature_module, verbose=False):
    """
    Build one feature snapshot per module-presentation at progress_pct.

    Returns a concatenated Student-Course dataframe with a cutoff_day column.
    """
    course_table = data["courses"].copy()
    pieces = []

    for _, course in course_table.iterrows():
        module = course["code_module"]
        presentation = course["code_presentation"]
        length = pd.to_numeric(
            pd.Series([course["module_presentation_length"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(length) or length <= 0:
            if verbose:
                print(f"Skip {module}/{presentation}: invalid presentation length")
            continue

        cutoff_day = int(np.floor(float(length) * float(progress_pct) / 100.0))
        d_sub = subset_presentation(data, module, presentation, feature_module)

        if d_sub["studentInfo"].empty:
            continue

        frame = feature_module.build(d_sub, cutoff_day)
        frame["progress_pct"] = float(progress_pct)
        frame["cutoff_day"] = cutoff_day
        frame["presentation_length"] = int(length)
        pieces.append(frame)

        if verbose:
            print(
                f"  {module}/{presentation}: {cutoff_day}/{int(length)} days, "
                f"{len(frame):,} eligible student-course rows"
            )

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def make_xy(df, columns=None):
    """One-hot encode categoricals and optionally align to training columns."""
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()

    cat_cols = [c for c in CAT if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, dtype=float)
    X.columns = safe_columns(X.columns)

    # Coerce any unexpected nonnumeric feature to numeric/missing.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def get_target(df, target_name):
    if target_name == "fail_only":
        return df["fail_only"].astype(int)
    return df["at_risk"].astype(int)


def build_models(y_train, seed):
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    return {
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    }


def get_estimator(model):
    """Return the fitted estimator inside sklearn pipelines."""
    return model[-1] if hasattr(model, "steps") else model


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def evaluate_probabilities(y, p):
    result = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else np.nan,
        "roc_auc": safe_auc(y, p),
        "pr_auc": (
            average_precision_score(y, p)
            if len(np.unique(y)) > 1 and np.sum(y) > 0
            else np.nan
        ),
        "brier": brier_score_loss(y, p) if len(y) else np.nan,
        "log_loss": (
            log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
            if len(y) and len(np.unique(y)) > 1
            else np.nan
        ),
    }

    # Capacity-based ranking metrics. No fixed probability threshold is used.
    for k in TOP_K_FRACTIONS:
        n_top = max(int(np.ceil(len(y) * k)), 1) if len(y) else 0
        idx = np.argsort(-p)[:n_top]
        y_top = np.asarray(y)[idx]
        tag = int(round(k * 100))
        result[f"precision_top{tag}"] = (
            float(y_top.mean()) if len(y_top) else np.nan
        )
        result[f"recall_top{tag}"] = (
            float(y_top.sum() / max(np.sum(y), 1)) if len(y_top) else np.nan
        )
    return result


def select_threshold_for_recall(y, p, target_recall):
    """
    Ngưỡng cao nhất sao cho recall >= target_recall trên (y, p).
    Cao nhất => precision tốt nhất trong số các ngưỡng đạt recall mục tiêu.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) == 0 or y.sum() == 0:
        return np.nan, np.nan
    _, recall, thresholds = precision_recall_curve(y, p)
    # recall có len = len(thresholds) + 1; phần tử cuối (recall=0) bỏ đi.
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds[0]), float(recall[0])
    i = ok[-1]  # thresholds tăng dần => recall giảm dần => i cuối = ngưỡng cao nhất
    return float(thresholds[i]), float(recall[i])


def oof_probabilities(train_df, y_train, args):
    """
    Xác suất out-of-fold trên tập train bằng leave-one-presentation-out.
    Dùng để chọn ngưỡng mà KHÔNG chạm vào nhãn tập test và không bị
    overfit như xác suất dự đoán lại chính tập train.
    Trả về dict model_name -> mảng xác suất (NaN nếu hàng không được chấm).
    """
    oof = {}
    pres = train_df["code_presentation"].to_numpy()
    for hold in sorted(set(pres)):
        va = pres == hold
        tr = ~va
        if tr.sum() == 0 or va.sum() == 0:
            continue
        y_tr = y_train.to_numpy()[tr]
        if len(np.unique(y_tr)) < 2:
            continue
        X_tr = make_xy(train_df.loc[tr])
        X_va = make_xy(train_df.loc[va], columns=X_tr.columns)
        for name, model in build_models(pd.Series(y_tr), args.seed).items():
            model.fit(X_tr, y_tr)
            oof.setdefault(name, np.full(len(train_df), np.nan))
            oof[name][va] = model.predict_proba(X_va)[:, 1]
    return oof


def evaluate_at_threshold(y, p, threshold):
    """Precision/recall/... tại một ngưỡng xác suất cố định."""
    y = np.asarray(y).astype(int)
    if len(y) == 0 or np.isnan(threshold):
        return {}
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) > 0 else np.nan)
    return {
        "threshold": float(threshold),
        "precision_at_thr": prec,
        "recall_at_thr": rec,
        "f1_at_thr": f1,
        "flagged_rate": float(pred.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def save_feature_importance(model, feature_names, out_dir, model_name, progress):
    estimator = get_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        return
    imp = pd.Series(
        estimator.feature_importances_, index=feature_names, name="importance"
    ).sort_values(ascending=False)
    folder = Path(out_dir) / "importance"
    folder.mkdir(parents=True, exist_ok=True)
    imp.to_csv(folder / f"importance_{model_name}_p{progress:g}.csv")


def run_shap(model, X_train, X_test, out_dir, model_name, progress, seed,
             max_background=100, max_explain=200):
    """Optional SHAP export; gracefully skip if SHAP is not installed."""
    try:
        shap = import_module("shap")
    except Exception as exc:
        print(f"  SHAP unavailable ({exc}); skipping.")
        return

    if X_test.empty or X_train.empty:
        return

    rng = np.random.default_rng(seed)
    bg_n = min(max_background, len(X_train))
    ex_n = min(max_explain, len(X_test))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_test), size=ex_n, replace=False)
    background = X_train.iloc[bg_idx]
    explain = X_test.iloc[ex_idx]

    estimator = get_estimator(model)
    model_dir = Path(out_dir) / "shap"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        if model_name == "XGBoost":
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(explain)
            if isinstance(values, list):
                values = values[-1]
        elif model_name == "RandomForest":
            # The RF pipeline imputes missing values before the tree estimator.
            transformed_bg = model[0].transform(background)
            transformed_ex = model[0].transform(explain)
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed_ex)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            elif np.asarray(values).ndim == 3:
                values = np.asarray(values)[:, :, 1]
        else:
            # SHAP linear explainer operates on transformed data for the pipeline.
            transformed_bg = model[0].transform(background)
            transformed_ex = model[0].transform(explain)
            explainer = shap.LinearExplainer(estimator, transformed_bg)
            values = explainer.shap_values(transformed_ex)

        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        mean_abs = np.abs(values).mean(axis=0)
        summary = pd.DataFrame({
            "feature": list(X_train.columns),
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        summary.to_csv(
            model_dir / f"shap_importance_{model_name}_p{progress:g}.csv",
            index=False,
        )

        # Local explanation values, aligned with student-course test records.
        local = pd.DataFrame(values, columns=X_train.columns)
        local.insert(0, "test_row_index", explain.index.to_numpy())
        local.to_csv(
            model_dir / f"shap_values_{model_name}_p{progress:g}.csv",
            index=False,
        )
    except Exception as exc:
        print(f"  SHAP failed for {model_name} at {progress:g}%: {exc}")


def run_lime(model, X_train, X_test, metadata, out_dir, model_name, progress,
             seed, max_explain=20):
    """Optional local LIME explanations for a small sample of test rows."""
    try:
        LimeTabularExplainer = import_module(
            "lime.lime_tabular"
        ).LimeTabularExplainer
    except Exception as exc:
        print(f"  LIME unavailable ({exc}); skipping.")
        return

    if X_test.empty or X_train.empty:
        return

    # LIME expects a numeric matrix. Impute only for the explanation interface.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_arr = imputer.fit_transform(X_train)
    test_arr = imputer.transform(X_test)

    explainer = LimeTabularExplainer(
        training_data=train_arr,
        feature_names=list(X_train.columns),
        class_names=["not_fail", "fail"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    n = min(max_explain, len(test_arr))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(test_arr), size=n, replace=False)
    rows = []

    for idx in selected:
        try:
            explanation = explainer.explain_instance(
                test_arr[idx],
                model.predict_proba,
                num_features=min(15, X_train.shape[1]),
                num_samples=3000,
            )
            for feature, weight in explanation.as_list():
                metadata_row = metadata.loc[X_test.index[idx]]
                rows.append({
                    "test_row_index": int(X_test.index[idx]),
                    "id_student": metadata_row.get("id_student"),
                    "code_module": metadata_row.get("code_module"),
                    "code_presentation": metadata_row.get("code_presentation"),
                    "final_result": metadata_row.get("final_result"),
                    "actual_fail": metadata_row.get("actual_fail"),
                    "model": model_name,
                    "progress_pct": progress,
                    "lime_feature": feature,
                    "lime_weight": weight,
                })
        except Exception as exc:
            print(f"  LIME row {idx} skipped: {exc}")

    if rows:
        folder = Path(out_dir) / "lime"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            folder / f"lime_{model_name}_p{progress:g}.csv",
            index=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict student-course failure probabilities by course progress."
    )
    parser.add_argument("--data", default="data")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_MODULES,
        default="xai",
        help="Feature set to use (default: xai).",
    )
    parser.add_argument(
        "--progress", type=float, nargs="+",
        default=[0, 10, 20, 30, 40, 50],
        help="Course progress cutoffs in percent.",
    )
    parser.add_argument(
        "--target", choices=["at_risk", "fail_only"], default="fail_only",
        help="at_risk=Fail+Withdrawn; fail_only=Fail, Withdrawn treated as 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-recall", type=float, default=DEFAULT_TARGET_RECALL,
        help="Recall mục tiêu (0.8-0.9) để chọn ngưỡng positive/negative.",
    )
    parser.add_argument("--train-presentations", nargs="+",
                        default=DEFAULT_TRAIN_PRESENTATIONS)
    parser.add_argument("--test-presentation", default=DEFAULT_TEST_PRESENTATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shap-sample", type=int, default=200)
    parser.add_argument("--lime-sample", type=int, default=20)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-lime", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    feature_module = import_module(FEATURE_SET_MODULES[args.feature_set])
    print(f"Feature set: {args.feature_set}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = feature_module.load(args.data)
    prediction_frames = {name: [] for name in
                         ["XGBoost", "RandomForest", "LogisticRegression"]}
    evaluation_rows = []
    module_rows = []

    for progress in args.progress:
        print(f"\n=== Progress cutoff: {progress:g}% ===")
        df = build_all_presentations(
            data, progress, feature_module, verbose=args.verbose
        )

        if df.empty:
            print("No feature rows built at this progress cutoff; skipping.")
            continue

        train_mask = df["code_presentation"].isin(args.train_presentations)
        test_mask = df["code_presentation"] == args.test_presentation

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            print(
                f"Missing train/test rows at {progress:g}% "
                f"(train={len(train_df)}, test={len(test_df)}); skipping."
            )
            continue

        y_train = get_target(train_df, args.target)
        y_test = get_target(test_df, args.target)

        # Build train matrix first; align test columns to it.
        X_train = make_xy(train_df)
        X_test = make_xy(test_df, columns=X_train.columns)

        # Exclude labels and metadata from model matrices.
        # Features not present in the train set are deliberately ignored.
        print(
            f"Train={len(X_train):,} | test={len(X_test):,} | "
            f"features={X_train.shape[1]:,} | "
            f"positive train={y_train.mean():.3f} | positive test={y_test.mean():.3f}"
        )

        models = build_models(y_train, args.seed)

        # Chọn ngưỡng từ xác suất out-of-fold của tập train (không dùng test).
        oof = oof_probabilities(train_df, y_train, args)

        # Metadata is retained separately for the prediction outputs.
        metadata_cols = [
            "id_student", "code_module", "code_presentation",
            "progress_pct", "cutoff_day", "presentation_length",
            "final_result",
        ]
        metadata = test_df[[c for c in metadata_cols if c in test_df.columns]].copy()
        metadata["actual_fail"] = y_test.to_numpy()

        for model_name, model in models.items():
            print(f"  Fitting {model_name}...")
            model.fit(X_train, y_train)
            p_test = model.predict_proba(X_test)[:, 1]

            # Ngưỡng đạt recall mục tiêu trên OOF của train.
            thr, oof_recall = np.nan, np.nan
            if model_name in oof:
                valid = ~np.isnan(oof[model_name])
                thr, oof_recall = select_threshold_for_recall(
                    y_train.to_numpy()[valid], oof[model_name][valid],
                    args.target_recall,
                )

            pred = metadata.copy()
            pred["model"] = model_name
            pred["probability_fail"] = p_test
            pred["threshold"] = thr
            pred["predicted_positive"] = (
                (p_test >= thr).astype(int) if not np.isnan(thr) else np.nan
            )
            pred["target_definition"] = args.target
            prediction_frames[model_name].append(pred)

            metrics = evaluate_probabilities(y_test.to_numpy(), p_test)
            thr_metrics = evaluate_at_threshold(y_test.to_numpy(), p_test, thr)
            evaluation_rows.append({
                "progress_pct": progress,
                "model": model_name,
                "target": args.target,
                "target_recall": args.target_recall,
                "oof_recall_at_thr": oof_recall,
                **metrics,
                **thr_metrics,
            })

            if model_name == "XGBoost":
                for module in sorted(test_df["code_module"].dropna().unique()):
                    idx = test_df["code_module"].to_numpy() == module
                    if len(np.unique(y_test.to_numpy()[idx])) > 1:
                        module_rows.append({
                            "progress_pct": progress,
                            "model": model_name,
                            "module": module,
                            **evaluate_probabilities(
                                y_test.to_numpy()[idx], p_test[idx]
                            ),
                        })

            save_feature_importance(
                model, X_train.columns, out_dir, model_name, progress
            )

            if not args.no_shap:
                run_shap(
                    model, X_train, X_test, out_dir, model_name,
                    progress, args.seed, max_explain=args.shap_sample,
                )

            if not args.no_lime and args.lime_sample > 0:
                run_lime(
                    model, X_train, X_test, metadata, out_dir, model_name,
                    progress, args.seed, max_explain=args.lime_sample,
                )

    # Write prediction files in long format: one row per student-course-cutoff.
    for model_name, frames in prediction_frames.items():
        if frames:
            result = pd.concat(frames, ignore_index=True)
            result = result.sort_values(
                ["code_module", "code_presentation", "id_student", "progress_pct"]
            )
            result.to_csv(
                out_dir / f"predictions_{model_name}.csv", index=False
            )

    evaluation = pd.DataFrame(evaluation_rows)
    if not evaluation.empty:
        evaluation.to_csv(out_dir / "evaluation.csv", index=False)
        print("\n=== Evaluation ===")
        print(evaluation.round(4).to_string(index=False))

    by_module = pd.DataFrame(module_rows)
    if not by_module.empty:
        by_module.to_csv(out_dir / "by_module_XGBoost.csv", index=False)

    # One combined file is convenient for downstream dashboard ingestion.
    combined = [
        frame for frames in prediction_frames.values() for frame in frames
    ]
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(
            out_dir / "predictions_all_models.csv", index=False
        )
        manifest = {
            "run_id": str(uuid.uuid4()),
            "feature_set": args.feature_set,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "progress": sorted({
                float(value)
                for frames in prediction_frames.values()
                for frame in frames
                for value in frame["progress_pct"].unique()
            }),
        }
        manifest_path = out_dir / "manifest.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_manifest.replace(manifest_path)
        print(f"Run manifest saved to: {manifest_path.resolve()}")

    print(f"\nOutputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()