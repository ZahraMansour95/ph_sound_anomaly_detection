import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RobustCalibrator:
    """Median/IQR score calibration."""
    median: np.ndarray
    iqr: np.ndarray
    eps: float = 1e-6
    clip_min: float = 0.0

    @classmethod
    def fit(cls, components, eps=1e-6, clip_min=0.0):
        values = np.asarray(components, dtype=np.float64)
        med = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        scale = q75 - q25
        fallback = np.nanstd(values, axis=0)
        scale = np.where(scale > eps, scale, np.where(fallback > eps, fallback, 1.0))
        return cls(med, scale, eps, clip_min)

    def transform(self, components):
        z = (np.asarray(components) - self.median) / (self.iqr + self.eps)
        z = np.maximum(z, self.clip_min)
        return z, np.nanmean(z, axis=-1)

    def save(self, path):
        Path(path).write_text(json.dumps({"median": self.median.tolist(), "iqr": self.iqr.tolist(),
                                          "eps": self.eps, "clip_min": self.clip_min}, indent=2))


@dataclass
class EmpiricalTailCalibrator:
    """Convert NLL values to healthy upper-tail scores."""
    sorted_values: list

    @classmethod
    def fit(cls, components):
        values = np.asarray(components, dtype=np.float64)
        return cls([np.sort(values[np.isfinite(values[:, j]), j]) for j in range(values.shape[1])])

    def transform(self, components):
        values = np.asarray(components, dtype=np.float64)
        surprises = np.full_like(values, np.nan)
        tails = np.full_like(values, np.nan)
        for j, reference in enumerate(self.sorted_values):
            if not len(reference):
                continue
            valid = np.isfinite(values[:, j])
            exceed = len(reference) - np.searchsorted(reference, values[valid, j], side="left")
            tails[valid, j] = (exceed + 1.0) / (len(reference) + 1.0)
            surprises[valid, j] = -np.log(tails[valid, j])
        return tails, surprises, np.nanmax(surprises, axis=1)

    def to_dict(self, names):
        return {name: values.tolist() for name, values in zip(names, self.sorted_values)}


def select_threshold(healthy_scores, cfg):
    scores = np.asarray(healthy_scores, float)
    scores = scores[np.isfinite(scores)]
    q = 1.0 - cfg.target_fpr if cfg.threshold_method == "target_fpr" else cfg.threshold_percentile / 100.0
    return float(np.quantile(scores, q))


def _safe(call):
    try: return float(call())
    except ValueError: return float("nan")


def binary_metrics(y_true, y_score, threshold):
    y_true, y_score = np.asarray(y_true, int), np.asarray(y_score, float)
    finite = np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]
    if not len(y_true):
        return {"threshold": float(threshold), "auroc": np.nan, "auprc": np.nan,
                "accuracy": np.nan, "balanced_accuracy": np.nan,
                "f1_macro": np.nan, "mcc": np.nan, "sensitivity": np.nan,
                "specificity": np.nan, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "auroc": _safe(lambda: roc_auc_score(y_true, y_score)),
            "auprc": _safe(lambda: average_precision_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_interval(y, score, metric, iterations=1000, seed=42):
    y, score = np.asarray(y), np.asarray(score)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    if not len(y) or np.unique(y).size < 2:
        return (np.nan, np.nan)
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            estimates.append(metric(y[idx], score[idx]))
    return tuple(np.percentile(estimates, [2.5, 97.5])) if estimates else (np.nan, np.nan)


def _component_metrics(val_df, test_df, component_cols, cfg, level, fixed_thresholds=None):
    y = (test_df.Group.str.lower() != "healthy").astype(int).to_numpy()
    rows = []
    pairs = [(c.removesuffix("_nll"), c) for c in component_cols] + [("combined", "score")]
    for component, score_col in pairs:
        threshold = (fixed_thresholds[component] if fixed_thresholds is not None
                     else select_threshold(val_df[score_col], cfg))
        metrics = binary_metrics(y, test_df[score_col], threshold)
        lo, hi = bootstrap_interval(y, test_df[score_col], roc_auc_score,
                                    cfg.bootstrap_iterations, cfg.seed)
        pr_lo, pr_hi = bootstrap_interval(y, test_df[score_col], average_precision_score,
                                          cfg.bootstrap_iterations, cfg.seed + 1)
        rows.append({"level": level, "component": component, **metrics,
                     "auroc_ci_low": lo, "auroc_ci_high": hi,
                     "auprc_ci_low": pr_lo, "auprc_ci_high": pr_hi})
    return pd.DataFrame(rows)


def _calibrate_level(val_df, test_df, component_cols, cfg):
    calibrator = EmpiricalTailCalibrator.fit(val_df[component_cols].to_numpy())
    legacy = RobustCalibrator.fit(val_df[component_cols].to_numpy(), cfg.calibration_eps,
                                  cfg.calibration_clip_min)
    names = [c.removesuffix("_nll") for c in component_cols]
    for frame in (val_df, test_df):
        raw = frame[component_cols].to_numpy()
        tail, surprise, combined = calibrator.transform(raw)
        _, legacy_score = legacy.transform(raw)
        for j, name in enumerate(names):
            frame[f"{name}_tail_probability"] = tail[:, j]
            frame[f"{name}_tail_surprise"] = surprise[:, j]
        frame["score"] = combined
        frame["legacy_mean_score"] = legacy_score
        frame["dominant_component"] = np.asarray(names)[np.nanargmax(surprise, axis=1)]
    return calibrator, legacy


def linear_probe_evaluation(test_df, cfg, out_dir):
    embedding_cols = [c for c in test_df if c.startswith("embedding_")]
    if not cfg.linear_probe_enabled or not embedding_cols:
        return None
    subject = test_df.groupby(["Patient", "Group"], as_index=False)[embedding_cols].mean()
    x = subject[embedding_cols].to_numpy()
    y = (subject.Group.str.lower() != "healthy").astype(int).to_numpy()
    counts = np.bincount(y) if np.unique(y).size == 2 else np.array([0])
    folds = min(cfg.linear_probe_folds, int(counts.min()))
    if folds < 2: return None
    cv = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                    class_weight="balanced"))
    probability = np.zeros(len(y), dtype=float)
    for train, test in cv.split(x, y):
        fitted = clone(estimator).fit(x[train], y[train])
        probability[test] = fitted.predict_proba(x[test])[:, 1]
    subject["label"] = y; subject["linear_probe_probability"] = probability
    subject.to_csv(Path(out_dir) / "linear_probe_oof_predictions.csv", index=False)
    metrics = binary_metrics(y, probability, 0.5)
    lo, hi = bootstrap_interval(y, probability, roc_auc_score, cfg.bootstrap_iterations, cfg.seed)
    result = pd.DataFrame([{**metrics, "auroc_ci_low": lo, "auroc_ci_high": hi,
                            "folds": folds, "unit": "subject",
                            "interpretation": "post-hoc supervised separability"}])
    result.to_csv(Path(out_dir) / "linear_probe_metrics.csv", index=False)
    return result


def save_scalar_evaluation(val_df, test_df, cfg, out_dir):
    """Evaluate pattern-level GNLL scores."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    recording_threshold = select_threshold(val_df.score, cfg)
    y = (test_df.Group.str.lower() != "healthy").astype(int).to_numpy()
    metrics = binary_metrics(y, test_df.score, recording_threshold)
    pd.DataFrame([metrics]).to_csv(out / "recording_metrics.csv", index=False)
    test_df.assign(prediction=(test_df.score >= recording_threshold).astype(int)).to_csv(
        out / "recording_scores.csv", index=False)
    reducer = "max" if cfg.subject_aggregation == "max" else "mean"
    val_subjects = val_df.groupby(["Patient", "Group"], as_index=False).score.agg(reducer)
    subjects = test_df.groupby(["Patient", "Group"], as_index=False).score.agg(reducer)
    subject_threshold = recording_threshold
    ys = (subjects.Group.str.lower() != "healthy").astype(int).to_numpy()
    subject_metrics = binary_metrics(ys, subjects.score, subject_threshold)
    pd.DataFrame([subject_metrics]).to_csv(out / "subject_metrics.csv", index=False)
    subjects.assign(prediction=(subjects.score >= subject_threshold).astype(int)).to_csv(
        out / "subject_scores.csv", index=False)
    val_df.to_csv(out / "validation_recording_scores.csv", index=False)
    val_subjects.to_csv(out / "validation_subject_scores.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps({
        "recording_threshold": recording_threshold, "subject_threshold": subject_threshold,
        "shared_recording_subject_threshold": True,
        "threshold_method": cfg.threshold_method, "target_fpr": cfg.target_fpr,
        "subject_aggregation": reducer}, indent=2))
    return metrics


def save_evaluation(val_df, test_df, cfg, out_dir, component_cols=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    component_cols = component_cols or [c for c in ("event_nll", "duration_nll", "gap_nll") if c in test_df]
    recording_calibrator, legacy = _calibrate_level(val_df, test_df, component_cols, cfg)
    recording_metrics = _component_metrics(val_df, test_df, component_cols, cfg, "recording")
    recording_metrics.to_csv(out / "recording_component_metrics.csv", index=False)
    combined = recording_metrics.loc[recording_metrics.component == "combined"].drop(columns=["level", "component"])
    combined.to_csv(out / "recording_metrics.csv", index=False)
    threshold = float(combined.iloc[0].threshold)
    raw_thresholds = {c.removesuffix("_nll"): select_threshold(val_df[c], cfg) for c in component_cols}
    for component, component_threshold in raw_thresholds.items():
        test_df[f"{component}_anomaly"] = (test_df[f"{component}_nll"] >= component_threshold).astype(int)
    test_df.assign(prediction=(test_df.score >= threshold).astype(int)).to_csv(out / "recording_scores.csv", index=False)

    reducer = "max" if cfg.subject_aggregation == "max" else "mean"
    numeric = component_cols + ["score", "legacy_mean_score"] + [
        c for c in test_df if c.startswith("embedding_")
    ]
    val_subjects = val_df.groupby(["Patient", "Group"], as_index=False)[numeric].agg(reducer)
    test_subjects = test_df.groupby(["Patient", "Group"], as_index=False)[numeric].agg(reducer)
    fixed_thresholds = {**raw_thresholds, "combined": threshold}
    subject_metrics = _component_metrics(
        val_subjects, test_subjects, component_cols, cfg, "subject",
        fixed_thresholds=fixed_thresholds,
    )
    subject_metrics.to_csv(out / "subject_component_metrics.csv", index=False)
    subject_combined = subject_metrics.loc[subject_metrics.component == "combined"].drop(columns=["level", "component"])
    subject_combined.to_csv(out / "subject_metrics.csv", index=False)
    subject_threshold = threshold
    for component, component_threshold in raw_thresholds.items():
        test_subjects[f"{component}_anomaly"] = (
            test_subjects[f"{component}_nll"] >= component_threshold
        ).astype(int)
    test_subjects.assign(prediction=(test_subjects.score >= subject_threshold).astype(int)).to_csv(out / "subject_scores.csv", index=False)

    val_df.to_csv(out / "validation_recording_scores.csv", index=False)
    val_subjects.to_csv(out / "validation_subject_scores.csv", index=False)
    payload = {"score_method": "max_healthy_tail_surprise",
               "recording_threshold": threshold, "subject_threshold": subject_threshold,
               "shared_recording_subject_threshold": True,
               "component_raw_nll_thresholds": raw_thresholds,
               "subject_component_raw_nll_thresholds": raw_thresholds,
               "threshold_method": cfg.threshold_method, "target_fpr": cfg.target_fpr,
               "subject_aggregation": reducer,
               "recording_tail_references": recording_calibrator.to_dict(component_cols)}
    (out / "thresholds.json").write_text(json.dumps(payload, indent=2))
    legacy.save(out / "legacy_score_calibration.json")
    linear_probe_evaluation(test_df, cfg, out)
    if cfg.save_publication_figures:
        from visualization import save_sequence_figures
        save_sequence_figures(val_df, test_df, recording_metrics, cfg, out)
    from disease_analysis import run_disease_score_analysis
    run_disease_score_analysis(test_df, cfg, out)
    return combined.iloc[0].to_dict()
