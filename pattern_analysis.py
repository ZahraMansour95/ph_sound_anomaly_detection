import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from applications import get_preset


def _id(value):
    text = str(value).strip()
    if text.endswith(".0"): text = text[:-2]
    return text.lstrip("0") or "0"


def _disease(patient, group, cfg):
    if str(group).strip().lower() == "healthy": return "Healthy"
    target = _id(patient)
    for disease, identifiers in get_preset(cfg.application).get("disease_ids", {}).items():
        if target in {_id(value) for value in identifiers}: return disease
    return "Other_Patient"


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0,
                                                               class_weight="balanced"))


def _fold_count(y, groups, maximum, minimum=2):
    counts = pd.DataFrame({"y": y, "g": groups}).drop_duplicates().groupby("y").g.nunique()
    return min(maximum, int(counts.min())) if len(counts) >= 2 and counts.min() >= minimum else 0


def _group_oof(x, y, groups, folds, seed):
    classes = np.asarray(sorted(np.unique(y))); probability = np.zeros((len(y), len(classes)))
    prediction = np.empty(len(y), dtype=object)
    # Split by patient to avoid leakage.
    group_frame = pd.DataFrame({"group": groups, "label": y}).drop_duplicates()
    inconsistent = group_frame.groupby("group").label.nunique()
    if (inconsistent > 1).any():
        bad = inconsistent[inconsistent > 1].index.tolist()
        raise ValueError(f"Patients assigned to multiple disease labels: {bad[:10]}")
    group_frame = group_frame.drop_duplicates("group").reset_index(drop=True)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for train_groups, test_groups in cv.split(group_frame.group, group_frame.label):
        train_ids = set(group_frame.group.iloc[train_groups])
        test_ids = set(group_frame.group.iloc[test_groups])
        train = np.flatnonzero(np.isin(groups, list(train_ids)))
        test = np.flatnonzero(np.isin(groups, list(test_ids)))
        if np.unique(y[train]).size < 2:
            raise ValueError("Subject-stratified training fold contains fewer than two classes")
        fitted = clone(_model()).fit(x[train], y[train])
        local = fitted.predict_proba(x[test])
        for j, label in enumerate(fitted.classes_):
            probability[test, np.where(classes == label)[0][0]] = local[:, j]
        prediction[test] = fitted.predict(x[test])
    return classes, prediction, probability


def _auc_ci(y, score, iterations, seed):
    rng, values = np.random.default_rng(seed), []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2: values.append(roc_auc_score(y[idx], score[idx]))
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)


def _evaluate(x, y, groups, cfg, name, out, feature_set):
    folds = _fold_count(y, groups, cfg.disease_cv_folds)
    if folds < 2: return None, None, None
    classes, prediction, probability = _group_oof(x, y, groups, folds, cfg.seed)
    result = {"analysis": name, "feature_set": feature_set, "n_recordings": len(y),
              "n_subjects": len(np.unique(groups)), "folds": folds,
              "balanced_accuracy": balanced_accuracy_score(y, prediction),
              "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
              "mcc": matthews_corrcoef(y, prediction)}
    binary = label_binarize(y, classes=classes)
    if len(classes) == 2: binary = np.column_stack([1-binary[:, 0], binary[:, 0]])
    rows = []
    for index, label in enumerate(classes):
        auc = roc_auc_score(binary[:, index], probability[:, index])
        lo, hi = _auc_ci(binary[:, index], probability[:, index], cfg.bootstrap_iterations,
                         cfg.seed + index)
        rows.append({"analysis": name, "feature_set": feature_set, "class": label,
                     "auroc": auc, "auroc_ci_low": lo, "auroc_ci_high": hi})
    result["macro_ovr_auroc"] = float(np.mean([row["auroc"] for row in rows]))
    oof = pd.DataFrame({"Patient": groups, "label": y, "prediction": prediction})
    for index, label in enumerate(classes): oof[f"probability_{label}"] = probability[:, index]
    oof.to_csv(out / f"{name}_{feature_set}_oof_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{name}_{feature_set}_class_metrics.csv", index=False)
    return result, (classes, prediction, probability), pd.DataFrame(rows)


def run_pattern_analysis(validation_scores, test_scores, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    embedding_cols = [c for c in test_scores if c.startswith("embedding_")]
    numeric = ["score", "gnll", "standardized_residual", "predicted_log_variance"] + embedding_cols
    recordings = test_scores.groupby(["Patient", "Wav_path", "Group", "Pattern"], as_index=False)[numeric].mean()
    recordings["Disease"] = [_disease(p, g, cfg) for p, g in zip(recordings.Patient, recordings.Group)]
    recordings.to_csv(out / "pattern_recording_scores.csv", index=False)
    groups = recordings.Patient.astype(str).to_numpy()
    results = []

    health_y = np.where(recordings.Group.astype(str).str.lower().eq("healthy"), "Healthy", "Patient")
    binary_result, binary_oof, _ = _evaluate(recordings[["score"]].to_numpy(), health_y, groups, cfg,
                                              "healthy_vs_all", out, "score_only")
    if binary_result: results.append(binary_result)

    subject_counts = recordings[["Patient", "Disease"]].drop_duplicates().Disease.value_counts()
    eligible = subject_counts[subject_counts >= cfg.disease_min_class_size].index
    multi = recordings[recordings.Disease.isin(eligible)].reset_index(drop=True)
    multiclass_oof = multiclass_metrics = None
    if multi.Disease.nunique() >= 2:
        score_result, multiclass_oof, multiclass_metrics = _evaluate(
            multi[["score"]].to_numpy(), multi.Disease.to_numpy(), multi.Patient.astype(str).to_numpy(),
            cfg, "disease_multiclass", out, "score_only")
        if score_result: results.append(score_result)
        if embedding_cols:
            latent_result, _, _ = _evaluate(
                multi[["score"] + embedding_cols].to_numpy(), multi.Disease.to_numpy(),
                multi.Patient.astype(str).to_numpy(), cfg, "disease_multiclass", out, "latent_plus_score")
            if latent_result: results.append(latent_result)

    binary_rows = []
    healthy = recordings[recordings.Disease == "Healthy"]
    for disease in [name for name in subject_counts.index if name != "Healthy"]:
        disease_frame = recordings[recordings.Disease == disease]
        n_subjects = disease_frame.Patient.nunique()
        if n_subjects < cfg.disease_min_class_size or healthy.Patient.nunique() < cfg.disease_min_class_size:
            binary_rows.append({"Disease": disease, "status": "excluded_insufficient_subjects",
                                "n_disease_subjects": n_subjects})
            continue
        comparison = pd.concat([healthy, disease_frame], ignore_index=True)
        y = np.where(comparison.Disease.eq(disease), disease, "Healthy")
        result, oof, metrics = _evaluate(comparison[["score"]].to_numpy(), y,
                                         comparison.Patient.astype(str).to_numpy(), cfg,
                                         f"healthy_vs_{disease}", out, "score_only")
        if result:
            disease_row = metrics.loc[metrics["class"] == disease].iloc[0]
            binary_rows.append({"Disease": disease, "status": "completed",
                                "n_disease_subjects": n_subjects, **result,
                                "auroc": disease_row.auroc, "auroc_ci_low": disease_row.auroc_ci_low,
                                "auroc_ci_high": disease_row.auroc_ci_high})
    pd.DataFrame(binary_rows).to_csv(out / "pattern_healthy_vs_each_disease_metrics.csv", index=False)
    pd.DataFrame(results).to_csv(out / "pattern_classifier_summary.csv", index=False)
    _figures(recordings, validation_scores, binary_oof, multi, multiclass_oof,
             multiclass_metrics, pd.DataFrame(binary_rows), embedding_cols, cfg, out)
    return results


def _figures(recordings, validation, binary_oof, multi, multiclass_oof, multiclass_metrics,
             binary_metrics, embedding_cols, cfg, out):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6))
    for label, selected, color in (("Healthy", recordings.Group.str.lower().eq("healthy"), "#0072B2"),
                                   ("Patient", ~recordings.Group.str.lower().eq("healthy"), "#D55E00")):
        axes[0,0].hist(recordings.loc[selected,"score"], bins="auto", density=True, alpha=.5, color=color, label=label)
    axes[0,0].set(title="A  Pattern anomaly-score distribution", xlabel="GNLL anomaly score", ylabel="Density")
    axes[0,0].legend(frameon=False)
    if binary_oof:
        classes, prediction, probability = binary_oof; y=(recordings.Group.str.lower() != "healthy").astype(int).to_numpy()
        patient_index=np.where(classes=="Patient")[0][0]; p=probability[:,patient_index]
        fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
        axes[0,1].plot(fpr,tpr,label=f"ROC AUC={roc_auc_score(y,p):.2f}")
        axes[0,1].plot(recall,precision,label=f"PR AUC={average_precision_score(y,p):.2f}")
        axes[0,1].plot([0,1],[0,1],"--",color="0.6",lw=1)
        axes[0,1].legend(frameon=False)
    axes[0,1].set(title="B  Healthy versus all patients", xlabel="False-positive rate / Recall", ylabel="True-positive rate / Precision")
    if embedding_cols and len(recordings) >= 3:
        coords=PCA(2).fit_transform(StandardScaler().fit_transform(recordings[embedding_cols]))
        for i,disease in enumerate(recordings.Disease.unique()):
            selected=recordings.Disease.eq(disease).to_numpy(); axes[1,0].scatter(coords[selected,0],coords[selected,1],s=18,alpha=.7,label=disease.replace("_"," "))
        axes[1,0].legend(frameon=False)
    axes[1,0].set(title="C  Temporal-bottleneck latent PCA",xlabel="PC1",ylabel="PC2")
    order=recordings.groupby("Disease").score.median().sort_values().index
    axes[1,1].boxplot([recordings.loc[recordings.Disease==d,"score"] for d in order],vert=False,labels=[d.replace("_"," ") for d in order],showfliers=False)
    axes[1,1].set(title="D  Disease-specific anomaly scores",xlabel="GNLL anomaly score")
    fig.tight_layout(); fig.savefig(out/"figure_pattern_overview.pdf",bbox_inches="tight"); fig.savefig(out/"figure_pattern_overview.png",dpi=cfg.figure_dpi,bbox_inches="tight"); plt.close(fig)

    completed=binary_metrics[binary_metrics.status=="completed"] if not binary_metrics.empty else pd.DataFrame()
    if completed.empty: return
    completed=completed.sort_values("auroc"); pos=np.arange(len(completed))
    fig,axes=plt.subplots(1,2,figsize=(7.16,max(2.5,.4*len(completed)+1.2)))
    auc=completed.auroc.to_numpy(); lo=completed.auroc_ci_low.to_numpy(); hi=completed.auroc_ci_high.to_numpy()
    labels=[x.replace("_"," ") for x in completed.Disease]
    axes[0].errorbar(auc,pos,xerr=np.vstack([auc-lo,hi-auc]),fmt="o",capsize=3,color="#0072B2"); axes[0].axvline(.5,ls="--",color=".6")
    axes[0].set(title="A  Healthy versus each disease",xlabel="AUROC (95% CI)",xlim=(0,1.03),yticks=pos,yticklabels=labels)
    axes[1].plot(completed.balanced_accuracy,pos,"o",color="#D55E00"); axes[1].axvline(.5,ls="--",color=".6")
    axes[1].set(title="B  Grouped-CV discrimination",xlabel="Balanced accuracy",xlim=(0,1.03),yticks=pos,yticklabels=labels)
    fig.tight_layout(); fig.savefig(out/"figure_pattern_disease_classification.pdf",bbox_inches="tight"); fig.savefig(out/"figure_pattern_disease_classification.png",dpi=cfg.figure_dpi,bbox_inches="tight"); plt.close(fig)
