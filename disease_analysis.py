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
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from applications import get_preset


def _patient_id(value):
    text = str(value).strip()
    if text.endswith(".0"): text = text[:-2]
    return text.lstrip("0") or "0"


def _disease(patient, group, cfg):
    if str(group).strip().lower() == "healthy": return "Healthy"
    target = _patient_id(patient)
    for name, identifiers in get_preset(cfg.application).get("disease_ids", {}).items():
        if target in {_patient_id(value) for value in identifiers}:
            return name
    return "Other_Patient"


def _estimator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
    )


def _oof_probabilities(x, y, classes, folds, seed):
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    probabilities = np.zeros((len(y), len(classes)), float)
    predictions = np.empty(len(y), dtype=object)
    for train, test in splitter.split(x, y):
        model = clone(_estimator()).fit(x[train], y[train])
        fold_probability = model.predict_proba(x[test])
        for local, label in enumerate(model.classes_):
            probabilities[test, np.where(classes == label)[0][0]] = fold_probability[:, local]
        predictions[test] = model.predict(x[test])
    return predictions, probabilities


def _bootstrap_auc(y_binary, probability, iterations, seed):
    rng, values = np.random.default_rng(seed), []
    for _ in range(iterations):
        idx = rng.integers(0, len(y_binary), len(y_binary))
        if np.unique(y_binary[idx]).size == 2:
            values.append(roc_auc_score(y_binary[idx], probability[idx]))
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)


def _ellipse(ax, points, color):
    if len(points) < 3: return
    covariance = np.cov(points.T)
    if not np.isfinite(covariance).all(): return
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    order = values.argsort()[::-1]; values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(points.mean(0), 2 * np.sqrt(values[0]), 2 * np.sqrt(values[1]),
                      angle=angle, facecolor="none", edgecolor=color, lw=1, alpha=.75)
    ax.add_patch(ellipse)


def _permutation_test(x, y, observed, folds, iterations, seed):
    rng, null = np.random.default_rng(seed), []
    classes = np.unique(y)
    for iteration in range(iterations):
        shuffled = rng.permutation(y)
        pred, _ = _oof_probabilities(x, shuffled, classes, folds, seed + iteration + 1)
        null.append(balanced_accuracy_score(shuffled, pred))
    null = np.asarray(null)
    return null, float((1 + np.sum(null >= observed)) / (len(null) + 1))


def run_disease_score_analysis(test_df, cfg, out_dir):
    if not cfg.disease_analysis_enabled: return None
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    score_cols = [c for c in ("event_tail_surprise", "duration_tail_surprise", "gap_tail_surprise")
                  if c in test_df]
    if len(score_cols) < 2: return None
    subject = test_df.groupby(["Patient", "Group"], as_index=False)[score_cols + ["score"]].mean()
    subject["Disease"] = [_disease(p, g, cfg) for p, g in zip(subject.Patient, subject.Group)]
    counts = subject.Disease.value_counts()
    eligible_names = counts[counts >= cfg.disease_min_class_size].index
    subject["quantitative_eligible"] = subject.Disease.isin(eligible_names)
    subject.to_csv(out / "disease_score_subjects.csv", index=False)
    pd.DataFrame({"Disease": counts.index, "n_subjects": counts.values,
                  "quantitative_eligible": counts.index.isin(eligible_names)}).to_csv(
        out / "disease_class_counts.csv", index=False)

    eligible = subject[subject.quantitative_eligible].reset_index(drop=True)
    classes = np.asarray(sorted(eligible.Disease.unique()))
    quantitative = len(classes) >= 2
    class_metrics = pd.DataFrame()
    matrix = None; predictions = probabilities = None
    summary = {"status": "descriptive_only", "features": score_cols,
               "minimum_class_size": cfg.disease_min_class_size,
               "excluded_groups": [name for name in counts.index if name not in eligible_names]}
    if quantitative:
        y = eligible.Disease.to_numpy(); x = eligible[score_cols].to_numpy(float)
        folds = min(cfg.disease_cv_folds, int(eligible.Disease.value_counts().min()))
        predictions, probabilities = _oof_probabilities(x, y, classes, folds, cfg.seed)
        balanced = balanced_accuracy_score(y, predictions)
        macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y, predictions)
        binary = label_binarize(y, classes=classes)
        if len(classes) == 2: binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
        rows = []
        for index, label in enumerate(classes):
            auc = roc_auc_score(binary[:, index], probabilities[:, index])
            lo, hi = _bootstrap_auc(binary[:, index], probabilities[:, index],
                                    cfg.bootstrap_iterations, cfg.seed + index)
            rows.append({"Disease": label, "n_subjects": int((y == label).sum()),
                         "ovr_auroc": auc, "auroc_ci_low": lo, "auroc_ci_high": hi})
        class_metrics = pd.DataFrame(rows)
        class_metrics.to_csv(out / "disease_classifier_class_metrics.csv", index=False)
        macro_auc = float(class_metrics.ovr_auroc.mean())
        null, permutation_p = _permutation_test(x, y, balanced, folds,
                                                 cfg.disease_permutations, cfg.seed + 1000)
        pd.DataFrame({"permuted_balanced_accuracy": null}).to_csv(
            out / "disease_classifier_permutation_null.csv", index=False)
        summary.update({"status": "completed", "n_subjects": len(eligible), "classes": classes.tolist(),
                        "folds": folds, "balanced_accuracy": balanced, "macro_f1": macro_f1,
                        "multiclass_mcc": mcc, "macro_ovr_auroc": macro_auc,
                        "permutation_iterations": cfg.disease_permutations,
                        "permutation_p_value": permutation_p})
        pd.DataFrame([{key: value for key, value in summary.items()
                       if np.isscalar(value) and key not in {"status"}}]).to_csv(
            out / "disease_classifier_overall_metrics.csv", index=False)
        matrix = confusion_matrix(y, predictions, labels=classes, normalize="true")
        oof = eligible[["Patient", "Group", "Disease"] + score_cols].copy()
        oof["prediction"] = predictions
        for index, label in enumerate(classes): oof[f"probability_{label}"] = probabilities[:, index]
        oof.to_csv(out / "disease_classifier_oof_predictions.csv", index=False)
    (out / "disease_classifier_summary.json").write_text(json.dumps(summary, indent=2))
    _save_figure(subject, score_cols, eligible, classes, matrix, class_metrics, summary, cfg, out)
    _run_healthy_vs_disease(subject, score_cols, cfg, out)
    return summary


def _run_healthy_vs_disease(subject, score_cols, cfg, out):
    """Compare healthy subjects with each disease group."""
    healthy = subject[subject.Disease == "Healthy"]
    disease_counts = subject.loc[subject.Disease != "Healthy", "Disease"].value_counts()
    rows, prediction_frames = [], []
    for comparison_index, (disease, n_disease) in enumerate(disease_counts.items()):
        if n_disease < cfg.disease_min_class_size or len(healthy) < cfg.disease_min_class_size:
            rows.append({"Disease": disease, "status": "excluded_insufficient_subjects",
                         "n_healthy": len(healthy), "n_disease": int(n_disease)})
            continue
        comparison = pd.concat([healthy, subject[subject.Disease == disease]], ignore_index=True)
        x = comparison[score_cols].to_numpy(float)
        y = comparison.Disease.to_numpy()
        classes = np.asarray(["Healthy", disease], dtype=object)
        folds = min(cfg.disease_cv_folds, len(healthy), int(n_disease))
        predictions, probabilities = _oof_probabilities(
            x, y, classes, folds, cfg.seed + comparison_index
        )
        disease_probability = probabilities[:, 1]
        y_binary = (y == disease).astype(int)
        pred_binary = (predictions == disease).astype(int)
        auc = roc_auc_score(y_binary, disease_probability)
        lo, hi = _bootstrap_auc(y_binary, disease_probability, cfg.bootstrap_iterations,
                                cfg.seed + 2000 + comparison_index)
        balanced = balanced_accuracy_score(y_binary, pred_binary)
        macro_f1 = f1_score(y_binary, pred_binary, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_binary, pred_binary)
        null, permutation_p = _permutation_test(
            x, y, balanced, folds, cfg.disease_permutations,
            cfg.seed + 3000 + comparison_index,
        )
        pd.DataFrame({"permuted_balanced_accuracy": null}).to_csv(
            out / f"healthy_vs_{disease}_permutation_null.csv", index=False
        )
        rows.append({"Disease": disease, "status": "completed", "n_healthy": len(healthy),
                     "n_disease": int(n_disease), "folds": folds, "auroc": auc,
                     "auroc_ci_low": lo, "auroc_ci_high": hi,
                     "balanced_accuracy": balanced, "macro_f1": macro_f1, "mcc": mcc,
                     "permutation_iterations": cfg.disease_permutations,
                     "permutation_p_value": permutation_p})
        prediction = comparison[["Patient", "Group", "Disease"] + score_cols].copy()
        prediction["Comparison"] = f"Healthy_vs_{disease}"
        prediction["binary_label"] = y_binary
        prediction["prediction"] = predictions
        prediction["disease_probability"] = disease_probability
        prediction_frames.append(prediction)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "healthy_vs_each_disease_metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            out / "healthy_vs_each_disease_oof_predictions.csv", index=False
        )
    _save_binary_figure(metrics, cfg, out)


def _save_binary_figure(metrics, cfg, out):
    completed = metrics[metrics.status == "completed"].copy()
    if completed.empty: return
    completed = completed.sort_values("auroc").reset_index(drop=True)
    labels = [value.replace("_", " ") for value in completed.Disease]
    positions = np.arange(len(completed))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, max(2.5, .42 * len(completed) + 1.25)))
    auc = completed.auroc.to_numpy(); low = completed.auroc_ci_low.to_numpy(); high = completed.auroc_ci_high.to_numpy()
    axes[0].errorbar(auc, positions, xerr=np.vstack([auc-low, high-auc]), fmt="o",
                     color="#0072B2", ecolor="0.3", capsize=3)
    axes[0].axvline(.5, color="0.6", ls="--", lw=1)
    axes[0].set(title="A  Healthy versus each disease", xlabel="AUROC (95% bootstrap CI)",
                xlim=(0, 1.03), yticks=positions, yticklabels=labels)
    balanced = completed.balanced_accuracy.to_numpy()
    axes[1].plot(balanced, positions, "o", color="#D55E00")
    axes[1].axvline(.5, color="0.6", ls="--", lw=1)
    axes[1].set(title="B  Out-of-fold balanced accuracy", xlabel="Balanced accuracy",
                xlim=(0, 1.03), yticks=positions, yticklabels=labels)
    for index, row in completed.iterrows():
        axes[1].annotate(f"p={row.permutation_p_value:.3f}",
                         (row.balanced_accuracy, index), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "figure_healthy_vs_each_disease.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_healthy_vs_each_disease.png", dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _save_figure(subject, score_cols, eligible, classes, matrix, class_metrics, summary, cfg, out):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6))
    x = StandardScaler().fit_transform(subject[score_cols]); coords = PCA(n_components=2).fit_transform(x)
    labels = list(subject.Disease.unique()); cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    for index, label in enumerate(labels):
        selected = subject.Disease.eq(label).to_numpy(); color = cmap(index % cmap.N)
        axes[0, 0].scatter(coords[selected, 0], coords[selected, 1], s=22, alpha=.75,
                           color=color, edgecolors="none", label=f"{label.replace('_', ' ')} (n={selected.sum()})")
        center = coords[selected].mean(0)
        axes[0, 0].scatter(*center, marker="X", s=80, color=color, edgecolors="black", linewidths=.6)
        _ellipse(axes[0, 0], coords[selected], color)
    axes[0, 0].set(title="A  PCA of anomaly-score components", xlabel="PC1", ylabel="PC2")
    axes[0, 0].legend(frameon=False, loc="best")

    profile = subject.groupby("Disease")[score_cols].median().reindex(labels)
    image = axes[0, 1].imshow(profile.to_numpy(), aspect="auto", cmap="viridis")
    axes[0, 1].set(title="B  Median disease anomaly profiles",
                   xticks=np.arange(len(score_cols)), xticklabels=[c.split("_")[0].capitalize() for c in score_cols],
                   yticks=np.arange(len(profile)), yticklabels=[f"{x.replace('_', ' ')} (n={(subject.Disease == x).sum()})" for x in profile.index])
    for i in range(len(profile)):
        for j in range(len(score_cols)):
            axes[0, 1].text(j, i, f"{profile.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if profile.iloc[i, j] > np.nanmedian(profile.to_numpy()) else "black")
    fig.colorbar(image, ax=axes[0, 1], fraction=.046, pad=.04, label="Tail surprise")

    if matrix is not None:
        image = axes[1, 0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axes[1, 0].set(title="C  Out-of-fold confusion matrix", xlabel="Predicted", ylabel="Observed",
                       xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
                       xticklabels=[x.replace("_", " ") for x in classes],
                       yticklabels=[x.replace("_", " ") for x in classes])
        axes[1, 0].tick_params(axis="x", rotation=35)
        for i in range(len(classes)):
            for j in range(len(classes)):
                axes[1, 0].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if matrix[i,j] > .5 else "black")
        fig.colorbar(image, ax=axes[1, 0], fraction=.046, pad=.04)
        values = class_metrics.ovr_auroc.to_numpy(); low = class_metrics.auroc_ci_low.to_numpy(); high = class_metrics.auroc_ci_high.to_numpy()
        pos = np.arange(len(values))
        axes[1, 1].errorbar(values, pos, xerr=np.vstack([values-low, high-values]), fmt="o", color="black", capsize=3)
        axes[1, 1].axvline(.5, color="0.6", ls="--", lw=1)
        axes[1, 1].set(title=(f"D  Disease discrimination (macro AUC={summary['macro_ovr_auroc']:.2f}, "
                              f"permutation p={summary['permutation_p_value']:.3f})"),
                       xlabel="One-vs-rest AUROC (95% CI)", xlim=(0, 1.03), yticks=pos,
                       yticklabels=[x.replace("_", " ") for x in class_metrics.Disease])
    else:
        for ax, title in ((axes[1, 0], "C  Confusion matrix"), (axes[1, 1], "D  Disease discrimination")):
            ax.text(.5, .5, "Insufficient disease-group sample size", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title); ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "figure_disease_score_discrimination.pdf", bbox_inches="tight")
    fig.savefig(out / "figure_disease_score_discrimination.png", dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)
