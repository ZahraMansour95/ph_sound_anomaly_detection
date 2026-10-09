from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler


COLORS = {"event": "#0072B2", "duration": "#E69F00", "gap": "#009E73",
          "combined": "#CC79A7", "Healthy": "#0072B2", "Patient": "#D55E00"}


def _style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42})


def _save(fig, out, stem, dpi):
    fig.tight_layout()
    fig.savefig(Path(out) / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(Path(out) / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _ecdf(values):
    x = np.sort(np.asarray(values, float)); return x, np.arange(1, len(x) + 1) / max(len(x), 1)


def _normalise_patient_id(value):
    text = str(value).strip()
    if text.endswith(".0"): text = text[:-2]
    return text.lstrip("0") or "0"


def _disease_label(patient, group, cfg):
    if str(group).strip().lower() == "healthy": return "Healthy"
    from applications import get_preset
    target = _normalise_patient_id(patient)
    for disease, identifiers in get_preset(cfg.application).get("disease_ids", {}).items():
        if target in {_normalise_patient_id(value) for value in identifiers}:
            return disease
    return "Other_Patient"


def _disease_score_space_figure(test_df, components, cfg, out):
    """Plot disease groups using anomaly-score components."""
    score_cols = [f"{component}_tail_surprise" for component in components]
    subject = test_df.groupby(["Patient", "Group"], as_index=False)[score_cols + ["score"]].mean()
    if len(subject) < 3: return
    subject["Disease"] = [_disease_label(p, g, cfg) for p, g in zip(subject.Patient, subject.Group)]
    features = StandardScaler().fit_transform(subject[score_cols])
    method = "PCA"
    try:
        import umap
        if len(subject) >= 4:
            coords = umap.UMAP(n_components=2, random_state=cfg.seed,
                               n_neighbors=max(2, min(15, len(subject)-1))).fit_transform(features)
            method = "UMAP"
        else:
            coords = PCA(n_components=2).fit_transform(features)
    except (ImportError, ValueError):
        coords = PCA(n_components=2).fit_transform(features)
    subject["x"], subject["y"] = coords[:, 0], coords[:, 1]
    subject.to_csv(out / "disease_anomaly_score_space_coordinates.csv", index=False)
    labels = list(dict.fromkeys(subject.Disease.tolist()))
    cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    scores = subject.score.to_numpy(float)
    span = np.ptp(scores)
    sizes = 22 + 70 * ((scores - scores.min()) / span if span > 0 else np.zeros_like(scores))
    centers = []
    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    for index, label in enumerate(labels):
        selected = subject.Disease.eq(label).to_numpy(); color = cmap(index % cmap.N)
        ax.scatter(coords[selected, 0], coords[selected, 1], s=sizes[selected], alpha=.72,
                   color=color, edgecolors="white", linewidths=.35,
                   label=f"{label.replace('_', ' ')} (n={selected.sum()})")
        center = coords[selected].mean(axis=0)
        centers.append({"Disease": label, "n_subjects": int(selected.sum()),
                        "x_center": center[0], "y_center": center[1],
                        "mean_anomaly_score": scores[selected].mean()})
        ax.scatter(center[0], center[1], marker="X", s=95, color=color,
                   edgecolors="black", linewidths=.7, zorder=5)
    pd.DataFrame(centers).to_csv(out / "disease_anomaly_score_space_centers.csv", index=False)
    ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2",
           title="Disease groups in anomaly-score space\n(marker size = combined anomaly score)")
    ax.legend(frameon=False, fontsize=6, loc="best")
    _save(fig, out, "figure_disease_anomaly_score_space", cfg.figure_dpi)


def save_sequence_figures(val_df, test_df, metrics, cfg, out_dir):
    _style(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    components = [c.removesuffix("_nll") for c in ("event_nll", "duration_nll", "gap_nll") if c in test_df]
    y = (test_df.Group.str.lower() != "healthy").astype(int).to_numpy()

    fig, axes = plt.subplots(1, len(components), figsize=(2.35 * len(components), 2.15), squeeze=False)
    for ax, component in zip(axes[0], components):
        col = f"{component}_nll"
        for label, mask in (("Healthy", y == 0), ("Patient", y == 1)):
            x, cdf = _ecdf(test_df.loc[mask, col]); ax.plot(x, cdf, lw=1.5, color=COLORS[label], label=label)
        threshold = np.quantile(val_df[col].dropna(), 1 - cfg.target_fpr)
        ax.axvline(threshold, color="0.25", ls="--", lw=1, label="Threshold")
        ax.set(title=component.capitalize(), xlabel="Mean observed-category NLL", ylabel="Cumulative probability")
    axes[0][0].legend(frameon=False)
    _save(fig, out, "figure_component_distributions", cfg.figure_dpi)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))
    pairs = [(c, f"{c}_nll") for c in components] + [("combined", "score")]
    for name, col in pairs:
        values = test_df[col].to_numpy(float); finite = np.isfinite(values)
        if finite.sum() == 0 or np.unique(y[finite]).size < 2: continue
        fpr, tpr, _ = roc_curve(y[finite], values[finite]); axes[0].plot(fpr, tpr, lw=1.4, color=COLORS[name], label=name.capitalize())
        precision, recall, _ = precision_recall_curve(y[finite], values[finite]); axes[1].plot(recall, precision, lw=1.4, color=COLORS[name], label=name.capitalize())
    axes[0].plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for ax in axes: ax.legend(frameon=False)
    _save(fig, out, "figure_roc_pr_components", cfg.figure_dpi)

    order = metrics.component.tolist(); x = np.arange(len(order))
    values = metrics.auroc.to_numpy(); low = metrics.auroc_ci_low.to_numpy(); high = metrics.auroc_ci_high.to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    ax.errorbar(x, values, yerr=np.vstack([values-low, high-values]), fmt="o", capsize=3,
                color="black", ecolor="0.35")
    for i, name in enumerate(order): ax.plot(i, values[i], "o", color=COLORS.get(name, "black"))
    ax.axhline(.5, color="0.6", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=[v.capitalize() for v in order], ylabel="AUROC (95% bootstrap CI)", ylim=(0, 1.03))
    _save(fig, out, "figure_component_auroc", cfg.figure_dpi)

    embedding_cols = [c for c in test_df if c.startswith("embedding_")]
    if embedding_cols:
        subject = test_df.groupby(["Patient", "Group"], as_index=False)[embedding_cols].mean()
        x = StandardScaler().fit_transform(subject[embedding_cols])
        method = "PCA"
        try:
            import umap
            coords = umap.UMAP(n_components=2, random_state=cfg.seed, n_neighbors=min(15, len(x)-1)).fit_transform(x)
            method = "UMAP"
        except (ImportError, ValueError):
            coords = PCA(n_components=2).fit_transform(x)
        np.savetxt(out / "embedding_coordinates.csv", np.column_stack([subject.Patient, subject.Group, coords]),
                   delimiter=",", fmt="%s", header="Patient,Group,x,y", comments="")
        fig, ax = plt.subplots(figsize=(3.5, 2.75))
        healthy = subject.Group.str.lower().eq("healthy").to_numpy()
        for label, mask in (("Healthy", healthy), ("Patient", ~healthy)):
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=.75, edgecolors="none",
                       color=COLORS[label], label=label)
        ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2", title=f"{method} (descriptive only)")
        ax.legend(frameon=False)
        _save(fig, out, "figure_embedding_descriptive", cfg.figure_dpi)

    _disease_score_space_figure(test_df, components, cfg, out)
