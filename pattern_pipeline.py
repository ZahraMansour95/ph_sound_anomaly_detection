from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation import RobustCalibrator, save_scalar_evaluation
from losses import gaussian_nll
from models import PatternGNLLAE


def load_audio(path, start, end, target_sr):
    try:
        import torchaudio
        info = torchaudio.info(path); offset = max(0, int(start * info.sample_rate)); frames = max(1, int((end - start) * info.sample_rate))
        wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=frames)
        wav = wav.mean(0)
        if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.float()
    except Exception:
        import librosa
        wav, _ = librosa.load(path, sr=target_sr, mono=True, offset=float(start), duration=max(float(end-start), 1/target_sr))
        return torch.as_tensor(wav, dtype=torch.float32)


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg; self.backend = cfg.pattern_feature_backend.lower(); self.processor = self.model = None
        if self.backend in {"wav2vec2", "ast"}:
            from transformers import AutoFeatureExtractor, AutoModel
            name = "facebook/wav2vec2-base-960h" if self.backend == "wav2vec2" else "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.processor = AutoFeatureExtractor.from_pretrained(name); self.model = AutoModel.from_pretrained(name).eval()
            for parameter in self.model.parameters(): parameter.requires_grad_(False)
    def __call__(self, wav):
        target = int(self.cfg.pattern_fixed_sec * self.cfg.pattern_sr)
        wav = torch.nn.functional.pad(wav[:target], (0, max(0, target - len(wav))))
        if self.backend == "raw":
            frame = 256; usable = max(frame, (len(wav) // frame) * frame)
            wav = torch.nn.functional.pad(wav, (0, max(0, usable-len(wav))))[:usable]
            return wav.view(-1, frame)
        if self.backend == "mfcc":
            import torchaudio
            return torchaudio.transforms.MFCC(self.cfg.pattern_sr, n_mfcc=self.cfg.pattern_n_mfcc)(wav).transpose(0, 1)
        values = self.processor(wav.numpy(), sampling_rate=self.cfg.pattern_sr, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**values).last_hidden_state.squeeze(0)
        return output.float()


class PatternDataset(Dataset):
    def __init__(self, frame, cfg, extractor, aliases, target_pattern):
        target_pattern = str(target_pattern).strip().upper()
        aliases = {
            str(key).strip().upper(): str(value).strip().upper()
            for key, value in aliases.items()
        }
        items = []
        for row in frame.itertuples(index=False):
            raw_label = str(row.Label).strip().upper()
            label = aliases.get(raw_label, raw_label)
            if label == target_pattern:
                items.append((row, label))
        self.items, self.cfg, self.extractor = items, cfg, extractor
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        row, pattern = self.items[index]
        wav = load_audio(row.Wav_path, row.Start, row.End, self.cfg.pattern_sr)
        return {"x": self.extractor(wav), "patient": row.Patient, "wav": row.Wav_path,
                "group": row.Group, "pattern": pattern, "start": float(row.Start)}


def collate_pattern(batch):
    if not batch: return None
    length = max(x["x"].size(0) for x in batch); dim = batch[0]["x"].size(1)
    values = torch.zeros(len(batch), length, dim); mask = torch.zeros(len(batch), length)
    for i, item in enumerate(batch):
        n = item["x"].size(0); values[i, :n] = item["x"]; mask[i, :n] = 1
    return {"x": values, "mask": mask, **{k: [x[k] for x in batch] for k in ["patient", "wav", "group", "pattern", "start"]}}


def _device(cfg): return torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")


def _split_segments(df, train_records, val_records, test_records):
    def keys(records): return set(zip(records.Patient, records.Wav_path))
    def take(records):
        wanted = keys(records); return df[[pair in wanted for pair in zip(df.Patient, df.Wav_path)]].reset_index(drop=True)
    return take(train_records), take(val_records), take(test_records)


def _filter_missing_audio(frame, split_name):
    """Remove rows with missing audio files."""
    frame = frame.copy()
    if "Wav_path" not in frame.columns:
        raise KeyError("Pattern mode requires a Wav_path column")

    paths = frame["Wav_path"].astype(str).str.strip()
    unique_paths = paths.drop_duplicates()
    exists = {path: Path(path).is_file() for path in unique_paths}
    keep = paths.map(exists).fillna(False)

    missing_rows = int((~keep).sum())
    missing_files = int(paths[~keep].nunique())
    if missing_files:
        print(
            f"[PATTERN][{split_name.upper()}][AUDIO] "
            f"skipping {missing_rows} rows from {missing_files} missing WAV files"
        )
        for path in paths[~keep].drop_duplicates().head(10):
            print(f"  missing: {path}")

    return frame.loc[keep].reset_index(drop=True), missing_rows, missing_files


def _banner(cfg, architecture, pattern=None):
    print("\n" + "=" * 68)
    print(f"Application : {cfg.application.upper()}\nMode        : PATTERN\nTask        : HETEROSCEDASTIC RECONSTRUCTION")
    if pattern is not None:
        print(f"Pattern     : {str(pattern).upper()} (independent model)")
    print(f"Model       : {architecture.upper()}\nBottleneck  : temporal compression (4x)")
    print("Decoder     : mirrored ConvTranspose1d upsampling with Conv1d refinement")
    print("Loss/score  : masked Gaussian NLL (stabilised mean/variance initialisation)")
    print("=" * 68)


def train_pattern(model, train_loader, val_loader, cfg, out, pattern):
    device = _device(cfg)
    model.to(device)
    warmup_epochs = min(
        max(int(getattr(cfg, "gnll_warmup_epochs", 0)), 0),
        max(int(cfg.pattern_epochs) - 1, 0),
    )
    model.set_variance_trainable(warmup_epochs == 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), cfg.lr, weight_decay=cfg.weight_decay
    )
    best, bad, checkpoint = float("inf"), 0, Path(out) / "best_model.pt"
    for epoch in range(1, cfg.pattern_epochs + 1):
        if epoch == warmup_epochs + 1 and warmup_epochs > 0:
            model.set_variance_trainable(True)
            # Restart checkpoint selection after warm-up.
            best, bad = float("inf"), 0
        model.train(); train_values = []
        for batch in train_loader:
            x, mask = batch["x"].to(device), batch["mask"].to(device); optimizer.zero_grad()
            loss, _, _, _ = gaussian_nll(model(x, mask), x, mask, cfg.variance_regularization)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step(); train_values.append(loss.item())
        model.eval(); validation = []
        with torch.no_grad():
            for batch in val_loader:
                x, mask = batch["x"].to(device), batch["mask"].to(device)
                validation.append(gaussian_nll(model(x, mask), x, mask)[0].item())
        val = float(np.mean(validation))
        if val < best: best, bad = val, 0; torch.save(model.state_dict(), checkpoint)
        else: bad += 1
        phase = "mean warm-up" if epoch <= warmup_epochs else "joint GNLL"
        print(
            f"[{cfg.application.upper()}][PATTERN][{str(pattern).upper()}]"
            f"[{model.architecture.upper()}] Epoch {epoch:03d}/{cfg.pattern_epochs} "
            f"| phase={phase} train={np.mean(train_values):.4f} val={val:.4f}"
        )
        if bad >= cfg.pattern_patience: break
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.set_variance_trainable(True)
    return device


@torch.no_grad()
def pattern_scores(model, loader, device):
    model.eval(); rows = []
    for batch in loader:
        x, mask = batch["x"].to(device), batch["mask"].to(device)
        output = model(x, mask)
        _, nll, residual, uncertainty = gaussian_nll(output, x, mask)
        denom = mask.sum(1).clamp_min(1)
        values = torch.stack([(nll.sum(1)/denom), (residual.sum(1)/denom), (uncertainty.sum(1)/denom)], 1).cpu().numpy()
        for i, v in enumerate(values):
            embedding = output.embedding[i].mean(0).cpu().numpy()
            row = {"Patient": batch["patient"][i], "Wav_path": batch["wav"][i], "Group": batch["group"][i],
                         "Pattern": batch["pattern"][i], "Start": batch["start"][i], "gnll": v[0],
                         "standardized_residual": v[1], "predicted_log_variance": v[2]}
            row.update({f"embedding_{j:03d}": float(value) for j, value in enumerate(embedding)})
            rows.append(row)
    return pd.DataFrame(rows)


def run_pattern_experiment(df, splits, preset, architecture, cfg, out_dir):
    """Train one healthy-only GNLL model for each pattern."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "validation", "test")
    raw_frames = _split_segments(df, *splits)
    frames = []
    audio_report = []
    for split_name, frame in zip(split_names, raw_frames):
        filtered, missing_rows, missing_files = _filter_missing_audio(frame, split_name)
        frames.append(filtered)
        audio_report.append({
            "split": split_name,
            "rows_before": len(frame),
            "rows_after": len(filtered),
            "missing_rows": missing_rows,
            "missing_files": missing_files,
        })
    pd.DataFrame(audio_report).to_csv(out_root / "missing_audio_summary.csv", index=False)

    aliases = preset["label_aliases"]
    target_patterns = [
        str(pattern).strip().upper()
        for pattern in preset["target_patterns"]
    ]
    requested = [str(pattern).strip().upper() for pattern in (cfg.pattern_labels or ["all"])]
    if "ALL" not in requested:
        unknown = sorted(set(requested).difference(target_patterns))
        if unknown:
            raise ValueError(f"Unknown pattern(s) for {cfg.application}: {unknown}; choose from {target_patterns}")
        target_patterns = [pattern for pattern in target_patterns if pattern in requested]
    extractor = FeatureExtractor(cfg)
    summary = []

    # Keep training and validation healthy-only.
    model_frames = []
    for split_index, frame in enumerate(frames):
        if split_index < 2:
            healthy = (
                frame["Group"].astype(str).str.strip().str.lower() == "healthy"
            )
            excluded = int((~healthy).sum())
            if excluded:
                print(
                    f"[PATTERN][{split_names[split_index].upper()}] "
                    f"excluding {excluded} non-healthy rows"
                )
            model_frames.append(frame.loc[healthy].reset_index(drop=True))
        else:
            model_frames.append(frame)

    for pattern in target_patterns:
        _banner(cfg, architecture, pattern)
        pattern_out = out_root / pattern.lower()
        pattern_out.mkdir(parents=True, exist_ok=True)

        datasets = [
            PatternDataset(frame, cfg, extractor, aliases, pattern)
            for frame in model_frames
        ]
        n_train, n_val, n_test = [len(dataset) for dataset in datasets]
        test_groups = [
            str(row.Group).strip().lower()
            for row, _ in datasets[2].items
        ]
        n_test_healthy = sum(group == "healthy" for group in test_groups)
        n_test_patient = sum(group != "healthy" for group in test_groups)
        print(
            f"[{cfg.application.upper()}][PATTERN][{pattern}][FILTER] "
            f"train={n_train} validation={n_val} test={n_test} "
            f"(healthy={n_test_healthy}, patient={n_test_patient})"
        )

        status = {
            "application": cfg.application,
            "architecture": architecture,
            "pattern": pattern,
            "n_train": n_train,
            "n_validation": n_val,
            "n_test": n_test,
            "n_test_healthy": n_test_healthy,
            "n_test_patient": n_test_patient,
        }

        minimum_train = int(getattr(cfg, "pattern_min_train_samples", 1))
        minimum_val = int(getattr(cfg, "pattern_min_validation_samples", 1))
        minimum_test = int(getattr(cfg, "pattern_min_test_samples", 1))
        reasons = []
        if n_train < minimum_train:
            reasons.append(f"train {n_train} < {minimum_train}")
        if n_val < minimum_val:
            reasons.append(f"validation {n_val} < {minimum_val}")
        if n_test < minimum_test:
            reasons.append(f"test {n_test} < {minimum_test}")
        if n_test_healthy == 0:
            reasons.append("no healthy test samples")
        if n_test_patient == 0:
            reasons.append("no patient test samples")

        if reasons:
            status["status"] = "skipped"
            status["reason"] = "; ".join(reasons)
            summary.append(status)
            print(
                f"[{cfg.application.upper()}][PATTERN][{pattern}][SKIPPED] "
                f"{status['reason']}"
            )
            continue

        loaders = [
            DataLoader(
                dataset,
                batch_size=cfg.pattern_batch_size,
                shuffle=(index == 0),
                num_workers=cfg.num_workers,
                collate_fn=collate_pattern,
            )
            for index, dataset in enumerate(datasets)
        ]

        sample = datasets[0][0]["x"]
        model = PatternGNLLAE(sample.size(1), cfg, architecture)
        device = train_pattern(
            model, loaders[0], loaders[1], cfg, pattern_out, pattern
        )

        validation_scores = pattern_scores(model, loaders[1], device)
        test_scores = pattern_scores(model, loaders[2], device)
        calibrator = RobustCalibrator.fit(
            validation_scores[["gnll"]].to_numpy(),
            cfg.calibration_eps,
            cfg.calibration_clip_min,
        )
        calibrator.save(pattern_out / "score_calibration.json")
        for score_frame in (validation_scores, test_scores):
            score_frame["score"] = calibrator.transform(
                score_frame[["gnll"]].to_numpy()
            )[1]

        validation_scores.to_csv(
            pattern_out / "validation_pattern_scores.csv", index=False
        )
        test_scores.to_csv(pattern_out / "test_pattern_scores.csv", index=False)
        result = save_scalar_evaluation(
            validation_scores, test_scores, cfg, pattern_out
        )
        from pattern_analysis import run_pattern_analysis
        run_pattern_analysis(validation_scores, test_scores, cfg, pattern_out)

        status["status"] = "completed"
        status["reason"] = ""
        if isinstance(result, dict):
            for key, value in result.items():
                if np.isscalar(value) or value is None:
                    status[key] = value
        summary.append(status)

    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(out_root / "per_pattern_summary.csv", index=False)
    print(f"\n[{cfg.application.upper()}][PATTERN] Per-pattern summary")
    if not summary_frame.empty:
        print(summary_frame.to_string(index=False))
    return summary_frame
