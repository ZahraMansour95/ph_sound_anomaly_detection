import argparse
import random
from pathlib import Path

import numpy as np
import torch

from applications import event_vocabulary, get_preset
from config import Config
from data import build_sequences, load_events, recording_index, split_recordings
from pattern_pipeline import run_pattern_experiment
from sequence_pipeline import run_sequence_experiment
from temporal_bins import TemporalBinner


def parse_args():
    parser = argparse.ArgumentParser(description="Unified physiological-sound anomaly detection")
    parser.add_argument("--mode", choices=["label", "pattern", "all"], default="all")
    parser.add_argument("--application", choices=["bowel", "lung", "heart", "all"], default="all")
    parser.add_argument("--label-task", choices=["reconstruction", "prediction", "all"], default="all")
    parser.add_argument("--label-model", choices=["rnn", "tcn", "transformer", "all"], default="all")
    parser.add_argument("--pattern-model", choices=["lstm", "gru", "conv", "transformer", "all"], default="all")
    parser.add_argument("--pattern-label", nargs="+", default=["all"],
                        help="Pattern(s) to analyse, e.g. --pattern-label SB MB; default: all")
    parser.add_argument("--csv", nargs="+", default=None, help="Override preset CSV path(s); only with one application")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--healthy-val-frac", type=float, default=0.20)
    parser.add_argument("--temporal-bins", type=int, default=16)
    parser.add_argument("--pattern-feature-backend", choices=["raw", "mfcc", "wav2vec2", "ast"], default="raw")
    parser.add_argument("--gnll-initial-variance", type=float, default=1.0,
                        help="Initial constant variance predicted by the GNLL variance head")
    parser.add_argument("--gnll-warmup-epochs", type=int, default=2,
                        help="Epochs used to stabilise the reconstruction mean before learning variance")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def record_indices(all_records, subset):
    mapping = {(row.Patient, row.Wav_path): i for i, row in all_records.iterrows()}
    return [mapping[(row.Patient, row.Wav_path)] for _, row in subset.iterrows()]


def label_combinations(args):
    tasks = ["reconstruction", "prediction"] if args.label_task == "all" else [args.label_task]
    for task in tasks:
        allowed = ["rnn", "transformer"] if task == "reconstruction" else ["tcn", "transformer"]
        selected = allowed if args.label_model == "all" else [args.label_model]
        invalid = set(selected).difference(allowed)
        if invalid: raise ValueError(f"{sorted(invalid)} not valid for label task {task}; choose {allowed}")
        for model in selected: yield task, model


def run_application(application, args):
    cfg = Config(application=application, mode=args.mode, label_task=args.label_task,
                 label_model=args.label_model, pattern_model=args.pattern_model,
                 out_dir=args.out_dir, device=args.device, epochs=args.epochs,
                 pattern_epochs=args.epochs, batch_size=args.batch_size,
                 target_fpr=args.target_fpr, temporal_num_bins=args.temporal_bins,
                 healthy_val_frac=args.healthy_val_frac,
                 pattern_feature_backend=args.pattern_feature_backend,
                 gnll_initial_variance=args.gnll_initial_variance,
                 gnll_warmup_epochs=args.gnll_warmup_epochs).for_application(application)
    cfg.pattern_labels = args.pattern_label
    if cfg.pattern_feature_backend in {"wav2vec2", "ast"}:
        cfg.pattern_sr = 16000
    set_seed(cfg.seed); preset = get_preset(application)
    paths = args.csv if args.csv else preset["csv_paths"]
    df = load_events(paths, preset); records = recording_index(df, cfg.min_segments_per_recording)
    train_records, val_records, test_records = split_recordings(
        records, cfg.healthy_val_frac, cfg.seed
    )
    print(f"\n[{application.upper()}] recordings={len(records)} train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    if args.mode in {"label", "all"}:
        vocabulary = event_vocabulary(application)
        arrays = build_sequences(
            df,
            records,
            vocabulary,
            preset["label_aliases"],
            cfg.max_events,
            gap_label=preset.get("sequence_gap_label") if cfg.use_gap else None,
        )
        event_counts = arrays.mask.sum(axis=1)
        sequence_records = records.loc[
            event_counts >= cfg.min_sequence_events
        ].reset_index(drop=True)
        removed = int(len(records) - len(sequence_records))
        if sequence_records.empty:
            raise ValueError(
                "No label-sequence recordings remain after removing NONE and "
                "non-target rows"
            )
        if removed:
            print(
                f"[{application.upper()}][LABEL] excluded {removed} recordings "
                f"with fewer than {cfg.min_sequence_events} sound events"
            )
        sequence_train, sequence_val, sequence_test = split_recordings(
            sequence_records,
            cfg.healthy_val_frac,
            cfg.seed,
        )
        split_indices = {
            "train": record_indices(records, sequence_train),
            "val": record_indices(records, sequence_val),
            "test": record_indices(records, sequence_test),
        }
        train_idx = split_indices["train"]
        binner = TemporalBinner.fit(
            arrays.duration[train_idx],
            arrays.gap[train_idx],
            arrays.mask[train_idx],
            arrays.gap_mask[train_idx],
            cfg,
        )
        for task, model in label_combinations(args):
            out = cfg.run_dir / "label" / task / model
            run_sequence_experiment(arrays, split_indices, vocabulary, task, model, binner, cfg, out)

    if args.mode in {"pattern", "all"}:
        architectures = ["lstm", "gru", "conv", "transformer"] if args.pattern_model == "all" else [args.pattern_model]
        for architecture in architectures:
            out = cfg.run_dir / "pattern" / cfg.pattern_feature_backend / architecture
            summary = run_pattern_experiment(
                df,
                (train_records, val_records, test_records),
                preset,
                architecture,
                cfg,
                out,
            )
            completed = int((summary["status"] == "completed").sum()) if not summary.empty else 0
            skipped = int((summary["status"] == "skipped").sum()) if not summary.empty else 0
            print(
                f"[{application.upper()}][PATTERN][{architecture.upper()}] "
                f"completed={completed} skipped={skipped}"
            )


def main():
    args = parse_args()
    if args.application == "all" and args.csv:
        raise ValueError("--csv cannot be combined with --application all")
    applications = ["bowel", "lung", "heart"] if args.application == "all" else [args.application]
    failures = []
    for application in applications:
        try: run_application(application, args)
        except Exception as exc:
            if args.application != "all": raise
            failures.append((application, repr(exc))); print(f"[{application.upper()}][FAILED] {exc}")
    if failures:
        formatted = "\n".join(f"  {app}: {error}" for app, error in failures)
        raise RuntimeError(f"One or more applications failed:\n{formatted}")


if __name__ == "__main__": main()
