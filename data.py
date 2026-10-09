import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_ID = -100


def normalize_patient_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    stripped = text.lstrip("0")
    return stripped or "0"


def load_events(paths: List[str], preset):
    frames = []
    required = {"Patient", "Wav_path", "Start", "End", "Label"}
    for path in paths:
        frame = pd.read_csv(path, dtype={"Patient": str})
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["Patient"] = df["Patient"].map(lambda x: str(x).strip().removesuffix(".0"))
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df.dropna(subset=["Start", "End"]).copy()
    df["Dur"] = df["End"] - df["Start"]
    df = df[df["Dur"] >= 0].copy()
    healthy = {normalize_patient_id(x) for x in preset["healthy_ids"]}
    patient = {normalize_patient_id(x) for x in preset["patient_ids"]}
    overlap = healthy.intersection(patient)
    healthy.difference_update(overlap)
    norm = df["Patient"].map(normalize_patient_id)
    df["Group"] = np.where(norm.isin(patient), "patient", np.where(norm.isin(healthy), "healthy", "unknown"))
    unknown = df.loc[df.Group == "unknown", "Patient"].unique()
    if len(unknown):
        raise ValueError(f"Unknown patient IDs: {unknown[:20].tolist()}")
    return df.reset_index(drop=True)


def recording_index(df, min_segments=3):
    rows = []
    for (patient, wav), group in df.groupby(["Patient", "Wav_path"], sort=False):
        group = group.sort_values("Start")
        if len(group) >= min_segments:
            rows.append({"Patient": patient, "Wav_path": wav, "Group": group.Group.iloc[0], "idxs": group.index.tolist()})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No recordings remain after filtering")
    return result


def split_recordings(records, val_fraction, seed):
    rng = random.Random(seed)
    ids = records.loc[records.Group == "healthy", "Patient"].unique().tolist()
    if len(ids) < 2:
        raise ValueError("At least two healthy subjects are needed")
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    train = records[(records.Group == "healthy") & records.Patient.isin(train_ids)].reset_index(drop=True)
    val = records[(records.Group == "healthy") & records.Patient.isin(val_ids)].reset_index(drop=True)
    test = records[
        (records.Group == "patient") | records.Patient.isin(val_ids)
    ].reset_index(drop=True)
    return train, val, test


@dataclass
class SequenceArrays:
    event: np.ndarray
    duration: np.ndarray
    gap: np.ndarray
    gap_mask: np.ndarray
    mask: np.ndarray
    start: np.ndarray
    patients: np.ndarray
    groups: np.ndarray
    wavs: np.ndarray


def _interval_union_duration(intervals, lower, upper):
    """Return the covered NONE duration within an interval."""
    clipped = []
    for start, end in intervals:
        start = max(float(start), float(lower))
        end = min(float(end), float(upper))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            clipped.append((start, end))
    if not clipped:
        return 0.0
    clipped.sort()
    total = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def build_sequences(df, records, vocabulary, aliases, max_events, gap_label=None):
    """Build event, duration, and optional gap sequences."""
    label_to_id = {x: i for i, x in enumerate(vocabulary)}
    aliases = {
        str(key).strip().upper(): (
            None if value is None else str(value).strip().upper()
        )
        for key, value in aliases.items()
    }
    canonical_gap = None if gap_label is None else str(gap_label).strip().upper()
    n = len(records)
    event = np.full((n, max_events), PAD_ID, np.int64)
    duration = np.zeros((n, max_events), np.float32)
    gap = np.zeros((n, max_events), np.float32)
    gap_mask = np.zeros((n, max_events), np.float32)
    mask = np.zeros((n, max_events), np.float32)
    start = np.zeros((n, max_events), np.float32)
    meta = [[], [], []]
    for row_index, row in records.iterrows():
        recording = df.loc[row.idxs].sort_values(["Start", "End"])
        sound_events = []
        pending_gap_intervals = []
        previous_sound_end = None

        for item in recording.itertuples(index=False):
            raw = str(item.Label).strip().upper()
            canonical = aliases.get(raw, raw)
            if canonical is None:
                continue
            if canonical_gap is not None and canonical == canonical_gap:
                pending_gap_intervals.append((float(item.Start), float(item.End)))
                continue
            if canonical not in label_to_id:
                # Ignore labels outside the selected vocabulary.
                continue

            current_start = float(item.Start)
            current_end = float(item.End)
            if previous_sound_end is None:
                gap_value, gap_valid = 0.0, 0.0
            else:
                gap_value = _interval_union_duration(
                    pending_gap_intervals, previous_sound_end, current_start
                )
                gap_valid = 1.0 if canonical_gap is not None else 0.0

            sound_events.append(
                (
                    label_to_id[canonical],
                    max(current_end - current_start, 0.0),
                    gap_value,
                    gap_valid,
                    current_start,
                )
            )
            previous_sound_end = current_end
            pending_gap_intervals = []
            if len(sound_events) >= max_events:
                break

        length = len(sound_events)
        if length:
            values = np.asarray(sound_events, dtype=np.float64)
            event[row_index, :length] = values[:, 0].astype(np.int64)
            duration[row_index, :length] = values[:, 1].astype(np.float32)
            gap[row_index, :length] = values[:, 2].astype(np.float32)
            gap_mask[row_index, :length] = values[:, 3].astype(np.float32)
            start[row_index, :length] = values[:, 4].astype(np.float32)
            mask[row_index, :length] = 1
        meta[0].append(row.Patient); meta[1].append(row.Group); meta[2].append(row.Wav_path)
    return SequenceArrays(event, duration, gap, gap_mask, mask, start, *map(np.asarray, meta))


class SequenceDataset(Dataset):
    def __init__(self, arrays: SequenceArrays, indices):
        self.a, self.indices = arrays, np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        i = self.indices[item]
        return {"event": torch.as_tensor(self.a.event[i]), "duration": torch.as_tensor(self.a.duration[i]),
                "gap": torch.as_tensor(self.a.gap[i]), "gap_mask": torch.as_tensor(self.a.gap_mask[i]),
                "mask": torch.as_tensor(self.a.mask[i]),
                "start": torch.as_tensor(self.a.start[i]), "patient": self.a.patients[i],
                "group": self.a.groups[i], "wav": self.a.wavs[i]}
