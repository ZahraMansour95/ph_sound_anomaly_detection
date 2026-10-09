import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import SequenceDataset
from evaluation import save_evaluation
from losses import sequence_loss
from models import (CausalSequenceTCN, CausalSequenceTransformer, MaskedSequenceRNN,
                    MaskedSequenceTransformer)
from temporal_bins import TemporalBinner


def _device(cfg): return torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")


def _banner(cfg, task, model):
    print("\n" + "=" * 68)
    print(f"Application : {cfg.application.upper()}\nMode        : LABEL\nTask        : {task.upper()}")
    components = "event + duration + NONE-gap" if cfg.use_gap else "event + duration"
    losses = "focal event CE + duration-bin CE"
    if cfg.use_gap:
        losses += " + NONE-gap-bin CE"
    print(f"Model       : {model.upper()}\nComponents  : {components}\nLoss        : {losses}")
    if task == "reconstruction":
        print(f"Evaluation  : stochastic {cfg.reconstruction_mask_prob:.0%} masked-token scoring")
    print("Score       : maximum healthy-tail surprise across probabilistic components")
    print("=" * 68)


def _corrupt(event, db, gb, mask, gap_mask, mask_id, duration_mask_id,
             gap_mask_id, probability, use_gap, selected=None,
             guarantee_gap=False):
    if selected is None:
        chosen = (torch.rand_like(mask) < probability) & mask.bool()
        # Select at least one token per sequence.
        for i in range(chosen.size(0)):
            valid = torch.where(mask[i] > 0)[0]
            if len(valid) and not chosen[i].any():
                chosen[i, valid[0]] = True
            # Include a valid bowel gap when available.
            if guarantee_gap:
                valid_gap = torch.where(mask[i].bool() & gap_mask[i].bool())[0]
                if len(valid_gap) and not (chosen[i] & gap_mask[i].bool()).any():
                    random_position = torch.randint(len(valid_gap), (1,), device=mask.device)
                    chosen[i, valid_gap[random_position]] = True
    else:
        chosen = selected.bool() & mask.bool()
    corrupted_gap = gb
    if use_gap:
        corrupted_gap = gb.masked_fill(
            chosen & gap_mask.bool(), gap_mask_id
        )
    return (
        event.masked_fill(chosen, mask_id),
        db.masked_fill(chosen, duration_mask_id),
        corrupted_gap,
        chosen.float(),
    )


def _prepare(batch, binner, device):
    event = batch["event"].to(device); mask = batch["mask"].to(device)
    duration = batch["duration"].to(device); gap = batch["gap"].to(device)
    gap_mask = batch["gap_mask"].to(device)
    db, gb = binner.transform(duration, gap)
    return event, db, gb, mask, gap_mask


def make_model(task, name, vocab_size, binner, cfg):
    args = (vocab_size, binner.num_duration_bins, binner.num_gap_bins)
    if task == "reconstruction":
        return MaskedSequenceRNN(*args, cfg) if name == "rnn" else MaskedSequenceTransformer(*args, cfg.max_events, cfg)
    return CausalSequenceTCN(*args, cfg) if name == "tcn" else CausalSequenceTransformer(*args, cfg.max_events, cfg)


def _forward_loss(
    model, batch, binner, cfg, device, training, eval_partition=None
):
    event, db, gb, mask, gap_mask = _prepare(batch, binner, device)
    if model.task == "reconstruction":
        ei, di, gi, selected = _corrupt(
            event, db, gb, mask, gap_mask, model.mask_id,
            binner.num_duration_bins, binner.num_gap_bins,
            cfg.reconstruction_mask_prob, cfg.use_gap, selected=None,
            guarantee_gap=(cfg.use_gap and not training),
        )
        output = model(ei, di, gi, padding_mask=~mask.bool())
        score_mask = selected
        score_gap_mask = selected * gap_mask if cfg.use_gap else torch.zeros_like(selected)
        return (
            sequence_loss(
                output, event, db, gb, score_mask, score_gap_mask, cfg
            ),
            score_mask,
            score_gap_mask,
            output,
        )
    # Use the current event to predict the next one.
    output = model(event[:, :-1], db[:, :-1], gb[:, :-1], padding_mask=~mask[:, :-1].bool())
    score_mask = mask[:, 1:]
    score_gap_mask = gap_mask[:, 1:] if cfg.use_gap else torch.zeros_like(score_mask)
    return (
        sequence_loss(
            output, event[:, 1:], db[:, 1:], gb[:, 1:],
            score_mask, score_gap_mask, cfg,
        ),
        score_mask,
        score_gap_mask,
        output,
    )


def train(model, train_loader, val_loader, binner, cfg, out_dir):
    device = _device(cfg); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, checkpoint = float("inf"), Path(out_dir) / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        totals = []
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(); parts, _, _, _ = _forward_loss(model, batch, binner, cfg, device, True)
            parts.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
            totals.append([parts.total.item(), parts.event.item(), parts.duration.item(), parts.gap.item()])
        model.eval(); validation = []
        with torch.no_grad():
            for batch in val_loader:
                validation.append(
                    _forward_loss(model, batch, binner, cfg, device, False)[0].total.item()
                )
        means, val = np.mean(totals, axis=0), float(np.mean(validation))
        if val < best: best = val; torch.save(model.state_dict(), checkpoint)
        gap_text = f" gap={means[3]:.4f}" if cfg.use_gap else ""
        print(f"[{cfg.application.upper()}][LABEL][{model.task.upper()}] Epoch {epoch:03d}/{cfg.epochs} | "
              f"event={means[1]:.4f} duration={means[2]:.4f}{gap_text} total={means[0]:.4f} val={val:.4f}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state); return device


@torch.no_grad()
def raw_scores(model, loader, binner, cfg, device):
    model.eval(); rows, token_components = [], []
    for batch in loader:
        def entropy(logits):
            probability = torch.softmax(logits, -1)
            return -(probability * torch.log_softmax(logits, -1)).sum(-1)

        if model.task == "reconstruction":
            parts, score_mask, score_gap_mask, output = _forward_loss(
                model, batch, binner, cfg, device, False
            )
            component_losses = [parts.event_per_token, parts.duration_per_token]
            component_uncertainty = [entropy(output.event_logits), entropy(output.duration_logits)]
            if cfg.use_gap:
                component_losses.append(parts.gap_per_token.masked_fill(
                    ~score_gap_mask.bool(), float("nan")))
                component_uncertainty.append(entropy(output.gap_logits).masked_fill(
                    ~score_gap_mask.bool(), float("nan")))
            stack = torch.stack(component_losses, -1)
            uncertainty = torch.stack(component_uncertainty, -1)
            hidden = output.hidden
        else:
            parts, score_mask, score_gap_mask, output = _forward_loss(
                model, batch, binner, cfg, device, False
            )
            component_losses = [
                parts.event_per_token,
                parts.duration_per_token,
            ]
            component_uncertainty = [
                entropy(output.event_logits),
                entropy(output.duration_logits),
            ]
            if cfg.use_gap:
                component_losses.append(
                    parts.gap_per_token.masked_fill(
                        ~score_gap_mask.bool(), float("nan")
                    )
                )
                component_uncertainty.append(
                    entropy(output.gap_logits).masked_fill(
                        ~score_gap_mask.bool(), float("nan")
                    )
                )
            stack = torch.stack(component_losses, -1)
            uncertainty = torch.stack(component_uncertainty, -1)
            hidden = output.hidden

        for i in range(stack.size(0)):
            valid = score_mask[i].bool(); values = stack[i][valid].cpu().numpy()
            if not len(values): continue
            counts = np.isfinite(values).sum(axis=0)
            component_mean = np.divide(
                np.nansum(values, axis=0), counts,
                out=np.full(values.shape[1], np.nan), where=counts > 0,
            )
            uncertainty_values = uncertainty[i][valid].cpu().numpy()
            uncertainty_counts = np.isfinite(uncertainty_values).sum(axis=0)
            uncertainty_mean = np.divide(
                np.nansum(uncertainty_values, axis=0), uncertainty_counts,
                out=np.full(uncertainty_values.shape[1], np.nan),
                where=uncertainty_counts > 0,
            )
            embedding = torch.nanmean(hidden[i][valid], dim=0).cpu().numpy()
            token_components.append(values)
            row = {
                "Patient": batch["patient"][i],
                "Wav_path": batch["wav"][i],
                "Group": batch["group"][i],
                "event_nll": component_mean[0],
                "duration_nll": component_mean[1],
                "event_observed_probability": float(np.exp(-component_mean[0])),
                "duration_observed_probability": float(np.exp(-component_mean[1])),
                "event_uncertainty": uncertainty_mean[0],
                "duration_uncertainty": uncertainty_mean[1],
                "_token_values": values,
            }
            if cfg.use_gap:
                row["gap_nll"] = component_mean[2]
                row["gap_observed_probability"] = float(np.exp(-component_mean[2]))
                row["gap_uncertainty"] = uncertainty_mean[2]
            row.update({f"embedding_{j:03d}": float(value) for j, value in enumerate(embedding)})
            rows.append(row)
    return pd.DataFrame(rows), np.concatenate(token_components, axis=0)


def run_sequence_experiment(arrays, split_indices, vocabulary, task, model_name, binner, cfg, out_dir):
    _banner(cfg, task, model_name); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    loaders = {k: DataLoader(SequenceDataset(arrays, v), batch_size=cfg.batch_size, shuffle=(k == "train"),
                             num_workers=cfg.num_workers) for k, v in split_indices.items()}
    model = make_model(task, model_name, len(vocabulary), binner, cfg)
    device = train(model, loaders["train"], loaders["val"], binner, cfg, out)
    val_raw, val_tokens = raw_scores(model, loaders["val"], binner, cfg, device)
    test_raw, _ = raw_scores(model, loaders["test"], binner, cfg, device)
    # Calibration uses healthy recording scores.
    val_raw = val_raw.drop(columns="_token_values")
    test_raw = test_raw.drop(columns="_token_values")
    binner.save(out / "temporal_bins.json")
    component_cols = ["event_nll", "duration_nll"] + (["gap_nll"] if cfg.use_gap else [])
    metrics = save_evaluation(
        val_raw, test_raw, cfg, out, component_cols,
    )
    (out / "run_config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))
    return metrics
