import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = -100


@dataclass
class SequenceOutput:
    event_logits: torch.Tensor
    duration_logits: torch.Tensor
    gap_logits: Optional[torch.Tensor]
    hidden: torch.Tensor


@dataclass
class PatternOutput:
    mean: torch.Tensor
    log_variance: torch.Tensor
    embedding: torch.Tensor


class SequenceHeads(nn.Module):
    def __init__(self, hidden, event_classes, duration_bins, gap_bins):
        super().__init__()
        self.event = nn.Linear(hidden, event_classes)
        self.duration = nn.Linear(hidden, duration_bins)
        self.gap = nn.Linear(hidden, gap_bins) if gap_bins > 0 else None
    def forward(self, h):
        gap_logits = None if self.gap is None else self.gap(h)
        return SequenceOutput(self.event(h), self.duration(h), gap_logits, h)


class MaskedSequenceRNN(nn.Module):
    task = "reconstruction"
    def __init__(self, event_classes, duration_bins, gap_bins, cfg, cell="gru"):
        super().__init__()
        self.event_classes = event_classes
        self.mask_id = event_classes
        self.event_emb = nn.Embedding(event_classes + 1, cfg.hidden_size)
        self.duration_emb = nn.Embedding(duration_bins + 1, cfg.hidden_size)
        self.gap_emb = nn.Embedding(gap_bins + 1, cfg.hidden_size) if gap_bins > 0 else None
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.encoder = rnn(cfg.hidden_size, cfg.hidden_size, cfg.num_layers, batch_first=True,
                           dropout=cfg.dropout if cfg.num_layers > 1 else 0, bidirectional=True)
        self.project = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size)
        self.heads = SequenceHeads(cfg.hidden_size, event_classes, duration_bins, gap_bins)
    def forward(self, event, duration_bin, gap_bin, padding_mask=None):
        event = event.masked_fill(event == PAD_ID, 0)
        duration_bin = duration_bin.clamp_min(0); gap_bin = gap_bin.clamp_min(0)
        h = self.event_emb(event) + self.duration_emb(duration_bin)
        if self.gap_emb is not None:
            h = h + self.gap_emb(gap_bin)
        h, _ = self.encoder(h)
        return self.heads(self.project(h))


class MaskedSequenceTransformer(nn.Module):
    task = "reconstruction"
    def __init__(self, event_classes, duration_bins, gap_bins, max_len, cfg):
        super().__init__()
        self.event_classes, self.mask_id = event_classes, event_classes
        self.event_emb = nn.Embedding(event_classes + 1, cfg.d_model)
        self.duration_emb = nn.Embedding(duration_bins + 1, cfg.d_model)
        self.gap_emb = nn.Embedding(gap_bins + 1, cfg.d_model) if gap_bins > 0 else None
        self.position = nn.Embedding(max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.transformer_ff,
                                           cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.heads = SequenceHeads(cfg.d_model, event_classes, duration_bins, gap_bins)
    def forward(self, event, duration_bin, gap_bin, padding_mask=None):
        event = event.masked_fill(event == PAD_ID, 0)
        positions = torch.arange(event.size(1), device=event.device)[None]
        h = self.event_emb(event) + self.duration_emb(duration_bin.clamp_min(0)) + self.position(positions)
        if self.gap_emb is not None:
            h = h + self.gap_emb(gap_bin.clamp_min(0))
        return self.heads(self.encoder(h, src_key_padding_mask=padding_mask))


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__(); self.chomp = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=self.chomp, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=self.chomp, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels); self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
    def _cut(self, x): return x[..., :-self.chomp] if self.chomp else x
    def forward(self, x):
        y = self.dropout(F.gelu(self.norm1(self._cut(self.conv1(x)))))
        y = self.dropout(F.gelu(self.norm2(self._cut(self.conv2(y)))))
        return x + y


class CausalSequenceTCN(nn.Module):
    task = "prediction"
    def __init__(self, event_classes, duration_bins, gap_bins, cfg):
        super().__init__(); h = cfg.hidden_size
        self.event_emb = nn.Embedding(event_classes + 1, h)
        self.duration_emb = nn.Embedding(duration_bins + 1, h)
        self.gap_emb = nn.Embedding(gap_bins + 1, h) if gap_bins > 0 else None
        self.blocks = nn.Sequential(*[CausalTCNBlock(h, cfg.tcn_kernel, 2 ** i, cfg.dropout) for i in range(cfg.tcn_layers)])
        self.heads = SequenceHeads(h, event_classes, duration_bins, gap_bins)
    def forward(self, event, duration_bin, gap_bin, padding_mask=None):
        event = event.masked_fill(event == PAD_ID, 0)
        h = self.event_emb(event) + self.duration_emb(duration_bin.clamp_min(0))
        if self.gap_emb is not None:
            h = h + self.gap_emb(gap_bin.clamp_min(0))
        return self.heads(self.blocks(h.transpose(1, 2)).transpose(1, 2))


class CausalSequenceTransformer(nn.Module):
    task = "prediction"
    def __init__(self, event_classes, duration_bins, gap_bins, max_len, cfg):
        super().__init__(); d = cfg.d_model
        self.event_emb = nn.Embedding(event_classes + 1, d)
        self.duration_emb = nn.Embedding(duration_bins + 1, d)
        self.gap_emb = nn.Embedding(gap_bins + 1, d) if gap_bins > 0 else None
        self.position = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.transformer_ff, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.transformer_layers)
        self.heads = SequenceHeads(d, event_classes, duration_bins, gap_bins)
    def forward(self, event, duration_bin, gap_bin, padding_mask=None):
        event = event.masked_fill(event == PAD_ID, 0); length = event.size(1)
        positions = torch.arange(length, device=event.device)[None]
        h = self.event_emb(event) + self.duration_emb(duration_bin.clamp_min(0)) + self.position(positions)
        if self.gap_emb is not None:
            h = h + self.gap_emb(gap_bin.clamp_min(0))
        causal = torch.triu(torch.ones(length, length, device=event.device, dtype=torch.bool), diagonal=1)
        return self.heads(self.encoder(h, mask=causal, src_key_padding_mask=padding_mask))


class PatternGNLLAE(nn.Module):
    def __init__(self, in_dim, cfg, architecture="conv"):
        super().__init__(); h = cfg.hidden_size
        self.architecture = architecture
        if architecture in {"lstm", "gru"}:
            rnn = nn.LSTM if architecture == "lstm" else nn.GRU
            self.encoder = rnn(in_dim, h, cfg.num_layers, batch_first=True,
                               dropout=cfg.dropout if cfg.num_layers > 1 else 0)
            self.context_decoder = rnn(h, h, 1, batch_first=True)
        elif architecture == "transformer":
            self.input = nn.Linear(in_dim, h)
            layer = nn.TransformerEncoderLayer(h, cfg.nhead, cfg.transformer_ff, cfg.dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg.transformer_layers)
            decoder_layer = nn.TransformerEncoderLayer(h, cfg.nhead, cfg.transformer_ff, cfg.dropout, batch_first=True)
            self.context_decoder = nn.TransformerEncoder(decoder_layer, 1)
        else:
            self.encoder = nn.Sequential(nn.Conv1d(in_dim, h, 3, padding=1),
                                         nn.GELU(), nn.Dropout(cfg.dropout))
            self.context_decoder = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        # Fourfold temporal downsampling.
        self.temporal_bottleneck = nn.Sequential(
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 4, stride=2, padding=1), nn.GELU(),
        )
        # Mirrored temporal upsampling.
        self.temporal_decode = nn.Sequential(
            nn.ConvTranspose1d(h, h, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
            nn.ConvTranspose1d(h, h, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=1), nn.GELU(),
        )
        self.mean_head = nn.Linear(h, in_dim); self.log_var_head = nn.Linear(h, in_dim)
        self.log_var_min, self.log_var_max = cfg.log_var_min, cfg.log_var_max
        initial_variance = max(float(cfg.gnll_initial_variance), 1e-8)
        initial_log_variance = min(
            max(math.log(initial_variance), self.log_var_min), self.log_var_max
        )
        self._initialise_pattern_parameters(initial_log_variance)

    def _initialise_pattern_parameters(self, initial_log_variance):
        """Initialize the GNLL output layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        # Start with a constant variance.
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, initial_log_variance)

    def set_variance_trainable(self, trainable):
        for parameter in self.log_var_head.parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, x, mask):
        if self.architecture in {"lstm", "gru"}:
            h, _ = self.encoder(x)
        elif self.architecture == "transformer":
            h = self.encoder(self.input(x), src_key_padding_mask=~mask.bool())
        else:
            h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h = h * mask.unsqueeze(-1)
        original_length = x.size(1)
        # Pad for two stride-2 stages, then crop after decoding.
        temporal = h.transpose(1, 2)
        pad_length = (-original_length) % 4
        if pad_length:
            temporal = F.pad(temporal, (0, pad_length))
        z = self.temporal_bottleneck(temporal)
        decoded = self.temporal_decode(z)[..., :original_length]
        if self.architecture in {"lstm", "gru"}:
            decoded, _ = self.context_decoder(decoded.transpose(1, 2))
        elif self.architecture == "transformer":
            decoded = self.context_decoder(decoded.transpose(1, 2), src_key_padding_mask=~mask.bool())
        else:
            decoded = self.context_decoder(decoded).transpose(1, 2)
        decoded = decoded * mask.unsqueeze(-1)
        embedding = z.transpose(1, 2)
        return PatternOutput(self.mean_head(decoded),
                             self.log_var_head(decoded).clamp(self.log_var_min, self.log_var_max),
                             embedding)
