import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TemporalBinner:
    duration_edges: np.ndarray
    gap_edges: Optional[np.ndarray] = None

    @staticmethod
    def _fit(values, mask, num_bins, lower, upper, explicit=None):
        if explicit is not None:
            edges = np.asarray(explicit, np.float32)
        else:
            valid = np.asarray(values)[np.asarray(mask) > 0.5]
            valid = valid[np.isfinite(valid) & (valid >= 0)]
            if not len(valid): raise ValueError("Cannot fit temporal bins without valid healthy values")
            lo, hi = np.percentile(np.log1p(valid), [lower, upper])
            if hi <= lo: hi = lo + 1e-3
            edges = np.expm1(np.linspace(lo, hi, num_bins - 1)).astype(np.float32)
        if edges.ndim != 1 or len(edges) < 1 or np.any(np.diff(edges) <= 0):
            raise ValueError("Temporal bin edges must be strictly increasing")
        return edges

    @classmethod
    def fit(cls, duration, gap, mask, gap_mask, cfg):
        args = (cfg.temporal_num_bins, cfg.temporal_lower_percentile, cfg.temporal_upper_percentile)
        duration_edges = cls._fit(
            duration, mask, *args, cfg.explicit_duration_edges
        )
        gap_edges = None
        if cfg.use_gap:
            gap_edges = cls._fit(
                gap, gap_mask, *args, cfg.explicit_gap_edges
            )
        return cls(duration_edges, gap_edges)

    @property
    def num_duration_bins(self): return len(self.duration_edges) + 1
    @property
    def num_gap_bins(self): return 0 if self.gap_edges is None else len(self.gap_edges) + 1
    def transform(self, duration, gap):
        device = duration.device
        de = torch.as_tensor(self.duration_edges, device=device, dtype=duration.dtype)
        duration_bin = torch.bucketize(duration.contiguous(), de)
        if self.gap_edges is None:
            gap_bin = torch.zeros_like(duration_bin)
        else:
            ge = torch.as_tensor(self.gap_edges, device=device, dtype=gap.dtype)
            gap_bin = torch.bucketize(gap.contiguous(), ge)
        return duration_bin, gap_bin
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "duration_edges": self.duration_edges.tolist(),
                "gap_edges": None if self.gap_edges is None else self.gap_edges.tolist(),
                "use_gap": self.gap_edges is not None,
            }, f, indent=2)
