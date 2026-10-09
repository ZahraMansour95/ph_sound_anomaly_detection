from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Config:
    application: str = "bowel"
    mode: str = "all"
    label_task: str = "all"
    label_model: str = "all"
    pattern_model: str = "all"
    pattern_labels: Optional[List[str]] = None
    csv_paths: Optional[List[str]] = None
    out_dir: str = "outputs"
    seed: int = 42
    device: str = "cuda"
    healthy_val_frac: float = 0.2
    min_segments_per_recording: int = 3
    min_sequence_events: int = 2
    max_events: int = 256
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    hidden_size: int = 128
    latent_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    d_model: int = 128
    nhead: int = 4
    transformer_layers: int = 3
    transformer_ff: int = 256
    tcn_layers: int = 6
    tcn_kernel: int = 3
    focal_gamma: float = 1.5
    use_focal_event_loss: bool = True
    reconstruction_mask_prob: float = 0.20
    reconstruction_score_all_tokens: bool = True
    temporal_num_bins: int = 16
    temporal_lower_percentile: float = 0.5
    temporal_upper_percentile: float = 99.5
    explicit_duration_edges: Optional[List[float]] = None
    explicit_gap_edges: Optional[List[float]] = None
    # Gap scoring is used only for bowel sounds.
    use_gap: bool = False
    calibration_eps: float = 1e-6
    calibration_clip_min: float = 0.0
    threshold_method: str = "target_fpr"  # target_fpr | percentile
    target_fpr: float = 0.10
    threshold_percentile: float = 90.0
    subject_aggregation: str = "mean"
    score_method: str = "max_tail"  # max_tail | legacy_mean
    save_publication_figures: bool = True
    figure_dpi: int = 600
    linear_probe_enabled: bool = True
    linear_probe_folds: int = 5
    bootstrap_iterations: int = 1000
    disease_analysis_enabled: bool = True
    disease_min_class_size: int = 3
    disease_cv_folds: int = 5
    disease_permutations: int = 200
    pattern_feature_backend: str = "raw"  # raw | mfcc | wav2vec2 | ast
    pattern_sr: int = 16000
    pattern_fixed_sec: float = 2.0
    pattern_n_mfcc: int = 20
    pattern_batch_size: int = 16
    pattern_epochs: int = 20
    pattern_patience: int = 5
    pattern_min_train_samples: int = 1
    pattern_min_validation_samples: int = 1
    pattern_min_test_samples: int = 2
    log_var_min: float = -8.0
    log_var_max: float = 6.0
    # Warm up the reconstruction mean before learning variance.
    gnll_initial_variance: float = 1.0
    gnll_warmup_epochs: int = 2
    variance_regularization: float = 1e-4
    num_workers: int = 0
    application_overrides: Dict[str, Dict] = field(default_factory=lambda: {
        "bowel": {"pattern_sr": 4000, "pattern_fixed_sec": 1.0, "use_gap": True},
        "lung": {"pattern_sr": 16000, "pattern_fixed_sec": 2.0, "use_gap": False},
        "heart": {"pattern_sr": 4000, "pattern_fixed_sec": 0.8, "use_gap": False},
    })

    def for_application(self, application: str) -> "Config":
        values = dict(self.__dict__)
        values["application"] = application
        overrides = values.pop("application_overrides")
        values.update(overrides.get(application, {}))
        obj = Config(**values)
        obj.application_overrides = overrides
        return obj

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.application
