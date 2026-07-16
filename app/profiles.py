"""Training profiles loaded from config.yaml (with safe code defaults).

A profile is the single *fast ↔ good* dial. It controls the TF-IDF feature size
(char n-grams on/off, vocabulary caps), the inverse-regularization grid
(auto-selected on validation) and threshold tuning. Word-only + smaller caps =
faster and far less RAM (fits large data on 8 GB) at a small quality cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Profile:
    name: str
    description: str = ""
    tune_threshold: bool = True
    threshold_per_label: bool = True
    c_grid: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    # TF-IDF feature shape (RAM/quality/speed lever). None -> use settings default.
    use_char: bool = True
    max_word_features: int | None = None
    max_char_features: int | None = None


@dataclass
class TrainingConfig:
    default_profile: str = "auto"
    profiles: dict[str, Profile] = field(default_factory=dict)
    validation_size: float = 0.15
    test_size: float = 0.15
    # 0 = classic train/val/test split; >=2 = k-fold cross-validation (all rows
    # train + validate via out-of-fold, deploy on 100%).
    cv_folds: int = 0
    min_text_length: int = 5
    drop_duplicates: bool = True
    min_samples_per_label: int | None = None

    def get(self, name: str) -> Profile:
        if name not in self.profiles:
            raise KeyError(f"Unknown profile {name!r}. Available: {sorted(self.profiles)}")
        return self.profiles[name]


_DEFAULTS: dict[str, Profile] = {
    "fast": Profile("fast", "Word-only TF-IDF, few C values, global threshold. Fastest, lowest RAM.",
                    tune_threshold=True, threshold_per_label=False, c_grid=[2.0, 4.0],
                    use_char=False, max_word_features=50_000),
    "auto": Profile("auto", "Word+char TF-IDF, auto C-selection + per-label thresholds.",
                    tune_threshold=True, threshold_per_label=True, c_grid=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
    "thorough": Profile("thorough", "Word+char TF-IDF, wide C grid + per-label thresholds. Best, slowest.",
                        tune_threshold=True, threshold_per_label=True,
                        c_grid=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
}


def load_training_config(path: str | Path) -> TrainingConfig:
    """Parse config.yaml into a TrainingConfig, falling back to code defaults."""
    path = Path(path)
    if not path.exists():
        return TrainingConfig(profiles=dict(_DEFAULTS))

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles: dict[str, Profile] = {}
    for name, raw in (data.get("profiles") or {}).items():
        profiles[name] = Profile(
            name=name,
            description=raw.get("description", ""),
            tune_threshold=raw.get("tune_threshold", True),
            threshold_per_label=raw.get("threshold_per_label", True),
            c_grid=raw.get("C_grid", raw.get("c_grid", [1.0, 2.0, 4.0])),
            use_char=raw.get("use_char", True),
            max_word_features=raw.get("max_word_features"),
            max_char_features=raw.get("max_char_features"),
        )
    if not profiles:
        profiles = dict(_DEFAULTS)

    prep = data.get("preprocessing") or {}
    split = data.get("split") or {}
    return TrainingConfig(
        default_profile=data.get("default_profile", "auto"),
        profiles=profiles,
        validation_size=float(split.get("validation_size", 0.15)),
        test_size=float(split.get("test_size", 0.15)),
        cv_folds=int(split.get("cv_folds", 0)),
        min_text_length=int(prep.get("min_text_length_chars", 5)),
        drop_duplicates=bool(prep.get("drop_duplicates", True)),
        min_samples_per_label=prep.get("min_samples_per_label"),
    )
