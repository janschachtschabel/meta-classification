"""Training profiles loaded from config.yaml (with safe code defaults).

A profile is the single *fast ↔ good* dial, and the shipped set (`fast`, `auto`,
`best`) is ordered by cost so the name is a truthful price tag. It controls the
TF-IDF feature size (char n-grams on/off, vocabulary caps), the
inverse-regularization grid, threshold tuning, and the evaluation mode — the
three levers that actually drive wall-clock: number of C candidates, number of
CV folds, and whether character n-grams are built at all.
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
    # Evaluation mode that suits this profile's size class (0 = holdout split,
    # >=2 = k-fold CV). None -> fall back to the config-wide `split.cv_folds`.
    # A request's own `cv_folds` still wins over both.
    cv_folds: int | None = None


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
    # Dataset convention: how often a text column is repeated when the training text
    # is assembled. Applies only when the request omits `text_column_weights`, and
    # only to columns that request actually trains on.
    text_column_weights: dict[str, int] = field(default_factory=dict)

    def get(self, name: str) -> Profile:
        if name not in self.profiles:
            raise KeyError(f"Unknown profile {name!r}. Available: {sorted(self.profiles)}")
        return self.profiles[name]


# Mirrors config.yaml — see there for the measurements behind each value.
# Three rungs, strictly ordered by cost: fast < auto < best. The C range is fixed at
# 2 ... 32 for all of them (measured: quality DROPS above 32), so the rungs differ in the
# two levers that were measured to pay — character n-grams and the fold count. `fast`
# additionally halves its grid because it exists for iteration, not for deployment.
_DEFAULTS: dict[str, Profile] = {
    "fast": Profile("fast",
                    "Word-only, holdout split, 2 C values. Seconds — for iteration, "
                    "not for a model you deploy.",
                    tune_threshold=True, threshold_per_label=True, c_grid=[2.0, 32.0],
                    use_char=False, max_word_features=200_000, cv_folds=0),
    "auto": Profile("auto", "Word+char TF-IDF, 3-fold CV, 3 C values. The recommended default.",
                    tune_threshold=True, threshold_per_label=True,
                    c_grid=[2.0, 8.0, 32.0], cv_folds=3),
    "best": Profile("best", "Word+char TF-IDF, 5-fold CV. Most accurate evaluation; ~1.9x auto.",
                    tune_threshold=True, threshold_per_label=True,
                    c_grid=[2.0, 8.0, 32.0], cv_folds=5),
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
            cv_folds=raw.get("cv_folds"),
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
        text_column_weights={
            str(col): int(weight)
            for col, weight in (prep.get("text_column_weights") or {}).items()
        },
    )
