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

# What a run costs, anchored on the ONE full-scale measurement this project has:
# faecher_300k_auto, 156 373 rows x 60 labels, `auto`, 40.2 min wall-clock (README,
# "How large a dataset does this handle?"). Row scaling is linear — benchmark_row_scaling
# measured exponent 1.00 for memory and vectorization — so minutes scale with the rows.
ANCHOR_ROWS = 156_373
ANCHOR_MINUTES = 40.2
# Relative to `auto`, from the head-fit and vectorization-pass counts in that same table
# (~4 min / 40 min / ~1.2 h at the anchor size). The RATIO is solid because it follows
# from the loop in tuning.cross_val_evaluate; the absolute minutes are an estimate.
_PROFILE_COST = {"fast": 0.1, "auto": 1.0, "best": 1.8}


def estimated_minutes(profile_name: str, n_rows: int) -> float | None:
    """Roughly how long training ``n_rows`` rows on this profile takes, in minutes.

    An **estimate**, not a schedule, and it is worth knowing what it cannot see: the
    label count (the head's coefficients are ``n_labels x n_features``), the machine, and
    the corpus's non-zeros per document — 365 on the anchor run against 278 on a shorter
    corpus. What it is good for is the decision it exists to support: whether a run is a
    coffee break or an afternoon.

    ``None`` for a profile the cost model has no factor for: a profile added to
    ``config.yaml`` has no measured cost until somebody measures it, and answering with
    ``auto``'s number would put a figure on screen that nothing supports.
    """
    factor = _PROFILE_COST.get(profile_name)
    if factor is None:
        return None
    return round(ANCHOR_MINUTES * factor * (n_rows / ANCHOR_ROWS), 1)


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
    # Refit the vectorizer per CV fold (no feature leakage), or fit one matrix over all
    # rows and slice it (one pass instead of k). 🟢 Measured 2026-09-09 on data_30k_ai
    # (26 450 rows x 48 labels, `auto` shape): sharing scored macro 0.7337 against 0.7320
    # refitting — +0.00171, inside the 0.002 gate but 86% of its budget, and in exactly
    # the direction leakage predicts. It saved 6.1% of the CV phase. The plan's gate asks
    # for a second target (university subjects) whose source export is not on this
    # machine, so the default stays the leak-free one until that can be measured.
    refit_vectorizer_per_fold: bool = True
    # Convergence tolerance for the SELECTION fits only; the deploy fit always uses
    # scikit-learn's 1e-4. Every C candidate is fit to be scored and then thrown away,
    # so the last digits of its convergence look like waste. ``None`` = sklearn's default.
    # 🔴 Measured 2026-09-09 on data_30k_ai (26 450 rows x 48 labels, `auto` shape,
    # benchmark_selection_tol.py). It is HARMLESS — same best_C, macro F1 moved
    # -0.000029 — and it is not worth having: 1e-3 saved a median 4.4% of the selection
    # phase over four runs (-2.7 / +3.4 / +5.4 / +11.2), against a 15% gate. Not because
    # the tail is thin: it halves the solver (7.02 -> 3.69 newton-cg iterations per
    # label, one full-row fit 26s -> 12s). The phase is simply mostly other work — k
    # vectorization passes, scoring, the threshold search. Kept as a knob because the
    # ratio changes with row count and fold count, and re-measuring is one flag away.
    selection_tol: float | None = None
    # Score every C candidate under thresholds tuned for THAT candidate, instead of
    # under a flat 0.5 cut, so the pair (C, thresholds) is chosen together. Otherwise a
    # candidate whose probabilities are RANKED well but scaled low is eliminated before
    # its thresholds exist. Costs no extra model fits: in CV mode every candidate's
    # out-of-fold probabilities are already in memory. Multilabel only —
    # binary/multiclass serving is argmax and reads no threshold.
    # 🟢 Measured 2026-09-09 on data_30k_ai (26 450 rows x 48 labels, `auto` shape,
    # benchmark_selection_rule.py) over THREE seeds, each drawing its own held-out split
    # and its own folds; the numbers come from 5 290 rows that neither the C search nor
    # the threshold tuning ever saw:
    #     seed   42: macro 0.6884 -> 0.6984  (+0.0101), labels/row -2.4%
    #     seed    7: macro 0.7243 -> 0.7271  (+0.0028), labels/row -0.6%
    #     seed 1234: macro 0.7221 -> 0.7254  (+0.0033), labels/row +0.3%
    # Median +0.0033 against the plan's >= 0.002 gate, positive on all three, and the
    # "no more than 10% extra labels per row" half is met with room to spare — the model
    # asserts FEWER labels, closer to the ~1.45 the data carries. All three seeds moved
    # best_C 8 -> 32, so the effect is systematic rather than one lucky draw, and it
    # costs no measurable time (233 vs 234 s, inside this machine's run-to-run spread).
    # Two things to carry forward: MICRO F1 is flat (+0.0100 / -0.0003 / -0.0003), so the
    # gain sits in the rare labels that macro weights equally — which is what per-label
    # thresholds are for, not a bonus everywhere. And the rule picks the TOP of the C
    # grid every time, so widening that grid past 32 (where quality was measured to
    # drop) has to be re-measured together with this flag.
    # The gain above is the CV path (`auto`, `best`). On the HOLDOUT path (`fast`) the
    # same real-data run found no disagreement at all: the flat cut already picks the top
    # C there, even with the grid widened down to 0.25, so both rules deploy the same
    # model. `fast` therefore carries the flag for consistency — it exists to predict what
    # `auto` will do — at a measured cost of nothing rather than an assumed one.
    select_c_on_tuned_thresholds: bool = True
    # Pull each label's tuned threshold toward the GLOBAL one by how much that label's
    # own cut can be trusted: weight ``n_pos / (n_pos + k)`` over its validation
    # positives. A cut fitted on four positives is mostly noise, one fitted on six
    # hundred is not, and the F1-argmax rule cannot tell them apart. ``None`` = off.
    # 🔴 Measured 2026-09-10 on data_30k_ai (26 450 rows x 48 labels, `auto` shape,
    # benchmark_threshold_shrinkage.py), k=10, three held-out seeds:
    #     seed   42: macro 0.6984 -> 0.6990  (+0.0006)
    #     seed    7: macro 0.7271 -> 0.7284  (+0.0013)
    #     seed 1234: macro 0.7254 -> 0.7230  (-0.0024)
    # Median +0.0006 against a >= 0.002 gate, straddling zero. NOT adopted.
    # The mechanism does fire — the spread of cuts narrows every time (sd 0.150 -> 0.126,
    # 0.135 -> 0.123, 0.141 -> 0.116) and the degenerate cut at the grid minimum 0.05
    # disappears — it simply buys no held-out quality here. The reason is
    # `select_c_on_tuned_thresholds`: measured against the PRE-C1 baseline the same k was
    # worth +0.0144 / +0.0035 / +0.0033, which is the gain C1 already banked. C1 removes
    # the degenerate cuts from the other side, by picking a C whose probabilities do not
    # produce them. Two fixes for one problem; the plan predicted exactly this and the
    # first run of the benchmark walked into it by leaving the baseline pre-C1.
    # Kept as a knob because the amount to shrink scales with how many positives the
    # TUNING split has, and that differs by evaluation mode: out-of-fold over the whole
    # pool gives a label ~420 positives here, a 20% holdout split ~107. A fixed-C probe
    # at holdout shape put shrinkage at +0.0040 — single seed, not the pipeline, so it
    # justifies keeping the flag and nothing more.
    threshold_shrinkage_k: float | None = None


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
            refit_vectorizer_per_fold=raw.get("refit_vectorizer_per_fold", True),
            selection_tol=raw.get("selection_tol"),
            select_c_on_tuned_thresholds=raw.get("select_c_on_tuned_thresholds", True),
            threshold_shrinkage_k=raw.get("threshold_shrinkage_k"),
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
