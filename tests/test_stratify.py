"""Tests for multilabel-aware partitioning.

The property that matters is not "the subsets are the right size" — random splitting
already does that — but that each LABEL is spread across them in proportion, including
the rare ones that macro F1 weights just as heavily as the common ones.
"""

from pathlib import Path

import numpy as np
import pytest

from app import stratify
from app.vectorizers import TfidfBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _labelled(counts: list[int], n: int = 30) -> np.ndarray:
    """`n` rows whose column i carries `counts[i]` positives, spread deterministically."""
    y = np.zeros((n, len(counts)), dtype=int)
    for col, count in enumerate(counts):
        y[np.linspace(0, n - 1, count, dtype=int), col] = 1
    return y


def test_partition_covers_every_row_exactly_once():
    """Anything else silently drops rows from training or scores one twice."""
    y = _labelled([3, 15, 30])
    parts = stratify.stratified_partition(y, [1 / 3, 1 / 3, 1 / 3], seed=1)

    joined = np.concatenate(parts)
    assert sorted(joined.tolist()) == list(range(len(y)))
    assert len(joined) == len(set(joined.tolist()))


def test_a_label_with_one_positive_per_fold_gets_exactly_one_in_each():
    """The case random splitting gets wrong. Three positives over three folds is the
    tightest possible test: proportional means 1/1/1, and a random shuffle lands on
    2/1/0 or 3/0/0 often enough to leave a fold with nothing to learn the label from.
    """
    y = _labelled([3, 15, 30])
    parts = stratify.stratified_partition(y, [1 / 3, 1 / 3, 1 / 3], seed=1)

    per_fold = np.array([y[part].sum(axis=0) for part in parts])
    assert per_fold[:, 0].tolist() == [1, 1, 1], f"rare label spread {per_fold[:, 0]}"
    assert per_fold[:, 1].tolist() == [5, 5, 5]
    assert per_fold[:, 2].tolist() == [10, 10, 10]


def test_uneven_proportions_split_each_label_in_the_same_ratio():
    """A 70/15/15 three-way split is the same algorithm with unequal targets, which is
    why the holdout path and the fold path share one implementation."""
    y = _labelled([20, 100], n=200)
    parts = stratify.stratified_partition(y, [0.7, 0.15, 0.15], seed=7)

    assert [len(p) for p in parts] == [140, 30, 30]
    rare = np.array([int(y[part][:, 0].sum()) for part in parts])
    assert rare.tolist() == [14, 3, 3], f"20 positives at 70/15/15 should be 14/3/3, got {rare}"


def test_partition_is_deterministic_for_a_seed_and_varies_across_seeds():
    """Reproducibility is the point of the exercise: the same seed has to give the same
    evaluation, or comparing two runs measures the splitter instead of the change."""
    y = _labelled([7, 15, 30])
    first = stratify.stratified_partition(y, [0.5, 0.5], seed=3)
    again = stratify.stratified_partition(y, [0.5, 0.5], seed=3)
    other = stratify.stratified_partition(y, [0.5, 0.5], seed=4)

    assert [p.tolist() for p in first] == [p.tolist() for p in again]
    assert [p.tolist() for p in first] != [p.tolist() for p in other]


def test_rows_carrying_no_label_still_land_somewhere():
    """`prepare_targets` drops these before we ever see them, so this is about the
    function being honest on its own terms rather than about the pipeline: a partition
    that quietly loses rows is a bug wherever it is used."""
    y = np.zeros((10, 2), dtype=int)
    y[:4, 0] = 1
    parts = stratify.stratified_partition(y, [0.5, 0.5], seed=1)

    assert sorted(np.concatenate(parts).tolist()) == list(range(10))
    assert [len(p) for p in parts] == [5, 5]


def test_it_beats_a_random_shuffle_on_the_thing_it_exists_for():
    """Not a tautology check: the claim is that proportional placement is measurably
    tighter than shuffling, across seeds, for the rarest label. If it were not, the
    ~60 lines would not earn their place.
    """
    y = _labelled([6, 40, 120], n=120)
    stratified_spread, random_spread = [], []
    for seed in range(20):
        parts = stratify.stratified_partition(y, [1 / 3, 1 / 3, 1 / 3], seed=seed)
        stratified_spread.append(np.ptp([int(y[p][:, 0].sum()) for p in parts]))

        shuffled = np.random.default_rng(seed).permutation(len(y))
        thirds = np.array_split(shuffled, 3)
        random_spread.append(np.ptp([int(y[t][:, 0].sum()) for t in thirds]))

    assert max(stratified_spread) == 0, (
        f"6 positives over 3 folds is exactly 2 each; got spreads {stratified_spread}"
    )
    assert max(random_spread) > 0, (
        "a random shuffle should sometimes misplace the rare label — if it never does, "
        "this fixture is too easy to be evidence"
    )


def test_partition_rejects_proportions_it_cannot_honour():
    y = _labelled([3, 15, 30])
    with pytest.raises(ValueError, match="proportions"):
        stratify.stratified_partition(y, [], seed=1)
    with pytest.raises(ValueError, match="proportions"):
        stratify.stratified_partition(y, [0.5, -0.5, 1.0], seed=1)


def test_three_way_split_stratifies_only_when_given_the_labels():
    """`y=None` keeps the random splitter every caller had before, so the flag is the
    only thing that changes behaviour. With labels, the rare class must land in every
    subset in proportion instead of wherever the shuffle drops it."""
    from app.data import three_way_split

    y = np.zeros((200, 2), dtype=int)
    y[np.linspace(0, 199, 20, dtype=int), 0] = 1
    y[:, 1] = 1

    random_rare = [int(y[part][:, 0].sum())
                   for part in three_way_split(200, val_size=0.15, test_size=0.15, seed=5)]
    strat_rare = [int(y[part][:, 0].sum())
                  for part in three_way_split(200, val_size=0.15, test_size=0.15, seed=5, y=y)]

    assert sum(random_rare) == sum(strat_rare) == 20
    assert strat_rare == [14, 3, 3], f"20 positives at 70/15/15 should be 14/3/3, got {strat_rare}"


def test_cross_val_evaluate_can_take_stratified_folds(monkeypatch):
    """The fold path has its own splitter (KFold), so it needs its own switch — and
    the folds must still partition every row exactly once, which is the property a
    hand-rolled splitter is most likely to break."""
    from app import tuning

    seen: list[np.ndarray] = []
    real = tuning.stratified_partition

    def spy(y, proportions, *, seed):
        parts = real(y, proportions, seed=seed)
        seen.append(np.concatenate(parts))
        return parts

    monkeypatch.setattr(tuning, "stratified_partition", spy)

    texts = [f"alpha beta {i}" for i in range(30)] + [f"gamma delta {i}" for i in range(30)]
    y = np.zeros((60, 2), dtype=int)
    y[:30, 0] = 1
    y[30:, 1] = 1
    result = tuning.cross_val_evaluate(
        TfidfBackend, texts, y, ["c0", "c1"], k=3, c_grid=[1.0], stratified=True)

    assert result is not None
    assert len(seen) == 1, "the fold split should be computed once, not per candidate"
    assert sorted(seen[0].tolist()) == list(range(60))


def test_a_profile_can_ask_for_stratified_splits_and_defaults_not_to(tmp_path):
    """B3's on-switch, off until the benchmark's gate is met. The yaml half catches a
    parser that ignores the key, which the defaults alone cannot see."""
    from app.profiles import Profile, load_training_config
    from app.settings import get_settings

    assert Profile("x").stratified_splits is False
    for profile in load_training_config(get_settings().config_file).profiles.values():
        assert profile.stratified_splits is False, f"{profile.name} must not stratify yet"

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """profiles:
  strat:
    C_grid: [1.0]
    stratified_splits: true
""",
        encoding="utf-8",
    )
    assert load_training_config(config_file).get("strat").stratified_splits is True


def test_both_training_paths_hand_the_profile_flag_to_their_splitter(tmp_path, monkeypatch):
    """Two splitters, two links, and neither is visible from the other: the holdout
    split happens in `prepare_data`, the folds in `cross_val_evaluate`. A flag wired
    into one and not the other would look done from either side.
    """
    from app import deploy, prepare, training
    from app.profiles import Profile, TrainingConfig
    from app.registry import Registry
    from app.settings import Settings

    seen: dict[str, object] = {}
    real_split = prepare.data_mod.three_way_split
    monkeypatch.setattr(
        prepare.data_mod, "three_way_split",
        lambda n, **kw: (seen.__setitem__("holdout_got_labels", kw.get("y") is not None),
                         real_split(n, **kw))[1])
    real_cv = deploy.cross_val_evaluate
    monkeypatch.setattr(
        deploy, "cross_val_evaluate",
        lambda *a, **kw: (seen.__setitem__("cv_stratified", kw.get("stratified", "NOT PASSED")),
                          real_cv(*a, **kw))[1])

    settings = Settings(data_dir=FIXTURES, models_dir=tmp_path / "models", auth_enabled=False,
                        config_file=tmp_path / "absent.yaml")
    profile = Profile("s", "", True, True, [1.0], cv_folds=2, stratified_splits=True)
    config = TrainingConfig(default_profile="s", profiles={"s": profile},
                            validation_size=0.2, test_size=0.2, min_samples_per_label=2)
    training.run_training(
        {"dataset_name": "tiny.csv", "model_name": "strat_probe",
         "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
         "label_column": "properties.ccm:taxonid", "csv_separator": ";",
         "label_separator": ",", "label_filter": None, "profile": "s"},
        settings, config, profile,
        Registry(settings.models_dir, settings.max_models_in_memory),
        on_progress=lambda **kwargs: None, should_stop=lambda: False,
    )

    assert seen == {"holdout_got_labels": True, "cv_stratified": True}
