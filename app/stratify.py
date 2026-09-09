"""Splitting rows so every LABEL is spread in proportion, not just the row count.

A random shuffle balances how many rows each subset gets and nothing else. With 48
labels whose support ranges from 20 to 2 476 on the same run, that leaves the rare ones
to chance: measured over 20 seeds on data_30k_ai, the rarest label lands anywhere from 1
to 4 positives in a 15 % validation split whose proportional share is 3. Macro F1 weights
that label exactly as heavily as the one with 2 476, so the noise it contributes is
noise in the headline number.

This is the iterative stratification of Sechidis, Tsoumakas and Vlahavas (2011): repeatedly
take the label with the fewest positives still unplaced and give each of its rows to the
subset that is furthest behind on that label. Handling the scarcest label first is the
whole idea — once the common labels have been spread there is no freedom left to fix a
rare one.

Folds and a train/val/test split are the same operation with different targets, so both
call sites share one implementation: ``[1/k] * k`` for k folds, ``[0.7, 0.15, 0.15]`` for
a three-way split.

A leaf: imports nothing of ours.
"""

from __future__ import annotations

import numpy as np


def _pick_subset(
    label_debt: np.ndarray, total_debt: np.ndarray, rng: np.random.Generator
) -> int:
    """The subset most in need of one more row of this label.

    Ties go to whichever subset is furthest from its overall size target, and a
    remaining tie is broken by the seeded generator rather than by index order. That
    last step is not cosmetic: with a deterministic tie-break the seed would barely
    reach the partition, every seed would produce nearly the same split, and a
    benchmark asking whether stratification lowers variance ACROSS SEEDS would be
    answered by a splitter that had simply stopped listening to the seed.
    """
    best = np.flatnonzero(label_debt == label_debt.max())
    if best.size > 1:
        room = total_debt[best]
        best = best[room == room.max()]
    return int(best[0] if best.size == 1 else rng.choice(best))


def stratified_partition(
    y: np.ndarray, proportions: list[float] | np.ndarray, *, seed: int = 42
) -> list[np.ndarray]:
    """Partition the rows of ``y`` (n x n_labels, 0/1) into subsets of the given shares.

    Returns one sorted index array per proportion. Every row appears in exactly one
    subset — including rows carrying no label at all, which have nothing to stratify on
    and go wherever the row count is furthest behind.

    :raises ValueError: if ``proportions`` is empty, contains a negative, or sums to 0 —
        each of which would otherwise produce a silently malformed partition.
    """
    shares = np.asarray(proportions, dtype=float)
    if shares.size == 0 or (shares < 0).any() or shares.sum() <= 0:
        raise ValueError(
            f"proportions must be a non-empty list of non-negative numbers that sum to "
            f"more than 0; got {list(proportions)!r}"
        )
    shares = shares / shares.sum()

    n = y.shape[0]
    rng = np.random.default_rng(seed)
    # "Debt" = how many rows each subset still wants, overall and per label. Both are
    # decremented as rows are placed, so the greediest subset is always the neediest.
    total_debt = n * shares
    label_debt = np.outer(y.sum(axis=0), shares)

    subsets: list[list[int]] = [[] for _ in shares]
    unplaced = np.ones(n, dtype=bool)

    while True:
        remaining = (y[unplaced] > 0).sum(axis=0)
        live = np.flatnonzero(remaining > 0)
        if live.size == 0:
            break
        # The scarcest label first: it has the least room to be placed well later.
        label = int(live[np.argmin(remaining[live])])
        for row in np.flatnonzero(unplaced & (y[:, label] > 0)):
            chosen = _pick_subset(label_debt[label], total_debt, rng)
            subsets[chosen].append(int(row))
            unplaced[row] = False
            label_debt[y[row] > 0, chosen] -= 1
            total_debt[chosen] -= 1

    for row in np.flatnonzero(unplaced):
        chosen = int(np.argmax(total_debt))
        subsets[chosen].append(int(row))
        total_debt[chosen] -= 1

    return [np.sort(np.array(rows, dtype=int)) for rows in subsets]
