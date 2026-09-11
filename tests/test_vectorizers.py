"""The TF-IDF backend: what it builds, and that building it stays within one peak.

The word and character vocabularies used to be fitted in two threads at once. Their
memory peaks then add up — +59 % at 30k rows for no measurable time saving, because the
analyzer loop holds the GIL (docs/plans/2026-09-11-training-memory.md). They are fitted
one after the other now, and the matrix must not notice.
"""

import threading
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import random as sparse_random
from sklearn.feature_extraction.text import TfidfVectorizer

from app.vectorizers import TfidfBackend, merge_columns

WORDS = ["Bruchrechnung", "Gleichung", "Geometrie", "Dreieck", "Fläche", "Römer", "Antike",
         "Kaiser", "Reich", "Photosynthese", "Zelle", "Pflanze", "Energie", "Strom", "Spannung",
         "Gedicht", "Roman", "Grammatik"]
TEXTS = [" ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(5 + i % 6)) + f" Nr {i % 9}"
         for i in range(60)]


def _same_arrays(a, b) -> bool:
    """Same shape, and the same data/indices/indptr arrays — values, dtypes, stored order."""
    return (a.shape == b.shape
            and all(np.array_equal(getattr(a, name), getattr(b, name))
                    and getattr(a, name).dtype == getattr(b, name).dtype
                    for name in ("data", "indices", "indptr")))


def _max_concurrent(monkeypatch, owner, method: str) -> list[int]:
    """Patch ``owner.<method>`` to record how many calls overlap in time (and how many
    there were: ``[peak, calls]``)."""
    state = {"active": 0}
    record = [0, 0]
    lock = threading.Lock()
    real = getattr(owner, method)

    def spy(*args, **kwargs):
        with lock:
            state["active"] += 1
            record[0] = max(record[0], state["active"])
            record[1] += 1
        try:
            time.sleep(0.2)  # a sibling running concurrently gets ample time to enter
            return real(*args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(owner, method, spy)
    return record


def test_fit_transform_builds_the_matrix_of_the_two_vectorizers_side_by_side():
    backend = TfidfBackend(max_word_features=40, max_char_features=60)
    matrix = backend.fit_transform(TEXTS)

    reference = TfidfBackend(max_word_features=40, max_char_features=60)
    word = reference._make("word", reference.word_ngram, 40).fit_transform(TEXTS)
    char = reference._make("char_wb", reference.char_ngram, 60).fit_transform(TEXTS)
    assert _same_arrays(matrix, hstack([word, char], format="csr"))


def test_transform_matches_what_fit_transform_built():
    backend = TfidfBackend(max_word_features=40, max_char_features=60)
    matrix = backend.fit_transform(TEXTS)
    again = backend.transform(TEXTS)
    assert (matrix != again).nnz == 0


def test_the_word_only_profile_builds_the_word_matrix_alone():
    backend = TfidfBackend(use_char=False, max_word_features=40)
    matrix = backend.fit_transform(TEXTS)
    word = TfidfBackend()._make("word", (1, 2), 40).fit_transform(TEXTS)
    assert backend.char_vec is None
    assert _same_arrays(matrix, word.tocsr())


def _scrambled(matrix, rng):
    """The same CSR matrix with every row's entries in random stored order — the shape of
    a training matrix, whose rows are in first-seen order, not sorted."""
    matrix = matrix.tocsr()
    order = np.concatenate([
        rng.permutation(np.arange(matrix.indptr[r], matrix.indptr[r + 1]))
        for r in range(matrix.shape[0])
    ] or [np.empty(0, dtype=np.intp)]).astype(np.intp)
    return csr_matrix((matrix.data[order], matrix.indices[order], matrix.indptr),
                      shape=matrix.shape)


@pytest.mark.parametrize(("rows", "left_cols", "right_cols", "density"), [
    (60, 30, 45, 0.15), (9, 5, 3, 0.0), (1, 1, 1, 1.0), (40, 20, 10, 0.4), (0, 4, 6, 0.5),
])
def test_merging_columns_is_scipys_hstack_array_for_array(rows, left_cols, right_cols, density):
    """scipy's hstack holds both inputs, a concatenated copy of their arrays AND the
    result (3x the matrix: +994 MB peak at 100k rows); the merge writes each row block
    straight into the result. It must build the very same arrays — stored order too."""
    rng = np.random.default_rng(rows + left_cols)
    left = _scrambled(sparse_random(rows, left_cols, density=density, format="csr",
                                    dtype=np.float32, random_state=rng), rng)
    right = _scrambled(sparse_random(rows, right_cols, density=density, format="csr",
                                     dtype=np.float32, random_state=rng), rng)
    assert _same_arrays(merge_columns(left, right, block_rows=7),
                        hstack([left, right], format="csr"))


def test_merging_in_blocks_bounded_by_entries_is_hstack_too(monkeypatch):
    """In production the entry bound, not the row count, cuts the merge into blocks; a
    bound of three entries puts a block boundary inside every other row here."""
    from app import vocabulary

    monkeypatch.setattr(vocabulary, "_BLOCK_ENTRIES", 3)
    rng = np.random.default_rng(11)
    left = _scrambled(sparse_random(40, 20, density=0.3, format="csr", dtype=np.float32,
                                    random_state=rng), rng)
    right = _scrambled(sparse_random(40, 30, density=0.3, format="csr", dtype=np.float32,
                                     random_state=rng), rng)
    assert _same_arrays(merge_columns(left, right), hstack([left, right], format="csr"))


def test_merging_promotes_dtypes_and_refuses_ragged_rows_like_hstack():
    """hstack promotes a float32 block beside a float64 one instead of truncating the
    wider values, and it refuses blocks of different heights rather than guess."""
    rng = np.random.default_rng(3)
    left = sparse_random(8, 5, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    right = sparse_random(8, 4, density=0.4, format="csr", dtype=np.float64, random_state=rng)
    assert _same_arrays(merge_columns(left, right), hstack([left, right], format="csr"))
    with pytest.raises(ValueError, match="rows"):
        merge_columns(left, right[:5])


def test_fit_transform_fits_both_vocabularies_in_two_passes_one_after_the_other(monkeypatch):
    """Both vocabularies go through the two-pass fit (vocabulary.py) — the reference
    fit_transform is the peak this exists to avoid — and never at the same time."""
    from app import vectorizers

    record = _max_concurrent(monkeypatch, vectorizers, "fit_transform_exact")
    TfidfBackend(max_word_features=40, max_char_features=60).fit_transform(TEXTS)
    assert record == [1, 2]  # never two at once; one call per vocabulary


def test_fit_never_fits_the_two_vocabularies_at_once(monkeypatch):
    record = _max_concurrent(monkeypatch, TfidfVectorizer, "fit")
    TfidfBackend(max_word_features=40, max_char_features=60).fit(TEXTS)
    assert record[0] == 1
