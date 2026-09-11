"""The TF-IDF backend: what it builds, and that building it stays within one peak.

The word and character vocabularies used to be fitted in two threads at once. Their
memory peaks then add up — +59 % at 30k rows for no measurable time saving, because the
analyzer loop holds the GIL (docs/plans/2026-09-11-training-memory.md). They are fitted
one after the other now, and the matrix must not notice.
"""

import threading
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from app.vectorizers import TfidfBackend

WORDS = ["Bruchrechnung", "Gleichung", "Geometrie", "Dreieck", "Fläche", "Römer", "Antike",
         "Kaiser", "Reich", "Photosynthese", "Zelle", "Pflanze", "Energie", "Strom", "Spannung",
         "Gedicht", "Roman", "Grammatik"]
TEXTS = [" ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(5 + i % 6)) + f" Nr {i % 9}"
         for i in range(60)]


def _same_arrays(a, b) -> bool:
    return (a.shape == b.shape and a.dtype == b.dtype
            and all(np.array_equal(getattr(a, name), getattr(b, name))
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
