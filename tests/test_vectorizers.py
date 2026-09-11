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


def _max_concurrent(monkeypatch, method: str) -> list[int]:
    """Patch ``TfidfVectorizer.<method>`` to record how many calls overlap in time."""
    state = {"active": 0}
    peak = [0]
    lock = threading.Lock()
    real = getattr(TfidfVectorizer, method)

    def spy(self, *args, **kwargs):
        with lock:
            state["active"] += 1
            peak[0] = max(peak[0], state["active"])
        try:
            time.sleep(0.2)  # a sibling running concurrently gets ample time to enter
            return real(self, *args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(TfidfVectorizer, method, spy)
    return peak


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


def test_fit_transform_never_fits_the_two_vocabularies_at_once(monkeypatch):
    peak = _max_concurrent(monkeypatch, "fit_transform")
    TfidfBackend(max_word_features=40, max_char_features=60).fit_transform(TEXTS)
    assert peak[0] == 1


def test_fit_never_fits_the_two_vocabularies_at_once(monkeypatch):
    peak = _max_concurrent(monkeypatch, "fit")
    TfidfBackend(max_word_features=40, max_char_features=60).fit(TEXTS)
    assert peak[0] == 1
