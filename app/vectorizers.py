"""TF-IDF feature backend (word + character n-grams).

Sparse, low-RAM, no model download. This is the single feature backend in
api_v3: benchmarking showed embeddings did not beat TF-IDF + LogisticRegression
on this metadata, so the project deliberately settled on TF-IDF.

Feature caps (``max_word_features`` / ``max_char_features``) are the main
RAM/quality lever and are wired to settings.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfBackend:
    """Word + character n-gram TF-IDF. Strong on short metadata text."""

    kind = "tfidf"

    def __init__(
        self,
        word_ngram: tuple[int, int] = (1, 2),
        # A single 5-gram length, not the usual (3, 5). Measured on data_30k_ai.csv the
        # shorter n-grams are pure cost: (5, 5) scores BEST clean (0.7917 micro / 0.7084
        # macro vs 0.7910 / 0.7060 for (3, 5)) at 278 instead of 726 non-zeros per
        # document. Since the matrix scales with non-zeros, that is 1.2 GB instead of
        # 3.2 GB at 600k rows and 3x faster fits. (4, 5) held up marginally better under
        # heavy character noise (-0.0278 vs -0.0316 micro at 8% typos) — the one axis
        # where the shorter grams pay off, and within single-split noise.
        char_ngram: tuple[int, int] = (5, 5),
        max_word_features: int = 80_000,
        max_char_features: int = 120_000,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_char: bool = True,
    ) -> None:
        self.word_ngram: tuple[int, int] = (word_ngram[0], word_ngram[1])
        self.char_ngram: tuple[int, int] = (char_ngram[0], char_ngram[1])
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_char = use_char
        self.word_vec: TfidfVectorizer | None = None
        self.char_vec: TfidfVectorizer | None = None

    def _make(self, analyzer: str, ngram: tuple[int, int], max_features: int) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            max_features=max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            dtype=np.float32,
        )

    def fit(self, texts: list[str]) -> TfidfBackend:
        self.word_vec = self._make("word", self.word_ngram, self.max_word_features)
        self.char_vec = (
            self._make("char_wb", self.char_ngram, self.max_char_features) if self.use_char else None
        )
        vecs = [self.word_vec, *([self.char_vec] if self.char_vec is not None else [])]
        with ThreadPoolExecutor(max_workers=len(vecs)) as pool:
            for future in [pool.submit(v.fit, texts) for v in vecs]:
                future.result()
        return self

    def transform(self, texts: list[str]):
        if self.word_vec is None:
            raise RuntimeError("TfidfBackend.transform called before fit")
        word = self.word_vec.transform(texts)
        if self.char_vec is None:
            return word.tocsr()
        return hstack([word, self.char_vec.transform(texts)], format="csr")

    def fit_transform(self, texts: list[str]):
        # Word + char vocabularies are independent: build them concurrently, and
        # each fit_transform tokenizes the texts only once (vs fit + transform).
        self.word_vec = self._make("word", self.word_ngram, self.max_word_features)
        if not self.use_char:
            self.char_vec = None
            return self.word_vec.fit_transform(texts).tocsr()
        self.char_vec = self._make("char_wb", self.char_ngram, self.max_char_features)
        with ThreadPoolExecutor(max_workers=2) as pool:
            word_future = pool.submit(self.word_vec.fit_transform, texts)
            char_future = pool.submit(self.char_vec.fit_transform, texts)
            word, char = word_future.result(), char_future.result()
        return hstack([word, char], format="csr")
