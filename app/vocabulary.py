"""Fit a TF-IDF vectorizer in two passes: scikit-learn's result without scikit-learn's peak.

``TfidfVectorizer.fit_transform`` counts EVERY n-gram it meets into a matrix — 2.65 M word
terms for the 80 000 it keeps at 100k rows — with per-document index lists and a sorted
tuple list over all of them, and prunes only then: a peak of 4-19x the final matrix
(docs/plans/2026-09-11-training-memory.md). Here pass 1 only counts each term's document
and term frequency, the selection repeats scikit-learn's ``_sort_features`` and
``_limit_features`` step for step, and pass 2 counts the kept terms alone, chunk by chunk.

"The same" means array for array. The reference numbers its columns by first sight while
counting, sorts every row by that number, and only then renumbers the columns
alphabetically — so each row of its training matrix stores its entries in FIRST-SEEN
order. A sparse ``X @ w`` sums a row in stored order: the same values in another order
would give the solver last-bit-different sums. Pass 2 therefore counts with the kept
terms numbered by first-seen rank and renumbers afterwards, exactly as the reference
does, and ``tests/test_vocabulary.py`` compares data, indices and indptr outright.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Callable, Sequence
from numbers import Integral

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Rows per chunk in pass 2 and in transform_chunked: scikit-learn builds Python lists of
# every entry before it builds the arrays, so a chunk bounds those lists.
CHUNK_ROWS = 20_000
# The reference ranks terms by float32 column sums, which are exact integers only below
# 2**24 occurrences of one term; past that, exact integer counts could break a tie
# differently. Such a fit is handed back to the reference instead.
_EXACT_TF_LIMIT = 2**24
_COUNT_PARAMS = frozenset(CountVectorizer().get_params(deep=False))


def count_terms(
    analyze: Callable[[str], list[str]], texts: Sequence[str]
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Every term's first-seen id, and its document and term frequency by that id.

    Ids follow the reference's numbering: first document first, within a document the
    order of first occurrence. Nothing per document outlives its document.
    """
    index: dict[str, int] = {}
    dfs = array("q")
    tfs = array("q")
    for doc in texts:
        for term, count in Counter(analyze(doc)).items():
            known = index.get(term)
            if known is None:
                index[term] = len(dfs)
                dfs.append(1)
                tfs.append(count)
            else:
                dfs[known] += 1
                tfs[known] += count
    return index, np.frombuffer(dfs, dtype=np.int64), np.frombuffer(tfs, dtype=np.int64)


def select_terms(
    index: dict[str, int], dfs: np.ndarray, tfs: np.ndarray, n_docs: int, *,
    min_df: float, max_df: float, max_features: int | None,
) -> tuple[list[str], np.ndarray]:
    """The kept terms in column (alphabetical) order, and each one's first-seen id.

    ``_sort_features`` then ``_limit_features`` of scikit-learn 1.6, step for step: the
    same bounds, the same mask, and the same ``argsort`` call on the same float32 array,
    so ties at the ``max_features`` cut fall the same way.
    """
    high = max_df if isinstance(max_df, Integral) else max_df * n_docs
    low = min_df if isinstance(min_df, Integral) else min_df * n_docs
    if high < low:
        raise ValueError("max_df corresponds to < documents than min_df")
    terms = sorted(index)
    first_seen = np.fromiter((index[term] for term in terms), dtype=np.intp, count=len(terms))
    doc_freq = dfs[first_seen]
    mask = (doc_freq <= high) & (doc_freq >= low)
    if max_features is not None and mask.sum() > max_features:
        term_freq = tfs[first_seen].astype(np.float32)
        mask_inds = (-term_freq[mask]).argsort()[:max_features]
        new_mask = np.zeros(len(terms), dtype=bool)
        new_mask[np.where(mask)[0][mask_inds]] = True
        mask = new_mask
    kept = np.flatnonzero(mask)
    if kept.size == 0:
        raise ValueError(
            "After pruning, no terms remain. Try a lower min_df or a higher max_df."
        )
    return [terms[i] for i in kept], first_seen[kept]


def _count_kept(vec: TfidfVectorizer, texts: Sequence[str], terms: list[str],
                first_seen: np.ndarray, chunk_rows: int) -> sp.csr_matrix:
    """Pass 2: the kept terms' counts, laid out exactly like the reference's matrix."""
    column_of_rank = np.argsort(first_seen, kind="stable")
    params = {k: v for k, v in vec.get_params(deep=False).items() if k in _COUNT_PARAMS}
    params["vocabulary"] = {terms[column]: rank for rank, column in enumerate(column_of_rank)}
    counter = CountVectorizer(**params)
    chunks = []
    for start in range(0, max(len(texts), 1), chunk_rows):
        by_rank = counter.transform(texts[start:start + chunk_rows])  # rows sorted by rank
        chunks.append(sp.csr_matrix(
            (by_rank.data, column_of_rank.astype(by_rank.indices.dtype)[by_rank.indices],
             by_rank.indptr),
            shape=by_rank.shape,
        ))
    return chunks[0] if len(chunks) == 1 else sp.vstack(chunks, format="csr")


def fit_transform_exact(
    vec: TfidfVectorizer, texts: Sequence[str], chunk_rows: int = CHUNK_ROWS
) -> sp.csr_matrix:
    """Fit ``vec`` on ``texts`` and return the TF-IDF matrix: the state and the matrix
    ``vec.fit_transform(texts)`` produces, array for array, at a fraction of its peak.

    ``texts`` is read twice, so it must be a sequence, not an iterator. Configurations
    this codebase never builds — a preset vocabulary, ``binary``, ``use_idf=False``, an
    inverted ``ngram_range`` — and a term counted past float32 precision go to the
    reference unchanged. Invalid input fails with the reference's own errors.
    """
    low_n, high_n = vec.ngram_range
    if (vec.vocabulary is not None or vec.binary or not vec.use_idf or low_n > high_n
            or isinstance(texts, str)):
        return vec.fit_transform(texts)
    # What the reference's @_fit_context runs before it fits: its parameter constraints.
    vec._validate_params()
    index, dfs, tfs = count_terms(vec.build_analyzer(), texts)
    if not index:
        raise ValueError("empty vocabulary; perhaps the documents only contain stop words")
    if vec.max_features is not None and int(tfs.max()) >= _EXACT_TF_LIMIT:
        return vec.fit_transform(texts)
    terms, first_seen = select_terms(index, dfs, tfs, len(texts), min_df=vec.min_df,
                                     max_df=vec.max_df, max_features=vec.max_features)
    del index, dfs, tfs  # the un-pruned vocabulary: the one large thing pass 1 holds
    counts = _count_kept(vec, texts, terms, first_seen, chunk_rows)
    transformer = TfidfTransformer(norm=vec.norm, use_idf=vec.use_idf,
                                   smooth_idf=vec.smooth_idf, sublinear_tf=vec.sublinear_tf)
    transformer.fit(counts)
    vec.vocabulary_ = {term: column for column, term in enumerate(terms)}
    vec.idf_ = transformer.idf_  # public setter; also records fixed_vocabulary_ = False
    return transformer.transform(counts, copy=False)


def transform_chunked(
    vec: TfidfVectorizer, texts: Sequence[str], chunk_rows: int = CHUNK_ROWS
) -> sp.csr_matrix:
    """``vec.transform(texts)`` — the same matrix — with scikit-learn's per-entry Python
    lists bounded by a chunk instead of the whole split. Rows are independent in TF-IDF,
    so stacking per-chunk results changes nothing."""
    if len(texts) <= chunk_rows:
        return vec.transform(texts)
    return sp.vstack([vec.transform(texts[start:start + chunk_rows])
                      for start in range(0, len(texts), chunk_rows)], format="csr")
