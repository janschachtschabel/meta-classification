"""Fit a TF-IDF vectorizer in two passes: scikit-learn's result without scikit-learn's peak.

``TfidfVectorizer.fit_transform`` counts EVERY n-gram it meets — 2.65 M word terms for the
80 000 it keeps at 100k rows — into Python lists of Python ints, builds a matrix over all
of them plus a sorted tuple list, and prunes only then: a peak of 4-19x the final matrix
(docs/plans/2026-09-11-training-memory.md). Here pass 1 runs the analyzer once and keeps
the un-pruned counts as two numpy arrays (8 bytes per entry), the selection repeats
scikit-learn's ``_sort_features`` and ``_limit_features`` step for step, and pass 2 picks
the kept entries out of those arrays with numpy — no second analyzer run.

"The same" means array for array. The reference numbers its columns by first sight while
counting, sorts every row by that number, and only then renumbers the columns
alphabetically — so each row of its training matrix stores its entries in FIRST-SEEN
order. A sparse ``X @ w`` sums a row in stored order: the same values in another order
would give the solver last-bit-different sums. Pass 2 therefore sorts each row by
first-seen id and renumbers afterwards, exactly as the reference does, and
``tests/test_vocabulary.py`` compares data, indices and indptr outright.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# Rows per block in pass 2 and in transform_chunked: bounds the temporaries of a block
# (and, in transform, the Python lists scikit-learn builds per entry).
CHUNK_ROWS = 20_000
# The reference ranks terms by float32 column sums, which are exact integers only below
# 2**24 occurrences of one term; past that, exact integer counts could break a tie
# differently. Such a fit is handed back to the reference instead.
_EXACT_TF_LIMIT = 2**24
# Entries per pass-2 block. Its temporaries (row numbers, sort order, masks: ~40 bytes an
# entry) stay near 40 MB however long the documents are — 20 000 rows of char 5-grams
# are 5.6 M entries, and blocks that size cost ~250 MB of scratch at 100k rows.
_BLOCK_ENTRIES = 1 << 20


@dataclass
class TermCounts:
    """Pass 1: the un-pruned document-term counts, 8 bytes per entry.

    Row ``r`` holds the entries ``indptr[r]:indptr[r + 1]`` of ``ids`` (a term's
    first-seen id) and ``counts`` (its occurrences in that document). Ids follow the
    reference's numbering: first document first, within a document the order of first
    occurrence.
    """

    index: dict[str, int]
    n_terms: int  # len(index) — kept apart, the index is released before pass 2
    indptr: np.ndarray
    ids: np.ndarray
    counts: np.ndarray

    def document_frequencies(self) -> np.ndarray:
        return np.bincount(self.ids, minlength=self.n_terms)

    def term_frequencies(self) -> np.ndarray:
        # float64 sums of integer counts: exact far beyond _EXACT_TF_LIMIT.
        return np.bincount(self.ids, weights=self.counts, minlength=self.n_terms)


def count_terms(analyze: Callable[[str], list[str]], texts: Sequence[str]) -> TermCounts:
    """Run the analyzer once over ``texts`` and keep every document's term counts."""
    index: dict[str, int] = {}
    setdefault = index.setdefault
    ids = array("i")
    counts = array("i")
    indptr = array("q", [0])
    for doc in texts:
        grams = Counter(analyze(doc))
        # len(index) is read per term, so a new term gets the next id at the moment it
        # is first seen — the reference's defaultdict(vocabulary.__len__) numbering.
        ids.extend([setdefault(term, len(index)) for term in grams])
        counts.extend(grams.values())
        indptr.append(len(ids))
    return TermCounts(index=index, n_terms=len(index),
                      indptr=np.frombuffer(indptr, dtype=np.int64),
                      ids=np.frombuffer(ids, dtype=np.intc),
                      counts=np.frombuffer(counts, dtype=np.intc))


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


def _row_blocks(indptr: np.ndarray, max_rows: int) -> list[tuple[int, int]]:
    """Consecutive row ranges of at most ``max_rows`` rows and ``_BLOCK_ENTRIES`` entries
    (a single longer row still makes a block of its own)."""
    n_rows, blocks, start = len(indptr) - 1, [], 0
    while start < n_rows:
        fits = int(np.searchsorted(indptr, indptr[start] + _BLOCK_ENTRIES, side="right")) - 1
        stop = min(max(fits, start + 1), start + max_rows, n_rows)
        blocks.append((start, stop))
        start = stop
    return blocks


def _kept_counts(counted: TermCounts, first_seen: np.ndarray, dtype: type,
                 chunk_rows: int) -> sp.csr_matrix:
    """Pass 2: the kept terms' counts, laid out exactly like the reference's matrix —
    each row's entries by first-seen id, columns numbered alphabetically.

    In place: the kept entries overwrite pass 1's arrays front to back — a block never
    writes past its own start, and reads all it needs before it writes — so no second
    set of arrays is allocated. The result therefore sits in buffers sized for the
    un-pruned entries, and ``counted`` is used up.
    """
    column_of_id = np.full(counted.n_terms, -1, dtype=np.intc)
    column_of_id[first_seen] = np.arange(len(first_seen), dtype=np.intc)
    source = counted.indptr
    n_rows = len(source) - 1
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    blocks = _row_blocks(source, chunk_rows)
    for start, stop in blocks:  # sweep 1: how many entries each row keeps
        kept = column_of_id[counted.ids[source[start]:source[stop]]] >= 0
        rows = np.repeat(np.arange(stop - start), np.diff(source[start:stop + 1]))
        indptr[start + 1:stop + 1] = np.bincount(rows[kept], minlength=stop - start)
    np.cumsum(indptr, out=indptr)
    indices = counted.ids
    # float32 data fits the int32 count buffer it replaces; any other width gets its own.
    in_place = np.dtype(dtype).itemsize == counted.counts.itemsize
    data = counted.counts.view(dtype) if in_place else np.empty(indptr[-1], dtype=dtype)
    for start, stop in blocks:  # sweep 2: write them, front to back
        lo, hi = source[start], source[stop]
        ids = counted.ids[lo:hi]
        columns = column_of_id[ids]
        kept = columns >= 0
        rows = np.repeat(np.arange(stop - start), np.diff(source[start:stop + 1]))[kept]
        order = np.lexsort((ids[kept], rows))  # by row, then first-seen id
        values = counted.counts[lo:hi][kept][order]  # copies, taken before the writes
        kept_columns = columns[kept][order]
        indices[indptr[start]:indptr[stop]] = kept_columns
        data[indptr[start]:indptr[stop]] = values
    nnz = int(indptr[-1])
    return sp.csr_matrix((data[:nnz], indices[:nnz], indptr), shape=(n_rows, len(first_seen)))


def fit_transform_exact(
    vec: TfidfVectorizer, texts: Sequence[str], chunk_rows: int = CHUNK_ROWS
) -> sp.csr_matrix:
    """Fit ``vec`` on ``texts`` and return the TF-IDF matrix: the state and the matrix
    ``vec.fit_transform(texts)`` produces, array for array, at a fraction of its peak.

    Configurations this codebase never builds — a preset vocabulary, ``binary``,
    ``use_idf=False``, an inverted ``ngram_range`` — and a term counted past float32
    precision go to the reference unchanged. Invalid input fails with the reference's
    own errors.

    The matrix may sit in buffers sized for the UN-pruned entries (pass 2 works in
    place): a caller that keeps it for long copies it, which ``TfidfBackend`` does or
    merges it into a fresh matrix anyway.
    """
    low_n, high_n = vec.ngram_range
    if (vec.vocabulary is not None or vec.binary or not vec.use_idf or low_n > high_n
            or isinstance(texts, str)):
        return vec.fit_transform(texts)
    # What the reference's @_fit_context runs before it fits: its parameter constraints.
    vec._validate_params()
    counted = count_terms(vec.build_analyzer(), texts)
    if not counted.index:
        raise ValueError("empty vocabulary; perhaps the documents only contain stop words")
    tfs = counted.term_frequencies()
    if vec.max_features is not None and int(tfs.max()) >= _EXACT_TF_LIMIT:
        return vec.fit_transform(texts)
    terms, first_seen = select_terms(
        counted.index, counted.document_frequencies(), tfs, len(texts),
        min_df=vec.min_df, max_df=vec.max_df, max_features=vec.max_features,
    )
    del tfs
    # The kept terms sit scattered among millions of dropped ones in the allocator's
    # arenas and would pin them: the memory would never go back to the OS (measured
    # +230 MB at 100k rows, under everything that follows). Copy them out through one
    # string, drop every original — the un-pruned index holds them all — and rebuild
    # them in fresh memory.
    joined, ends = "".join(terms), np.cumsum([len(term) for term in terms]).tolist()
    del terms
    counted.index = {}
    vocabulary = {joined[start:end]: column
                  for column, (start, end) in enumerate(zip([0, *ends[:-1]], ends, strict=True))}
    del joined
    counts = _kept_counts(counted, first_seen, vec.dtype, chunk_rows)
    del counted
    transformer = TfidfTransformer(norm=vec.norm, use_idf=vec.use_idf,
                                   smooth_idf=vec.smooth_idf, sublinear_tf=vec.sublinear_tf)
    transformer.fit(counts)
    vec.vocabulary_ = vocabulary
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
