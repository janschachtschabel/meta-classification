"""The two-pass TF-IDF fit must BE scikit-learn's fit — array for array, not value for value.

Same vocabulary, same idf (values and dtype), and a training matrix whose data, indices
and indptr are identical: a sparse `X @ w` sums a row in stored order, so the same values
in another order would already hand the solver different last bits. Every case runs the
reference `TfidfVectorizer.fit_transform` beside `vocabulary.fit_transform_exact` on the
same texts and compares everything a model is built from.
"""

import warnings

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from app import vocabulary
from app.vocabulary import fit_transform_exact, row_blocks, transform_chunked

SYLLABLES = ["ba", "ke", "li", "mo", "nu", "ra", "se", "ti", "vo", "zu", "sch", "ei", "au",
             "ö", "ü", "ß", "é"]
_rng = np.random.default_rng(7)
WORDS = sorted({"".join(_rng.choice(SYLLABLES, size=_rng.integers(1, 4))) for _ in range(700)})


def corpus(n_docs: int, seed: int) -> list[str]:
    """Zipf-distributed pseudo-German: a few frequent words, a long tail of rare ones,
    umlauts and accents for strip_accents, some capitals for lowercase — and plenty of
    equal frequencies, so the max_features cut has ties to break."""
    rng = np.random.default_rng(seed)
    weights = 1 / np.arange(1, len(WORDS) + 1)
    weights /= weights.sum()
    docs = []
    for _ in range(n_docs):
        words = rng.choice(WORDS, size=rng.integers(3, 30), p=weights)
        docs.append(" ".join(w.capitalize() if rng.random() < 0.2 else w for w in words))
    return docs


TRAIN = corpus(300, seed=1)
UNSEEN = corpus(80, seed=2)


def params(**overrides) -> dict:
    """TfidfBackend's settings (strip_accents, sublinear_tf, float32), overridable."""
    base = dict(analyzer="word", ngram_range=(1, 2), max_features=500, min_df=2, max_df=0.95,
                sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    return {**base, **overrides}


def _same_arrays(a, b) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype
    for name in ("data", "indices", "indptr"):
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtype == right.dtype, name
        assert np.array_equal(left, right), name


def _assert_same_fit(kwargs: dict, texts=TRAIN, chunk_rows: int = vocabulary.CHUNK_ROWS):
    reference = TfidfVectorizer(**kwargs)
    expected = reference.fit_transform(texts)
    vec = TfidfVectorizer(**kwargs)
    matrix = fit_transform_exact(vec, texts, chunk_rows=chunk_rows)

    assert vec.vocabulary_ == reference.vocabulary_
    assert vec.idf_.dtype == reference.idf_.dtype
    assert np.array_equal(vec.idf_, reference.idf_)
    assert vec.fixed_vocabulary_ == reference.fixed_vocabulary_
    _same_arrays(matrix, expected)
    _same_arrays(vec.transform(UNSEEN), reference.transform(UNSEEN))
    return vec


@pytest.mark.parametrize("kwargs", [
    params(),                                                   # word (1, 2), capped
    params(analyzer="char_wb", ngram_range=(5, 5), max_features=800),
    params(max_features=None),                                  # the uncapped branch
    params(min_df=0.02, max_df=250),                            # float min_df, int max_df
    params(ngram_range=(1, 1), max_features=40, min_df=1),      # a cut deep in the ties
], ids=["word-capped", "char_wb-capped", "uncapped", "float-min-int-max", "deep-ties"])
def test_the_two_pass_fit_is_scikit_learns_fit(kwargs):
    _assert_same_fit(kwargs)


def test_ties_at_the_cap_are_broken_exactly_like_scikit_learn():
    """300 terms, each in exactly two documents, and a cap of 100: which 100 survive is
    decided by nothing but how the argsort orders equal values."""
    tied = np.full(300, -2.0, dtype=np.float32)
    assert set(tied.argsort()[:100]) != set(tied.argsort(kind="stable")[:100]), (
        "premise: here numpy's default sort keeps other terms than a stable one would — "
        "without that, this test cannot tell a different tie-breaking apart"
    )
    docs = [f"w{i:03d} w{(i + 1) % 300:03d}" for i in range(300)]
    _assert_same_fit(params(ngram_range=(1, 1), max_features=100, min_df=1, max_df=1.0), docs)


@pytest.mark.parametrize("chunk_rows", [1, 7, 10_000])
def test_the_chunk_size_changes_nothing(chunk_rows):
    _assert_same_fit(params(), chunk_rows=chunk_rows)
    _assert_same_fit(params(analyzer="char_wb", ngram_range=(5, 5), max_features=800),
                     chunk_rows=chunk_rows)


@pytest.mark.parametrize("chunk_rows", [1, 7, 10_000])
def test_a_chunked_transform_is_the_transform(chunk_rows):
    vec = TfidfVectorizer(**params()).fit(TRAIN)
    _same_arrays(transform_chunked(vec, UNSEEN, chunk_rows=chunk_rows), vec.transform(UNSEEN))
    # scikit-learn refuses an empty batch outright; chunking must not turn that into
    # something else.
    with pytest.raises(ValueError) as expected:
        vec.transform([])
    with pytest.raises(ValueError) as caught:
        transform_chunked(vec, [], chunk_rows=chunk_rows)
    assert str(caught.value) == str(expected.value)


def _reference_error(kwargs: dict, texts: list[str]) -> Exception:
    with pytest.raises(Exception) as caught:
        TfidfVectorizer(**kwargs).fit_transform(texts)
    return caught.value


@pytest.mark.parametrize(("kwargs", "texts"), [
    (params(), ["", "   ", "!!"]),                                        # empty vocabulary
    (params(min_df=5), ["eins", "zwei", "drei"]),                           # nothing survives
    (params(min_df=1, max_df=0.4), ["eins zwei", "drei vier"]),             # max_df < min_df
    (params(max_features=-1), TRAIN),                                       # invalid parameter
    (params(ngram_range=(2, 1)), TRAIN),                                    # invalid range
    (params(ngram_range=None), TRAIN),                                      # not a range at all
], ids=["empty", "pruned-away", "max-below-min", "bad-max-features", "bad-ngram-range",
        "no-ngram-range"])
def test_a_fit_that_cannot_work_fails_the_way_scikit_learn_fails(kwargs, texts):
    expected = _reference_error(kwargs, texts)
    with pytest.raises(type(expected)) as caught:
        fit_transform_exact(TfidfVectorizer(**kwargs), texts)
    assert str(caught.value) == str(expected)


def _count_reference_fits(monkeypatch) -> list[int]:
    calls = []
    real = TfidfVectorizer.fit_transform

    def spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", spy)
    return calls


def test_a_term_counted_past_float32_precision_falls_back_to_the_reference(monkeypatch):
    """The reference ranks terms by float32 column sums, exact integers only below 2**24
    occurrences. Past that the exact counts could break a tie differently, so the fit is
    handed back to scikit-learn rather than risk a different vocabulary."""
    monkeypatch.setattr(vocabulary, "_EXACT_TF_LIMIT", 3)
    calls = _count_reference_fits(monkeypatch)
    _assert_same_fit(params())
    assert len(calls) == 2, "the reference fit, then the fallback inside fit_transform_exact"


def test_the_guard_only_matters_where_a_cap_ranks_by_frequency(monkeypatch):
    monkeypatch.setattr(vocabulary, "_EXACT_TF_LIMIT", 3)
    calls = _count_reference_fits(monkeypatch)
    _assert_same_fit(params(max_features=None))
    assert len(calls) == 1, "without a cap no frequency ranking happens, so no fallback"


def _state(transformer) -> dict:
    return {key: (value.tolist(), value.dtype) if isinstance(value, np.ndarray) else value
            for key, value in vars(transformer).items()}


def test_the_fitted_state_is_the_references_all_the_way_down():
    """What skops persists is the whole vectorizer, its fitted TfidfTransformer included —
    the reference installs one fitted on the count matrix (n_features_in_ and all)."""
    reference = TfidfVectorizer(**params())
    reference.fit_transform(TRAIN)
    vec = TfidfVectorizer(**params())
    fit_transform_exact(vec, TRAIN)
    assert _state(vec._tfidf) == _state(reference._tfidf)


def test_refitting_a_used_vectorizer_leaves_nothing_of_the_first_fit():
    """A vectorizer fitted before — here by scikit-learn itself, with other parameters —
    and fitted again must transform like a fresh one: not with the first fit's norm, and
    not refusing a vocabulary of another size."""
    vec = TfidfVectorizer(**params(max_features=300))
    vec.fit_transform(TRAIN)
    vec.set_params(max_features=500, norm="l1")
    matrix = fit_transform_exact(vec, TRAIN)

    reference = TfidfVectorizer(**params(norm="l1"))
    _same_arrays(matrix, reference.fit_transform(TRAIN))
    _same_arrays(vec.transform(UNSEEN), reference.transform(UNSEEN))


def _words_as_tuples(doc: str) -> list[tuple[str]]:
    return [(word,) for word in doc.split()]


@pytest.mark.parametrize("kwargs", [params(binary=True), params(use_idf=False),
                                    params(vocabulary=["ba", "ke", "li"]),
                                    params(dtype=np.float64),
                                    params(analyzer=_words_as_tuples, ngram_range=(1, 1))],
                         ids=["binary", "no-idf", "preset-vocabulary", "float64",
                              "callable-analyzer"])
def test_what_the_codebase_never_builds_is_left_to_scikit_learn(monkeypatch, kwargs):
    """float64 among them: the reference ranks terms by column sums in the matrix's own
    dtype, and how argsort orders equal values is a property of the dtype's sort kernel —
    the two-pass selection ranks float32, which is what TfidfBackend builds."""
    reference = TfidfVectorizer(**kwargs)
    expected = reference.fit_transform(TRAIN)
    calls = _count_reference_fits(monkeypatch)
    vec = TfidfVectorizer(**kwargs)
    _same_arrays(fit_transform_exact(vec, TRAIN), expected)
    assert len(calls) == 1


def test_a_one_pass_iterator_is_left_to_scikit_learn(monkeypatch):
    """The reference takes any iterable of documents. Two passes need a collection they
    can walk twice — the fallback for a counted-out term walks it again — so a
    generator goes to the reference whole."""
    expected = TfidfVectorizer(**params()).fit_transform(TRAIN)
    calls = _count_reference_fits(monkeypatch)
    _same_arrays(fit_transform_exact(TfidfVectorizer(**params()), iter(TRAIN)), expected)
    assert len(calls) == 1


def _warnings_of(fit) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit()
    return [str(warning.message) for warning in caught]


def test_the_references_warnings_come_along():
    """stop_words on a character analyzer does nothing, and scikit-learn says so. A fit
    that is the reference's fit says so too."""
    kwargs = params(analyzer="char_wb", ngram_range=(5, 5), max_features=800, stop_words=["ba"])
    expected = _warnings_of(lambda: TfidfVectorizer(**kwargs).fit_transform(TRAIN))
    assert expected, "premise: scikit-learn warns about this configuration"
    assert _warnings_of(lambda: fit_transform_exact(TfidfVectorizer(**kwargs), TRAIN)) == expected


def test_blocks_are_never_empty_and_never_overflow():
    """A block of no rows would never advance (an endless loop behind chunk_rows=0), and
    int32 row offsets near 2**31 must not wrap when the next block's end is computed."""
    with pytest.raises(ValueError, match="at least one row"):
        row_blocks(np.array([0, 2, 4]), 0)
    top = np.array([0, 2**31 - 30, 2**31 - 20, 2**31 - 10], dtype=np.int32)
    assert row_blocks(top, 10) == [(0, 1), (1, 3)]
