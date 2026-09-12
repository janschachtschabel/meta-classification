"""Does German-specific text preprocessing buy anything — stopwords, stemming, lemmas?

Asked 2026-09-12 after a leave-one-out explanation listed the connector "und" among the
strongest words of a subject prediction. Measured on the deployed vocabulary, the word
feature "und" is near weightless (idf 1.46 against 7.16 for a content word, |coef| 0.13-4)
and the weight sits in the character n-grams that span the phrase around it — so the
explanation attributes context to the token, and the preprocessing is not obviously at
fault. Whether the three classic normalisations nevertheless BUY anything is a
measurement, not an argument. This is the measurement.

Identical rows, split, seed, C grid and threshold procedure across variants; only the
text the vectorizer sees changes:

  stopwords   on the WORD analyzer only. Removing them from the text would place words
              next to each other that never were, and that junction is exactly what the
              character n-grams read.
  stemming    Snowball German, applied per word inside the text (word AND char features
              see it), punctuation and spacing untouched.
  lemmas      simplemma, the same way — a dictionary lemma instead of a cut suffix.

A custom preprocessor REPLACES scikit-learn's own, so every variant re-applies what
TfidfBackend relies on: lower-casing and unicode accent stripping. The stopword list is
normalised the same way, or it would not match the tokens it is meant to remove.

snowballstemmer and simplemma are development-only (not in requirements.lock, not in the
image): a variant whose package is missing is skipped and said so. Only a variant that
wins here is worth the dependency question at all — and it would then apply to every
/predict call as well, not just to training.

Usage:
    python scripts/benchmark_preprocessing.py [--dataset data_30k_ai.csv] [--rows N]
        [--c-grid 4,16] [--threads 6]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
from joblib import parallel_backend  # noqa: E402
from sklearn.feature_extraction.text import strip_accents_unicode  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from app.classifier import make_head  # noqa: E402
from app.data import prepare_targets, three_way_split  # noqa: E402
from app.dataset_load import load_dataset  # noqa: E402
from app.vectorizers import TfidfBackend  # noqa: E402

SEED = 42
TEXT_COLUMNS = ["properties.cclom:title", "properties.cclom:general_description",
                "properties.cclom:general_keyword"]
LABEL_COLUMN = "properties.ccm:taxonid"
DISCIPLINE = "http://w3id.org/openeduhub/vocabs/discipline/"
MIN_SAMPLES = 20
_WORD = re.compile(r"\w+", re.UNICODE)

# German function words: articles, pronouns, prepositions, conjunctions, auxiliaries and
# the fillers a metadata title is full of. Deliberately WITHOUT domain words ("Unterricht",
# "Aufgabe", "Material"), which carry subject signal — this measures stopwords, not a
# hand-pruned vocabulary.
_STOPWORD_BLOCK = """
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer
anderes auch auf aus bei beim bin bis bist da damit dann das dass dasselbe dazu dein
deine dem den denn der derselbe des dessen dich die dies diese dieselbe diesem diesen
dieser dieses dir doch dort du durch ein eine einem einen einer eines einig einige er es
etwas euch euer eure für gegen gewesen hab habe haben hat hatte hatten hier hin hinter
ich ihm ihn ihnen ihr ihre ihrem ihren ihrer im in indem ins ist ja jede jedem jeden
jeder jedes jene jenem jenen jener jenes jetzt kann kannst kein keine keinem keinen
keiner keines können könnte man manche manchem manchen mancher manches mein meine mich
mir mit muss musste nach nicht nichts noch nun nur ob oder ohne sehr sein seine seinem
seinen seiner seines selbst sich sie sind so solche solchem solchen solcher solches soll
sollte sondern sonst über um und uns unse unsem unsen unser unses unter viel vom von vor
war waren warst was weg weil weiter welche welchem welchen welcher welches wenn werde
werden wie wieder will wir wird wirst wo wollen wollte würde würden zu zum zur zwar
zwischen
"""
STOPWORDS = _STOPWORD_BLOCK.split()


def normalize(text: str) -> str:
    """What scikit-learn's own preprocessor does for TfidfBackend: lower-case, no accents."""
    return strip_accents_unicode(text.lower())


def per_word(transform) -> "callable":
    """A preprocessor that rewrites every word in place, leaving punctuation and spacing."""
    return lambda text: _WORD.sub(lambda m: transform(m.group()), normalize(text))


def stemming_preprocessor():
    import snowballstemmer

    stem = snowballstemmer.stemmer("german").stemWord
    return per_word(stem)


def lemma_preprocessor():
    import simplemma

    def lemma(word: str) -> str:
        return simplemma.lemmatize(word, lang="de")

    return per_word(lemma)


class PreprocessedBackend(TfidfBackend):
    """``TfidfBackend`` with a stopword list on the word analyzer and/or a preprocessor."""

    def __init__(self, *, stop_words=None, preprocessor=None, **kwargs) -> None:
        self._stop_words = stop_words
        self._preprocessor = preprocessor
        super().__init__(**kwargs)

    def _make(self, analyzer: str, ngram: tuple[int, int], max_features: int):
        vec = super()._make(analyzer, ngram, max_features)
        if self._preprocessor is not None:
            vec.set_params(preprocessor=self._preprocessor)
        if self._stop_words is not None and analyzer == "word":
            vec.set_params(stop_words=sorted({normalize(w) for w in self._stop_words}))
        return vec


def variants() -> list[tuple[str, dict]]:
    """The variants to compare; one whose package is missing is left out with a note."""
    out: list[tuple[str, dict]] = [("baseline", {})]
    out.append(("stopwords (word analyzer)", {"stop_words": STOPWORDS}))
    for name, factory in (("stemming (Snowball)", stemming_preprocessor),
                          ("lemmas (simplemma)", lemma_preprocessor)):
        try:
            preprocessor = factory()
        except ImportError as exc:
            log(f"skipping {name}: {exc.name} is not installed (development-only)")
            continue
        out.append((name, {"preprocessor": preprocessor}))
        if name.startswith("stemming"):
            out.append((f"stopwords + {name}",
                        {"stop_words": STOPWORDS, "preprocessor": preprocessor}))
    return out


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-label cut that maximises F1 on the validation split (as ``thresholds.py`` does)."""
    cuts = np.zeros(proba.shape[1])
    qs = np.linspace(0.50, 0.9995, 60)
    for col in range(proba.shape[1]):
        column, truth = proba[:, col], y_true[:, col]
        best_f1, best_t = -1.0, float(np.quantile(column, 0.5))
        for t in np.unique(np.quantile(column, qs)):
            f1 = f1_score(truth, (column >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        cuts[col] = best_t
    return cuts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data_30k_ai.csv")
    parser.add_argument("--rows", type=int, default=0, help="cap the usable rows (0 = all)")
    parser.add_argument("--c-grid", default="4,16")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    c_grid = [float(c) for c in args.c_grid.split(",")]

    log(f"loading {args.dataset} ...")
    loaded = load_dataset(BASE / "data" / args.dataset, TEXT_COLUMNS, LABEL_COLUMN,
                          separator=";", label_separator=",", label_filter=DISCIPLINE)
    y_all, classes, keep = prepare_targets(loaded.label_lists, min_samples=MIN_SAMPLES)
    texts = [t for t, k in zip(loaded.texts, keep, strict=False) if k]
    if args.rows:
        texts, y_all = texts[: args.rows], y_all[: args.rows]
    log(f"{len(texts):,} rows, {y_all.shape[1]} labels")

    tr, va, te = three_way_split(len(texts), val_size=0.15, test_size=0.15, seed=SEED)
    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
    txt_tr = [texts[i] for i in tr]
    txt_va = [texts[i] for i in va]
    txt_te = [texts[i] for i in te]

    results = []
    for name, kwargs in variants():
        backend = PreprocessedBackend(**kwargs)
        started = time.time()
        x_tr = backend.fit_transform(txt_tr)
        vectorize_s = time.time() - started
        x_va, x_te = backend.transform(txt_va), backend.transform(txt_te)
        n_word = len(backend.word_vec.vocabulary_)
        n_char = len(backend.char_vec.vocabulary_) if backend.char_vec is not None else 0
        log(f"{name}: vocabulary word={n_word:,} char={n_char:,} "
            f"| {x_tr.nnz / x_tr.shape[0]:.0f} non-zeros/row | vectorize {vectorize_s:.0f}s")

        best = None
        for c in c_grid:
            head = make_head(c, n_jobs=args.threads, solver="newton-cg")
            with parallel_backend("threading", n_jobs=args.threads):
                head.fit(x_tr, y_tr)
            proba_va = head.predict_proba(x_va)
            cuts = tune_thresholds(y_va, proba_va)
            val = f1_score(y_va, (proba_va >= cuts).astype(int), average="macro",
                           zero_division=0)
            log(f"  C={c}: val macro {val:.4f}")
            if best is None or val > best["val"]:
                best = {"c": c, "val": val, "cuts": cuts, "test": head.predict_proba(x_te)}
            del head
        assert best is not None
        preds = (best["test"] >= best["cuts"]).astype(int)
        row = {
            "variant": name, "vocabulary_word": n_word, "vocabulary_char": n_char,
            "nnz_per_row": round(x_tr.nnz / x_tr.shape[0], 1),
            "vectorize_seconds": round(vectorize_s, 1), "best_C": best["c"],
            "val_f1_macro": round(float(best["val"]), 4),
            "f1_macro": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_te, preds, average="micro", zero_division=0)), 4),
        }
        results.append(row)
        log(f"  -> {name}: TEST macro {row['f1_macro']:.4f} micro {row['f1_micro']:.4f}")
        del backend, x_tr, x_va, x_te

    baseline = results[0]
    print("\n| Variant | word vocabulary | test macro F1 | vs baseline | test micro F1 |"
          "\n|---|---:|---:|---:|---:|")
    for row in results:
        delta = row["f1_macro"] - baseline["f1_macro"]
        print(f"| {row['variant']} | {row['vocabulary_word']:,} | {row['f1_macro']:.4f} "
              f"| {delta:+.4f} | {row['f1_micro']:.4f} |")
    out = BASE / "preprocessing_results.json"
    out.write_text(json.dumps({"dataset": args.dataset, "n_rows": len(texts),
                               "n_labels": int(y_all.shape[1]), "seed": SEED,
                               "c_grid": c_grid, "results": results}, indent=2),
                   encoding="utf-8")
    log(f"written: {out}")


if __name__ == "__main__":
    main()
