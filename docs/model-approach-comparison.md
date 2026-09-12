# Modeling approach comparison: TF-IDF+LogReg vs. Linear SVM vs. Bayesian ProdSLDA

Three approaches to the same problem — classifying WLO metadata (subject,
educational level, resource type, ...) from title/description/keywords —
compared on architecture, multi-task handling, compute cost, accuracy, and
serving characteristics.

**Evidence levels used throughout** (so the reader can tell fact from
expectation at a glance):

- 🟢 **Measured** — run in this project, numbers below are real output.
- 🔵 **Documented** — stated by the source project's own docs/repo, not
  independently reproduced.
- ⚪ **Theoretical** — general ML-literature expectation; not benchmarked here.
  Never treat as a result.

---

## TL;DR

For this task (sparse, keyword-heavy metadata, CPU-only deployment target),
**TF-IDF + Logistic Regression (the current MetaClassify approach) remains the
soundest choice** — now with the SVM comparison actually run rather than assumed.
On identical features and splits, `LinearSVC` came out **+0.013 macro F1** ahead
on a single test split ([measured](#answered-svm-measured-on-this-projects-data-2026-07-25)),
a gap well inside split noise for a 48-label macro average and not worth giving up
natively calibrated probabilities for — which the whole API (thresholds,
`baseline_diff`, `label_f1`, `/predict/multi`) is built around; restoring them on
top of an SVM costs a nested calibration CV and +33 % fit time. LogReg is also the
only one of the three with 🟢 measured accuracy *and* training time on this exact
data plus deterministic sub-second serving. The Bayesian ProdSLDA approach offers a
genuinely richer signal (posterior credibility intervals) but costs GPU training,
~4 GB serving RAM, non-deterministic per-request inference, and — by the source
project's own admission — pickle persistence and a train/serve preprocessing
mismatch.

---

## At a glance

| | TF-IDF + LogReg (MetaClassify) | Linear SVM | Bayesian ProdSLDA (its-jointprobability) |
|---|---|---|---|
| Evidence level (accuracy) | 🟢 Measured | ⚪ Not benchmarked here | 🔵 None published |
| Multi-task design | One model **per facet**, combined via `/predict/multi` | Same as LogReg (drop-in head swap) | **One joint model**, 6 facets simultaneously |
| Native probability output | Yes (`predict_proba`) | No (needs Platt/isotonic calibration) | Yes — full posterior, not just a point |
| Training compute | CPU-only | CPU-only | GPU "highly recommended" |
| Inference determinism | Deterministic | Deterministic | **Stochastic** (varies run to run) |
| Serving RAM | ~0.5–1 GB (🟢 measured, README) | Similar order, unmeasured | ~4 GB (🔵 documented) |
| Persistence | skops (pickle-free) | skops (pickle-free, same head wrapper) | Python `pickle` (🔵 source repo calls this unsafe) |
| Uncertainty signal | `baseline_diff` (point estimate) | Point estimate (needs calibration first) | Mean/median + 80% credibility interval |
| Auth on the API | API-key, admin/readonly roles | (same app) | 🔵 None documented |

---

## 1. TF-IDF + Logistic Regression — the current MetaClassify approach

**Architecture:** sparse float32 TF-IDF (word n-grams, optional char n-grams,
up to 200k features in the `auto` profile) → `OneVsRestClassifier(LogisticRegression)`,
one binary classifier per label, `C` auto-selected via grid search
(validation split or out-of-fold in CV mode).

**Multi-task handling:** one model *per metadata facet* (`faecher_ai_cv5` for
subject, `bildungsstufe_ai_cv5` for educational level, ...), combined at the
API layer via `POST /predict/multi` (≤5 models per call, each keeping its own
tuned thresholds). `task_type` (multilabel / multiclass / binary) is
auto-detected per facet; multiclass/binary decide via pure argmax (skipping
threshold tuning entirely — the metrics measure exactly the rule serving
uses), multilabel uses tuned per-label thresholds.

**Parallelism/RAM engineering:** the `newton-cg`/`saga` solvers preserve
float32 *and* release the GIL, so joblib's `threading` backend fits every
label's classifier against **one shared sparse matrix** — all cores, ~1×
RAM instead of a copy per worker.

### 🟢 Measured numbers (this project, this session, CPU-only, no GPU)

| Model | Labels | Rows | Eval | Fit time | f1_macro | f1_micro |
|---|---:|---:|---|---:|---:|---:|
| `faecher_ai_cv5` | 48 | 26,450 | 5-fold CV (30 fits: 6×C-grid × 5 folds) | 1282.6 s (~21.4 min) | 0.740 | 0.808 |
| `bildungsstufe_ai_cv5` | 12 | 25,763 | 5-fold CV (30 fits) | 523.9 s (~8.7 min) | 0.578 | 0.813 |

(README also documents a single-`C` benchmark, no grid search: 15k rows /
200k dims, `newton-cg`/threading, `n_jobs=8` → ≈13 s fit time, ~1 GB RAM,
F1 micro 0.81 — a different measurement than the two above, included for
scale reference only.)

**Serving:** one deterministic sparse-matrix transform + sigmoid per
`/predict` call. No sampling — the same input always returns the same
output. No GPU anywhere in the stack (the project is explicitly torch-free).

**Uncertainty signal:** `baseline_diff` = confidence − confidence on an
empty-text input for the same label. A **point-estimate** heuristic, not a
distribution — it tells you the text moved the needle, not how *sure* the
model is about that movement. Per-label F1 in the bundle's `metrics.json`
serves as the closest thing to a reliability signal.

**Persistence & security:** skops (pickle-free, load-safety-checked,
allowlisted member types), API-key auth (admin/readonly roles), rate
limiting, constant-time key comparison.

---

## 2. Linear SVM — architecturally close, now benchmarked

> Accuracy and speed for this approach were **measured on 2026-07-25** — see
> [Answered: SVM measured](#answered-svm-measured-on-this-projects-data-2026-07-25).
> The architectural discussion below predates that run; where a ⚪ expectation in it
> conflicts with the measurement, the measurement wins.

**What would change:** swap `LogisticRegression` for `LinearSVC` inside the
same `OneVsRestClassifier` wrapper, same TF-IDF features. In this codebase
that is a near drop-in change to `classifier.py`'s `make_head()` — the rest
of the pipeline (`tuning.py`'s C-grid search, `prepare.py`, `deploy.py`)
would not need to know the difference for the *fitting* step.

**Where it stops being a drop-in swap:**

- **No native probabilities.** `LinearSVC` gives `decision_function()` — a
  signed distance to the separating hyperplane, not a `[0,1]` probability.
  For **multiclass/binary** (argmax rule, post the B9 fix) this is actually
  fine: `decision_function().argmax()` works directly, no calibration
  needed. For **multilabel** (tuned per-label thresholds) it is not
  optional — thresholding a hyperplane distance sensibly requires
  `CalibratedClassifierCV` (Platt/isotonic scaling) wrapped around the SVM,
  which adds an *inner* cross-validated calibration step on top of the
  existing outer `C`-grid search — more training cost, another layer of
  approximation.
- **`baseline_diff` would need calibration too.** A diff between two raw SVM
  margins is not on an interpretable 0–1 scale; the feature as currently
  designed assumes probabilities.
- **Parallelism story likely regresses.** `liblinear` (the solver behind
  `LinearSVC`) is GIL-bound — the same constraint this codebase already
  documents for LogReg's own `liblinear` option ("parallelises only via
  processes, copying the matrix per worker"). Fitting labels in parallel
  would probably need the `loky` (process) backend instead of `threading`,
  losing the "all cores share one matrix, ~1× RAM" property `newton-cg` was
  specifically chosen for.

**Accuracy — ⚪ theoretical only, do not treat as a result:** in the general
ML/NLP literature, linear SVM and L2-regularized logistic regression on
sparse high-dimensional TF-IDF features typically land within a small margin
of each other; margin-maximizing SVM sometimes edges ahead on cleaner/smaller
datasets, log-loss LogReg is sometimes more robust on noisy multilabel data.
Which one wins is dataset-dependent and genuinely unknown for *this* data —
**no SVM run has ever been made on this project's datasets.** See the open
question below.

---

## 3. Bayesian ProdSLDA (`openeduhub/its-jointprobability`)

Independently re-verified against the live repository this session (not
taken from a secondhand summary); consistent with an internal note from an
earlier investigation of the same repo.

**Architecture:** ProdLDA / ProdSLDA (🔵 [arXiv:1703.01488](https://arxiv.org/abs/1703.01488)),
implemented in Pyro using black-box variational inference. Bag-of-words →
VAE → 500-topic mixture (hidden layer size 1000) → linear map to each target
category, categories across different metadata fields modeled independently.

**Multi-task handling — the key structural difference from MetaClassify:**
**one joint model predicts all 6 metadata fields simultaneously** (school
discipline, university discipline, resource type, target audience,
educational context, WLO topic tree), semi-supervised via an observation
mask. MetaClassify's answer to the same problem is the opposite: separate
models per facet, composed at the API layer. Neither is objectively
"correct" — joint learning can share signal across related facets but
couples their training and update cadence; separate models are simpler to
retrain, version, and reason about individually (exactly what
`/predict/multi` is for).

**Training (🔵 documented):** GPU "highly recommended" for retraining; the
repo's own words: *"this will take a long time"* (no duration given). Two
sequential phases — unsupervised topic-model training, then the supervised
linear coupling. Memory usage scales directly with data size.

**Inference (🔵 documented) — costly and non-deterministic per request, not
just at training time:** *"computation time is roughly proportional to the
number of [posterior] samples"* (default 500, configurable far higher), and
*"predictions are stochastic, so another run on the same text may yield
slightly different predictions."* This is a genuine serving-latency and
reproducibility concern, separate from the one-time training cost.

**Output:** mean, median, and an 80%-credibility interval per category —
real posterior uncertainty from sampling, richer than a point estimate. This
is the one dimension where ProdSLDA is unambiguously ahead of both linear
approaches: neither LogReg nor SVM can produce a calibrated *interval*
without bolting on a separate Bayesian layer.

**Self-admitted limitations (🔵 from the repo's own docs), each independently
relevant to a choice MetaClassify already made:**

- *"Loading these files is generally considered to be unsafe, as they could
  execute arbitrary Python code"* — persistence is Python `pickle`. Confirms
  the case for skops.
- *"texts given processed through the REST-API do not run through the same
  pre-processing pipeline as the training data"* — an acknowledged
  train/serve skew. MetaClassify runs the same `clean_text` at train time
  and predict time specifically to avoid this class of bug.
- *"hierarchies are currently flattened, such that any information stemming
  from the hierarchies is discarded"* — information loss independent of the
  model family.
- No authentication is documented on the REST API.

**Accuracy — 🔵 none published.** The repo computes quality metrics during
training but reports no F1/precision/recall numbers in its documentation.
An informal, user-reported comparison from this project's own testing put it
at roughly 6% lower macro accuracy than the current TF-IDF+LogReg approach
on the same data, and subjectively slower to train — **this is this
project's own finding, not a number from the source repo**, and has not been
independently reproduced in a controlled side-by-side run.

---

## Full comparison table

| Dimension | TF-IDF + LogReg | Linear SVM | Bayesian ProdSLDA |
|---|---|---|---|
| Model family | Linear (log-loss) | Linear (hinge-loss) | Deep generative (VAE) + linear coupling |
| Framework | scikit-learn | scikit-learn (same wrapper) | Pyro (PyTorch) |
| Feature representation | Sparse TF-IDF, word+char n-grams | Same | Bag-of-words tensor |
| Multi-task design | One model per facet + API composition | Same as LogReg | One joint model, 6 facets |
| Training compute | CPU | CPU | GPU strongly recommended |
| Training time (this data) | 🟢 8.7–21.4 min (5-fold CV, per facet) | ⚪ unmeasured | 🔵 "a long time", unspecified |
| Accuracy (this data) | 🟢 f1_macro 0.58–0.74 (facet-dependent) | ⚪ unmeasured | 🔵 no published number; ~6% lower per informal internal comparison |
| Serving latency | Sub-second, deterministic | Sub-second, deterministic (expected) | Scales with posterior-sample count; documented as slow-ish and variable |
| Determinism | Always identical output | Always identical output | **Non-deterministic** — repeat runs vary |
| Serving RAM | 🟢 ~0.5–1 GB | ⚪ similar order, unmeasured | 🔵 ~4 GB |
| Native probabilities | Yes | No — needs calibration | Yes — full posterior |
| Uncertainty signal | Point-estimate diff (`baseline_diff`) | Point-estimate (post-calibration) | Mean/median + credibility interval |
| Persistence | skops (pickle-free) | skops (same wrapper) | Python pickle (🔵 flagged unsafe by source) |
| Train/serve preprocessing parity | Enforced (`clean_text` shared) | Enforced (same pipeline) | 🔵 Mismatch admitted by source |
| API auth | API-key, roles, rate limiting | Same app | 🔵 None documented |
| GPU dependency | None (torch-free) | None | Required for practical retraining |

---

## Answered: SVM measured on this project's data (2026-07-25)

🟢 **Measured.** The open question below has been closed by an actual run on
`data_30k_ai.csv` (26,450 rows, 48 subject labels). Everything except the head was
held identical: the **same TF-IDF matrix** (fit once on train and reused), the same
split and seed, the same per-label threshold procedure, metrics on the same
untouched test split. Each head selected its own `C` on validation, since `C` is
not comparable across the two losses.

| Head | best C | F1 macro | F1 micro | Precision macro | Recall macro | Fit time |
|------|-------:|---------:|---------:|----------------:|-------------:|---------:|
| `LogisticRegression` (deployed) | 32.0 | 0.7063 | 0.7937 | 0.7500 | 0.7041 | 33.0 s |
| `LinearSVC` (raw margins) | 1.0 | **0.7193** | **0.7998** | 0.7506 | 0.7093 | **27.5 s** |
| `LinearSVC` + `CalibratedClassifierCV` | 1.0 | 0.7158 | 0.7975 | 0.7515 | 0.7098 | 65.3 s |
| Ensemble: 0.25·LogReg + 0.75·calibrated SVC | – | 0.7112 | 0.7956 | 0.7394 | **0.7174** | 98.3 s |

**What this does and does not establish:**

- Linear SVM is **not worse**, and comes out **+0.013 macro F1 / +0.006 micro F1**
  ahead. That is a *single split with a single seed* — for a 48-label macro average
  over ~4,000 test rows, a gap that size is well inside what split noise can
  produce. Read it as "SVM is at least competitive here", **not** as "SVM wins".
- The gain is almost entirely **recall** (0.7093 vs 0.7041); precision is a tie
  to three decimals.
- **Calibration does not cost "some speed" — it reverses the entire comparison.**
  api_v3's contract needs probabilities in `[0,1]` (`confidence`, `baseline_diff`,
  the threshold grid, `/predict/multi`), which raw margins are not. `LinearSVC`'s
  speed advantage only exists in the variant api_v3 **cannot use**. With the
  calibration it does need, fit time goes from 27.5 s to 65.3 s — *slower* than the
  deployed LogReg's 33.0 s. And the cost is structural, not incidental:
  `CalibratedClassifierCV(cv=3)` keeps **three fitted sub-models per label** and
  evaluates all three per prediction. 🟢 Measured on a synthetic
  3000×20 000 / 20-label setup (the *ratios* transfer, the absolutes do not):

  | Head | Coefficients | Single-request latency | Batch of 200 |
  |------|-------------:|-----------------------:|-------------:|
  | LogReg `newton-cg` | 1.5 MB | **2.83 ms** | **3.5 ms** |
  | `LinearSVC` raw (no probabilities) | 3.1 MB | 2.07 ms | 3.5 ms |
  | `LinearSVC` + `Calibrated(cv=3)` | **9.2 MB** | **26.08 ms** | 30.1 ms |

  That is **6× the memory and ~9× the per-request latency** — and latency is what a
  serving API pays on every call, unlike training time. Projected to 300 labels ×
  200 k features the head alone would be **1373 MB** (vs. 229 MB for LogReg), which
  does not fit the 2 GB deployment target at all.
- ⚠️ These numbers are **not** comparable to the deployed `faecher_ai_cv5` figure
  (0.740 macro): that comes from 5-fold CV over all rows with the production
  threshold grid. Only the three rows of this table are comparable to each other.
- 🟢 **Correction to an expectation stated earlier in this document:** `liblinear`
  being GIL-bound was expected to cost parallelism or RAM. It does serialize the
  48 label fits — and was still *faster in wall clock* than 6-way-parallel
  `newton-cg`, at the same ~1× matrix RAM (threads share one matrix either way).
  The GIL mechanism is real; the practical penalty is not.

**Conclusion:** not worth swapping the head for ~0.01 macro F1 on one split,
against the cost of giving up native calibrated probabilities and adding a nested
calibration CV. Worth revisiting only with a full 5-fold CV run (~1 h) if that last
point of macro F1 becomes decision-relevant.

### Does combining LogReg *and* SVM help? — No, and the reason is measurable

The obvious follow-up is an ensemble. It was run (last row above: probability
average, weight picked on validation) and it lands **between** its two members,
not above them: 0.7112 macro — better than LogReg alone (0.7063), **worse than the
SVC alone** (0.7193/0.7158). For twice the training and twice the inference.

The diagnosis explains why, and generalizes beyond this one run:

| Diversity measure (test split) | Value |
|--------------------------------|------:|
| Probability correlation, LogReg vs calibrated SVC | **0.9855** |
| Jaccard over the labels either head asserts | **0.9064** |

An ensemble pays off when its members make *different* mistakes. These two are both
**linear** models on the **identical** feature matrix, differing only in the loss
function (logistic vs. hinge). They agree on 91 % of asserted labels and their
probability outputs correlate at 0.99 — there is almost no independent signal to
combine, so averaging mostly smooths the operating point rather than adding
information. That shows in the numbers: the ensemble has the **highest recall**
(0.7174) and the **lowest precision** (0.7394) of all four heads — it moved the
threshold trade-off, it did not learn anything new.

**Practical reading:** diversity would have to come from a genuinely different
feature space (embeddings, a different tokenization) or a different model class
(trees), not from a second linear head on the same TF-IDF matrix. And this project
already measured that embeddings do not beat TF-IDF here — so there is no cheap
ensemble partner available. Skip the ensemble.

### Scaling to a large vocab (48 → 300 labels)

WLO vocabs are not all 48 labels wide — `ccm:curriculum` alone carries 478 labels
above 20 samples. 🟢 Measured on that column (21,887 rows, same texts, same TF-IDF
matrix, same split; only the label set changes — `scripts/benchmark_label_scaling.py`).
One fixed `C` per head, so the F1 columns are cost-context, not a tuned verdict:

| Labels | Fit | Coef size | dtype | RSS growth | Threshold tuning | F1 macro | F1 micro |
|-------:|----:|----------:|-------|-----------:|-----------------:|---------:|---------:|
| 48 | 29.1 s | 36.6 MB | float32 | +36 MB | 4.3 s | 0.4576 | 0.6254 |
| 300 | 168.5 s | 228.9 MB | float32 | **+242 MB** | 24.3 s | 0.3507 | 0.3191 |

Scaling for 6.25× more labels: fit ×5.79, coefficients ×6.25, threshold tuning ×5.65.
**Everything is linear in the label count** — nothing blows up super-linearly, so a
wide vocab is a budgeting question, not an architectural one.

- **Memory is the planning number.** A 300-label head is 229 MB of float32
  coefficients, on top of ~148 MB of vectorizer per model. With
  `APIV3_MAX_MODELS_IN_MEMORY=2` that is roughly 750 MB of resident model data before
  the Python baseline — plan the container limit accordingly.
- *(Historical: the same run was originally executed with a `LinearSVC` head alongside.
  It landed within noise on quality — 0.3482 vs 0.3507 macro — while costing exactly 2×
  the coefficient memory through float64. That confirmed the head choice and the
  comparison was retired; see the section above.)*

#### The real problem at 300 labels is not the head

| Labels | Precision micro | Recall micro | Labels asserted per row | True labels per row |
|-------:|----------------:|-------------:|------------------------:|--------------------:|
| 48 | 0.634 | 0.617 | 1.14 | 1.17 |
| 300 | **0.220** | 0.581 | **4.22** | 1.60 |

Micro F1 halves (0.6254 → 0.3191) and even drops *below* macro, which is the reverse
of the usual pattern. The cause is visible in the last two columns: recall barely
moves, **precision collapses**, and the model asserts 4.2 labels where the truth has
1.6. Per-label thresholds are tuned to maximize *that label's own* F1 — for a rare
label, recall is cheap and precision is inherently poor, so the F1-optimal cut sits
low and buys recall with false positives. Averaged per label (macro) each rare label
contributes 1/300; pooled (micro) all those false positives land in one denominator.
Both heads land on 4.22 vs 4.23 asserted labels, so this is a property of the
**thresholding regime at high label counts**, not of the classifier.

**What to do about it** — using what the API already has, no new modelling:

- Serve wide vocabs in **ranking mode** (`top_k`, e.g. `0` = the training set's typical
  label count) instead of threshold mode. That caps the suggestion list at something
  near 1.6 instead of 4.2, and every entry carries `above_threshold` so forced picks
  stay visible.
- Attach **`label_f1`** (`include_label_f1=true`) and let the client drop or flag labels
  whose own F1 is poor — at 300 labels the per-label spread is what matters, and the
  bundle already carries it.
- Consider a higher **`min_samples_per_label`** for wide vocabs: fewer, better-supported
  labels beat a long tail that mostly contributes false positives.

*Status:* the question is **closed** — the project stays on `LogisticRegression`. The
one-off benchmark script was removed after the decision rather than carried along as
dead exploration code; this section is the record. Its design is described above in
enough detail to rebuild it should the question ever be reopened. What remains in
`scripts/` is `benchmark_label_scaling.py`, which studies the deployed head only.

---

## Answered: does German preprocessing help? — No (2026-09-12)

The question came from an explanation, not from a metric: a leave-one-out explanation of
a subject prediction listed the connector **"und"** among the strongest words, which
looks like preprocessing too thin to be taken seriously. Two measurements, because the
first one changes what the second one means.

**1. What "und" actually weighs in a deployed model** (48 subjects, 26 450 rows):

| | value |
|---|---:|
| `idf("und")` | **1.46** (a content word, "basen": 7.16) |
| word feature "und", \|coefficient\| | 0.13 – 4.2 depending on the label |
| character 5-gram `" und "` (the isolated connector) | 1 feature, median 0.70 |
| 5-grams touching it (`"n und"`, `"und b"` …) | 19 features, median 2.55 |
| 5-grams with "und" inside other words (Gr**und**lagen, K**und**e) | 334 features, median 42.8 |

So the model does not lean on the connector: `idf` has already pushed it to the floor and
`max_df=0.95` drops anything that appears in more than 95 % of documents. What the
explanation measures is something else — it removes a word and re-predicts, and with
character n-grams that also destroys every 5-gram spanning the phrase ("Säuren **und**
Basen"). The drop says "the phrase is gone", and the connector is charged for it. That is
a property of leave-one-out on character features, not of the preprocessing.

**2. Whether the three classic normalisations buy anything anyway**
(🟢 `scripts/benchmark_preprocessing.py`, identical rows, split, seed, C grid and
threshold procedure; only the text the vectorizer sees changes):

| Variant | non-zeros/row | test macro F1 | vs baseline | test micro F1 | vectorize |
|---|---:|---:|---:|---:|---:|
| baseline | 278 | **0.7084** | — | 0.7917 | 9 s |
| stopwords (word analyzer only) | 255 | 0.7067 | −0.0017 | 0.7907 | 8 s |
| stemming (Snowball German) | 247 | 0.7017 | −0.0067 | 0.7882 | **116 s** |
| stopwords + stemming | 224 | 0.7060 | −0.0024 | 0.7904 | 120 s |
| lemmas (simplemma) | 271 | 0.7081 | −0.0003 | **0.7941** | 14 s |

**Practical reading:** none of them wins. Stopwords and lemmas land inside the noise band
this project uses elsewhere (±0.002 macro), stemming loses beyond it. The reason is in the
architecture: `char_wb` 5-grams already carry the morphology a stemmer would produce —
which is exactly why the 334 in-word "und" features above have real weight — and `idf`
plus `max_df` already handle function words. Stemming's one measurable gain is a 19 %
smaller matrix (278 → 224 non-zeros per row), but the feature caps are the cheaper lever
for that and are measured above.

And the cost is not only training: any preprocessing change has to run on **every**
`/predict` call as well, or model and query stop matching. Snowball costs 13× the
vectorization time here; lemmas would add a language-data dependency to a torch-free
image. Neither is worth −0.000 to −0.007 macro F1.

**Caveats:** one split, one seed — the differences below ~0.002 are not separable from
noise. The word cap stays saturated at 80 000 in every variant, so the possible second
benefit of stemming (more room under the cap on much larger corpora) is not measured
here; on 26 450 rows every variant hits the same cap. `snowballstemmer` and `simplemma`
are development-only and are not in `requirements.lock` or the image.

## Sources

- MetaClassify: this repository (`app/classifier.py`, `app/tuning.py`,
  `app/deploy.py`, `README.md`), plus this session's own training runs
  (`faecher_ai_cv5`, `bildungsstufe_ai_cv5`).
- Bayesian approach: [github.com/openeduhub/its-jointprobability](https://github.com/openeduhub/its-jointprobability),
  independently re-fetched and verified this session; cross-checked against
  an internal note from an earlier investigation of the same repository.
- Linear SVM: architectural characterization from scikit-learn's documentation of
  `LinearSVC` vs `LogisticRegression`, **plus a project-specific benchmark run on
  `data_30k_ai.csv` on 2026-07-25** — see
  [Answered: SVM measured](#answered-svm-measured-on-this-projects-data-2026-07-25).
  Sections written before that run are labelled ⚪ Theoretical; where they conflict
  with the measurement, the measurement wins. **The question is closed** (2026-07-25):
  the project stays on `LogisticRegression` and the SVM benchmark code was removed;
  this document is the surviving record of why.
- Label-count scaling: `scripts/benchmark_label_scaling.py` (deployed head only).
