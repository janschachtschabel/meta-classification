# Plan — train on AI-marked rows, validate only on real ones (api_v3), 2026-09-19

**Status: requested 2026-09-19 ("plane die Anpassung und beginne mit der Umsetzung");
Phases A–D implemented the same day on branch `feat/ai-marked-rows`, each task test-first;
the UI checked in a preview with its own data and models directories. Two independent
reviews followed; their findings are fixed, and the thin-label threshold became the
user's choice (decision 7).**
Companion to data-prep's `docs/plan-2026-09-19-ai-provenance-prompts.md`, which writes the
marks this plan reads. Same conventions as `2026-09-11-training-memory.md`.

## Goal

A dataset prepared in data-prep says which rows an LLM wrote or touched. api_v3 must
**train on those rows but never validate, tune or score on them** — in k-fold mode and in
the holdout mode — stay **bit-identical for datasets without marks**, let the user **drop
the generated rows per training run**, and **say in the model** that AI data was involved.

## Context

Today api_v3 reads only the text and label columns (`dataset_load.load_dataset`, `usecols`),
so the marks never reach the pipeline. k-fold CV validates every row out-of-fold
(`tuning.cross_val_evaluate`), the holdout split draws val/test from all rows
(`data.three_way_split`). A balanced label lifted from 3 to 100 rows is therefore scored
mostly on text the same LLM wrote from the same examples — "AI measured with AI" — and the
F1 flatters itself by a margin nobody can see afterwards. The only clean number today is
`POST /models/{name}/evaluate` on a separate real dataset.

## The contract (written by data-prep, `app/refine/provenance.py`)

| column | non-blank means |
|---|---|
| `generated_for` | the row was written by an LLM (balancing; Runs exports after the data-prep change) |
| `example_for` | a REAL row shown to the generator as an example — its paraphrases are in the data |
| `enriched_fields` | a REAL row where an LLM filled fields (comma-separated column names) |

Blank, whitespace or a missing cell is no mark; a CSV without these columns has none.

## Decisions (made here; the owner can overrule each)

1. **Where the choice lives:** a per-run training option `synthetic_rows`, applied when the
   dataset is read (`load_dataset`) — not at upload. The file stays intact, so "with" and
   "without" can be trained from the same dataset and compared.
2. **Row roles:**

   | mark | `synthetic_rows="train"` (default) | `synthetic_rows="exclude"` |
   |---|---|---|
   | `generated_for` | trains, never validates | dropped at read time, before dedupe |
   | `example_for` | trains, never validates | an ordinary row — nothing derived from it is left |
   | `enriched_fields` | trains, never validates | trains, never validates |
   | same cleaned text as a train-only row | train-only too (text group) | same rule |

   "exclude" removes generated rows only. Enriched rows are real rows with AI-filled cells;
   the original cell is not recorded, so they cannot be restored — they stay, train-only.
3. **Labels no real row of the evaluated rows carries** ("not validated" — for k-fold every
   real row, for the holdout its test split, amended after review): trained as before (the
   min-samples threshold counts every row, so balancing CAN lift a label into training —
   wanted), threshold = the global one (the existing `n_pos == 0` rule in
   `thresholds.tune_threshold_columns`), **no F1** (omitted from `per_label_f1`, UI shows
   "–"), excluded from the macro averages, listed in the metadata.
4. **Too few real rows** (none at all — a pure Runs export pushed from data-prep — or fewer
   than `cv_folds`, or a holdout whose val/test part would be empty): fall back to today's
   evaluation over all rows and **flag it** (`validated_on: "all_rows"` + reason; the UI
   says the numbers are not meaningful and points to "Bewerten gegen…"). Refusing instead
   would end the synthetic-only workflow data-prep's Runs push exists for.
5. **`POST /models/{name}/evaluate` skips every marked row** and reports how many: an
   evaluation is a validation, and "never measure AI with AI" applies there as well.
6. Micro F1 keeps all label columns on the validated rows (a false positive of an
   unvalidated label is still a real error); macro P/R/F1 and `per_label_f1` cover the
   validated labels only.
7. **Thin labels — the owner's call** (added after review): a label with fewer real rows
   than `min_samples_per_label` reached training only through marked rows, so its own
   threshold is tuned on a handful of real rows. The review flagged those cuts; the owner
   decided the user chooses per run — `thin_label_threshold`, `own`
   (default: the behaviour before this option) or `global` (the threshold search leaves
   those columns at the global cut, `thresholds.tune_threshold_columns(keep_global=)`,
   threaded through `tuning.select_c` / `cross_val_evaluate`). The bundle lists
   `thin_labels` and the choice; the model detail warns under `own` and notes under
   `global`. Single-label tasks read no threshold, so the choice changes nothing there.

## Scope

In: reading the marks; the eligible-row split for both evaluation modes; scored labels; the
fallback; `synthetic_rows`; the `synthetic_data` metadata block; evaluate skipping marked
rows; training-form option and hint; model-detail notice; docs.

Out: per-label AI row counts in the label table; the pre-flight (`/datasets/analyze`)
counting marks; restoring enriched cells; recognising `source=synthetic` without
`generated_for` (legacy Runs exports — re-export in data-prep instead).

## Architecture

### Files

| file | change |
|---|---|
| `app/metrics.py` (new) | `compute_metrics`, `is_single_label`, `argmax_onehot` moved out of `tuning.py` (behaviour-preserving; `tuning.py` is at 324 lines and this feature adds to it) — then `scored=` added |
| `app/provenance.py` (new, leaf) | the mark contract, the per-block reading, `RowProvenance` (+ its metadata summary) |
| `app/dataset_load.py` | read the mark columns if present, drop generated rows under "exclude", the text-group rule, return `marks` |
| `app/data.py` | `three_way_split(..., train_only=None)` |
| `app/prepare.py` | carry `marks` through every row drop; decide validate rows / scored labels / fallback |
| `app/real_rows.py` (new, split out of `prepare.py` at the size guide) | the real-row split, the fallback, scored and thin labels |
| `app/thresholds.py` | `keep_global=`: thin labels at the global cut when the run chose it (decision 7) |
| `app/tuning.py` | `cross_val_evaluate(..., validate=None, scored=None)` |
| `app/deploy.py` | pass `validate` / `scored` through (a few lines; the file stays "left whole") |
| `app/training.py` | `synthetic_data` block, evaluation text |
| `app/schemas.py`, `app/routes/training.py` | `synthetic_rows` field, `_REQ_KEYS` |
| `app/evaluate.py` | skip marked rows, count them |
| `app/static/ui/{index.html,training.js,model-detail.js,strings-de.js,strings-en.js}` | option, hint, notice |
| `docs/ui-guide.md`, `CHANGELOG.md` | docs |

### Interfaces

```python
# app/provenance.py
GENERATED_FOR, EXAMPLE_FOR, ENRICHED_FIELDS = "generated_for", "example_for", "enriched_fields"
MARK_COLUMNS = (GENERATED_FOR, EXAMPLE_FOR, ENRICHED_FIELDS)
GENERATED, EXAMPLE, ENRICHED, SAME_TEXT = 1, 2, 4, 8          # int8 bitmask per row
SYNTHETIC_MODES = ("train", "exclude")

def _marked(column: pd.Series) -> np.ndarray   # private, per row: blank, whitespace or a missing cell = no mark
def block_marks(frame: pd.DataFrame, *, mode: str) -> np.ndarray   # int8 per row; EXAMPLE only in "train"

@dataclass
class RowProvenance:
    marks: np.ndarray              # int8 per kept row
    mode: str
    excluded_generated: int
    validate: np.ndarray | None    # bool per row that may validate; None = fallback to all rows
    scored: np.ndarray | None      # bool per class with a real row; None = all classes
    fallback: str | None           # why the metrics include AI-marked rows
    def train_only(self) -> np.ndarray
    def summary(self, classes: list[str], *, scored_rows: int) -> dict

# app/dataset_load.py
LoadedData.marks: list[int] | None          # None: the CSV has no mark column
LoadedData.excluded_generated: int
load_dataset(..., synthetic_rows: str = "train")

# app/data.py
three_way_split(n, *, val_size, test_size, seed, y=None, train_only: np.ndarray | None = None)

# app/prepare.py
Prepared.provenance: RowProvenance | None = None

# app/tuning.py
cross_val_evaluate(..., validate: np.ndarray | None = None, scored: np.ndarray | None = None)

# app/metrics.py
compute_metrics(y_true, proba, classes, global_threshold, per_label, task_type="multilabel",
                *, scored: np.ndarray | None = None)
```

`schemas.TrainRequest.synthetic_rows: Literal["train", "exclude"] = "train"` and
`thin_label_threshold: Literal["own", "global"] = "own"` (decision 7).

### Data model — the `synthetic_data` block (only when the dataset had marked rows)

```json
"synthetic_data": {
  "mode": "train",
  "generated_rows": 120, "example_rows": 16, "enriched_rows": 40,
  "excluded_generated_rows": 0,
  "train_only_rows": 178,
  "validated_on": "real_rows",
  "scored_rows": 812,
  "labels_not_validated": ["http://…/380"],
  "fallback": null,
  "thin_labels": ["http://…/380"],
  "thin_label_threshold": "own"
}
```

## Non-functional

- Memory: one int8 per row plus a set of the marked rows' texts — no second copy of the data.
- Security: mark cells are read, never evaluated; names in the metadata come from the label
  space the run already stores.
- i18n: every new UI string in `strings-de.js` and `strings-en.js`.
- Accessibility: the new select has a `<label>`; notices are text, not colour.

## Risks

| risk | mitigation |
|---|---|
| an unmarked dataset changes its numbers | legacy code path when there are no marks; a test trains the same data with and without all-blank mark columns and requires identical C, thresholds and metrics; an A/B run on a real dataset before and after |
| heavy enrichment leaves few real rows to validate on | reported (`scored_rows`, `train_only_rows`); the fallback covers "too few" |
| row drops misalign `marks` with `texts` | every drop in `prepare` is a mask/index applied to both, tested through `_drop_unlearnable` |

## Tasks

Phase A — foundation (no behaviour change)
- Step 0: invoke /better-coding-workflow
- A1 refactor: move `compute_metrics`, `is_single_label`, `_argmax_onehot` to `app/metrics.py`; update imports (`tuning`, `deploy`, `evaluate`, tests). Verify: full suite green, `git diff` shows moves only.
- A2 `app/provenance.py` + `tests/test_provenance.py`: blank/NaN/whitespace are no marks; missing columns → zeros; EXAMPLE only in "train"; `summary()` shape.

Phase B — reading the marks
- Step 0: invoke /better-coding-workflow
- B1 `load_dataset` (tests first, `tests/test_dataset_load_marks.py`): no mark column → `marks is None`, texts identical; each mark → its bit; a marked duplicate dropped by dedupe marks the kept row SAME_TEXT; "exclude" drops generated rows before dedupe (a real duplicate after a generated first occurrence survives) and ignores `example_for`; result independent of `chunk_rows`.

Phase C — validation on real rows
- Step 0: invoke /better-coding-workflow
- C1 `three_way_split(train_only=)`: None → identical arrays; else val/test ⊆ real rows, every train-only row in train.
- C2 `prepare_data`: marks follow `row_keep` and `_drop_unlearnable`; `validate`, `scored`, fallback (no real rows; fewer than k; empty val/test).
- C3 `metrics.compute_metrics(scored=)`: None → identical; else macro/per-label over scored columns, micro over all.
- C4 `cross_val_evaluate(validate=, scored=)`: None → identical folds; else a train-only row is never in a held-out block and always in the training part; OOF, C, thresholds, metrics from validate rows.
- C5 `deploy` wiring + end-to-end test: the same dataset with and without all-blank mark columns → identical results; a marked dataset → `per_label_f1` lacks the unvalidated label, `scored_rows` = real rows.

Phase D — option, metadata, evaluate, UI, docs
- Step 0: invoke /better-coding-workflow (+ /better-coding-frontend for D4–D5)
- D1 `synthetic_rows` in `TrainRequest` + `_REQ_KEYS`; 422 on other values; "exclude" shrinks `n_samples`.
- D2 `_build_metadata`: `synthetic_data` block + evaluation text; absent for unmarked data.
- D3 evaluate skips marked rows, `ai_marked_rows_skipped` in the record.
- D4 training form: the select and its note whenever the dataset has a mark column — the note
  says marked rows never validate, which holds for `enriched_fields` alone too — with
  "leave out" disabled when there is no `generated_for` column.
- D5 model detail: "KI-Daten" row, fallback warning, per-label note.
- D6 docs: `docs/ui-guide.md`, `CHANGELOG.md`, this plan's status.

Phase E — review and verify
- /better-coding-review on the branch diff; /better-coding-verify: `python -m pytest tests -q`,
  `python -m ruff check app tests scripts`, `python -m mypy app --config-file pyproject.toml`;
  UI checked in the preview; an A/B training on a real unmarked dataset (identical
  metrics.json apart from timestamps/durations).

## Acceptance criteria

1. Unmarked dataset: C, thresholds, metrics identical to `main` (test + A/B run).
2. Marked dataset, k-fold and holdout: no train-only row is ever scored (test inspects the
   held-out blocks and the test split); metrics carry `scored_rows` = real rows.
3. `synthetic_rows="exclude"`: generated rows absent from training, example rows validate.
4. A label with only train-only rows: trained, no F1, listed in `labels_not_validated`.
5. No real rows: training completes, `validated_on: "all_rows"`, UI warns.
6. Evaluate on a marked dataset: marked rows skipped and counted.
7. Gate green: pytest, ruff, mypy.
