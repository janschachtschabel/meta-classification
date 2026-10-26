# Design: corrections you can see and reuse, model names you can fix, the AI share up front

## Goal
Make the correction loop usable from the admin UI (see, export, import, turn into a dataset),
let a model's name be corrected and a display name be set, and show how much of a dataset an
LLM wrote before anyone trains on it.

## Context
Corrections are recorded (`POST /feedback`, append-only JSONL) and exportable as a training CSV
(`GET /feedback/export`, admin) -- but only through the API: the UI can record one, never see,
export or reuse them. A model's name is its directory name and nothing else; there is no way to
fix a typo after training or import. The share of AI-written rows is recorded in a bundle only
after training (`synthetic_data`); the training form offers "train / leave out" without saying
how many rows the choice is about. Share links can already be listed and revoked (`GET /share`,
`DELETE /share/{id}`, the link tables in the Models and Datasets tabs) -- nothing to build there.

## Scope
In scope:
- Corrections: list (paged, per model), full backup export (JSONL) and import of it (exact
  duplicates skipped), the existing training CSV export, "save as dataset", a Corrections tab.
- Models: rename (refused while a run or import holds either name; the old name's share links
  are revoked, the UI warns with their count first); a display name in the documentation block,
  unique across models and never another model's name; the documentation shown in the detail.
- Datasets: the number of AI-marked rows in `GET /datasets/{name}`, shown in the training form
  with its meaning (trains, never validates) next to the existing train/leave-out choice.

Out of scope (assessment in the summary): renaming datasets, per-dataset info cards, packaging a
dataset with its card for sharing; deleting single corrections (the store is append-only by
design -- training data, not a log); faster column loading.

## Approach
Extend what exists instead of adding parallel paths:
- Corrections stay one append-only JSONL. The backup format IS that format (one record per
  line, `recorded_at` kept), so an import is an append with validation and a duplicate check.
  "Save as dataset" writes the existing training CSV through the dataset import's publish path
  (staging file + `os.link`, so an existing name is refused, never replaced).
- Rename = one directory rename under the registry's disk lock, cache eviction of every
  spelling, `revoke_for("model", old)`. Revoking is not optional: a link resolves BY NAME, so a
  link to the old name would serve whatever model is later given that name.
- Display name lives in `metrics.json` `info` (the documentation block `PUT /models/{name}/info`
  already edits), so export, import and the model card carry it without a new file.
- AI share: count the mark columns with the reader training uses (`provenance`), in the same
  dataset-info call the training form already waits for.

Rejected: a separate rename endpoint per field (one documentation PUT already exists); a
display name in `config.json` (machine data the loader checks; a human field does not belong
there); a second corrections store for imports (two files to keep in step).

## Global constraints (CLAUDE.md)
English code/comments/API texts; UI strings only in strings-de.js / strings-en.js; routes thin,
logic in core modules; `safe_name` for every user-supplied name; files under ~300 lines, split by
responsibility; tests first; no new dependency; single worker.

## Architecture
### Files
- `app/feedback.py` (242 lines): add `page(model, limit, offset)` (newest first) and `import_lines(lines)`.
  If it passes ~300 lines, the reading side moves to `app/feedback_read.py`.
- `app/routes/feedback.py`: `GET /feedback` (admin, paged, `model` filter), `GET /feedback/export?format=csv|jsonl`,
  `POST /feedback/import` (admin, multipart JSONL, size-capped), `POST /feedback/dataset` (admin, `{dataset_name}`).
- `app/registry.py` (595): `rename(old, new)`; `app/routes/models.py`: `POST /models/{name}/rename` (admin).
- `app/schemas/training.py` `ModelInfo`: `display_name` (<=100). Uniqueness checked in the info route.
- `app/dataset_stats.py` + `routes/datasets.py`: `ai_rows` in `GET /datasets/{name}`.
- UI: new `corrections.js` (the tab), `models.js`/`model-detail.js` (rename, display name, documentation
  shown), `training.js` (AI share line), strings DE/EN, `index.html` (tab + markup).

### Interfaces
- `GET /feedback?model=&limit=50&offset=0` -> `{"total": int, "items": [{"recorded_at", "model_name", "text", "predicted": [...], "corrected": [...], "source"}]}`
- `GET /feedback/export?format=jsonl` -> the store as lines; `format=csv` (default) as today.
- `POST /feedback/import` (file) -> `{"imported": n, "duplicates": n, "invalid": n}`
- `POST /feedback/dataset` `{"dataset_name": "..."}` -> `{"dataset_name", "rows"}`; 409 if the name exists, 400 if no correction carries labels.
- `POST /models/{name}/rename` `{"new_name": "..."}` -> `{"name": new, "revoked_links": n}`; 404/409.
- `PUT /models/{name}/info` gains `display_name`; 409 if another model uses it as display or real name.
- `GET /datasets/{name}` gains `"ai_rows": {"generated": n, "enriched": n, "example": n}` when mark columns exist.

## Non-functional
- Security: listing, export, import, dataset and rename are admin; import validates every line
  against the same schema as `POST /feedback` and is capped by `max_upload_mb`; names via `safe_name`.
- Accessibility/i18n: tables with headers, labelled controls, loading/empty/error states, both languages.

## Risks
- Rename while a share link is downloading: the rename waits for the disk lock like a publish does.
- A corrections file of 1 GB: listing reads newest-first by scanning; acceptable for an admin view, paged.

## Open questions
None blocking. Interpretation to confirm: "ein Label um Anzeigenamen gesondert zu ändern" read as
a model display name separate from the technical name (label display names are already editable
through `PUT /label-names`).
