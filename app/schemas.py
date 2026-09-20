"""Pydantic request models for the API (the fixed-shape responses are in ``responses``,
the rest are plain dicts)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, field_validator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


# A blank string (common from Swagger form fields) collapses to None.
OptionalFilter = Annotated[str | None, BeforeValidator(_empty_to_none)]

# One wording per concept: several requests take the same field, and two descriptions of
# one thing read as two different things. Text only — every field keeps its own type,
# default and constraints at its declaration.
_DATASET_NAME = (
    "File name of a dataset in the data directory, as `GET /datasets` lists it (add one with "
    "`POST /datasets/import`). A plain name: path characters are refused."
)
_LABEL_COLUMN = "Column holding each row's labels, several per cell separated by `label_separator`."
_CSV_SEPARATOR = "The CSV's field delimiter: exactly one character."
_LABEL_SEPARATOR = (
    "Separator between the labels in one `label_column` cell, matched literally, at least one "
    "character. Labels are trimmed; empty ones are dropped."
)
_LABEL_FILTER = (
    "Keep only labels containing this substring (case-sensitive), e.g. a vocabulary's URI "
    "prefix; rows left without a label are dropped. Blank or null = every label."
)
_SERVING_MODEL = (
    "The model to classify with, as `GET /models` lists it. The default names a model called "
    "`default`, which exists only if one was trained or imported under that name (else 404)."
)


class ModelInfo(BaseModel):
    """What only the person training a model knows, carried inside the bundle.

    Everything else in a model's metadata is measured by the pipeline. These four are
    assertions by a human, kept in their own block so a reader can tell the two apart
    — and they exist for one moment in particular: handing the model to someone else,
    where `dataset: "data_300k.csv"` says nothing about where the data came from.

    Deliberately NOT here: the label vocabulary. It follows from the class URIs
    (`GET /models/{name}` reports `label_vocabulary`), so it cannot be typed in wrong.

    All fields optional and length-bounded: this is free text from a client that ends
    up rendered in the admin UI and shipped inside an exportable archive.
    """

    model_config = ConfigDict(extra="forbid")  # an unknown key is a typo worth a 422

    author: str | None = Field(
        None, max_length=200, examples=["Redaktion WLO <redaktion@example.org>"],
        description="Who trained the model, and how to reach them.",
    )
    description: str | None = Field(
        None, max_length=4_000,
        examples=["Subject classifier for school material. Not for grading learners."],
        description="What the model is for — and, more usefully, what it is NOT for.",
    )
    data_source: str | None = Field(
        None, max_length=1_000, examples=["WLO prod export 2026-07-26, school disciplines only"],
        description=(
            "Where the training data came from. The bundle records the file NAME, which "
            "is meaningless on another machine; this is the provenance behind it."
        ),
    )
    license: str | None = Field(
        None, max_length=200, examples=["CC BY-SA 4.0"],
        description="Terms the model may be used under, for models that leave the house.",
    )


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., examples=["data_30k.csv"], description=_DATASET_NAME)
    model_name: str = Field(
        ..., examples=["taxonid"],
        description=(
            "Name the new model is stored and served under (`model_name` in `/predict`): at most "
            "100 characters, no path characters. Refused (409) while a model, a queued run or the "
            "running one already has it."
        ),
    )
    text_columns: list[str] = Field(
        ...,
        description=(
            "Columns merged into the input text (title + description + keywords recommended). "
            "A column the CSV lacks is skipped silently — `POST /datasets/{name}/validate` "
            "reports it."
        ),
        examples=[[
            "properties.cclom:title",
            "properties.cclom:general_description",
            "properties.cclom:general_keyword",
        ]],
    )
    # Values capped at 10: every repeat concatenates another full copy of that
    # column across all rows, so an unbounded multiplier is a memory lever.
    text_column_weights: dict[str, Annotated[int, Field(ge=1, le=10)]] | None = Field(
        None,
        examples=[{"properties.cclom:title": 2, "properties.cclom:general_keyword": 2}],
        description=(
            "How often each text column is repeated in the training text. "
            'Format: one entry per column you want to boost, e.g. '
            '`{"properties.cclom:title": 2, "properties.cclom:general_keyword": 2}` — '
            "columns you leave out stay at 1 (unchanged), so only list what you boost. "
            "Allowed values 1-10; every key must appear in `text_columns`.\n\n"
            "**Defaults:** `null` applies `preprocessing.text_column_weights` from "
            "config.yaml (shipped as title + keywords at 2x, the measured optimum), "
            "narrowed to the columns you are training on. Send `{}` to train unweighted. "
            "`GET /train/profiles` reports the active default.\n\n"
            "Why: a title or keyword list carries far more signal per word than a long "
            "description, but the description supplies more words and drowns it out; "
            "repeating a field gives it that weight back. `sublinear_tf` damps repetition "
            "logarithmically, so 2 is worth ~1.7x, not 2x.\n\n"
            "**Training-time only:** the trained model expects input built the same way, so "
            "assemble the text you send to /predict with the same repetitions (the weights "
            "are recorded in the model's metadata, see GET /models/{name})."
        ),
    )
    label_column: str = Field(
        ..., examples=["properties.ccm:taxonid"],
        description=(
            _LABEL_COLUMN + " An optional `<label_column>_DISPLAYNAME` column — same order, same "
            "separator — supplies readable names."
        ),
    )
    optimize_parameters: str = Field(
        "auto",
        description=(
            "Quality/effort profile, cheapest first: fast | auto | best — or any other profile "
            "config.yaml defines (`GET /train/profiles`). An unknown name answers 400."
        ),
    )
    task_type: str | None = Field(
        None, examples=["auto"],
        description="Override task type: 'multilabel' | 'multiclass' | 'binary'. None/'auto' = auto-detect.",
    )
    label_filter: OptionalFilter = Field(
        None, examples=["http://w3id.org/openeduhub/vocabs/discipline/"],
        description=_LABEL_FILTER,
    )
    min_samples_per_label: int | None = Field(
        20, ge=1, examples=[20],
        description=(
            "Minimum number of tagged samples a label must have to be included in training; "
            "rarer labels are dropped. Declared (not auto-scaled) because dropping labels is a "
            "decision worth seeing: 20 suits datasets of a few thousand rows and up, but a small "
            "dataset needs a lower value or training aborts with 'not enough data'. "
            "null = auto (scales with dataset size: 2 / 5 / 20 / 35)."
        ),
    )
    cv_folds: int | None = Field(
        None, ge=0, le=20, examples=[3],
        description=(
            "Evaluation mode: 0 = classic train/val/test split (the deployed model is fit on "
            "train+val, i.e. the test share is never learned from), >= 2 = k-fold "
            "cross-validation (every row trains AND validates via out-of-fold metrics — "
            "AI-marked rows only train, see `synthetic_rows`; the "
            "deployed model is fit on 100% of the data). k only controls how much data the "
            "evaluation models see (67% at k=3, 80% at k=5), so a lower k is slightly "
            "pessimistic, not less honest. null = the profile's own setting "
            "(fast: 0, auto: 3, best: 5), which falls back to split.cv_folds."
        ),
    )
    synthetic_rows: Literal["train", "exclude"] = Field(
        "train",
        description=(
            "What this run does with rows an LLM wrote, as data-prep marks them "
            "(`generated_for`). `train` (default): they train, but never validate — the "
            "folds, the validation and the test split are drawn from real rows only, so "
            "the metrics measure the model on real data. `exclude`: they are left out when "
            "the dataset is read, and the real rows shown to the generator as examples "
            "(`example_for`) validate again. Rows whose fields an LLM completed "
            "(`enriched_fields`) train but never validate in either mode. Too few real rows "
            "to validate on, and the metrics include the marked rows after all "
            "(`synthetic_data.fallback` says why). A dataset "
            "without these columns trains exactly as before. The bundle's `synthetic_data` "
            "block records what was done."
        ),
    )
    thin_label_threshold: Literal["own", "global"] = Field(
        "own",
        description=(
            "Where a thin label is cut: one that reaches `min_samples_per_label` only through "
            "AI-marked rows, so fewer of its rows can validate it than that minimum. `own` "
            "(default): the per-label threshold tuned on those few real rows — tailored to "
            "the label, but it can swing. `global`: the global threshold — steadier, but not "
            "fitted to the label. Your decision: the bundle's `synthetic_data` block lists the "
            "thin labels (`thin_labels`) and the choice made. No effect on a dataset without "
            "data-prep's marks, nor on binary/multiclass tasks (argmax reads no threshold)."
        ),
    )
    # Upper bound 2_000_000: the vocabulary caps are the main RAM lever (the head
    # holds n_labels x n_features float32, the vectorizer the vocabulary itself), so
    # an unbounded value is an out-of-memory request, not a quality setting.
    max_word_features: int | None = Field(
        None, ge=1_000, le=2_000_000, examples=[80_000],
        description=(
            "Word-n-gram vocabulary cap for THIS run. null = the profile's value, else "
            "`APIV3_TFIDF_MAX_WORD_FEATURES` (default 80000). Raising it can recover signal "
            "when the cap is saturated (compare `tfidf.n_features` against "
            "`tfidf.max_word_features + max_char_features` in the model metadata: equal "
            "means the vocabulary was truncated) — at a proportional cost in RAM, bundle "
            "size and cold-load time."
        ),
    )
    max_char_features: int | None = Field(
        None, ge=1_000, le=2_000_000, examples=[120_000],
        description=(
            "Character-n-gram vocabulary cap for THIS run. null = the profile's value, else "
            "`APIV3_TFIDF_MAX_CHAR_FEATURES` (default 120000). Ignored by word-only profiles "
            "(`use_char: false`, e.g. `fast`)."
        ),
    )
    # Exactly one character: pandas parses a multi-char sep as a REGEX (python
    # engine) — a crafted one can backtrack catastrophically (ReDoS), and in
    # /train it would hang the training thread outside any stop checkpoint.
    csv_separator: str = Field(";", min_length=1, max_length=1, description=_CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=_LABEL_SEPARATOR)
    info: ModelInfo | None = Field(
        None,
        description=(
            "Optional documentation to store in the bundle: author, purpose and limits, "
            "data provenance, license. Editable afterwards via `PUT /models/{name}/info`, "
            "so a typo never costs a retrain."
        ),
    )

    @field_validator("text_column_weights")
    @classmethod
    def _weights(cls, value: dict | None, info: ValidationInfo) -> dict | None:
        """Reject weights for columns that are not being trained on.

        Silently ignoring them (the loader would) hides a typo: the user believes a
        field is boosted, the model was never told, and nothing in the result says so.
        """
        # "text_columns" missing from info.data means IT failed validation; there is
        # nothing to compare against, so stay quiet rather than stack a second,
        # unfounded error on top of the real one.
        if not value or "text_columns" not in info.data:
            return value
        unknown = sorted(set(value) - set(info.data["text_columns"]))
        if unknown:
            raise ValueError(f"text_column_weights names columns not in text_columns: {unknown}")
        return value

    @field_validator("task_type", mode="before")
    @classmethod
    def _tt(cls, value: object) -> object:
        if value in (None, "", "auto"):
            return None
        if value not in ("multilabel", "multiclass", "binary"):
            raise ValueError("task_type must be one of: multilabel, multiclass, binary, auto")
        return value

    @field_validator("cv_folds")
    @classmethod
    def _cv(cls, value: int | None) -> int | None:
        if value == 1:  # neither a split nor a cross-validation
            raise ValueError("cv_folds must be 0 (holdout split) or >= 2 (k-fold cross-validation)")
        return value


class _PredictOptions(BaseModel):
    """Options shared by single- and multi-model prediction requests."""

    # Bounded at the trust boundary: cap the batch size and per-text length so a
    # single request cannot exhaust the single worker's RAM/CPU (predict builds
    # n_texts x n_labels objects). Empty list -> 422.
    texts: list[Annotated[str, Field(max_length=100_000)]] = Field(
        ..., min_length=1, max_length=1000,
        description=(
            "The texts to classify: 1-1000 per request, each at most 100,000 characters. Build "
            "each the way the model's training text was built — the same fields, repeated by its "
            "`text_column_weights` (`GET /models/{name}`); markup is cleaned as in training."
        ),
    )
    top_k: int | None = Field(
        None, ge=0, le=1000,
        description=(
            "null (default) = the model DECIDES: multilabel returns every label above its "
            "tuned per-label threshold, multiclass/binary the single best label. "
            "N = RANKING: exactly the N most probable labels regardless of thresholds "
            "(for a multilabel model each carries `above_threshold`, so forced entries stay "
            "distinguishable). 0 = ranking of the training set's typical label count."
        ),
    )
    threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description=(
            "One confidence cut (0-1) for every label, replacing the model's tuned per-label "
            "thresholds for this request; null (default) = the tuned ones. Multilabel only: "
            "binary/multiclass decide by argmax. In ranking mode (`top_k` set) it only decides "
            "`above_threshold`."
        ),
    )
    label_filter: OptionalFilter = Field(
        None,
        description=(
            "Return only labels whose URI contains this substring (case-sensitive). Applied "
            "first, so a ranking — and a binary/multiclass model's single answer — is taken "
            "among the matching labels. Blank or null = every label."
        ),
    )
    include_baseline_diff: bool = Field(
        False,
        description=(
            "Attach `baseline_diff` (confidence minus the model's empty-text prediction) to every "
            "prediction: separates what the text contributes from the label's base rate."
        ),
    )
    include_label_f1: bool = Field(
        False,
        description=(
            "Attach `label_f1` (this label's F1 from the training evaluation) to every prediction. "
            "Confidence says how sure the model is HERE, `label_f1` how much that is worth: a 0.95 "
            "on a label that only scores 0.60 overall is worth a human look. Left out (not null) "
            "where the bundle has no F1 for the label: one no real row could validate "
            "(`synthetic_data.labels_not_validated` in `GET /models/{name}`), or any label of a "
            "bundle trained before per-label F1 was recorded."
        ),
    )


class PredictRequest(_PredictOptions):
    model_name: str = Field("default", description=_SERVING_MODEL)


class MultiPredictRequest(_PredictOptions):
    model_names: list[str] = Field(
        ..., min_length=1, max_length=5,
        description=(
            "Models (= target fields) to classify with in one call, e.g. subjects + resource type. "
            "Capped at 5: each model may need a cold load into the LRU cache."
        ),
    )


class ExplainRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=100_000,
        description=(
            "The one text to classify and explain, built like a `/predict` text. Word importance "
            "reads its first 60 words."
        ),
    )
    model_name: str = Field("default", description=_SERVING_MODEL)
    top_n_words: int = Field(
        5, ge=1, le=50,
        description=(
            "How many of the most influential words to list per predicted label; word importance "
            "covers the top 5 predicted labels at most."
        ),
    )


class FeedbackRequest(BaseModel):
    """One correction an editor made to a prediction.

    The bounds are the point: this is the only write a *readonly* key can make, so they
    are what stands between a key and an unbounded file on the volume.
    """

    text: str = Field(
        ..., min_length=1, max_length=100_000,
        description="The text that was classified; it becomes the `text` column of `GET /feedback/export`.",
    )
    model_name: str = Field(
        ...,
        description=(
            "The model whose prediction is corrected. Recorded with the correction, not looked "
            "up; path characters are refused."
        ),
    )
    # What the model said, so a later reader can see WHAT was corrected, not only to
    # what. Optional: a correction is still a correction if nobody recorded the guess.
    predicted: list[str] = Field(
        default_factory=list, max_length=100,
        description=(
            "Label URIs the model predicted for `text` (up to 100), so a reader sees what was "
            "corrected, not only to what. Optional; not part of the export."
        ),
    )
    # Empty means "none of these apply" — a real thing to say, and not trainable.
    corrected: list[str] = Field(
        default_factory=list, max_length=100,
        description=(
            "The label URIs that are right for `text` (up to 100): the row's `labels` in "
            "`GET /feedback/export`. Empty = none of the labels apply — recorded, but left out "
            "of the export, since a row without labels cannot train."
        ),
    )
    # Where it came from ("ui", a script, an integration), so a later merge can weigh
    # or filter by origin instead of guessing.
    source: str = Field(
        "ui", max_length=50,
        description="Where the correction came from (`ui`, a script, an integration), so it can be weighed by origin.",
    )


class EvaluateRequest(BaseModel):
    """Score an existing model on a dataset. The model's own label space decides what
    can be scored, so no label settings are accepted here beyond a filter."""

    dataset_name: str = Field(..., description=_DATASET_NAME)
    text_columns: list[str] = Field(
        ..., min_length=1, max_length=20,
        description=(
            "Columns merged into the input text — the ones the model was trained on "
            "(`metadata.text_columns` in `GET /models/{name}`). A column the CSV lacks is skipped."
        ),
    )
    label_column: str = Field(
        ...,
        description=(
            _LABEL_COLUMN + " Read as the truth, in the model's own label space: labels it never "
            "learned are reported (`unknown_labels`), not scored."
        ),
    )
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=_CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=_LABEL_SEPARATOR)
    label_filter: OptionalFilter = Field(None, description=_LABEL_FILTER)
    # The model was fit on text assembled a particular way; scoring it on text
    # assembled differently measures a distribution it was not tuned on.
    text_column_weights: dict[str, int] | None = Field(
        None,
        description=(
            "How often each text column is repeated in the evaluation text, as in `/train` "
            '(e.g. `{"properties.cclom:title": 2}`; unlisted columns once). Omitted or null '
            "(default) takes the model's own weights (`metadata.text_column_weights` in "
            "`GET /models/{name}`), narrowed to `text_columns`, so it is scored on text built "
            "the way it was trained; an explicit `{}` repeats nothing. Values below 1 count as "
            "1; keys not in `text_columns` are ignored. The recorded evaluation says which "
            "weights it used."
        ),
    )


class AnalyzeRequest(BaseModel):
    dataset_name: str = Field(..., description=_DATASET_NAME)
    text_columns: list[str] = Field(
        ...,
        description="Columns merged into the input text, for the text statistics. A column the CSV lacks is skipped.",
    )
    label_column: str = Field(..., description=_LABEL_COLUMN)
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=_CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=_LABEL_SEPARATOR)
    label_filter: OptionalFilter = Field(None, description=_LABEL_FILTER)


class ValidateRequest(BaseModel):
    """Body of ``POST /datasets/{name}/validate`` — mirrors ``AnalyzeRequest``
    minus the fields the endpoint does not use (the dataset name travels in
    the path; validation has no label filter)."""

    text_columns: list[str] = Field(
        ..., description="Columns merged into the input text; each one the CSV lacks is reported in `errors`.",
    )
    label_column: str = Field(..., description=_LABEL_COLUMN)
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=_CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=_LABEL_SEPARATOR)


class ExportRequest(BaseModel):
    generate_share_url: bool = Field(
        False,
        description=(
            "false (default): the response is the file itself. true: an expiring share link "
            "instead (`share_url`, `share_id`, `expires_at`) — `GET /share/{id}` then serves the "
            "file to anyone holding the link, without an API key."
        ),
    )
    expires_hours: int = Field(
        24, ge=1, le=168,
        description="Lifetime of the share link in hours, 1-168 (one week). Read only with `generate_share_url=true`.",
    )
