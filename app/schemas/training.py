"""What a training run is asked for: the dataset, the text and label columns, and the
levers of the run itself."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .common import CSV_SEPARATOR, DATASET_NAME, LABEL_COLUMN, LABEL_FILTER, LABEL_SEPARATOR, OptionalFilter


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
    dataset_name: str = Field(..., examples=["data_30k.csv"], description=DATASET_NAME)
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
            LABEL_COLUMN + " An optional `<label_column>_DISPLAYNAME` column — same order, same "
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
        description=LABEL_FILTER,
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
    csv_separator: str = Field(";", min_length=1, max_length=1, description=CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=LABEL_SEPARATOR)
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
