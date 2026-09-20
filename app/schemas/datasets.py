"""Requests that work on a dataset or on a bundle: evaluate, analyze, validate, export."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import (
    CSV_SEPARATOR,
    DATASET_NAME,
    LABEL_COLUMN,
    LABEL_FILTER,
    LABEL_SEPARATOR,
    OptionalFilter,
)


class EvaluateRequest(BaseModel):
    """Score an existing model on a dataset. The model's own label space decides what
    can be scored, so no label settings are accepted here beyond a filter."""

    dataset_name: str = Field(..., description=DATASET_NAME)
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
            LABEL_COLUMN + " Read as the truth, in the model's own label space: labels it never "
            "learned are reported (`unknown_labels`), not scored."
        ),
    )
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=LABEL_SEPARATOR)
    label_filter: OptionalFilter = Field(None, description=LABEL_FILTER)
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
    dataset_name: str = Field(..., description=DATASET_NAME)
    text_columns: list[str] = Field(
        ...,
        description="Columns merged into the input text, for the text statistics. A column the CSV lacks is skipped.",
    )
    label_column: str = Field(..., description=LABEL_COLUMN)
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=LABEL_SEPARATOR)
    label_filter: OptionalFilter = Field(None, description=LABEL_FILTER)


class ValidateRequest(BaseModel):
    """Body of ``POST /datasets/{name}/validate`` — mirrors ``AnalyzeRequest``
    minus the fields the endpoint does not use (the dataset name travels in
    the path; validation has no label filter)."""

    text_columns: list[str] = Field(
        ..., description="Columns merged into the input text; each one the CSV lacks is reported in `errors`.",
    )
    label_column: str = Field(..., description=LABEL_COLUMN)
    # Single char only — see TrainRequest.csv_separator (regex/ReDoS guard).
    csv_separator: str = Field(";", min_length=1, max_length=1, description=CSV_SEPARATOR)
    label_separator: str = Field(",", min_length=1, description=LABEL_SEPARATOR)


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
