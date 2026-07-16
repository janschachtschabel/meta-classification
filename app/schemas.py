"""Pydantic request models for the API (responses are plain dicts)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, field_validator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


# A blank string (common from Swagger form fields) collapses to None.
OptionalFilter = Annotated[str | None, BeforeValidator(_empty_to_none)]


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., examples=["data_30k.csv"])
    model_name: str = Field(..., examples=["taxonid"])
    text_columns: list[str] = Field(
        ...,
        description="Columns merged into the input text (title + description + keywords recommended).",
        examples=[[
            "properties.cclom:title",
            "properties.cclom:general_description",
            "properties.cclom:general_keyword",
        ]],
    )
    label_column: str = Field(..., examples=["properties.ccm:taxonid"])
    optimize_parameters: str = Field("auto", description="Quality/effort profile: fast | auto | thorough")
    task_type: str | None = Field(
        None, examples=["auto"],
        description="Override task type: 'multilabel' | 'multiclass' | 'binary'. None/'auto' = auto-detect.",
    )
    label_filter: OptionalFilter = Field(
        None, examples=["http://w3id.org/openeduhub/vocabs/discipline/"],
        description="Keep only labels containing this substring (optional).",
    )
    min_samples_per_label: int | None = Field(
        None, ge=1, examples=[20],
        description=(
            "Minimum number of tagged samples a label must have to be included in training; "
            "rarer labels are dropped. null = auto (scales with dataset size, ~20 for typical sets)."
        ),
    )
    cv_folds: int | None = Field(
        None, ge=0, le=20, examples=[5],
        description=(
            "Evaluation mode: 0 = classic train/val/test split, >= 2 = k-fold cross-validation "
            "(every row trains AND validates via out-of-fold metrics; the deployed model is fit "
            "on 100% of the data). null = config default (split.cv_folds)."
        ),
    )
    csv_separator: str = ";"
    label_separator: str = ","

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
    texts: list[Annotated[str, Field(max_length=100_000)]] = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(
        None, ge=0, le=1000,
        description=(
            "null (default) = the model DECIDES: multilabel returns every label above its "
            "tuned per-label threshold, multiclass/binary the single best label. "
            "N = RANKING: exactly the N most probable labels regardless of thresholds "
            "(each carries `above_threshold` so forced entries stay distinguishable). "
            "0 = ranking of the training set's typical label count."
        ),
    )
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    label_filter: OptionalFilter = None
    include_baseline_diff: bool = Field(
        False,
        description=(
            "Attach `baseline_diff` (confidence minus the model's empty-text prediction) to every "
            "prediction: separates what the text contributes from the label's base rate."
        ),
    )


class PredictRequest(_PredictOptions):
    model_name: str = "default"


class MultiPredictRequest(_PredictOptions):
    model_names: list[str] = Field(
        ..., min_length=1, max_length=5,
        description=(
            "Models (= target fields) to classify with in one call, e.g. subjects + resource type. "
            "Capped at 5: each model may need a cold load into the LRU cache."
        ),
    )


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model_name: str = "default"
    top_n_words: int = Field(5, ge=1, le=50)


class AnalyzeRequest(BaseModel):
    dataset_name: str
    text_columns: list[str]
    label_column: str
    csv_separator: str = ";"
    label_separator: str = ","
    label_filter: OptionalFilter = None


class ExportRequest(BaseModel):
    generate_share_url: bool = False
    expires_hours: int = Field(24, ge=1, le=168)
