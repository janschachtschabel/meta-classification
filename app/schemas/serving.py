"""What a prediction, an explanation or a correction may ask of a trained model."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from .common import SERVING_MODEL, OptionalFilter


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
    model_name: str = Field("default", description=SERVING_MODEL)


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
    model_name: str = Field("default", description=SERVING_MODEL)
    top_n_words: int = Field(
        5, ge=1, le=50,
        description=(
            "How many of the most influential words to list per predicted label; word importance "
            "covers the top 5 predicted labels at most."
        ),
    )


# A label URI, bounded per ITEM. `max_length` on a list field bounds the list, not what
# is in it, so the 100-entry caps below left the request as a whole unbounded — and this
# is the one write a readonly key can make, into a file that is deliberately never capped.
LabelUri = Annotated[str, Field(max_length=500)]


class FeedbackRequest(BaseModel):
    """One correction an editor made to a prediction.

    The bounds are the point: this is the only write a *readonly* key can make, so they
    are what stands between a key and an unbounded file on the volume — the list caps
    below bound how many, ``LabelUri`` bounds how big each one is.
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
    predicted: list[LabelUri] = Field(
        default_factory=list, max_length=100,
        description=(
            "Label URIs the model predicted for `text` (up to 100, each at most 500 characters), "
            "so a reader sees what was corrected, not only to what. Optional; not part of the export."
        ),
    )
    # Empty means "none of these apply" — a real thing to say, and not trainable.
    corrected: list[LabelUri] = Field(
        default_factory=list, max_length=100,
        description=(
            "The label URIs that are right for `text` (up to 100, each at most 500 characters): "
            "the row's `labels` in `GET /feedback/export`. Empty = none of the labels apply — "
            "recorded, but left out of the export, since a row without labels cannot train."
        ),
    )
    # Where it came from ("ui", a script, an integration), so a later merge can weigh
    # or filter by origin instead of guessing.
    source: str = Field(
        "ui", max_length=50,
        description="Where the correction came from (`ui`, a script, an integration), so it can be weighed by origin.",
    )
