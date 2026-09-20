"""What the request models share: the blank-to-null validator, and one wording per
concept — several requests take the same field, and two descriptions of one thing read
as two different things."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


# A blank string (common from Swagger form fields) collapses to None.
OptionalFilter = Annotated[str | None, BeforeValidator(_empty_to_none)]

# One wording per concept: several requests take the same field, and two descriptions of
# one thing read as two different things. Text only — every field keeps its own type,
# default and constraints at its declaration.
DATASET_NAME = (
    "File name of a dataset in the data directory, as `GET /datasets` lists it (add one with "
    "`POST /datasets/import`). A plain name: path characters are refused."
)
LABEL_COLUMN = "Column holding each row's labels, several per cell separated by `label_separator`."
CSV_SEPARATOR = "The CSV's field delimiter: exactly one character."
LABEL_SEPARATOR = (
    "Separator between the labels in one `label_column` cell, matched literally, at least one "
    "character. Labels are trimmed; empty ones are dropped."
)
LABEL_FILTER = (
    "Keep only labels containing this substring (case-sensitive), e.g. a vocabulary's URI "
    "prefix; rows left without a label are dropped. Blank or null = every label."
)
SERVING_MODEL = (
    "The model to classify with, as `GET /models` lists it. The default names a model called "
    "`default`, which exists only if one was trained or imported under that name (else 404)."
)
