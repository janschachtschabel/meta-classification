"""What the request models share: the blank-to-null validator, and one wording per
concept — several requests take the same field, and two descriptions of one thing read
as two different things."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BeforeValidator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


# A blank string (common from Swagger form fields) collapses to None.
OptionalFilter = Annotated[str | None, BeforeValidator(_empty_to_none)]


def separator_problem(value: str) -> str | None:
    """Why ``value`` cannot delimit a CSV's fields, or None when it can.

    Exactly one character: pandas parses a longer ``sep`` as a regular expression on its
    python engine, so the separator would be caller-supplied pattern code run over the
    whole file -- a crafted one backtracks catastrophically, and in a training it would hang
    the thread outside any stop checkpoint. Not a line break: a row ends there, and pandas
    refuses ``\\n`` with a ValueError the API answered as a 500 (audit 2026-09-30, V02).
    The query and form parameters answer a problem with 400, the request models with 422.
    """
    if len(value) != 1:
        return "separator must be a single character."
    if value in ("\r", "\n"):
        return "separator must not be a line break: a row ends there."
    return None


def _usable_separator(value: str) -> str:
    if problem := separator_problem(value):
        raise ValueError(problem)
    return value


# A request model's CSV delimiter. Each field keeps its length bounds, which OpenAPI shows;
# this adds the rule they cannot express.
CsvSeparator = Annotated[str, AfterValidator(_usable_separator)]

# One wording per concept: several requests take the same field, and two descriptions of
# one thing read as two different things. Text only — every field keeps its own type,
# default and constraints at its declaration.
DATASET_NAME = (
    "File name of a dataset in the data directory, as `GET /datasets` lists it (add one with "
    "`POST /datasets/import`). A plain name: path characters are refused."
)
LABEL_COLUMN = "Column holding each row's labels, several per cell separated by `label_separator`."
CSV_SEPARATOR = "The CSV's field delimiter: exactly one character, not a line break."
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
