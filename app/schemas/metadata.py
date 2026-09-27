"""What a caller may ask of the descriptive-metadata generators."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from ..metadata import DEFAULT_SETTINGS

# The batch cap is measured, not copied from /predict, which allows 1000: generation costs
# 12-22 ms per text where a TF-IDF transform costs well under one, so a full batch here is
# already ~2 s of CPU on a worker that serves everything else meanwhile.
MAX_TEXTS = 100


class MetadataRequest(BaseModel):
    """One batch of texts, and the budgets the proposals have to fit."""

    texts: list[Annotated[str, Field(max_length=100_000)]] = Field(
        ..., min_length=1, max_length=MAX_TEXTS,
        description=(
            f"The texts to describe: 1-{MAX_TEXTS} per request, each at most 100,000 characters. "
            "**Keep the line breaks**: the generators recover headings and sentences from them, "
            "undo line wrapping from PDFs and drop page chrome, so a text flattened to one line "
            "yields noticeably worse fields. HTML and Markdown are removed for you, and so are "
            "the bodies of `script`, `style`, `nav` and `footer` — a block-level tag counts as a "
            "line break, so a scraped page works as it is. Unlike `/predict` there is no model "
            "to match: nothing here depends on how a model was trained."
        ),
    )
    title_max: int = Field(
        DEFAULT_SETTINGS.title_max, ge=20, le=300,
        description=(
            "Longest title in characters. Cut at a clause boundary with an ellipsis if the "
            "source line is longer. The default is what the methods were measured at."
        ),
    )
    desc_max: int = Field(
        DEFAULT_SETTINGS.desc_max, ge=80, le=5_000,
        description=(
            "Longest description in characters. Whole sentences only — the last one that does "
            "not fit is left out rather than cut off, so a small budget can come back well "
            "under it. The default is what the methods were measured at."
        ),
    )
    n_keywords: int = Field(
        DEFAULT_SETTINGS.n_keywords, ge=1, le=50,
        description=(
            "How many keywords at most, best first. Fewer come back when the text yields "
            "fewer: keywords are noun phrases found IN the text, so a short text has few."
        ),
    )
