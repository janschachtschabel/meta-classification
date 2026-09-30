"""What a caller may ask of the descriptive-metadata generators."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from ..metadata import DEFAULT_SETTINGS

# The batch cap is measured, not copied from /predict, which allows 1000: generation costs
# 12-22 ms per text of ordinary length, where a TF-IDF transform costs well under one.
MAX_TEXTS = 100
# ...but a long text costs by its length, and the per-text caps multiplied out to 10 million
# characters: 0.4 s per 100,000 characters of ordinary prose, up to 4.8 s in the shapes pysbd
# is slowest on (259 ms per 4,000-character piece) -- about eight CPU-minutes for one request
# at the caps (audit 2026-09-30, M03). The sum is what a request may ask for.
MAX_REQUEST_CHARS = 1_000_000


class MetadataRequest(BaseModel):
    """One batch of texts, and the budgets the proposals have to fit."""

    texts: list[Annotated[str, Field(max_length=100_000)]] = Field(
        ..., min_length=1, max_length=MAX_TEXTS,
        description=(
            f"The texts to describe: 1-{MAX_TEXTS} per request, each at most 100,000 characters "
            f"and together at most {MAX_REQUEST_CHARS:,}. "
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

    @model_validator(mode="after")
    def _within_the_character_budget(self) -> MetadataRequest:
        total = sum(len(text) for text in self.texts)
        if total > MAX_REQUEST_CHARS:
            raise ValueError(
                f"{total:,} characters in one request; at most {MAX_REQUEST_CHARS:,} across all "
                "texts. Split the batch."
            )
        return self
