"""Descriptive metadata for a text: a title, a description and keywords.

Classification says what a text is *about* in the vocabulary of a trained model. This says
what it *is*, in its own words, so an item can be handed on as a complete metadata record
rather than as labels alone. The two are independent: nothing here reads a model bundle, the
registry or the volume — the text in the request is the only input.

The generators are a port of the default combination of `static-metadata-generators`, which
measured 52 methods on two test sets; `docs/plans/2026-09-26-descriptive-metadata.md` records
which three were ported, the one deviation from that app's own default and why, and the three
dependencies they need. Only that default combination exists here: there is no method
registry and nothing to choose between.

All three are extractive — every word they output occurs in the input — so they propose, they
do not write. That is the point: the source app's conclusion after measuring the generative
alternatives was that small language models invent facts, and these do not.
"""

from __future__ import annotations

from . import fields
from .textprep import prepare
from .types import DescriptiveMetadata, Document, MetadataSettings

DEFAULT_SETTINGS = MetadataSettings()

__all__ = [
    "DEFAULT_SETTINGS",
    "DescriptiveMetadata",
    "Document",
    "MetadataSettings",
    "generate",
    "prepare",
]


def generate(
    text: str, settings: MetadataSettings = DEFAULT_SETTINGS
) -> DescriptiveMetadata:
    """Propose a title, a description and keywords for one text.

    The text is prepared once — cleaned, split into lines and sentences — and all three
    generators read that one `Document`, which is also where the ranked keyword candidates are
    cached: the title is built from the same ranking the keywords come from, so a text with no
    heading would otherwise rank its candidates twice.

    Returns empty strings and an empty list for a text that yields nothing (blank input, or
    only page chrome). That is an answer, not an error: the caller asked what this text
    suggests, and the answer is nothing.
    """
    doc = prepare(text)
    return DescriptiveMetadata(
        title=fields.title(doc, settings),
        description=fields.description(doc, settings),
        keywords=fields.keywords(doc, settings),
    )
