"""What the generators read and what they hand back.

Kept apart from the modules that use them so the text layer and the field methods can both
name a ``Document`` without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetadataSettings:
    """The output budgets. Every method has to respect them.

    The defaults are the source app's (`metagen/types.py`), and the parity fixture was
    produced with them — changing one changes what the generators are expected to produce.
    """

    title_max: int = 90
    desc_max: int = 500
    n_keywords: int = 8


@dataclass
class Document:
    """A prepared text.

    ``cache`` holds derived data that more than one method would otherwise recompute: the
    title method and the keyword method both rank the same candidates, so the candidates and
    their scores are built once per document.
    """

    text: str  # cleaned, non-empty lines joined by "\n"
    lines: list[str]
    sentences: list[str]  # in document order; a line break is always a boundary
    cache: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class DescriptiveMetadata:
    """One proposal per field. Empty where the text gave nothing to work with."""

    title: str
    description: str
    keywords: list[str]
