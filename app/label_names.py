"""What a label URI is, and what it is called.

Split out of ``data`` (plan item A1): loading, cleaning and splitting a dataset is one
responsibility, and reasoning about the STRUCTURE of a label — is this a namespace or a
concept, which vocabulary do these labels come from, which display name belongs to which
URI — is another. The second is read by four modules that have nothing to do with CSVs
(``model_io``, ``model_archive``, ``model_report`` and the repair scripts), which is what
made the coupling visible.

A leaf: it imports nothing of ours, so ``data`` may depend on it and not the reverse.
"""

from __future__ import annotations

import os


def is_container_label(label: str) -> bool:
    """Does this value name a NAMESPACE rather than a concept?

    A trailing ``/`` means "members of", not "a member" — in URIs as in paths. Such a value
    is a tagging accident, never a class worth learning: measured on ``data_300k.csv``, the
    bare vocabulary root ``…/vocabs/discipline/`` was attached to 522 rows and trained as
    an ordinary label scoring F1 0.4096, diluting macro F1 and letting ``/predict`` answer
    with a label that carries no meaning.

    ``min_samples_per_label`` cannot catch this — 522 rows clears any sane threshold — so
    the guard has to be structural. It is deliberately narrow: a genuine broader concept
    has an id (``…/discipline/120``) and is kept, because the label hierarchy is real
    signal (the higher-education vocabulary averages 2.25 levels per row).
    """
    return label.endswith("/")


def label_vocabulary(classes: list[str]) -> str | None:
    """The namespace all label URIs belong to, or ``None`` when they do not agree.

    "Which vocabulary does this model classify into" is a question a recipient of a
    shared bundle asks, and it must never be answered by a typed-in field that can be
    wrong: it follows from the labels themselves. A vocabulary is the DIRECT parent of
    every concept id, which is what separates one vocabulary from the common ancestor
    of two — ``…/vocabs/discipline/380`` and ``…/vocabs/educationalContext/sek`` share
    ``…/vocabs/``, which names neither, so this reports nothing rather than a
    misleading fragment.

    🟢 Measured against the 14 bundles of the local model store: every one resolved to
    a single namespace (discipline, educationalContext, hochschulfaechersystematik).
    """
    if not classes:
        return None
    prefix = os.path.commonprefix(classes)
    if "/" not in prefix:
        return None
    prefix = prefix.rsplit("/", 1)[0] + "/"
    if any("/" in uri[len(prefix):] for uri in classes):
        return None
    return prefix


def _rejoin_split_names(fragments: list[str]) -> list[str]:
    """Undo a split caused by a separator character INSIDE a display name.

    Label URIs and their display names arrive as two lists sharing one separator, but
    only URIs are guaranteed free of it. German orthography then says which fragment is
    a continuation rather than a new name: a lowercase start, a preceding compound half
    ("Rechts-"), or an unclosed parenthesis.

    Measured on ``data_300k.csv``: this reconstructs 29.6% of the damaged rows, and where
    the result could be cross-checked against rows that were never damaged it agreed
    75/75 times — i.e. precise but not complete, which is why ``_pair_names`` still
    refuses to guess when it does not reconcile.
    """
    merged: list[str] = []
    for fragment in fragments:
        continues = bool(merged) and (
            fragment[:1].islower()
            or merged[-1].endswith("-")
            or merged[-1].count("(") > merged[-1].count(")")
        )
        if continues:
            merged[-1] = f"{merged[-1]}, {fragment}"
        else:
            merged.append(fragment)
    return merged


def pair_names(uris: list[str], names: list[str]) -> list[tuple[str, str]]:
    """Pair URIs with display names, but ONLY when the two provably line up.

    The URI count is authoritative. A positional zip of unequal lists silently shifts
    every later name onto the wrong URI — on ``data_300k.csv`` that gave 34 of 119
    higher-education labels the name of a *different* subject, which reads as a
    confident statement rather than as missing data. So when the counts cannot be
    reconciled, this contributes nothing and callers fall back to the URI.
    """
    if len(uris) == len(names):
        return list(zip(uris, names, strict=True))
    repaired = _rejoin_split_names(names)
    if len(repaired) == len(uris):
        return list(zip(uris, repaired, strict=True))
    return []
