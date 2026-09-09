"""Render a model bundle's metadata as a human-readable card.

An exported bundle is what a third party receives: two skops containers and some
JSON. That is enough for a machine and nothing for a person, so the archive carries
a generated ``README.md`` — the first file anyone opens.

Pure rendering, no I/O: the caller passes the bundle's own documents, which keeps
this testable and keeps ``registry`` the only place that touches disk.
"""

from __future__ import annotations

_WEAKEST_LABELS = 10


def _row(label: str, value: object) -> str:
    return f"| {label} | {value} |"


def _authored_section(info: dict) -> list[str]:
    """The author's own statements, or an honest note that there are none."""
    if not info:
        return [
            "> No documentation was supplied for this model. Who trained it, what it is",
            "> for and where its data came from are therefore unknown — ask the sender,",
            "> or set it via `PUT /models/{name}/info`.",
        ]
    fields = [("Author", "author"), ("License", "license"), ("Data source", "data_source")]
    lines = ["| | |", "|---|---|"]
    lines += [_row(title, info[key]) for title, key in fields if info.get(key)]
    if info.get("description"):
        lines += ["", info["description"]]
    return lines


def _training_section(metadata: dict) -> list[str]:
    tfidf = metadata.get("tfidf") or {}
    weights = metadata.get("text_column_weights") or {}
    lines = [
        "| | |", "|---|---|",
        _row("Dataset", f"`{metadata.get('dataset', '—')}`"),
        _row("Text columns", ", ".join(f"`{c}`" for c in metadata.get("text_columns", [])) or "—"),
        _row("Label column", f"`{metadata.get('label_column', '—')}`"),
        _row("Rows used", f"{metadata.get('n_samples', 0):,}"),
        _row("Profile", metadata.get("profile", "—")),
        _row("Regularization C", f"{metadata.get('best_C', '—')} (searched {metadata.get('c_grid', '—')})"),
        _row("Features", f"{tfidf.get('n_features', '—')} "
                         f"({'word + character' if tfidf.get('use_char') else 'word only'}) n-grams"),
        _row("Min. samples per label", metadata.get("min_samples_per_label", "—")),
        _row("Training time", f"{metadata.get('training_time_seconds', 0) / 60:.1f} min"),
    ]
    if metadata.get("label_filter"):
        lines.insert(-1, _row("Label filter", f"`{metadata['label_filter']}`"))
    if weights:
        lines += [
            "",
            "**Text was assembled with repeated fields** "
            + ", ".join(f"`{col}` ×{n}" for col, n in weights.items())
            + ". A model fit this way expects input built the same way — repeat those "
            "fields in the text you send to `/predict`, or its tuned thresholds sit on "
            "a different feature distribution than they were tuned on.",
        ]
    return lines


def _quality_section(metadata: dict, config: dict) -> list[str]:
    metrics = metadata.get("metrics") or {}
    lines = [
        "| | |", "|---|---|",
        _row("Evaluation", metadata.get("evaluation", "—")),
        _row("Decision rule", metrics.get("decision_rule", "—")),
    ]
    for title, key in (("F1 macro", "f1_macro"), ("F1 micro", "f1_micro")):
        if isinstance(metrics.get(key), int | float):
            lines.append(_row(title, f"{metrics[key]:.4f}"))
    if metrics.get("predicted_labels_per_row") is not None:
        lines.append(_row("Labels asserted per row",
                          f"{metrics['predicted_labels_per_row']} "
                          f"(data carries {metrics.get('true_labels_per_row', '—')})"))

    per_label = metrics.get("per_label_f1") or {}
    if per_label:
        names = config.get("uri_to_label") or {}
        weakest = sorted(per_label.items(), key=lambda kv: kv[1])[:_WEAKEST_LABELS]
        lines += [
            "", f"**The {len(weakest)} weakest labels.** A high confidence on one of these "
            "is worth less than the same number on a strong label — check them before "
            "relying on the model for those subjects.", "",
            "| Label | F1 |", "|---|---:|",
        ]
        lines += [_row(names.get(uri, uri), f"{score:.3f}") for uri, score in weakest]
    return lines


def render(name: str, config: dict, metadata: dict, vocabulary: str | None) -> str:
    """Build the Markdown card for one exported bundle."""
    classes = config.get("classes") or []
    summary = (
        f"A {config.get('task_type', 'text')} classifier over **{len(classes)} labels**"
        + (f" from `{vocabulary}`" if vocabulary else "")
        + "."
    )
    lines = [
        f"# {name}",
        "",
        summary,
        "",
        "Trained with **MetaClassify** (TF-IDF + logistic regression, CPU-only). "
        "The archive is self-contained: it loads with scikit-learn and skops alone, "
        "without this application.",
        "",
        "## Provided by the author",
        "",
        *_authored_section(metadata.get("info") or {}),
        "",
        "## What it was trained on",
        "",
        *_training_section(metadata),
        "",
        "## How well it works",
        "",
        *_quality_section(metadata, config),
        "",
        "## Using it",
        "",
        "```bash",
        'curl -X POST http://<host>/predict -H "X-API-Key: $KEY" \\',
        '  -H "Content-Type: application/json" \\',
        f'  -d \'{{"texts": ["..."], "model_name": "{name}"}}\'',
        "```",
        "",
        "The scores come from the model's own tuned thresholds. `manifest.json` lists a "
        "SHA-256 for every other file in this archive; the import path verifies it.",
        "",
        f"_Generated from the bundle's metadata on export. Model created "
        f"{metadata.get('created_at', 'unknown')}._",
        "",
    ]
    return "\n".join(lines)
