"""Render a model bundle's metadata as a human-readable card.

An exported bundle is what a third party receives: two skops containers and some
JSON. That is enough for a machine and nothing for a person, so the archive carries
a generated ``README.md`` — the first file anyone opens.

Pure rendering, no I/O: the caller passes the bundle's own documents, which keeps
this testable and keeps ``registry`` the only place that touches disk.
"""

from __future__ import annotations

from .bundle_meta import as_mapping, as_names, as_number, per_label_f1

_WEAKEST_LABELS = 10


def _cell(value: object) -> str:
    # Cells carry author- and bundle-supplied text: a literal "|" would end the cell
    # and a newline the row, so both are neutralized rather than trusted to behave.
    return str(value).replace("|", "\\|").replace("\n", " ")


def _row(label: str, value: object) -> str:
    return f"| {label} | {_cell(value)} |"


def _measured(value: object, template: str) -> str:
    """Format a number from the metrics document, or say it is not known.

    Values that cannot be formatted are the common case for old bundles, not an
    exotic one, and the card must still render: an em dash is the honest output.
    """
    number = as_number(value)
    return template.format(number) if number is not None else "—"


def _minutes(seconds: object) -> str:
    number = as_number(seconds)
    return f"{number / 60:.1f} min" if number is not None else "—"


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
    tfidf = as_mapping(metadata.get("tfidf"))
    weights = as_mapping(metadata.get("text_column_weights"))
    lines = [
        "| | |", "|---|---|",
        _row("Dataset", f"`{metadata.get('dataset', '—')}`"),
        _row("Text columns", ", ".join(f"`{c}`" for c in as_names(metadata.get("text_columns"))) or "—"),
        _row("Label column", f"`{metadata.get('label_column', '—')}`"),
        _row("Rows used", _measured(metadata.get("n_samples"), "{:,.0f}")),
        _row("Profile", metadata.get("profile", "—")),
        _row("Regularization C", f"{metadata.get('best_C', '—')} (searched {metadata.get('c_grid', '—')})"),
        _row("Features", f"{tfidf.get('n_features', '—')} "
                         f"({'word + character' if tfidf.get('use_char') else 'word only'}) n-grams"),
        _row("Min. samples per label", metadata.get("min_samples_per_label", "—")),
        _row("Training time", _minutes(metadata.get("training_time_seconds"))),
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
    metrics = as_mapping(metadata.get("metrics"))
    lines = [
        "| | |", "|---|---|",
        _row("Evaluation", metadata.get("evaluation", "—")),
        _row("Decision rule", metrics.get("decision_rule", "—")),
    ]
    for title, key in (("F1 macro", "f1_macro"), ("F1 micro", "f1_micro")):
        score = as_number(metrics.get(key))
        if score is not None:  # absent or unusable: omit the row, don't imply a measurement
            lines.append(_row(title, f"{score:.4f}"))
    if metrics.get("predicted_labels_per_row") is not None:
        lines.append(_row("Labels asserted per row",
                          f"{metrics['predicted_labels_per_row']} "
                          f"(data carries {metrics.get('true_labels_per_row', '—')})"))

    per_label = per_label_f1(metadata)
    if per_label:
        names = as_mapping(config.get("uri_to_label"))
        support = as_mapping(metadata.get("per_label_support"))
        weakest = sorted(per_label.items(), key=lambda kv: kv[1])[:_WEAKEST_LABELS]
        lines += [
            "", f"**The {len(weakest)} weakest labels.** A high confidence on one of these "
            "is worth less than the same number on a strong label — check them before "
            "relying on the model for those subjects. The row count says why a score is "
            "low: too few examples is a different problem from a hard distinction.", "",
            "| Label | F1 | Rows |", "|---|---:|---:|",
        ]
        lines += [
            f"| {_cell(names.get(uri, uri))} | {score:.3f} | {_cell(support.get(uri, '—'))} |"
            for uri, score in weakest
        ]
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
        *_authored_section(as_mapping(metadata.get("info"))),
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
