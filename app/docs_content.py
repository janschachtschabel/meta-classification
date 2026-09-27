"""The text and assets the API presents about itself.

Inert content, kept out of `main.py` so the factory there reads as wiring. These change when
the API's story changes or when the vendored Swagger assets are updated — not when a router
or a middleware does.
"""

from __future__ import annotations

TAGS_METADATA = [
    {"name": "System", "description": "Health check and (safe) configuration."},
    {"name": "Training", "description": "Train models asynchronously and monitor progress."},
    {"name": "Prediction", "description": "Classify texts with a trained model."},
    {"name": "Models", "description": "List, inspect, evaluate, delete, export/import and share models."},
    {"name": "Datasets", "description": "Manage, analyze and validate CSV datasets."},
    {"name": "Feedback", "description": "Corrections to predictions, and the CSV they train from."},
]

DESCRIPTION = (
    "**MetaClassify** — torch-free, CPU-only text-classification API. Train on "
    "your metadata, serve multiple models via REST.\n\n"
    "**Authentication:** `X-API-Key` header. Role *readonly* for classification "
    "and status, *admin* for training and management actions. `/health` is public.\n\n"
    "**Typical flow:** provide a dataset → `POST /train` → `GET /train/status` "
    "(phase, progress, estimated time remaining) → `POST /predict`. Multiple models "
    "coexist; select per request via `model_name`.\n\n"
    "**AI-written rows:** a dataset prepared in data-prep marks the rows an LLM wrote or "
    "touched (`generated_for`, `example_for`, `enriched_fields`). Training learns from "
    "them but validates on real rows only; when too few real rows force an exception, "
    "the model says so (`synthetic_data` in `GET /models/{name}`)."
)


# Self-hosted Swagger UI page: vendored assets + an external init script (no inline
# JS), so it renders same-origin under CSP without any CDN. Assets live in
# static/swagger and are pinned (see static/swagger/README.md).
SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetaClassify — API docs</title>
  <link rel="stylesheet" href="/swagger-static/swagger-ui.css">
  <link rel="icon" href="/favicon.ico" type="image/svg+xml">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="/swagger-static/swagger-ui-bundle.js"></script>
  <script src="/swagger-static/swagger-init.js"></script>
</body>
</html>"""


# Tab icon for /ui and /docs, inline so the app stays free of binary assets. Served
# as SVG under the .ico name browsers request unprompted (they honour the content
# type, not the extension) — otherwise every visit logs a 404.
FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="7" fill="#0f5cad"/>'
    b'<path d="M8 11h16M8 16h11M8 21h7" stroke="#fff" stroke-width="3" '
    b'stroke-linecap="round"/></svg>'
)


