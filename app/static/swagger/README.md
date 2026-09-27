# Vendored Swagger UI assets

`swagger-ui.css` and `swagger-ui-bundle.js` are vendored from
[`swagger-ui-dist`](https://www.npmjs.com/package/swagger-ui-dist) so the `/docs`
page is served **same-origin** (no CDN → no IP leak, works offline, runs under
the app's Content-Security-Policy). `swagger-init.js` is our own external init
(kept out of the HTML so `/docs` needs no inline `<script>`).

- **Pinned version:** `swagger-ui-dist@5.17.14`

## Integrity

These files are served to every `/docs` visitor and are committed rather than declared in a
manifest, so `pip-audit` does not see them and no dependency bot watches them. The hashes below
are what `tests/test_vendored_assets.py` enforces — the point is not that the version is current
but that an edit cannot pass unnoticed, and that a re-download can be compared against what was
here before.

| File | SHA-256 |
|---|---|
| `swagger-ui-bundle.js` | `sha256:c2e4a9ef08144839ff47c14202063ecfe4e59e70a4e7154a26bd50d880c88ba1` |
| `swagger-ui.css` | `sha256:40170f0ee859d17f92131ba707329a88a070e4f66874d11365e9a77d232f6117` |
| `swagger-init.js` | `sha256:70dd325077f06181b01eb301cc246731997a92078969976ff735c2f7493e476f` |

To update, re-download at the new version, **record the new hashes**, and bump the version note
above — the test will fail until you do, which is the reminder:

```sh
VER=5.17.14
base="https://cdn.jsdelivr.net/npm/swagger-ui-dist@$VER"
curl -sSfo swagger-ui.css        "$base/swagger-ui.css"
curl -sSfo swagger-ui-bundle.js  "$base/swagger-ui-bundle.js"
sha256sum swagger-ui.css swagger-ui-bundle.js swagger-init.js
```

Compare those digests against the ones npm publishes for the tarball before committing; a file
fetched over a CDN is only as trustworthy as the check you run on it.

After updating, load `/docs` and confirm it renders with **zero** CSP violations
(the page runs under a scoped CSP that allows only `style-src 'unsafe-inline'`
and `img-src data:` beyond the strict same-origin default).
