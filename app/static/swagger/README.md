# Vendored Swagger UI assets

`swagger-ui.css` and `swagger-ui-bundle.js` are vendored from
[`swagger-ui-dist`](https://www.npmjs.com/package/swagger-ui-dist) so the `/docs`
page is served **same-origin** (no CDN → no IP leak, works offline, runs under
the app's Content-Security-Policy). `swagger-init.js` is our own external init
(kept out of the HTML so `/docs` needs no inline `<script>`).

- **Pinned version:** `swagger-ui-dist@5.17.14`

To update, re-download the two assets at the new version and bump this note:

```sh
VER=5.17.14
base="https://cdn.jsdelivr.net/npm/swagger-ui-dist@$VER"
curl -sSfo swagger-ui.css        "$base/swagger-ui.css"
curl -sSfo swagger-ui-bundle.js  "$base/swagger-ui-bundle.js"
```

After updating, load `/docs` and confirm it renders with **zero** CSP violations
(the page runs under a scoped CSP that allows only `style-src 'unsafe-inline'`
and `img-src data:` beyond the strict same-origin default).
