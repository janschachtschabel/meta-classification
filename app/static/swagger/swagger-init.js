"use strict";
// External Swagger UI init (kept out of the /docs HTML) so the page needs no
// inline <script> and can run under the same-origin CSP. BaseLayout + the apis
// preset are provided by swagger-ui-bundle.js — no standalone preset needed.
window.ui = SwaggerUIBundle({
  url: "/openapi.json",
  dom_id: "#swagger-ui",
  deepLinking: true,
  presets: [SwaggerUIBundle.presets.apis],
  layout: "BaseLayout",
  persistAuthorization: true,
});
