/* Transport layer: API-key handling + fetch wrapper. Same-origin, no CORS.
   The key lives in sessionStorage only (gone when the tab closes) and is sent
   as the X-API-Key header on every request — exactly the API's auth model. */
"use strict";

const Api = (() => {
  const KEY = "apiv3-key";

  /* Site data can be blocked outright — private windows, locked-down profiles — and
     there READING sessionStorage throws rather than returning null. `Api.getKey()` is the
     first thing boot() does, so an unguarded read left both views hidden: a blank page.
     i18n.js guards localStorage for the same reason.

     The key is held in memory as well, and that is not belt-and-braces: a guard that only
     swallowed the write would leave `getKey()` answering "" on such a profile, so a
     correct key would be typed, stored nowhere, and rejected as 401 on the very next
     request. In memory the session works normally; only a reload asks for the key again,
     which is what sessionStorage was buying anyway. */
  let held = "";
  const getKey = () => {
    if (held) return held;
    try { return sessionStorage.getItem(KEY) || ""; } catch { return ""; }
  };
  const setKey = (k) => {
    held = k;
    try { sessionStorage.setItem(KEY, k); } catch { /* this tab only, then */ }
  };
  const clearKey = () => {
    held = "";
    try { sessionStorage.removeItem(KEY); } catch { /* nothing was stored */ }
  };

  class ApiError extends Error {
    constructor(status, detail, cause) {
      // `detail` is the server's own wording and stays as it came: it names the
      // specific thing that was wrong, which a generic translated line cannot.
      super(detail || t("errors.requestFailed", { status }));
      // status 0 = the request never reached a server, so there is no HTTP status to give.
      this.status = status;
      // Kept for the console: the translated message is for the operator, the original
      // TypeError is what a developer needs, and dropping it loses the only clue.
      if (cause) this.cause = cause;
    }
  }

  async function request(path, { method = "GET", json, form } = {}) {
    const headers = {};
    const key = getKey();
    if (key) headers["X-API-Key"] = key; // keyless mode: rely on the server's auth setting
    let body;
    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(json);
    } else if (form !== undefined) {
      body = form; // browser sets the multipart boundary itself
    }
    let res;
    try {
      res = await fetch(path, { method, headers, body });
    } catch (cause) {
      // `fetch` rejects with `TypeError: Failed to fetch` on a dropped connection, a DNS
      // failure or a CORS refusal — browser internals, in the BROWSER's language, which
      // eight display sites were showing verbatim as if the API had said it. Translated
      // here rather than at each of them: there is one transport, so there is one message.
      throw new ApiError(0, t("errors.network"), cause);
    }
    if (res.status === 401) {
      clearKey();
      window.dispatchEvent(new Event("apiv3-unauthorized"));
      throw new ApiError(401, t("errors.unauthorized"));
    }
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail; } catch { /* non-JSON error body */ }
      if (res.status === 403) detail = detail || t("errors.adminRequired");
      if (res.status === 429) detail = detail || t("errors.rateLimited");
      throw new ApiError(res.status, detail);
    }
    const type = res.headers.get("content-type") || "";
    return type.includes("json") ? res.json() : res;
  }

  /* Hand a blob to the browser as a download. */
  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: filename });
    a.click();
    URL.revokeObjectURL(url);
  }

  /* Authenticated file download: fetch as blob, hand to the browser. */
  async function download(path, filename) {
    saveBlob(await (await request(path, { method: "POST" })).blob(), filename);
  }

  /* The same, for endpoints that answer a multipart upload with a file. The headers come
     back with the blob: whether a streamed answer is complete is only readable there
     (`X-Input-Rows` on /predict/csv) -- a stream cut short ends like a finished one. */
  async function downloadForm(path, form, filename) {
    const res = await request(path, { method: "POST", form });
    const blob = await res.blob();
    saveBlob(blob, filename);
    return { blob, headers: res.headers };
  }

  return {
    getKey, setKey, clearKey, ApiError, download, downloadForm, saveBlob,
    get: (p) => request(p),
    post: (p, json) => request(p, { method: "POST", json }),
    put: (p, json) => request(p, { method: "PUT", json }),
    postForm: (p, form) => request(p, { method: "POST", form }),
    del: (p) => request(p, { method: "DELETE" }),
  };
})();
