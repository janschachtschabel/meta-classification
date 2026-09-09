/* Transport layer: API-key handling + fetch wrapper. Same-origin, no CORS.
   The key lives in sessionStorage only (gone when the tab closes) and is sent
   as the X-API-Key header on every request — exactly the API's auth model. */
"use strict";

const Api = (() => {
  const KEY = "apiv3-key";

  const getKey = () => sessionStorage.getItem(KEY) || "";
  const setKey = (k) => sessionStorage.setItem(KEY, k);
  const clearKey = () => sessionStorage.removeItem(KEY);

  class ApiError extends Error {
    constructor(status, detail) {
      super(detail || `Request failed (${status})`);
      this.status = status;
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
    const res = await fetch(path, { method, headers, body });
    if (res.status === 401) {
      clearKey();
      window.dispatchEvent(new Event("apiv3-unauthorized"));
      throw new ApiError(401, "API key invalid or expired — please sign in again.");
    }
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail; } catch { /* non-JSON error body */ }
      if (res.status === 403) detail = detail || "This action needs the admin API key.";
      if (res.status === 429) detail = detail || "Rate limit reached — please wait a moment.";
      throw new ApiError(res.status, detail);
    }
    const type = res.headers.get("content-type") || "";
    return type.includes("json") ? res.json() : res;
  }

  /* Authenticated file download: fetch as blob, hand to the browser. */
  async function download(path, filename) {
    const res = await request(path, { method: "POST" });
    const url = URL.createObjectURL(await res.blob());
    const a = Object.assign(document.createElement("a"), { href: url, download: filename });
    a.click();
    URL.revokeObjectURL(url);
  }

  return {
    getKey, setKey, clearKey, ApiError, download,
    get: (p) => request(p),
    post: (p, json) => request(p, { method: "POST", json }),
    put: (p, json) => request(p, { method: "PUT", json }),
    postForm: (p, form) => request(p, { method: "POST", form }),
    del: (p) => request(p, { method: "DELETE" }),
  };
})();
