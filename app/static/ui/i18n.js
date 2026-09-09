/* German and English for this UI, and the one place that decides which.

   No framework and no fetch: the two maps are ordinary scripts loaded before
   everything else, so `t()` answers from the first line of the first module and
   the page never paints a string it has to correct a moment later. The price is
   both languages on the wire — some 20 kB for an admin tool on the same origin,
   which buys away an async boot and its failure mode.

   `t()` returns TEXT, never markup, and never escapes: the caller escapes at the
   innerHTML boundary exactly as it did before this layer existed. So a value that
   comes from a model bundle or a person still goes in as `t(key, {name: esc(x)})`
   when the result lands in innerHTML, and as `t(key, {name: x})` when it lands in
   textContent. */
"use strict";

const I18n = (() => {
  const STORAGE_KEY = "apiv3-lang";
  const SUPPORTED = ["de", "en"];
  // German, not English: the people running this read German, and a browser set
  // to a third language should land on the audience's tongue rather than on the
  // one the code happens to be written in.
  const FALLBACK = "de";
  const MAPS = { de: STRINGS_DE, en: STRINGS_EN };

  /* Site data can be blocked outright (private windows, locked-down profiles),
     where touching localStorage throws rather than returning null. A language
     preference is worth exactly zero broken pages. */
  function stored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
  }
  function remember(language) {
    try { localStorage.setItem(STORAGE_KEY, language); } catch { /* nothing to do */ }
  }

  function resolve() {
    const choice = stored();
    if (SUPPORTED.includes(choice)) return choice;
    for (const tag of navigator.languages || [navigator.language || ""]) {
      const base = String(tag).toLowerCase().split("-")[0];
      if (SUPPORTED.includes(base)) return base;
    }
    return FALLBACK;
  }

  let current = resolve();
  let plurals = new Intl.PluralRules(current);
  let numbers = new Intl.NumberFormat(current);
  const missing = new Set();

  /* One text. `params` fills {placeholders}; a `count` param also picks the
     plural form when the entry has one. Numbers are formatted for the active
     locale on the way in — 156373 reads as 156.373 in German and 156,373 in
     English, and no call site should have to remember that. */
  function t(key, params) {
    let value = MAPS[current][key];
    if (value === undefined) {
      if (!missing.has(key)) { missing.add(key); console.warn(`i18n: no string for ${key}`); }
      return key;  // visible in the UI, rather than a silent blank
    }
    if (typeof value === "object") {
      const form = plurals.select(Number(params && params.count));
      value = form in value ? value[form] : value.other;
    }
    if (!params) return value;
    return value.replace(/\{(\w+)\}/g, (whole, name) => {
      if (!(name in params)) return whole;
      const given = params[name];
      return typeof given === "number" && Number.isFinite(given) ? numbers.format(given) : given;
    });
  }

  /* Put the active language into the markup. Everything a person reads that is
     not built by JS carries one of these hooks; `data-i18n-html` is the one that
     writes markup, and it is fed a constant (no params) on purpose — see the
     test that holds that line. */
  function apply(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll("[data-i18n-html]").forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
    root.querySelectorAll("[data-i18n-placeholder]").forEach(
      (el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    root.querySelectorAll("[data-i18n-title]").forEach(
      (el) => { el.title = t(el.dataset.i18nTitle); });
    root.querySelectorAll("[data-i18n-aria-label]").forEach(
      (el) => { el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel)); });
    document.documentElement.lang = current;  // screen readers pronounce from this
  }

  /* Switching re-applies the markup and leaves already-rendered output alone: a
     results table is the answer to a question that was asked in the other
     language, and throwing away someone's work — or their half-filled form — to
     restate it is the worse trade. Any list refreshes on its next load, and the
     status card re-renders on the next poll. */
  function setLanguage(language) {
    if (!SUPPORTED.includes(language) || language === current) return;
    current = language;
    plurals = new Intl.PluralRules(current);
    numbers = new Intl.NumberFormat(current);
    remember(language);
    apply();
  }

  function init() {
    document.querySelectorAll(".lang-select").forEach((select) => {
      select.value = current;
      select.addEventListener("change", (ev) => {
        setLanguage(ev.target.value);
        document.querySelectorAll(".lang-select").forEach((other) => { other.value = current; });
      });
    });
    apply();
  }

  return { t, apply, init, setLanguage, locale: () => current };
})();

const t = I18n.t;
I18n.init();
