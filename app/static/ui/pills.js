/* Pill multi-select backed by a native <datalist> — platform-first: the browser
   provides the autocomplete dropdown and its keyboard handling; this widget only
   manages the selected pills. No dependencies, no ARIA gymnastics. */
"use strict";

function createPillPicker({ pills, input, datalist, emptyHint, onChange = () => {} }) {
  let available = [];
  let selected = [];
  let ready = false;  // suppress onChange during construction (TDZ safety)

  const escP = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function render() {
    pills.innerHTML = selected.map((c) => `
      <span class="pill">${escP(c)}
        <button type="button" class="pill-x" data-remove="${escP(c)}"
                aria-label="Remove ${escP(c)}">&times;</button></span>`).join("");
    pills.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => {
      selected = selected.filter((c) => c !== b.dataset.remove);
      render();
      input.focus();
    }));
    // Suggest only what is not picked yet.
    datalist.innerHTML = available.filter((c) => !selected.includes(c))
      .map((c) => `<option value="${escP(c)}">`).join("");
    input.disabled = !available.length;
    input.placeholder = available.length ? "Type to add …" : emptyHint;
    if (ready) onChange([...selected]);  // no event for the initial empty render
  }

  function tryAdd() {
    const value = input.value.trim();
    if (!value) return;
    const remaining = available.filter((c) => !selected.includes(c));
    const matches = remaining.filter((c) => c.toLowerCase().includes(value.toLowerCase()));
    // Exact pick (from the datalist) or an unambiguous substring wins.
    const pick = remaining.find((c) => c === value) || (matches.length === 1 ? matches[0] : null);
    if (pick) {
      selected.push(pick);
      input.value = "";
      render();
    }
  }

  input.addEventListener("change", tryAdd); // fires when a datalist entry is chosen
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); tryAdd(); }
    else if (ev.key === "Backspace" && !input.value && selected.length) {
      selected.pop();
      render();
    }
  });

  render();
  ready = true;
  return {
    setOptions(options) { available = [...options]; selected = []; render(); },
    values: () => [...selected],
  };
}
