/* Pill multi-select backed by a native <datalist> — platform-first: the browser
   provides the autocomplete dropdown and its keyboard handling; this widget only
   manages the selected pills. No dependencies, no ARIA gymnastics. */
"use strict";

/* `emptyHintKey` rather than a resolved string: the picker is built once at script
   load, and a language switched afterwards has to reach the hint it renders. */
function createPillPicker({ pills, input, datalist, emptyHintKey, onChange = () => {} }) {
  let available = [];
  let selected = [];
  let ready = false;  // suppress onChange during construction (TDZ safety)

  const escP = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function render() {
    pills.innerHTML = selected.map((c) => `
      <span class="pill">${escP(c)}
        <button type="button" class="pill-x" data-remove="${escP(c)}"
                aria-label="${escP(t("train.pills.remove", { name: c }))}">&times;</button></span>`).join("");
    pills.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => {
      selected = selected.filter((c) => c !== b.dataset.remove);
      render();
      input.focus();
    }));
    // Suggest only what is not picked yet.
    datalist.innerHTML = available.filter((c) => !selected.includes(c))
      .map((c) => `<option value="${escP(c)}">`).join("");
    input.disabled = !available.length;
    // The key travels with the element: the markup carries one for the pre-script
    // state, and this input has TWO states. Without it, switching language after a
    // dataset was chosen re-applied the markup's key and told the user to choose a
    // dataset in a field that was enabled and ready.
    const placeholderKey = available.length ? "train.pills.typeToAdd" : emptyHintKey;
    input.dataset.i18nPlaceholder = placeholderKey;
    input.placeholder = t(placeholderKey);
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
