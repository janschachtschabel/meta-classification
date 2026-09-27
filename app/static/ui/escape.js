/* HTML escaping for every `${…}` the UI interpolates — the app's single XSS defence.

   Its own file, and the first script the page loads, for a load-order reason: the pill
   pickers in training.js render at CONSTRUCTION time, i.e. while their own script runs,
   which is before app.js (loaded last) has executed. A `const` in a later script is not
   yet declared then, so pills.js carried a verbatim copy of this function to stay
   callable. Two copies of an escape is one that gets hardened and one that quietly does
   not, so the definition moved here instead — early enough for every caller, including
   the ones that render before the shell exists. */
"use strict";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
