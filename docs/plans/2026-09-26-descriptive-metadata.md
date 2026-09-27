# Descriptive metadata: title, description and keywords from the text

Status: implemented · 2026-09-26

## Goal

Give the API a way to propose a **title**, a **description** and **keywords** for a text, so
that a classified item can be handed on as a complete metadata record rather than as labels
alone. One extra endpoint, and a switch in the Query tab that asks for it alongside the
classification — or, since 2026-09-27, **instead of** one. The endpoint reads no model, so
refusing a submit without one made the feature unreachable on an instance with nothing
trained, which is every instance on its first day.

## Where it comes from

`C:\Users\jan\staging\Windsurf\static-metadata-generators` measured 52 methods from eight
families on two test sets and settled on a default combination
(`metagen/recommended.py`, argued in its `docs/bewertung.md`). **Only that default
combination is ported — no method registry, no alternatives, no selectable variants.**

| Field | Method there | Ported |
|---|---|---|
| Title | `title.heading_or_keywords` — a real heading at the top of the text, else the template `K1: K2 und K3` from the three best TF-IDF keywords | yes, unchanged |
| Keywords | `keywords.tfidf` — noun phrases weighted by frequency in the text and rarity in German | yes, unchanged |
| Description | `description.lead_centroid` — first sentence + the most central sentences (Model2Vec `potion`) | **no — `description.lead` instead**, see below |

## The one deviation, and why

`description.lead_centroid` needs `model2vec` and downloads
`minishlab/potion-multilingual-128M` from Hugging Face on first use. Measured locally:
**1002 MB** in the HF cache, 4.9 s cold start. Two problems, one of them a hard stop:

- It fetches from an external host at runtime. CLAUDE.md: *"model/dataset import via file
  upload only (no URL fetch)"*, asserted by `tests/test_no_url_fetch.py`. A 1 GB runtime
  download is exactly what that boundary exists to prevent.
- It is not the better method on real material. From the source repo's own measurements
  (`docs/methodenvergleich.md`, "Beschreibung → Messwerte"):

  | Method | T1 sem | T1 q | **T2 note** | T2 fits · rework · no | typical | cold start |
  |---|---|---|---|---|---|---|
  | `description.lead` | 0,68 | 1,2 | **1,2** | **4 · 4 · 2** | 1 ms | 2 ms |
  | `description.lead_centroid` | 0,72 | 1,4 | 1,0 | 1 · 8 · 1 | 7 ms | 4,9 s |

  Test set 2 is the one made of real mixed materials, each output rated individually, and
  `bewertung.md` says so itself: *"Der reine Textanfang liest sich in Testset 2 etwas besser
  (1,2 statt 1,0)"*. `lead_centroid` is steadier (it is rarely unusable), `lead` is right more
  often.

`description.lead` is the opening-sentence half of `lead_centroid` — same
`description_candidates`, same budget fitting — so this is the default combination minus its
model, not a different approach.

## Dependencies

Three new runtime deps, 13 → 16. CLAUDE.md requires the justification in writing:

| Package | Size | Why it earns its place | Licence |
|---|---|---|---|
| `wordfreq` | 58 MB | The IDF term of the keyword score: rarity of a word in German. Load-bearing — it is the difference between the best keyword method (q 1,8) and the best one without it (TextRank, q 1,2), and the title template is built from the same keywords. | Apache-2.0 (code) |
| `pysbd` | 1 MB, no deps | German sentence splitting. Every field reads `doc.sentences`; naive `[.!?]` splitting breaks on German abbreviations and ordinals ("Ludwig XVI.", "z. B."), which the source repo hit with an English splitter. | MIT |
| `snowballstemmer` | 1 MB, no deps | Stems merge the surface forms of one candidate ("Brüche"/"Bruch") and dedupe the keyword list. | BSD-3 |

Measured together: **872 ms** to import and do the first lookup, 274 modules, and nothing
network-ish beyond `urllib.parse` (string splitting, which `test_no_url_fetch.py` explicitly
allows).

### Two rejected alternatives, both measured

**Vendoring wordfreq's German table instead of depending on the package.** `large_de.msgpack.gz`
is only 3,6 MB of wordfreq's 58 MB, so this looked attractive. Rejected on licence: wordfreq's
*code* is Apache-2.0 but its *data* is CC-BY-SA 4.0 (Wikipedia, ParaCrawl, OpenSubtitles,
SUBTLEX). A derived table committed here would put a ShareAlike obligation on an MIT repo.
As a dependency the data stays with the package, exactly as in the source app.

**`nltk` for the stemmer**, which is what the source app uses
(`from nltk.stem.snowball import SnowballStemmer`). Rejected on two measurements:

- `import nltk.stem.snowball` takes **10,4 s** and loads 1741 modules, for one function.
- It pulls **`urllib.request`, `http.client`, `socket`, `ssl`** into the process, because
  `nltk/__init__.py` imports its downloader. `test_no_url_fetch.py` only scans `app/`, so
  nothing would have gone red — but the property it protects ("with no client imported, no
  route CAN fetch a URL") would have been quietly broken.

`snowballstemmer`'s `german` is the **German2** variant, so it is not bit-identical to nltk's.
Measured on 627,123 German words: the stems differ on 2,66 %. Measured where it matters — the
full pipeline over the source repo's five sample documents:

- all five titles identical, all five descriptions identical, all five keyword **sets** identical
- one document (`05-gedichtanalyse`) ranks the same eight keywords in a different order
  ("Stilmittel" 5th instead of 8th)

That is the whole cost, and it is recorded in the parity test so it cannot drift further unnoticed.

## Architecture

```
app/metadata/__init__.py     public surface: generate(text, settings) -> DescriptiveMetadata
app/metadata/textprep.py     clean, split into sentences, tokenize, stopwords, stem
app/metadata/phrases.py      noun-phrase candidates, TF-IDF score, keyword finalisation
app/metadata/budget.py       fit into a character budget; polish a title
app/metadata/fields.py       the three default methods
app/metadata/resources/      stopwords_de.txt, generic_terms_de.txt (vendored, 5 KB, MIT)
app/schemas/metadata.py      MetadataRequest
app/routes/metadata.py       POST /metadata — thin, delegates to app.metadata
```

Dependency direction stays inward: `routes → schemas → metadata`, and `app/metadata` imports
nothing from `app/`. It does not touch the registry, a model bundle or the volume — the text
in the request is all it reads.

## Contract

`POST /metadata`, auth **readonly**, rate limit `predict_limit`, body run in a worker thread
(single-worker design; `test_event_loop_is_not_blocked.py` guards this).

```
{ "texts": ["..."], "title_max": 90, "desc_max": 500, "n_keywords": 8 }
->
{ "results": [ { "text": "<truncated echo>", "title": "...",
                 "description": "...", "keywords": ["...", "..."] } ] }
```

Bounds at the trust boundary, per text ≤ 100,000 characters as `/predict` has. The batch cap
is set from a measurement (below), not copied from `/predict`: the work per text is far heavier
than a TF-IDF transform.

## Verification plan

1. **Parity** — the ported pipeline reproduces the source app's output for the three default
   methods on its five sample documents, byte for byte, with the one keyword-order difference
   recorded explicitly. Fixture generated from the source app *before* the port is written.
2. **Endpoint** — auth, bounds (empty list, oversized text, bad numbers), the echo truncation,
   and that the event loop is not blocked.
3. **No regression** — `test_no_url_fetch.py`, `test_dependency_locks.py` and the full suite
   stay green; the new imports add no outbound client.
4. **UI** — the switch asks for the metadata, the card renders it, both language files carry
   every new key (`test_ui_i18n.py`), and the a11y/contrast rules still hold.

## Follow-up, resolved 2026-09-26: markup and the warmstart

Two items were left open when the feature landed; both are now implemented.

### Markup stripping

The original note said this needed "a newline-preserving variant of `data.py`'s regexes,
which is a shared-helper design question". Measuring it answered the question and changed the
answer: the two pipelines cannot share one *function*, only the patterns.

- `app/markup.py` now owns the patterns and both compositions. `data.clean_text` keeps its
  original order and is unchanged — **differential-tested at 0 divergences over 1,546,832
  inputs**: 35 hand-picked adversarial cases, 400,000 fuzz strings from a markup-heavy
  alphabet, and 1.1 M real rows from `data_30k.csv` and `data_300k.csv`, each compared against
  the pre-change implementation reconstructed from the diff.
- **Tags are removed in ONE pass**, with a replacement function choosing `"\n"` for a
  block-level tag and `" "` for anything else. The first attempt used two passes — block tags
  first, then the general pattern — on the argument that `"\n"` versus `" "` is invisible once
  the whitespace collapse runs. That argument was wrong, and the review caught it: substituting
  the block tag *deletes the `<` and `>` that stop `HTML_TAG_RE`'s `[^<>]+`*, so the second
  pass joins a bare `<` before the tag to a bare `>` after it and eats the prose between.
  `Preis < 5 Euro <br> Menge > 3 Stück` came back as `Preis 3 Stück` — live data loss on the
  `/predict` path, since German prose comparing values around an HTML paragraph is ordinary
  input. Pinned in `tests/test_data.py`. Matching exactly the set a single tag pass always
  matched is what makes sharing safe.
- Two refinements belong to the metadata pipeline **only**, and applying them to
  `clean_text` would shift the features of every already-trained model:
  - **Emphasis is decided per run, in Python, by two explicit rules.** A run longer than three
    of the *same* marker is a rule or a form blank, not emphasis: measured on parity fixture 02,
    a real worksheet, `Name: ____________________   Datum: ____________` is 21% letters so
    `letter_ratio` drops it, but strip the underscores as `data.clean_text` does and it becomes
    `Name: Datum:` — 82% letters, kept, and a line that says nothing lands in the description.
    A run between two word characters belongs to the word: `arbeitsblatt_1_loesung.pdf` must not
    become `arbeitsblatt1loesung.pdf`, because the endpoint guarantees every word it returns
    occurs in the input, and that rewrite returns one that never did. Both were review findings
    against a first version that used lookarounds; the rules replaced them because a mixed run
    (`**_fett_**`, `~~**weg**~~`) also has to go, which no single lookaround expressed.
  - **The bodies of `script`, `style`, `nav` and `footer` are removed.** Both halves came
    from measurement, not foresight. With the tags gone but the body kept,
    `<h1>Bruchrechnung</h1><script>var tracker = {id: 42}; ...` produced the title
    `Bruchrechnung var tracker = {id: 42}; function send(){ ... }`; and an end-to-end check on
    a realistic scraped page showed a breadcrumb `<nav>` becoming the document's first line and
    winning the heading heuristic — proposed title `Startseite » Mathematik` instead of the
    `<h1>` directly below. `header` and `aside` are deliberately excluded: pages put the real
    `<h1>` inside `<header>`, and an `<aside>` often carries a definition box. The
    pattern is an unrolled loop rather than a lazy `.*?`, which would rescan to the end of the
    document once per unclosed `<script` — the quadratic shape this project already paid 52 s
    for once. It runs *before* entities are decoded, so a prose mention of `&lt;script&gt;`
    does not read as a real unclosed tag and take the rest of the sentence with it.
- Closed ATX headings (`### Titel ###`) lose both marker runs, anchored to lines that open
  with `#` so `Die Antwort steht in Aufgabe #5 #` keeps its last word. The trailing run is
  trimmed with two `rstrip`s rather than an optional group in the pattern, and that is the
  second thing the review caught: written as
  `#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$`, the lazy body, the optional closing run and the
  trailing whitespace all compete for the same characters, so a heading line ending in anything
  that is neither space nor `#` backtracks quadratically. **Measured 9.2 s for one
  25,000-character line**, extrapolating to ~150 s at the 100,000 a request may carry, with the
  GIL held — reachable with a readonly key, and the exact incident class this file's comments
  claim to have eliminated. Now 1.4 ms at 25,000 and 5.3 ms at 100,000.
- Quote markers are removed before heading markers, and repeatedly: `_MD_HEADING_RE` is anchored
  to `^`, so `> # Titel` was not a heading to it and kept its `#`.
- **The parity fixture still passes unchanged** (23/23). The markup step is a no-op on the
  five plain-text documents, which is what makes it additive rather than a re-specification.
- Every pattern here now carries a scaling measurement in
  `tests/test_metadata_markup.py::test_markup_removal_does_not_backtrack`. The first version of
  that test exercised only the patterns that were already fixed, which is precisely why the new
  quadratic one shipped green — the input list now covers each pattern the pipeline adds.

### Warmstart

`wordfreq` loads its German frequency table on the first lookup, not at import. Measured:

| | boot | RSS after boot | first request | second |
|---|---|---|---|---|
| `APIV3_WARMUP_METADATA=false` | 11–14 ms | 189 MB | **328–381 ms** | 6–11 ms |
| `APIV3_WARMUP_METADATA=true` | 302–344 ms | 248.6 MB | **10–11 ms** | 4 ms |

The table load alone is 245-286 ms over four fresh interpreters; the rest of the first
request is pysbd's regex compilation (~20 ms) and the generation itself.

**Off by default**, which is the one real decision here. The time is one-off but the
~58 MB is not — the table stays resident for the life of the process, and `ThreadBudget`
sizes how many head fits may run at once against the cgroup limit, so a table nothing reads
would quietly buy a training fewer parallel fits. A deployment serving `/metadata` pays that
memory on its first request anyway and should turn the setting on.

No result cache was added. `stem` already carries `lru_cache(200_000)` and `_idf`
`lru_cache(100_000)` — the two hot inner functions — and a warm generation is ~20 ms, which
does not justify holding request texts in memory in a process whose RAM is budgeted.
