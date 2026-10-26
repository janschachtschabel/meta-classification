# Design: descriptive metadata -- titles that make sense, no lowercase starts, a word on how

## Goal
Replace the title template that joins three unrelated keywords ("Valerie: Katzen und Micki"),
stop titles and descriptions from starting in lower case, and tell the user in the admin UI how
the proposals come about. Small changes inside `app/metadata`, no new dependency, no slower.

## What was wrong (reproduced 2026-10-26)
- **Title template** `K1: K2 und K3` from the three best keywords. On running text it is
  almost always that pattern, and the three are ranked by RARITY, so the title names details:
  "Komplexitätsklasse: Element, Standardbibliothek" for a text on binary search, "Katheten:
  Hypotenuse und Seitenlänge" for Pythagoras, inflected forms included ("Geheimdiensten: ...").
- **Lowercase starts**, three causes:
  1. a lowercase first line passes as a heading: "valerie und ihre katzen", a byline
     "von Anna Schmidt";
  2. a text that begins mid-sentence: the description starts "und dann kam ...";
  3. pysbd ends a sentence after an abbreviation it does not know ("im 19. Jh. und prägte",
     "ca. 4 kg. und war"): the next segment, and so the description, starts "und prägte ...".
- **Keyword candidates** miss nouns with a capital inside ("pH-Wert", six times in its text, never
  a candidate) and adjectives in "-är" ("binäre Suche").

## Approach
Measured on the 8-text gold set supplied with the external `textmeta` experiment plus the source
app's 5 own texts, title token-F1 against the editor's title (stemmed content words), each text
once as written and once with its heading removed:

| Title strategy | with heading | running text |
|---|---|---|
| today: heading, else `K1: K2 und K3` | 0.57 | 0.18 |
| heading, else definitional subject (top-5 check), else K1 | 0.58 | 0.21 |
| **heading, else definitional subject if its noun recurs, else the most frequent of the top-8 keywords** | **0.68** | **0.38** |
| same, plus "A und B" when both share a sentence | 0.68 | 0.38 |

Chosen: the third row. "A und B" adds nothing measurable and is the kind of pairing that was
reported. The definitional subject is taken from `textmeta` (`_DEFINITION_SUBJECT`): the article,
lowercase adjectives, a noun, an optional genitive attribute, then a lowercase verb --
"Der Satz des Pythagoras ist ..." -> "Der Satz des Pythagoras", "Die binäre Suche ist ..." ->
"Die binäre Suche". Accepted only when one of its nouns recurs in the text ("Die Lehrkraft
erklärt heute etwas." does not) and that noun is not generic -- "Die Schülerinnen und Schüler
lernen ...", "Das Arbeitsblatt enthält ..." recur precisely because they are generic (review
finding, the keywords' `GENERIC_TERMS` list decides).

Lowercase starts: a heading must not open with an all-lowercase word; a description sentence must
not either (a fragment); a pysbd segment that opens in lower case continues the sentence before it
on the same line when that one ends in an abbreviation -- a word of at most four letters or a
number before the period (the comma repair beside it does the same for "," and ";"; joining after
ANY period made a text written all in lower case one sentence, review finding); and as the last
guard, a title or description whose first word is
lowercase is capitalised (every word still occurs in the input -- the grounding tests compare
case-insensitively, as the keywords are already capitalised).

Keyword tagging: a token with a capital after its first letter is a noun ("pH-Wert", "eBook");
"-är" joins the adjective suffixes. Gold keyword F1@5: 0.42 -> 0.45 exact, 0.52 -> 0.55 by head
noun. The parity fixture's five documents produce exactly what they did before (all three fields).

Not taken from `textmeta`: its own sentence splitter (pysbd stays), its scikit-learn vectorizer
(the keyword score keeps German word frequencies, so no model is needed), MMR/centroid
descriptions (the lead description stays, as measured in the source app), the existing-title
splitting (the endpoint has no such input).

## Caveats
The gold set was written by the author of `textmeta`, and 13 texts are few: the numbers show
direction, not precision. The source app's semantic title score (embedding similarity, which
favoured the three-keyword template for packing in topic words) was not re-measured -- this
change trades it for titles that read as titles.

## Files
- `app/metadata/fields.py`: the title order (heading -> definitional subject -> frequent keyword
  -> first sentence); `_is_title_like` refuses a lowercase opener; the capitalisation guard.
- `app/metadata/phrases.py`: tagging (inner capital, "-är"); keywords with their counts.
- `app/metadata/sentences.py`: a lowercase segment continues the sentence before it.
- `app/metadata/textprep.py`: a description sentence opens with a capital (or a digit/quote).
- `app/metadata/budget.py`: `capitalised`.
- UI: a "how are these made" note in the metadata card (`query.js`, strings DE/EN).
- Tests: `tests/test_metadata_titles.py` (new); the parity test of the old template is replaced.
