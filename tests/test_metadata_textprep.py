"""`app/metadata/textprep`: undoing wrapped lines in linear time, with unchanged results.

M01 (audit 2026-09-30): joining a wrapped line searched the WHOLE text joined so far for a
line-final hyphen and copied it once per line. Lines that start in lower case -- vocabulary
lists, "der ... und die ..." -- all join, so the cost grew with the square of the length:
10k characters 0.04 s, 100k 4.0 s (measured before the fix), and a request at the documented
caps came to 13-60 CPU-minutes. No scaling test reached this path.
"""

import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.metadata import generate, textprep


def _wrapped(chars: int) -> str:
    """Distinct lines that all start in lower case, so every one continues the one before
    (identical lines would be dropped as repeats)."""
    lines, size = [], 0
    while size < chars:
        lines.append(f"der Begriff {len(lines)} und die Definition im Unterricht")
        size += len(lines[-1]) + 1
    return "\n".join(lines)


def test_joining_wrapped_lines_takes_linear_time():
    """A ratio, not a wall-clock budget (see test_metadata_sentences): eight times the input
    takes ~8x when linear and ~64x when quadratic."""

    def best_of_three(text: str) -> float:
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            textprep.clean_text(text)
            timings.append(time.perf_counter() - started)
        return min(timings)

    ratio = best_of_three(_wrapped(100_000)) / best_of_three(_wrapped(12_500))

    assert ratio < 14, f"8x the input took {ratio:.1f}x the time"


class _SearchSpy:
    def __init__(self, pattern: re.Pattern) -> None:
        self.pattern = pattern
        self.lengths: list[int] = []

    def search(self, text: str):
        self.lengths.append(len(text))
        return self.pattern.search(text)


def test_the_hyphen_check_never_reads_more_than_one_line(monkeypatch):
    """The root cause, deterministic: whatever has been joined, a line-final hyphen is a
    question about the END of it."""
    spy = _SearchSpy(textprep._WRAP_HYPHEN_RE)
    monkeypatch.setattr(textprep, "_WRAP_HYPHEN_RE", spy)
    text = _wrapped(20_000)

    textprep.clean_text(text)

    assert spy.lengths, "the hyphen check did not run"
    assert max(spy.lengths) <= max(len(line) for line in text.splitlines())


# The joiner as it was before M01, verbatim, as the reference the fix must agree with.
def _reference_join(joined: str, line: str) -> str:
    if textprep._WRAP_HYPHEN_RE.search(joined):
        next_word = line.split(maxsplit=1)[0].rstrip(".,")
        if line[0].islower() and next_word not in textprep._SUSPENDED_HYPHEN_NEXT:
            return joined[:-1] + line
        if line[0].isupper():
            return joined + line
    return f"{joined} {line}"


def _reference_join_wrapped_lines(lines: list[str]) -> list[str]:
    wrapped = textprep._is_hard_wrapped(lines)
    widths = [len(line) for line in lines if len(line) >= 20]
    full_width = 0.8 * statistics.median(widths) if widths else float("inf")
    joined: list[str] = []
    last = ""
    for line in lines:
        if (
            joined and last and line and not textprep.is_list_item(line)
            and not textprep._SENTENCE_END_RE.search(last)
            and (textprep._breaks_sentence(last, line) or (wrapped and len(last) >= full_width))
        ):
            joined[-1] = _reference_join(joined[-1], line)
        else:
            joined.append(line)
        last = line
    return joined


_WORDS = ["und", "oder", "bis", "Wald", "wasser", "Lern-", "Er-", "lebnis", "Nord-", "Süd-Dialog",
          "die", "Katze.", "Ende!", "Frage?", "Komma,", "x", "Übung", "über", "- Punkt", "1. Punkt",
          "a) klein", "Satz", "-", "é-", "3-"]


def _random_lines(rng: random.Random) -> list[str]:
    lines = []
    for _ in range(rng.randint(0, 14)):
        if rng.random() < 0.15:
            lines.append("")
            continue
        words = [rng.choice(_WORDS) for _ in range(rng.randint(1, 9))]
        lines.append(" ".join(words))
    return lines


def test_joining_gives_exactly_what_it_gave_before():
    """Behaviour-preserving: 20,000 generated line lists -- hyphens, suspended compounds,
    commas, sentence ends, list items, blank lines -- join exactly as before M01."""
    rng = random.Random(20260930)
    for _ in range(20_000):
        lines = _random_lines(rng)
        assert textprep.join_wrapped_lines(lines) == _reference_join_wrapped_lines(lines), lines



# --- M02: parallel requests ------------------------------------------------------------------


def _distinct_texts(count: int) -> list[str]:
    """Real prose, recombined: the parity fixture's sentences, shuffled into ``count`` texts."""
    fixture = Path(__file__).parent / "fixtures" / "metadata_parity.json"
    documents = json.loads(fixture.read_text(encoding="utf-8"))["documents"]
    sentences = [s for doc in documents for s in textprep.split_sentences(doc["text"]) if len(s) > 40]
    rng = random.Random(64)
    return ["\n".join(rng.sample(sentences, 12)) for _ in range(count)]


def test_parallel_requests_get_what_sequential_ones_get():
    """M02 (audit 2026-09-30): pysbd keeps the text it is segmenting on the segmenter, and the
    pure-Python Snowball stemmer keeps its word on the stemmer -- one instance of each, shared
    by every request thread. In parallel, all 64 results differed from the sequential ones,
    18 descriptions held sentences the input does not contain, 24 parallel HTTP requests got
    6-10 answers of 500, and a stem computed mid-race stayed in the cache for good
    ("Nährstoff" -> "franzos"). The stem cache is cleared before each round, or it would
    hide the stemmer's half of the race."""
    texts = _distinct_texts(64)
    textprep.cached_stem.cache_clear()
    sequential = [generate(text) for text in texts]

    with ThreadPoolExecutor(8) as pool:
        for _ in range(3):
            textprep.cached_stem.cache_clear()
            parallel = list(pool.map(generate, texts))
            wrong = sum(p != s for p, s in zip(parallel, sequential, strict=True))
            assert wrong == 0, f"{wrong} of {len(texts)} parallel results differ"



# --- M04: what the caches may hold ---------------------------------------------------------------

_LONG = "Donaudampfschifffahrtsgesellschaftskapitaensmuetze" * 40  # 2,000 characters, one token


def test_a_token_longer_than_any_word_is_not_cached():
    """M04 (audit 2026-09-30): the stem cache counts ENTRIES (200,000), and a token can be as
    long as a text line -- 25 long tokens in each of 200 texts added 55 MiB for 5,000 entries,
    over 2 GB at the cap, none of it visible to the training's memory budget."""
    textprep.cached_stem.cache_clear()
    before = textprep.cached_stem.cache_info().currsize

    assert textprep.stem(_LONG)  # still stemmed -- just not remembered

    assert textprep.cached_stem.cache_info().currsize == before


def test_a_token_longer_than_any_word_is_no_keyword():
    """Nor a candidate: a 2,000-character "keyword" is a fragment, not a term (M04, M06)."""
    text = (f"Die {_LONG} ist ein Beispiel. Die Französische Revolution veränderte Europa. "
            "Die Französische Revolution begann 1789 in Paris.")

    keywords = generate(text).keywords

    assert keywords, "the ordinary keywords are still found"
    assert all(len(word) <= textprep.MAX_TOKEN_CHARS for k in keywords for word in k.split()), keywords



# --- M06 (audit 2026-09-30): the proposals themselves --------------------------------------------


def _occurs(keyword: str, text: str) -> bool:
    """Does the keyword occur in the text as written (its first letter may be capitalised)?"""
    flat = " ".join(text.split()).lower()
    return keyword.lower() in flat


def test_a_keyword_never_runs_across_punctuation():
    """`Mathematik, Physik, Chemie` came back as the keyword "Mathematik Physik Chemie": each
    word occurs in the text, the phrase does not."""
    text = ("Mathematik, Physik, Chemie gehören zu den MINT-Fächern. Mathematik, Physik und "
            "Chemie werden oft gemeinsam unterrichtet. Die Fächer ergänzen sich im Unterricht.")

    keywords = generate(text).keywords

    assert keywords
    assert all(_occurs(k, text) for k in keywords), keywords


def test_a_ligature_is_read_as_its_letters():
    """A PDF writes "ﬂüssige" with one ligature character, whose capital is two letters: the
    keyword came back as "FLüssige Phase"."""
    text = ("Die ﬂüssige Phase entsteht beim Schmelzen. Die ﬂüssige Phase hat kein festes "
            "Volumen und keine feste Form. Beim Erhitzen verdampft die ﬂüssige Phase.")

    keywords = generate(text).keywords

    assert "Flüssige Phase" in keywords, keywords
    assert not any("FL" in k for k in keywords), keywords


def test_the_title_template_adds_no_word_the_text_does_not_have():
    """Every word the endpoint returns occurs in the input -- except the "und" the keyword
    template inserted, in English texts too."""
    text = ("Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs "
            "light in the leaves. Photosynthesis produces oxygen and glucose from carbon dioxide "
            "and water. Chlorophyll gives leaves their green colour.")

    title = generate(text).title

    assert " und " not in title, title


def test_the_description_takes_no_sentence_whose_reference_was_skipped():
    """The sentence introducing Napoleon does not fit the 500-character budget and is skipped;
    the one after it opens with "Er" -- which then pointed at the Revolution."""
    first = "Die Französische Revolution begann im Jahr 1789 mit dem Sturm auf die Bastille in Paris."
    second = ("Sie veränderte die politische Ordnung Europas, beendete die absolute Monarchie in "
              "Frankreich, die über Jahrhunderte bestanden hatte, und brachte neue Ideen von "
              "Freiheit und Gleichheit hervor.")
    napoleon = ("Napoleon Bonaparte, ein junger Offizier aus Korsika, nutzte die unruhigen Jahre "
                "nach der Revolution geschickt für seinen Aufstieg, gewann zahlreiche Schlachten "
                "in Italien und Ägypten, stürzte im Jahr 1799 das Direktorium in einem "
                "Staatsstreich und übernahm als Erster Konsul die Macht in der Republik.")
    crowned = "Er krönte sich im Jahr 1804 selbst zum Kaiser der Franzosen."

    description = generate(" ".join([first, second, napoleon, crowned])).description

    assert description.startswith(first), description
    assert napoleon not in description, "test setup: the Napoleon sentence has to be skipped"
    assert crowned not in description, description


def test_a_boilerplate_phrase_inside_a_paragraph_does_not_drop_the_paragraph():
    """"Datenschutzerklärung" in a sentence ABOUT data protection is content, not chrome; the
    phrases now mark only lines short enough to be chrome."""
    paragraph = ("Jede Website, die personenbezogene Daten verarbeitet, braucht eine "
                 "Datenschutzerklärung, in der steht, welche Daten erhoben werden, wozu sie "
                 "verwendet werden und wie lange sie gespeichert bleiben. Das verlangt die DSGVO.")

    assert "Datenschutzerklärung" in textprep.clean_text(paragraph)
