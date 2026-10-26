"""Titles that read as titles, and no title or description that starts in lower case.

A text without a heading got the title "K1: K2 und K3" from its three rarest keywords --
"Valerie: Katzen und Micki", "Komplexitätsklasse: Element, Standardbibliothek" for a text on
binary search. And a lowercase first line, a text starting mid-sentence or an abbreviation pysbd
does not know made the title or the description start in lower case. The measurements behind
the replacement are in docs/plans/2026-10-26-metadata-titles.md.
"""

from app.metadata import generate, prepare

# --------------------------------------------------------------------------------- titles ---


def test_a_definitional_opening_names_the_text():
    text = ("Die binäre Suche ist ein effizientes Verfahren, um ein Element in einer sortierten "
            "Liste zu finden. Statt die Liste zu durchlaufen, vergleicht der Algorithmus den Wert "
            "mit dem mittleren Element. Bei jedem Schritt halbiert sich der Suchbereich. In vielen "
            "Programmiersprachen ist die binäre Suche bereits in der Standardbibliothek enthalten.")

    assert generate(text).title == "Die binäre Suche"


def test_a_definitional_subject_keeps_its_genitive():
    text = ("Der Satz des Pythagoras ist einer der bekanntesten Sätze der Geometrie. Er gilt in "
            "jedem rechtwinkligen Dreieck. Die beiden kürzeren Seiten heißen Katheten, die längste "
            "heißt Hypotenuse. Mit dem Satz des Pythagoras lässt sich eine fehlende Seitenlänge "
            "berechnen.")

    assert generate(text).title == "Der Satz des Pythagoras"


def test_a_subject_the_text_never_returns_to_is_no_title():
    text = ("Die Lehrkraft erklärt heute etwas Neues über Pflanzen. Chlorophyll absorbiert Licht "
            "in den Chloroplasten der Blätter. Chlorophyll ist grün und steckt in jedem Blatt. Ohne "
            "Chlorophyll gäbe es keine Fotosynthese.")

    title = generate(text).title

    assert "Lehrkraft" not in title, title
    assert title == "Chlorophyll"


def test_without_a_subject_the_title_is_the_most_frequent_keyword_never_a_template():
    # The reported shape: a name, a colon, two more words.
    text = ("Valerie hat zwei Katzen. Die Katzen heißen Micki und Mausi und spielen gern im "
            "Garten. Morgens füttert Valerie die Katzen mit Futter aus der Dose. Abends schlafen "
            "die Katzen auf dem Sofa neben Valerie.")

    title = generate(text).title

    assert ":" not in title and " und " not in title, title
    assert "Katzen" in title, "the topic, not a side character"


def test_a_noun_with_an_inner_capital_is_a_keyword_and_can_be_the_title():
    text = ("Der pH-Wert gibt an, wie sauer oder basisch eine wässrige Lösung ist. Lösungen mit "
            "einem pH-Wert unter 7 sind sauer. Reines Wasser hat einen pH-Wert von 7. Im Unterricht "
            "wird der pH-Wert mit Indikatoren wie Rotkohlsaft bestimmt.")

    result = generate(text)

    assert result.title == "Der pH-Wert"
    assert "pH-Wert" in result.keywords, result.keywords


def test_an_adjective_in_aer_starts_a_keyword_phrase():
    text = ("Die binäre Suche halbiert den Suchbereich bei jedem Schritt. Eine binäre Suche "
            "braucht eine sortierte Liste. Ohne Sortierung funktioniert die binäre Suche nicht.")

    assert "Binäre Suche" in generate(text).keywords


# ------------------------------------------------------------------------ lowercase starts ---


def test_a_lowercase_first_line_is_no_heading():
    text = ("valerie und ihre katzen\n\nValerie hat zwei Katzen. Die Katzen heißen Micki und "
            "Mausi und spielen gern im Garten. Valerie füttert die Katzen jeden Morgen.")

    assert generate(text).title[:1].isupper()


def test_a_byline_is_no_heading():
    text = ("von Anna Schmidt\n\nKatzen sind beliebte Haustiere in vielen Familien. Sie brauchen "
            "Futter, Wasser und viel Zuwendung. Katzen schlafen bis zu sechzehn Stunden am Tag.")

    assert "Anna Schmidt" not in generate(text).title


def test_a_text_starting_mid_sentence_gets_a_description_from_its_first_full_sentence():
    text = ("und dann kam Micki endlich wieder nach Hause zu Valerie. Valerie freute sich sehr "
            "über die Rückkehr der Katze. Die Katzen spielen nun wieder zusammen im Garten.")

    assert generate(text).description.startswith("Valerie freute sich")


def test_an_unknown_abbreviation_does_not_split_the_sentence():
    text = ("Das geschah im 19. Jh. und prägte die Region bis heute nachhaltig. Die Katze wog "
            "damals ca. 4 kg. und war damit recht schwer für ihr Alter.")

    assert "Das geschah im 19. Jh. und prägte die Region bis heute nachhaltig." in prepare(text).sentences
    assert generate(text).description[:1].isupper()


def test_the_last_resort_capitalises_a_lowercase_start():
    # Too short for any sentence to qualify: the description falls back to what there is.
    result = generate("und dann schlief die Katze ein")

    assert result.title[:1].isupper(), result.title
    assert result.description[:1].isupper(), result.description


def test_a_word_that_opens_with_a_capital_inside_is_left_as_written():
    result = generate("pH-Wert messen im Unterricht\n\nDer pH-Wert zeigt, wie sauer eine Lösung "
                      "ist. Der pH-Wert wird mit Indikatoren bestimmt.")

    assert result.title == "pH-Wert messen im Unterricht"
