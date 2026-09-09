# Bedienungsleitfaden: Admin-Oberfläche (für Nicht-Techniker)

> *This end-user guide is intentionally written in German for the editorial team;
> developer documentation stays in English.*

Die Oberfläche erreichst du im Browser unter **`http://<server>:8000/ui/`**.
Jeder Bereich hat dort zusätzlich eine aufklappbare Hilfe („What do the results
mean?" usw.) — dieser Leitfaden erklärt alles im Zusammenhang.

## Anmeldung

Du bekommst vom Admin einen **API-Schlüssel** (eine Art Passwort). Es gibt zwei Sorten:

| Schlüssel | Darfst du damit |
|---|---|
| **Readonly** | Texte klassifizieren, Status ansehen, Listen ansehen |
| **Admin** | zusätzlich: trainieren, hochladen, löschen, exportieren |

Den Schlüssel einmal oben eintippen → **Sign in**. Er gilt nur in diesem
Browser-Tab und ist nach dem Schließen wieder weg (bewusst, aus Sicherheitsgründen).
Läuft der Server ohne Anmeldepflicht (lokaler Betrieb), erscheint statt der
Anmeldung direkt die Oberfläche mit dem Hinweis „auth disabled".

## Der Grundgedanke in einem Satz

Das System **lernt aus Beispielen**: Du gibst ihm eine Tabelle mit Texten und den
richtigen Labels (z. B. Schulfächern), es lernt den Zusammenhang — danach kann es
**neue** Texte selbstständig einordnen.

## Schritt für Schritt: das erste Modell

1. **Datasets** → CSV-Datei hochladen. Eine Zeile pro Material; Textspalten
   (Titel, Beschreibung, …) und mindestens eine Spalte mit den richtigen Labels
   (mehrere Labels in einer Zelle durch Komma getrennt).
2. **Training** → Dataset auswählen → bei *Text columns* die Spalten antippen,
   die das System **lesen** soll (Titel + Beschreibung + Schlagwörter ist meist
   die beste Wahl) → bei *Label fields* die Spalte mit den **richtigen
   Antworten** wählen.
   - Wählst du **mehrere** Label-Felder, entsteht **pro Feld ein eigenes
     Modell**; die Namen werden automatisch abgeleitet (`meinname_taxonid`, …)
     und die Trainings laufen nacheinander — **Tab offen lassen**, bis alle
     gestartet sind.
3. **Profile:** Drei Stufen, aufsteigend nach Rechenzeit — `auto` ist die
   Empfehlung. `fast` nur zum schnellen Ausprobieren (nicht für ein Modell, das
   in Betrieb geht), `best` für die genaueste Bewertung (~1,9× so lange wie
   `auto`; der Unterschied sind allein 5 statt 3 Bewertungsdurchläufe).
4. **Evaluation:** Wie ehrlich die Qualität gemessen wird. Jedes Profil bringt
   seine passende Einstellung schon mit — „Profile default" belässt es dabei.
   - *Train/val/test split*: ein Teil der Daten wird als „unbekannte Prüfung"
     beiseitegelegt. Der Preis: **dieser Teil fließt nie ins Training ein**, das
     ausgelieferte Modell lernt nur aus 85 % der Zeilen.
   - *Cross-validation*: jede Zeile ist abwechselnd Lernstoff **und** Prüfung —
     das ausgelieferte Modell wird auf **100 %** der Daten trainiert, und
     bewertet wird über alle Zeilen statt über eine Stichprobe. Kostet ein
     Mehrfaches an Rechenzeit; mehr Folds = die Prüfungsmodelle sehen mehr Daten.
5. **Start training** → der Fortschrittsbalken (auch oben als kleine Anzeige auf
   jedem Tab sichtbar) zeigt Phase, Prozent und Restzeit. Ein Training auf
   ~30.000 Zeilen dauert je nach Einstellung wenige Minuten bis ~1 Stunde; der
   Server bleibt dabei bedienbar (das Training nimmt sich höchstens ~60 % der
   Rechenleistung). **Stop training** bricht sauber ab.

## Ergebnisse lesen (Query-Tab)

Text eingeben, Modell(e) anhaken, **Classify**.

- **Confidence (Balken + Zahl 0–1):** Wie sicher das Modell ist, dass das Label
  passt. Die Werte der Labels sind unabhängig voneinander — sie müssen sich
  nicht zu 1 addieren, und **mehrere Labels gleichzeitig sind ein normales
  Ergebnis** (ein Text kann Physik *und* Elektrotechnik sein).
- **Top-k leer (Standard):** Du siehst nur Labels, hinter denen das Modell
  wirklich steht. Oft ist das genau eines — das ist dann die ehrliche Antwort,
  kein Fehler.
- **Top-k = N:** Du siehst immer die N wahrscheinlichsten Labels — auch schwache.
  **Ausgegraute Einträge** („below threshold") sind Kandidaten, die das Modell
  von sich aus *nicht* behaupten würde. Nützlich, um die „zweite Meinung" des
  Modells zu sehen.
- **diff (Baseline-Differenz, Häkchen „Show baseline diff"):** Wie viel der
  Sicherheit aus **deinem Text** kommt — und nicht daher, dass das Label einfach
  häufig ist. Ein hoher Confidence-Wert mit *diff nahe 0* heißt: das Modell rät
  auf das übliche Label, dein Text hat es nicht überzeugt.
- Mehrere Modelle gleichzeitig anhaken (Strg-Klick) → eine Antwort pro Modell,
  z. B. Fach **und** Materialart in einem Rutsch.
- Die **allererste** Abfrage nach einem Server-Neustart kann ~30 Sekunden
  dauern (das Modell wird von der Festplatte geladen) — danach kommen Antworten
  in Millisekunden.

## Qualität verstehen (Models-Tab)

- **F1 (0–1):** Wie gut die Antworten des Modells bei einer ehrlichen Prüfung
  mit ungesehenen Daten waren. 1,0 wäre perfekt; **~0,8 ist in der Praxis sehr
  brauchbar**.
- **micro vs. macro:** *micro* = Gesamt-Trefferquote (häufige Labels zählen
  mehr). *macro* = jedes Label zählt gleich viel — seltene Labels drücken den
  Wert, deshalb ist macro fast immer niedriger. Ein großer Abstand zwischen
  beiden heißt: bei seltenen Labels ist das Modell schwächer.
- **Evaluation** zeigt, wie gemessen wurde (Split oder Cross-Validation).
- **Klick auf den Modellnamen** öffnet die Detailansicht: worauf trainiert wurde
  (Datensatz, Textspalten samt Gewichtung, Profil, Zeilen, Regularisierung C und
  das durchsuchte Gitter, Dauer, Datum) und **jedes Label einzeln** — F1,
  Zeilenzahl und die Schwelle, die beim Antworten wirklich angewendet wird.
  Sortierbar; standardmäßig **schwächstes Label zuerst**, denn das ist die Stelle,
  an der eine Antwort einen zweiten Blick verdient.
  - Steht bei **C** der Hinweis *"the winner sits at the edge of the grid"*, lag
    der beste Wert am Rand des durchsuchten Bereichs — ein besserer könnte
    außerhalb liegen. Abhilfe ist ein breiteres Gitter, nicht mehr Folds.
  - **Zeilen** bleibt bei Modellen leer, die trainiert wurden, bevor diese Zahl
    aufgezeichnet wurde. **„argmax"** in der Schwellen-Spalte heißt: das Modell
    nimmt sein bestes Label und liest gar keine Schwelle.

## Teilen & Verschieben

- **Share link** (bei Modellen und Datasets): erzeugt einen Download-Link, der
  **ohne Schlüssel** funktioniert und nach 24 Stunden erlischt — gut, um jemandem
  etwas zu geben, der keinen Zugang hat. Der Link ist wie ein Schlüssel zu
  behandeln: wer ihn hat, kann herunterladen.
- **Download / Import:** Ein Modell wandert als ZIP-Datei zwischen Servern:
  auf Server A herunterladen (oder Share-Link), auf Server B über *Import model*
  hochladen. Ein direkter „Import per Link" ist **absichtlich** nicht möglich
  (Sicherheitsentscheidung — der Server ruft nie selbst fremde Adressen ab).

## Häufige Fragen

**Ich bekomme nur 1 Fach, obwohl ich Top-k=5 gesetzt habe?**
Das war ein früheres Verhalten und ist behoben — heute liefert Top-k=5 immer
5 Einträge (die schwachen ausgegraut). Einmal die Seite neu laden, falls du es
noch siehst.

**Warum sind es nach dem Hochladen weniger Zeilen als in meiner Datei?**
Doppelte Texte werden vor dem Training automatisch entfernt (sonst würde sich
das Modell selbst prüfen und die Qualität schönrechnen). Auch Zeilen ohne Label
oder mit fast leerem Text fallen raus.

**Kann ich den Tab während des Trainings schließen?**
Das *laufende* Training läuft auf dem Server weiter. Nur wenn du **mehrere**
Label-Felder gewählt hast, muss der Tab offen bleiben, bis alle Trainings
gestartet wurden (die Warteschlange lebt im Tab).

**Was passiert bei einem Tippfehler im Spaltennamen?**
Nichts Schlimmes — die Oberfläche bietet ohnehin nur existierende Spalten an,
und die API antwortet mit einer klaren Meldung inklusive der verfügbaren Spalten.
