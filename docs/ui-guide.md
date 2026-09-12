# Bedienungsleitfaden: Admin-Oberfläche (für Nicht-Techniker)

> *This end-user guide is intentionally written in German for the editorial team;
> developer documentation stays in English.*

Die Oberfläche erreichst du im Browser unter **`http://<server>:8000/ui/`**.
Sie spricht Deutsch und Englisch; oben rechts steht ein Umschalter, und die Wahl
bleibt für den nächsten Besuch gespeichert. Dieser Leitfaden nennt die **deutschen**
Beschriftungen. Jeder Bereich hat zusätzlich eine aufklappbare Hilfe („Was bedeuten
die Ergebnisse?“ usw.) — dieser Leitfaden erklärt alles im Zusammenhang.

## Anmeldung

Du bekommst vom Admin einen **API-Schlüssel** (eine Art Passwort). Es gibt zwei Sorten:

| Schlüssel | Darfst du damit |
|---|---|
| **Readonly** | Texte klassifizieren, Status ansehen, Listen ansehen |
| **Admin** | zusätzlich: trainieren, hochladen, löschen, exportieren |

Den Schlüssel einmal oben eintippen → **Anmelden**. Er gilt nur in diesem
Browser-Tab und ist nach dem Schließen wieder weg (bewusst, aus Sicherheitsgründen).
Läuft der Server ohne Anmeldepflicht (lokaler Betrieb), erscheint statt der
Anmeldung direkt die Oberfläche mit dem Hinweis „Auth deaktiviert“.

## Der Grundgedanke in einem Satz

Das System **lernt aus Beispielen**: Du gibst ihm eine Tabelle mit Texten und den
richtigen Labels (z. B. Schulfächern), es lernt den Zusammenhang — danach kann es
**neue** Texte selbstständig einordnen.

## Schritt für Schritt: das erste Modell

1. **Datensätze** → CSV-Datei hochladen. Eine Zeile pro Material; Textspalten
   (Titel, Beschreibung, …) und mindestens eine Spalte mit den richtigen Labels
   (mehrere Labels in einer Zelle durch Komma getrennt).
2. **Training** → Dataset auswählen → bei *Textspalten* die Spalten antippen,
   die das System **lesen** soll (Titel + Beschreibung + Schlagwörter ist meist
   die beste Wahl) → bei *Label-Felder* die Spalte mit den **richtigen
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
   seine passende Einstellung schon mit — „Profile default“ belässt es dabei.
   - *Train/val/test split*: ein Teil der Daten wird als „unbekannte Prüfung“
     beiseitegelegt. Der Preis: **dieser Teil fließt nie ins Training ein**, das
     ausgelieferte Modell lernt nur aus 85 % der Zeilen.
   - *Cross-validation*: jede Zeile ist abwechselnd Lernstoff **und** Prüfung —
     das ausgelieferte Modell wird auf **100 %** der Daten trainiert, und
     bewertet wird über alle Zeilen statt über eine Stichprobe. Kostet ein
     Mehrfaches an Rechenzeit; mehr Folds = die Prüfungsmodelle sehen mehr Daten.
5. **Training starten** → der Fortschrittsbalken (auch oben als kleine Anzeige auf
   jedem Tab sichtbar) zeigt Phase, Prozent und Restzeit. Ein Training auf
   ~30.000 Zeilen dauert je nach Einstellung wenige Minuten bis ~1 Stunde; der
   Server bleibt dabei bedienbar (das Training nimmt sich höchstens ~60 % der
   Rechenleistung). **Training stoppen** bricht sauber ab.

## Ergebnisse lesen (Reiter „Abfrage“)

Text eingeben, Modell(e) anhaken, **Klassifizieren**.

- **Konfidenz (Balken + Zahl 0–1):** Wie sicher das Modell ist, dass das Label
  passt. Die Werte der Labels sind unabhängig voneinander — sie müssen sich
  nicht zu 1 addieren, und **mehrere Labels gleichzeitig sind ein normales
  Ergebnis** (ein Text kann Physik *und* Elektrotechnik sein).
- **Top-k leer (Standard):** Du siehst nur Labels, hinter denen das Modell
  wirklich steht. Oft ist das genau eines — das ist dann die ehrliche Antwort,
  kein Fehler.
- **Top-k = N:** Du siehst immer die N wahrscheinlichsten Labels — auch schwache.
  **Ausgegraute Einträge** („below threshold“) sind Kandidaten, die das Modell
  von sich aus *nicht* behaupten würde. Nützlich, um die „zweite Meinung“ des
  Modells zu sehen.
- **„Kein Label über dem Schwellenwert des Modells.“** Dann hat das Modell
  nichts behauptet — und darunter steht jetzt **„Am nächsten dran, aber unter
  dem Schwellenwert“** mit den drei wahrscheinlichsten Labels, ausgegraut. Das
  ist keine Antwort des Modells, sondern die Auskunft, *wie knapp* es war.
  Häufig bei kurzen Eingaben („Im Unterricht experimentieren wir“): wenige
  Wörter ergeben wenig Signal. Dazu kommt, dass jeder Lauf seine Schwellen pro
  Label so wählt, dass die **F1-Bewertung** am höchsten ist — bei einem
  treffsicheren Modell liegt dieses Optimum *höher*. Ein auf mehr Daten
  trainiertes Modell kann deshalb **seltener** etwas sagen und trotzdem das
  bessere sein: in einem gemessenen Fall stand „Chemie“ bei 0,302, während die
  Schwelle dieses Labels bei 0,60 lag. Wer eine Antwort erzwingen will, setzt in der
  Oberfläche **Top-k** — dann kommt immer eine Rangliste. Einen eigenen
  Schwellenwert kennt die Oberfläche nicht; über die Schnittstelle geht er als
  `threshold` in `POST /predict` mit. Am meisten hilft aber der Text selbst:
  ein paar Fachwörter mehr.
- **Diff (Baseline-Differenz, Häkchen „Baseline-Diff anzeigen“):** Wie viel der
  Sicherheit aus **deinem Text** kommt — und nicht daher, dass das Label einfach
  häufig ist. Ein hoher Konfidenz-Wert mit *diff nahe 0* heißt: das Modell rät
  auf das übliche Label, dein Text hat es nicht überzeugt.
- Mehrere Modelle gleichzeitig anhaken (Strg-Klick) → eine Antwort pro Modell,
  z. B. Fach **und** Materialart in einem Rutsch.
- Die **allererste** Abfrage nach einem Server-Neustart kann ~30 Sekunden
  dauern (das Modell wird von der Festplatte geladen) — danach kommen Antworten
  in Millisekunden.

## „Korrigieren“ — eine falsche Antwort richtigstellen

Neben **Warum?** steht **Korrigieren**. Damit hält man fest, was das Modell hätte sagen
sollen: die Liste zeigt alle Labels des Modells, Mehrfachauswahl per Strg-Klick.

- **Nichts auswählen** heißt „keines davon passt“. Das wird ebenfalls gespeichert,
  taucht aber nicht im Trainings-Export auf — eine Zeile ohne Label kann ein Lauf nicht
  lernen.
- Korrekturen werden **nie verworfen**. Anders als die Lauf-Historie (gedeckelt bei 200)
  sind sie kein Protokoll, sondern die Daten, aus denen der nächste Lauf lernt.
- Der Export unter `GET /feedback/export` ist eine CSV mit den Spalten `text` und
  `labels`, die sich direkt als Datensatz hochladen und trainieren lässt.

Korrigieren darf jeder mit Lese-Schlüssel — genau die Redaktion, die die Fehler sieht.
Der Export bleibt Admin-Sache.

## „Warum?“ — die Erklärung zu einer Antwort

Neben dem Modellnamen im Ergebnis steht **Warum?**. Der Knopf zeigt, welche Wörter die
Antwort getragen haben: pro Label die einflussreichsten Wörter als Chips.

- Der Wert ist, **wie weit die Konfidenz fällt, wenn man das Wort weglässt**. Bei einer
  sicheren Antwort sind diese Zahlen winzig — ein Modell bei 0,999 bewegt sich für kein
  einzelnes Wort viel. Deshalb sind die Balken **pro Label** skaliert: entscheidend ist
  die Reihenfolge, nicht der Betrag.
- Die Werte sind untereinander vergleichbar, **addieren sich aber nicht zur Konfidenz**:
  verglichen wird die Wortliste des Textes mit und ohne ein Wort, während die Konfidenz
  oben den Text beschreibt, wie er eingegeben wurde (mit Satzzeichen, und ab 60 Wörtern
  gekürzt).
- **Füllwörter wie „und“ ganz oben sind kein Zeichen schlechter Datenaufbereitung.**
  Das Modell liest auch Zeichenketten über Wortgrenzen hinweg („Säuren **und** Basen“).
  Nimmt man das Wort weg, zerreißt man die ganze Wortfolge — der Einbruch wird dem
  Füllwort zugeschrieben, gehört aber der Phrase. Nachgemessen: „und“ selbst ist im
  Modell praktisch gewichtslos (`idf` 1,46 gegen 7,16 für ein Fachwort). Stoppwörter zu
  entfernen, zu stemmen oder zu lemmatisieren wurde gemessen und bringt hier **nichts**
  (`docs/model-approach-comparison.md`).
- Ein **−** (gestrichelter Rahmen, gedämpfter Balken) markiert ein Wort, das *gegen* das
  Label spricht: ohne dieses Wort wäre die Konfidenz höher.
- Jedes **Vorkommen** eines Wortes wird einzeln bewertet. Dasselbe Wort kann deshalb
  zweimal mit verschiedenen Werten auftauchen — das ist kein Fehler.
- Aufklappbar darunter: **alle Labels** mit Konfidenz, `diff` und F1, stärkstes zuerst.

Bei einem einzelnen Wort gibt es keine Wort-Zuordnung — weglassen braucht mindestens
zwei Wörter. Die Label-Tabelle steht trotzdem zur Verfügung.

## Zwei Modelle vergleichen (Reiter „Modelle“)

Im Modell-Detail steht **Bewerten gegen…**: Datensatz und Spalten wählen, starten. Der Lauf
geht als Hintergrundjob an den Server — hinter ein laufendes Training, sichtbar auf dem
Reiter „Training“. Das Ergebnis landet **neben** den Trainingsmetriken im Bundle, nie darüber.

In der Tabelle darüber steht dann pro Lauf: Datensatz, gewertete Zeilen, **Getroffene Labels**,
F1 macro und micro.

- **Getroffene Labels** ist entscheidend fürs Lesen: F1 Makro mittelt über *alle* Labels des
  Modells. Ein Datensatz, der 4 von 59 Labels berührt, drückt den Wert aus Gründen, die
  mit der Qualität nichts zu tun haben — gemessen: macro 0,068 neben micro 0,941.
  **Vergleiche also nur denselben Datensatz auf zwei Modellen**, nie zwei Datensätze.
- Zeilen, deren Labels das Modell nie gelernt hat, werden **ausgeschlossen und gezählt**
  — sie dem Modell als Fehler anzurechnen wäre unfair, sie stillschweigend wegzulassen
  würde schmeicheln.
- Steht bei F1 nur „–“, hatten die beiden nichts gemeinsam: das Vokabular des Datensatzes
  passt nicht zum Modell.

## Mehrere Läufe hintereinander (Training-Tab)

Werden mehrere Label-Felder ausgewählt, entsteht pro Feld ein Modell. Alle Läufe gehen
**sofort an den Server**, der sie der Reihe nach abarbeitet — der Tab darf zugehen, das
Notebook zuklappen. Was noch wartet, steht unter dem Status („Queued on the server: …“)
und kommt aus der Serverantwort, nicht aus dem Browser.

**Stop** beendet den laufenden Lauf *und* leert die Warteschlange: „Stop“ heißt „das soll
enden“, nicht „spring zum nächsten“. Es wirkt am nächsten Prüfpunkt — zwischen zwei
Anläufen des Modells, nicht mitten drin —, und ein Anlauf dauert bei großen Datenmengen
Minuten. Deshalb endet ein Lauf, der nicht von selbst reagiert, nach **~30 s** ohnehin;
die Statuszeile sagt das dann auch. Ein gestoppter Lauf hinterlässt in keinem Fall ein
Modell.

Während des Laufs zeigt die Statuskarte zwei Zeilen, die erklären, warum es manchmal
langsamer geht als erwartet:

- **Speicher: „1.842 MB · Spitzenwert 3.105 MB“** — links, was die Anwendung gerade
  belegt (Trainingsprozess mitgezählt), rechts das Maximum dieses Laufs. Der
  Spitzenwert ist die Zahl, mit der man die Frage „reichen 8 GB?“ beim nächsten Mal
  beantwortet; er steht später auch im Verlauf und im Modell selbst.
- **Threads: „3 von 9 — vom Speicherbudget gebremst“** — so viele Label-Anläufe dürfen
  gleichzeitig rechnen. Jeder braucht Arbeitsspeicher in der Größe der Textmatrix, also
  lässt das Speicherbudget bei vielen Labels weniger davon zu, als Kerne da wären. Das
  ist **Absicht**: der Lauf wird langsamer, statt am Speicher zu scheitern. Wer mehr
  Tempo braucht, gibt dem Container mehr Speicher.

Unter *Training* → **Verlauf** steht danach, was jeder Lauf ergeben hat — auch die
gescheiterten, die kein Modell hinterlassen und deren Grund es sonst nirgends mehr gäbe.

## Vor dem Start prüfen (Reiter „Training“)

Unter den Trainingseinstellungen steht **Vor dem Training prüfen**. Der Knopf liest den
Datensatz einmal durch und beantwortet die zwei Fragen, die man sonst erst nach dem Lauf
beantwortet bekommt:

- **Wie viele Labels überleben *deine* Schwelle?** Steht dort „keeps 0 of 3 labels“,
  würde der Lauf mit „not enough data“ abbrechen — sichtbar in Sekunden statt nach einer
  Stunde. Ein Knopf übernimmt den Vorschlag der Heuristik direkt ins Feld.
- **Was kostet der Lauf** mit dem gewählten Profil — und bei mehreren Label-Feldern:
  *pro Modell*.

Geprüft wird gegen das **erste** ausgewählte Label-Feld; bei mehreren können sich die
Zahlen unterscheiden.

## Einen Datensatz prüfen, bevor man ihn trainiert (Reiter „Datensätze“)

Ein Klick auf den Dateinamen zeigt die Spalten und die ersten Zeilen. Darunter wählt man
Textspalten und Label-Spalte und drückt **Analysieren** — das liest jede Zeile, dauert bei
einem großen Export also einen Moment.

Was dann dasteht, sind die zwei Zahlen, die vor einem Lauf zählen:

- **Wie viele Labels eine Schwelle überleben.** Alles unterhalb des gewählten
  `min_samples_per_label` fliegt aus dem Training und kann nie vorhergesagt werden — das
  ist eine Entscheidung, keine Nebensache. Die markierte Zeile ist der Vorschlag der
  Größenheuristik, ein Startpunkt, keine Antwort.
- **Was ein Lauf kostet**, je Profil. Hochgerechnet aus *einem* gemessenen Lauf
  (156 373 Zeilen mit `auto` in 40 Minuten) und linear in der Zeilenzahl. Die Schätzung
  kennt weder die Anzahl der Labels noch die Maschine — sie beantwortet „Kaffeepause oder
  Nachmittag“, keinen Termin.

Eine Warnung erscheint, wenn Labels mit weniger als 10 Beispielen dabei sind: die werden
schlecht abschneiden, egal wie gut der Lauf ist.

## Viele Texte auf einmal (Reiter „Abfrage“)

Der Reiter kennt drei Modi — die Auswahl steht oben im Formular:

- **Ein Text:** wie bisher, mit Balken, `diff` und Label-F1.
- **Viele Texte:** ein Text pro Zeile. Läuft in Paketen, zeigt eine Tabelle und lässt
  sie als CSV herunterladen. Leerzeilen werden übersprungen.
- **Eine CSV-Datei:** die Datei braucht die Textspalten, auf die das Modell trainiert
  wurde — **welche das sind und wie stark jede zählt, liest der Server aus dem Modell
  selbst**. Das ist kein Komfort, sondern Notwendigkeit: ein Modell, das auf
  „Titel Titel Beschreibung“ gefitted wurde, sitzt auf einer anderen Merkmalsverteilung
  als eines auf „Titel Beschreibung“ — und seine Schwellen sitzen mit darauf.

Die Antwort ist eine CSV mit `row,uri,label,confidence,above_threshold`, eine Zeile je
vorhergesagtem Label. `row` ist die 0-basierte Nummer der Eingabezeile — damit lassen
sich die Antworten wieder an die eigene Datei anfügen. **Zeilen, für die das Modell
nichts behauptet, stehen mit ab hier leeren Feldern trotzdem drin**: „welche hat es
abgelehnt“ gehört zur Antwort dazu.

Ab ~5000 Zeilen verweist der Textmodus auf die CSV-Variante — die streamt, statt jede
Antwort im Browser zu sammeln.

## Qualität verstehen (Reiter „Modelle“)

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
    aufgezeichnet wurde. **„argmax“** in der Schwellen-Spalte heißt: das Modell
    nimmt sein bestes Label und liest gar keine Schwelle.

## Teilen & Verschieben

- **Freigabelink** (bei Modellen und Datensätzen): erzeugt einen Download-Link, der
  **ohne Schlüssel** funktioniert und nach 24 Stunden erlischt — gut, um jemandem
  etwas zu geben, der keinen Zugang hat. Der Link ist wie ein Schlüssel zu
  behandeln: wer ihn hat, kann herunterladen.
- **Herunterladen / Importieren:** Ein Modell wandert als ZIP-Datei zwischen
  Servern: auf Server A herunterladen (oder Freigabelink), auf Server B über
  *Modell importieren* hochladen. Ein direkter „Import per Link“ ist **absichtlich** nicht möglich
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
