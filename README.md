# 📈 Exponentielle Glättung – Niveau, Trend und Wochenmuster lernen

Zweites Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger der [Naiven Prognose](https://github.com/sebastian-hanisch/naive-forecast-demo);
geplant sind neun weitere Stücke (ARIMA, Dynamische Regression, Croston-Verfahren, Boosting, Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Das Wochenmittel aus dem Vorgänger hat eine Stellschraube, die man von Hand drehen muss: über wie viele Wochen mitteln? Die **exponentielle Glättung** ersetzt sie durch ein Gedächtnis, das mit jedem Tag ein Stück vergisst: Aus jedem Prognosefehler lernt das Modell ein wenig
(**Niveau**, **Trend**, **Wochenmuster**), und wie schnell es vergisst, **schätzt es aus den Daten**. Die Demo läuft auf **denselben Tagesaufträgen eines Depots** wie das Vorgänger-Stück (dieselbe Reihe, im Test auf denselben Fingerabdruck geprüft), im selben
Rolling-Origin-Vergleich mit MASE und Orakel-Untergrenze, und vergleicht sechs Glättungsmodelle mit dem Wochenmittel und den naiven Verfahren. Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben; die Kreuzprobe im Test läuft gegen statsmodels.

**Bezug zu OR:** jede Bestands-, Personal- und Tourenplanung braucht eine Nachfrageprognose; die Glättung ist das Arbeitspferd dafür – schnell, robust und auf tausenden Reihen anwendbar.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie sah für dieses Stück vor: die Glättung löst das Fensterproblem des Wochenmittels und fügt Trend und Saison hinzu. Die Messung bestätigt das im Kern und differenziert es an vier Stellen:

1. **Der Vorsprung ist klein.** Holt-Winters multiplikativ erreicht im Mittel über zwölf Reihen MASE 0,88 gegen 0,95 beim Wochenmittel (k = 4) und **0,94 beim im Nachhinein besten k** (6): rund 6 % weniger Fehler, in allen zwölf Reihen. Der große Sprung gegenüber dem letzten Tag (2,73) kommt vom Wochenmuster, nicht vom Lernen – das hatte schon das Wochenmittel.
2. **Trend und Dämpfung bringen fast nichts.** Von der einfachen Glättung (2,07) über Holt (2,05) zur gedämpften Variante (2,04) fällt die MASE um 0,03. Erst das Wochenmuster (Holt-Winters additiv 0,95, multiplikativ 0,88) bringt den Sprung.
3. **Das Wochenmuster wird eingefroren, nicht ständig neu gelernt.** Das geschätzte γ liegt bei Holt-Winters multiplikativ in allen zwölf Reihen bei 0,0009 (die Untergrenze des Suchbereichs): im Vehikel ändert sich das Wochenmuster nie, und das Modell lernt genau das. Ob es bei einem wandernden Wochenmuster anders käme, ist hier nicht gemessen.
4. **Feiertage und Aktionen sind der Glättung unbekannt – und stecken dann im Niveau.** An den Ereignistagen ist sie so schlecht wie das Wochenmittel; in den 14 Tagen danach sogar **schlechter** (20,3 gegen 18,8 Aufträge Fehler bei Ereignisstärke 1,0), weil das Ereignis das Niveau verschmutzt hat. Das war nicht geplant und ist der Anlass für die Dynamische Regression (Stück 4).

Außerdem: **das α wird gebraucht, aber nicht fein.** Das beste k des Wochenmittels wandert mit dem Rauschen von 2 auf 12 Wochen, das geschätzte α von 0,35 auf 0,05 – die Glättung findet ihr Gedächtnis selbst. Von Hand gesetzt, ist die MASE über einen breiten Bereich (α = 0,05 bis 0,15) fast gleich.

## Modell

- **Die Reihe** (`es_scenario.py`): wortgleich zu `nf_scenario.py` im Vorgänger: 1 095 Tage, Niveau, Trend, Wochenmuster, Jahresmuster, Niveausprung, Feiertage, Aktionen, multiplikatives Rauschen; Ursprünge im letzten Jahr (ab Tag 730).
- **Modell** (`es_ets.py`): Zustände Niveau $\ell$, Trend $b$ und Saison $s$ (Periode 7). Prognose $\hat y = (\ell + \phi b) + s$ (multiplikativ: $\cdot\,s$), Fehler $e = y - \hat y$; Niveau $\ell' = \ell + \phi b + \alpha e$ (multiplikativ: $\alpha e / s$), Trend $b' = \phi b + \beta e$,
  Saison $s' = s + \gamma e$ (multiplikativ: $\gamma e / \ell'$, mit dem schon aktualisierten Niveau, wie in statsmodels). Sechs Modelle: **einfache Glättung** (nur Niveau), **Holt** (mit Trend), **Holt gedämpft** ($\phi < 1$), **Holt-Winters additiv**, **multiplikativ** und **multiplikativ gedämpft**.
- **Schätzen:** die Ein-Schritt-Fehlerquadrate der ersten 730 Tage minimieren – 3 000 feste Zufallspunkte im Einheitswürfel (für alle Kandidaten gleichzeitig durch die Reihe gerechnet), dann sechs Runden lokaler Verfeinerung, deterministisch. Anfangszustände aus der Zerlegung der Trainingstage (zentriertes 7-Tage-Mittel, Verhältnisse je Wochentag, Gerade durch die ersten acht saisonbereinigten Wochen).
  Danach laufen die Zustände mit den festen Parametern durch das Testjahr; zu jedem Ursprung kennt das Modell nur die Tage davor.
- **Kennzahlen, Orakel, Vergleichsverfahren:** wie im Vorgänger (MASE mit dem saisonal naiven Trainingsfehler, Orakel = wahrer Erwartungswert; naiv, saisonal naiv und Wochenmittel laufen zum Vergleich mit). **AICc** je Modell aus den Trainingstagen.

## Methodik

- **Handrechnungen:** einfache Glättung (10 → 11 → 9,5 → 9,75), Holt, gedämpfter Trend, additive und multiplikative Saison Schritt für Schritt, Rückgriff auf den jüngsten Saisonwert je Wochentag, Abschneiden bei 0.
- **Kreuzprobe gegen statsmodels:** für **alle sechs Modelle** stimmen Ein-Schritt-Fehler und Prognose bei denselben festen Parametern und Anfangszuständen auf 1e-12 überein (der Optimierer ist dabei nicht im Spiel). Dabei zeigte sich, dass statsmodels das multiplikative Saisonglied mit dem *neuen* Niveau teilt; diese Variante ist übernommen.
  Die geschätzten Fehlerquadrate liegen höchstens 2 % über denen, die statsmodels mit frei geschätzten Anfangszuständen erreicht (meist darunter).
- **Kein Blick in die Zukunft:** Ändert man die Tage ab einem Ursprung, bleibt der Zustand bis dahin unverändert; die Neuschätzung im Experiment sieht nur die Tage vor dem Block.
- **Die naiven Verfahren:** vektorisiert gegen eine unabhängige Schleife; im Standardfall dieselben Zahlen wie im Vorgänger (Wochenmittel 0,88, saisonal naiv 1,12, naiv 2,68, Orakel 0,75).
- **Schätzung:** deterministisch, über andere Suchseeds stabil (Fehlerquadrate < 0,5 % Abweichung); die Zulässigkeit ($\beta \le \alpha$, $\gamma \le 1-\alpha$, $0{,}8 \le \phi \le 0{,}98$) ist getestet; auf einer rauschfreien Wochenreihe findet das Modell die Reihe bis auf die Rundung.
- **Statistik:** zwölf feste Seeds, Fehlerbalken = Standardfehler. Das beste k des Wochenmittels wird im Nachhinein aus $k \in \{2,3,4,6,8,12\}$ gewählt – das ist großzügig für das Wochenmittel und macht den Vorsprung der Glättung eher zu klein als zu groß.
- **Literatur** (nicht nachgebaut): Hyndman/Athanasopoulos, *Forecasting: Principles and Practice* (3. Aufl., Kap. 8, Exponentielle Glättung); Hyndman et al., *Forecasting with Exponential Smoothing* (2008, Zustandsraum-Form); Hyndman/Koehler 2006 (MASE).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardreihen** (12 Seeds, Horizont 14, 352 Ursprünge) | MASE: einfache Glättung **2,07**, Holt 2,05, gedämpft 2,04, Holt-Winters additiv **0,95**, multiplikativ **0,88**, multiplikativ gedämpft 0,87; zum Vergleich Wochenmittel (k = 4) 0,95, saisonal naiv 1,17, naiv 2,73; Orakel **0,74**. Holt-Winters multiplikativ liegt in allen zwölf Reihen vor dem Wochenmittel. | `test_standard_over_twelve_series` |
| Standardfall (Preset, Seed 3) | Holt-Winters multiplikativ 0,84, gedämpft 0,84, additiv 0,89; Wochenmittel 0,88, saisonal naiv 1,13, einfache Glättung 2,02, naiv 2,68; Orakel 0,75; geschätztes α 0,12. | `test_standard_preset_and_agreement_with_the_naive_forecast_demo` |
| **Was bringt welcher Baustein? Das beste k des Wochenmittels** | Wochenmittel über k = 2 / 3 / 4 / 6 / 8 / 12: 1,01 / 0,96 / 0,95 / **0,94** / 0,96 / 1,04. Holt-Winters multiplikativ (0,88) liegt in 12 von 12 Reihen vor dem Wochenmittel mit k = 6. | `test_ladder_experiment` |
| **Modellwahl nach AICc** (Trainingstage) | Wählt in allen 12 Reihen ein multiplikatives Modell (gedämpft oder ungedämpft); gegenüber dem im Nachhinein besten Glättungsmodell kostet das im Mittel nur **0,002** MASE-Punkte. | `test_ladder_experiment` |
| **Geschätzte Parameter** (12 Seeds) | α (Holt-Winters multiplikativ) im Mittel **0,12**, γ in jeder Reihe unter 0,002 (das Wochenmuster wird eingefroren; bei additivem Ansatz im Mittel 0,036); α der einfachen Glättung um 0,04; φ des gedämpften Modells im Mittel 0,95 (Bereich 0,80 bis 0,98). | `test_fitted_parameters_over_twelve_series` |
| **Horizont** | Die Glättung liegt bei **jedem** Horizont vor dem Wochenmittel: Fehler in Aufträgen bei Horizont 1: 16,1 gegen 17,8, bei Horizont 14: 17,4 gegen 18,5. | `test_the_smoothing_model_beats_the_weekly_mean_at_every_horizon` |
| **Ein Fenster von Hand gegen ein geschätztes α** (Rauschen 0,04 / 0,08 / 0,14 / 0,24 / 0,40) | Bestes k des Wochenmittels **2 / 3 / 6 / 8 / 12** Wochen; geschätztes α **0,35 / 0,23 / 0,12 / 0,07 / 0,05**; MASE Holt-Winters 0,97 / 0,91 / 0,88 / 0,86 / 0,85 gegen 1,12 / 1,00 / 0,94 / 0,90 / 0,88 beim besten k. Vorsprung 0,15 bei Rauschen 0,04, **0,03** bei 0,40; in 60 von 60 Reihen-Stufen-Kombinationen vorn. | `test_window_alpha_experiment` |
| **Wie empfindlich ist α? Hilft Neuschätzen?** | α von Hand auf 0,02 / 0,05 / 0,09 / 0,15 / 0,25 / 0,40 / 0,60: MASE 0,98 / 0,89 / 0,88 / 0,88 / 0,90 / 0,94 / 1,00 – ein breiter Boden (α = 0,05 bis 0,15 innerhalb von 0,02). Das aus den Trainingstagen geschätzte α (0,12) erreicht 0,880; die Parameter alle 56 Ursprünge auf allen bekannten Tagen **neu zu schätzen ergibt 0,881**: kein Gewinn. | `test_alpha_experiment` |
| **Feiertage und Aktionen** (Ereignisstärke 0 / 0,5 / 1,0; 9 % Ereignistage, 27 % bis 14 Tage danach, 64 % sonst) | Ohne Ereignisse liegt Holt-Winters an normalen Tagen fast auf dem Orakel (14,8 gegen 13,9 Aufträge Fehler; Wochenmittel 16,3). Bei Stärke 1,0: **an Ereignistagen** 54,2 (Wochenmittel 55,0, Orakel 17,3), das 3,1-Fache des Orakels; **in den 14 Tagen danach** 20,3 gegen 18,8 beim Wochenmittel (das Ereignis steckt im Niveau); **sonst** 15,5 gegen 17,8. Bei Stärke 0,5: 30,0 / 15,9 / 15,2 gegen 30,4 / 16,7 / 16,8. | `test_events_experiment` |
| **Additiv oder multiplikativ?** (Wochenmuster 0,5 / 1,0 / 1,5) | Additiv 0,90 / 0,95 / 0,98, multiplikativ 0,88 / 0,88 / 0,88. Der additive Ansatz verliert, sobald das Niveau wandert: ohne Trend und Jahresmuster (Wochenmuster 1,5) liegen additiv und multiplikativ bei 0,74 und 0,73. | `test_weekly_experiment`, `test_additive_gap_needs_a_moving_level` |
| **Dämpfung bei starkem Trend** (+40 % je Jahr) | Holt-Winters multiplikativ / gedämpft / Wochenmittel: 1,11 / 1,10 / 1,19 bei Horizont 14, 1,15 / 1,14 / 1,23 bei Horizont 28 – die Dämpfung hilft ein wenig, obwohl der wahre Trend linear ist (der geschätzte ist verrauscht). | `test_damping_helps_a_little_at_a_long_horizon_with_a_strong_trend` |
| Starkes Rauschen (Preset, Seed 3, 0,4) | Holt-Winters multiplikativ 0,82, Wochenmittel 0,86, saisonal naiv 1,15, einfache Glättung 1,07; Orakel 0,81; α fällt von 0,12 auf 0,05. | `test_strong_noise_preset` |
| Ohne Feiertage und Aktionen (Preset, Seed 3) | Holt-Winters multiplikativ 0,89, Wochenmittel 0,93, saisonal naiv 1,20; Orakel 0,83 (die Glättung liegt 7 % darüber). | `test_no_events_preset_is_close_to_the_floor` |
| Feiertage und Aktionen stark (Preset, Seed 3) | Holt-Winters multiplikativ 0,81, Wochenmittel 0,84, Orakel 0,62 (Abstand 31 %). | `test_strong_events_preset` |
| Starkes Wochenmuster (Preset, Seed 3) | Additiv 0,91, multiplikativ 0,83, Wochenmittel 0,87, Orakel 0,75. | `test_strong_weekly_preset` |
| Niveausprung +30 % (Preset, Seed 3, Sprung an Tag 809) | Holt-Winters multiplikativ 1,03 (α 0,16), Wochenmittel 1,08, additiv 1,21, saisonal naiv 1,36, einfache Glättung 2,47; Orakel 0,90. | `test_level_shift_preset` |

Die Preset-Zeilen sind **Einzelreihen** (Seed 3); belastbar sind die Zeilen über zwölf Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Reihe selbst zählt** | Feiertage und Aktionen sind der Glättung unbekannt: an den Ereignistagen so schlecht wie das Wochenmittel, danach durch das verschmutzte Niveau sogar schlechter. | Dynamische Regression (geplant) |
| **Ein Wochenmuster genügt** | Das Jahresmuster (365 Tage) ist als Saison nicht schätzbar; die Glättung sieht es als langsame Niveauänderung und läuft ihm hinterher. | ARIMA, Dynamische Regression (geplant) |
| **Der Bedarf ist nie null** | Bei vielen Nullen gibt das Modell verschmierte oder negative Werte aus (hier auf 0 abgeschnitten); der Bedarf liegt bei 100 Aufträgen je Tag. | Croston, SBA, TSB (geplant) |
| **Die Parameter bleiben gültig** | Sie stammen aus zwei Jahren und ändern sich im Testjahr nicht – so ist es im Vehikel; ob das in einer echten Reihe stimmt, ist offen. Bei einem wandernden Wochenmuster ist ein größeres γ zu erwarten (nicht gemessen). | – |
| **Eine Reihe genügt** | Jedes Depot bekommt seine eigenen Parameter; ähnliche Depots teilen ihr Wissen nicht. | Globale Modelle: Boosting, Vortrainiertes Netz (geplant) |
| **Es gibt eine Punktprognose** | Das Modell nennt einen Wert; wie sicher er ist, sagt es (hier) nicht. | Prognoseintervalle (geplant) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Reihen. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 45 Sekunden, 93 Tests): der Kern von Hand (jeder Baustein Schritt für Schritt), Kreuzprobe gegen statsmodels für alle sechs Modelle, Schätzung (deterministisch, zulässig, kein Blick in die Zukunft), naive Verfahren gegen eine unabhängige Schleife, die Reihe (Fingerabdruck wie im Vorgänger),
Auswertung und Experimentzeilen, Preset- und Permalink-Klemmen (inkl. Modellwahl), AppTest-Rauchtests (Standard, jedes Preset, jedes Modell in der Ursprungsansicht, Ursprungs-Regler bei kürzerem Testbereich, Extremwerte, fünf Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen;
Reihen und Schätzung sind deterministisch, die Bänder großzügiger als die Rundung).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `es_constants.py` | Regler-Grenzen, Modelle, Suchparameter, Experiment-Gitter und -Seeds |
| `es_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `es_scenario.py` | Die Reihe (wortgleich zum Vorgänger) |
| `es_ets.py` | Zustandsraum-Modell, Schätzung, Zustände, Prognose |
| `es_forecast.py` | Rolling-Origin, naive Vergleichsverfahren, MAE/RMSE/ME/MASE |
| `es_evaluation.py` | Analyse, Orakel, fünf Experimente |
| `es_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Multiplikative Fehler (ETS(M,·,·)) und Modellwahl über die volle ETS-Familie; die Schreibweise mit additivem Fehler genügt für die Aussagen hier (Schätzung per kleinster Quadrate statt Likelihood).
- Eine Jahressaison (Periode 365) und Mehrfachsaisonalität (TBATS); das Jahresmuster bleibt bewusst unerklärt.
- Alles, was die nächsten Stücke bringen: ARIMA, Regression auf Kalendermerkmale, Intervalle, mehrere Reihen, Nullen.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Kreuzprobe im Test: statsmodels).
