"""Exponentielle Glättung - Niveau, Trend und Wochenmuster lernen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe (Nachfolger der Naiven Prognose): dieselben Tagesaufträge eines Depots, aber das Modell lernt aus jedem neuen Tag und schätzt seine Vergessensrate selbst.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import es_constants as C
import es_forecast as F
from es_evaluation import Settings, alpha_experiment, analyse, events_experiment, ladder_experiment, weekly_experiment, window_alpha_experiment
from es_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from es_visualization import SHORT, build_alpha, build_bars, build_events, build_horizon, build_ladder, build_origin, build_series, build_state, build_weekly, build_window_alpha, method_label

st.set_page_config(page_title="Exponentielle Glättung – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _ladder(seeds):
    return ladder_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _window_alpha(levels, seeds):
    return window_alpha_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _alpha(alphas, seeds):
    return alpha_experiment(alphas=alphas, seeds=seeds)


@st.cache_data(show_spinner=False)
def _events(levels, seeds):
    return events_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _weekly(levels, seeds):
    return weekly_experiment(levels=levels, seeds=seeds)


st.title("📈 Exponentielle Glättung – Niveau, Trend und Wochenmuster lernen")
st.markdown(
    """
Das Wochenmittel aus dem Vorgänger-Stück hat eine Stellschraube, die man von Hand drehen muss: **über wie viele Wochen mitteln?** Die **exponentielle Glättung** ersetzt sie durch ein Gedächtnis, das mit jedem neuen Tag ein Stück vergisst:
Aus jedem Prognosefehler lernt sie ein wenig (**Niveau**, **Trend**, **Wochenmuster**), und wie schnell sie vergisst, **schätzt sie aus den Daten**. Die Demo läuft auf denselben **Tagesaufträgen eines Depots** wie das Vorgänger-Stück, wieder im Rolling-Origin-Vergleich
und gegen die **Orakel-Untergrenze** - und zeigt, was das Lernen bringt, wo es nichts bringt und woran es scheitert (unten: Feiertage und Aktionen).
"""
)
st.caption(
    "Zweites Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe, Nachfolger der Naiven Prognose; alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (die Kreuzprobe im Test läuft gegen statsmodels). "
    "**Bezug zu OR:** jede Bestands-, Personal- und Tourenplanung braucht eine Nachfrageprognose; die Glättung ist das Arbeitspferd dafür - schnell, robust und auf tausenden Reihen anwendbar."
)

with st.expander("So funktioniert die exponentielle Glättung", expanded=True):
    st.markdown(
        """
1. **Ein Schritt.** Das Modell trägt drei Zustände mit sich: das **Niveau** $\\ell$, den **Trend** $b$ und für jeden Wochentag einen **Saisonwert** $s$. Aus ihnen folgt die Prognose für den nächsten Tag; danach wird der **Fehler** $e$ (Ist minus Prognose) in jeden Zustand
   zurückgegeben - mit einem Anteil, den ein Glättungsparameter festlegt: $\\alpha$ für das Niveau, $\\beta$ für den Trend, $\\gamma$ für das Wochenmuster. Ein kleines $\\alpha$ heißt langes Gedächtnis, ein großes heißt: der letzte Tag zählt viel.
2. **Bausteine.** *Einfache Glättung* hat nur ein Niveau. *Holt* fügt einen Trend hinzu, *gedämpft* lässt ihn abklingen. *Holt-Winters* fügt das Wochenmuster hinzu - **additiv** (Aufschlag in Aufträgen) oder **multiplikativ** (Faktor).
3. **Schätzen.** Die Parameter werden auf den ersten zwei Jahren so gewählt, dass die Ein-Schritt-Fehler möglichst klein sind; danach laufen die Zustände mit jedem Tag des Testjahres weiter, ohne dass etwas neu geschätzt wird.
4. **Bewertung.** Wie im Vorgänger: viele Ursprünge (Rolling-Origin), die MASE und die Orakel-Untergrenze; das Wochenmittel und die naiven Verfahren laufen zum Vergleich mit.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.caption("Die Reihe")
    trend = st.slider("Trend (% je Jahr)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Lineares Wachstum (oder Schrumpfen) der Aufträge, in Prozent des Ausgangsniveaus je Jahr.")
    weekly = st.slider("Wochenmuster", *bounds("weekly_slider"), key="weekly_slider", step=C.WEEKLY_STEP, help="Stärke des Wochentagsmusters (1 = Standard: Freitag am stärksten, Sonntag am schwächsten; 0 = keines).")
    yearly = st.slider("Jahresmuster", *bounds("yearly_slider"), key="yearly_slider", step=C.YEARLY_STEP, help="Amplitude der jahreszeitlichen Schwankung (Anteil des Niveaus). Die Glättung kennt nur das Wochenmuster; das Jahresmuster sieht sie als langsame Niveauänderung.")
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung des multiplikativen Rauschens (log-normal, im Mittel unverzerrt).")
    shift = st.slider("Niveausprung (%)", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="Sprung des Niveaus an einem zufälligen Tag im letzten Jahr (0 = keiner), z. B. ein neuer Großkunde.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Effekte: am Feiertag ruht das Depot (bis -50 %), am Folgetag Nachholeffekt, dazu drei Aktionswochen je Jahr (bis +50 %). Die Glättung kennt sie nicht.")
    st.caption("Der Vergleich")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    step = st.slider("Abstand der Ursprünge (Tage)", *bounds("step_slider"), key="step_slider", help="Alle wie viele Tage ein neuer Ursprung beginnt. 1 = jeder Tag des letzten Jahres.")
    k = st.slider("Wochen im Wochenmittel (k)", *bounds("k_slider"), key="k_slider", help="Über wie viele letzte Wochen das Vergleichsverfahren 'Wochenmittel' den Wochentag mittelt.")
    model_key = st.selectbox("Modell in der Ursprungsansicht", list(C.ETS_MODELS), key="model_select", format_func=lambda m: C.METHOD_NAMES[m], help="Dieses Glättungsmodell zeigt die Ansicht am Ursprung (Niveau, Prognose, gelerntes Wochenmuster). In der Auswertung laufen alle sechs.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Rauschen, Aktionstage, Phase des Jahresmusters und den Tag des Niveausprungs fest.")
    st.button("🎲 Neue Reihe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"trend_slider": int(trend), "weekly_slider": round(float(weekly), 2), "yearly_slider": round(float(yearly), 2), "noise_slider": round(float(noise), 2), "shift_slider": int(shift), "events_slider": round(float(events), 2),
                   "horizon_slider": int(horizon), "step_slider": int(step), "k_slider": int(k), "model_select": model_key, "seed_input": int(seed)})

settings = Settings(int(trend), round(float(weekly), 2), round(float(yearly), 2), round(float(noise), 2), int(shift), round(float(events), 2), int(horizon), int(step), int(k), int(seed))
a = analyse(settings)
s = a.series
K = settings.k
model, states = a.fitted[model_key]

# --- Die Reihe und das Modell -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Reihe und das Modell an einem Ursprung")
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, origin, model_key), width="stretch", key="series_chart")
st.plotly_chart(build_origin(a, origin, model_key), width="stretch", key="origin_chart")
weekday_name = C.WEEKDAYS[(origin - 1) % 7]
st.caption(
    f"Reihe mit {s.n} Tagen (Mittel {de(s.y.mean())} Aufträge je Tag); Ursprung an Tag {origin} (letzter bekannter Tag: ein {weekday_name}). Oben zeigt die dicke Linie das **Niveau**, das das Modell zu jedem Tag aus den Tagen davor gelernt hat; "
    f"unten die Prognose von {SHORT[model_key]} (dicke Linie) für die nächsten {settings.horizon} Tage neben Wochenmittel, einfacher Glättung und dem letzten Tag; die grüne gepunktete Linie ist der wahre Erwartungswert."
)

c1, c2, c3, c4, c5 = st.columns(5)
lvl_o = float(states.level[origin])
c1.metric("Niveau ℓ am Ursprung", f"{de(lvl_o)} Aufträge", help="Zustand des Modells nach dem letzten bekannten Tag.")
c2.metric("Trend b je Tag", f"{de(states.trend[origin], 2)}" if model.trend != "none" else "–", help="Geschätzter Zuwachs des Niveaus je Tag; Modelle ohne Trend haben keinen.")
c3.metric("α (Niveau)", de(model.alpha, 3), help="Anteil des Fehlers, der ins Niveau geht. Klein: langes Gedächtnis.")
half = np.log(0.5) / np.log(1 - model.alpha)
c4.metric("Halbwertszeit des Niveaus", f"{de(half)} Tage", help="Nach so vielen Tagen zählt eine Beobachtung nur noch halb so viel wie am ersten Tag: ln 0,5 / ln (1 − α).")
c5.metric("β · γ · φ", " · ".join(de(v, 3) if on else "–" for v, on in ((model.beta, model.trend != "none"), (model.gamma, model.season != "none"), (model.phi, model.trend == "damped"))),
          help="Trendglättung β, Glättung des Wochenmusters γ, Dämpfung φ des Trends; '–' heißt: das Modell hat den Baustein nicht.")

state_fig = build_state(a, origin, model_key)
if state_fig is not None:
    st.markdown(f"##### Das gelernte Wochenmuster ({SHORT[model_key]})")
    st.plotly_chart(state_fig, width="stretch", key="state_chart")
    st.caption("Die Balken sind der Saisonzustand am Ursprung, die Rauten das wahre Muster des Vehikels. Bei kleinem γ ändert sich das gelernte Muster kaum; bei großem folgt es jedem Ausschlag.")
else:
    st.info(f"{C.METHOD_NAMES[model_key]} hat keinen Saisonbaustein: das Wochenmuster bleibt im Fehler. Wählen Sie links ein Holt-Winters-Modell, um das gelernte Muster zu sehen.")
st.markdown("##### Die Modelle und ihre geschätzten Parameter")
prow = [{"Modell": C.METHOD_NAMES[m], "α": de(mo.alpha, 3), "β": de(mo.beta, 3) if mo.trend != "none" else "–", "γ": de(mo.gamma, 3) if mo.season != "none" else "–", "φ": de(mo.phi, 3) if mo.trend == "damped" else "–",
         "Trainingsfehler (RMSE)": de(mo.rmse, 1), "AICc": de(mo.aicc, 0)} for m, (mo, _) in a.fitted.items()]
st.dataframe(prow, hide_index=True)
best_aic = min(a.fitted, key=lambda m: a.fitted[m][0].aicc)
st.caption(f"Geschätzt auf den ersten {C.FIRST_TEST} Tagen. Das AICc bestraft zusätzliche Parameter und Anfangszustände; es wählt hier {C.METHOD_NAMES[best_aic]} (Test-MASE {de(a.summary[best_aic]['mase'], 2)}, bestes Glättungsmodell im Test: {SHORT[a.best_ets]} {de(a.summary[a.best_ets]['mase'], 2)}).")

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über alle Ursprünge")
n_org = len(a.origins)
sm = a.summary
best = a.best_ets
gap_k = sm[best]["mase"] - sm["snaive_k"]["mase"]
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Bestes: {SHORT[best]}", f"MASE {de(sm[best]['mase'], 2)}", delta=f"{gap_k:+.2f}".replace(".", ",") + " gegen Wochenmittel", delta_color="inverse", help=f"Bestes der sechs Glättungsmodelle ({C.METHOD_NAMES[best]}): kleinste MASE; der Pfeil vergleicht mit dem Wochenmittel (grün: besser).")
m2.metric(f"Wochenmittel (k = {K})", f"MASE {de(sm['snaive_k']['mase'], 2)}", help="Derselbe Wochentag, gemittelt über die letzten k Wochen - die beste Messlatte aus dem Vorgänger-Stück.")
m3.metric("Naiv (letzter Tag)", f"MASE {de(sm['naive']['mase'], 2)}", help="Der letzte beobachtete Tag, für alle Horizonte fortgeschrieben.")
m4.metric("Orakel-Untergrenze", f"MASE {de(a.oracle['mase'], 2)}", help="Fehler der Prognose 'wahrer Erwartungswert' gegen die beobachteten Werte: das Rauschen der Reihe. Kein Verfahren liegt im Mittel darunter.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": method_label(m, K), "MASE": de(sm[m]["mase"], 2), "MAE (Aufträge)": de(sm[m]["mae"], 1), "RMSE": de(sm[m]["rmse"], 1), "Verzerrung (Prognose minus Ist)": de(sm[m]["me"], 1)} for m in sorted(sm, key=lambda m: sm[m]["mase"])]
st.dataframe(rows, hide_index=True)
gap_or = sm[best]["mase"] / a.oracle["mase"] - 1
if gap_k < 0:
    st.success(f"✅ Die Glättung schlägt das Wochenmittel: {SHORT[best]} erreicht MASE {de(sm[best]['mase'], 2)} gegen {de(sm['snaive_k']['mase'], 2)} ({pct(-gap_k / sm['snaive_k']['mase'])} weniger) und liegt noch {pct(gap_or)} über der Orakel-Untergrenze "
               f"({de(a.oracle['mase'], 2)}). Ohne Wochenmuster im Modell wären es {de(sm['ses']['mase'], 2)} (einfache Glättung).")
else:
    st.warning(f"⚠️ Hier gewinnt das Wochenmittel: {de(sm['snaive_k']['mase'], 2)} gegen {de(sm[best]['mase'], 2)} beim besten Glättungsmodell ({SHORT[best]}). Die Glättung lernt aus jedem Tag - "
               "wo die Reihe dafür zu unruhig ist oder das Wochenmuster fehlt, hat sie keinen Vorsprung mehr.")
st.caption(f"{n_org} Ursprünge im Testjahr (Abstand {settings.step} Tage), je {settings.horizon} Tage Horizont; MASE-Nenner: saisonal naiver Fehler in den ersten {C.FIRST_TEST} Tagen ({de(F.mase_scale(s.y, C.FIRST_TEST), 1)} Aufträge).")

st.markdown("##### Wie der Fehler mit dem Horizont wächst")
st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was bringt welcher Baustein?")
st.caption(f"Standardreihe; die Verfahren von der einfachen Glättung bis Holt-Winters, dazu das Wochenmittel für jedes k aus {', '.join(str(x) for x in C.K_GRID)}. Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer wenige Sekunden.")
if st.button("Bausteine durchrechnen", key="ladder_start"):
    st.session_state["ladder_on"] = True
if st.session_state.get("ladder_on"):
    r = _ladder(C.EXP_SEEDS)
    st.plotly_chart(build_ladder(r), width="stretch", key="ladder_chart")
    n_mult = sum(p.startswith("hw_mult") for p in r["picks"])
    st.warning(
        f"**Befund:** Der große Sprung ist das Wochenmuster: {de(r['ses'], 2)} (einfache Glättung) gegen {de(r['hw_add'], 2)} (Holt-Winters additiv) und {de(r['hw_mult'], 2)} (multiplikativ); der letzte Tag liegt bei {de(r['naive'], 2)}. "
        f"Trend und Dämpfung bringen fast nichts ({de(r['ses'], 2)} → {de(r['holt'], 2)} → {de(r['damped'], 2)}), denn der Trend ist klein gegen das Wochenmuster und das Rauschen. Das Wochenmittel mit dem im Nachhinein besten k ({r['best_k']}) erreicht {de(r['best_k_mase'], 2)}; "
        f"Holt-Winters multiplikativ liegt in {r['wins_vs_best_k']} von {r['n_seeds']} Reihen davor (Orakel: {de(r['floor'], 2)}). Die Modellwahl nach AICc landet in {n_mult} von {r['n_seeds']} Reihen in der multiplikativen Familie und kostet gegenüber dem im Nachhinein besten Glättungsmodell "
        f"nur {de(r['aicc_gap'], 3)} MASE-Punkte."
    )

st.markdown("---")

st.subheader("🔬 Ein Fenster von Hand gegen ein geschätztes α")
st.caption(f"Rauschen {', '.join(de(x, 2) for x in C.NOISE_LEVELS)}; je Stufe das Wochenmittel mit dem im Nachhinein besten k (das ist großzügig für das Wochenmittel) gegen Holt-Winters multiplikativ. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Rauschen durchrechnen", key="window_start"):
    st.session_state["window_on"] = True
if st.session_state.get("window_on"):
    rows_w = _window_alpha(C.NOISE_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_window_alpha(rows_w), width="stretch", key="window_chart")
    lo, hi = rows_w[0], rows_w[-1]
    n_win = sum(r["wins"] for r in rows_w)
    n_all = sum(r["n_seeds"] for r in rows_w)
    st.warning(
        f"**Befund:** Bei Rauschen {de(lo['noise'], 2)} ist ein kurzes Gedächtnis richtig (bestes k: {lo['best_k']} Wochen, geschätztes α {de(lo['alpha'], 2)}), bei {de(hi['noise'], 2)} ein langes (k = {hi['best_k']}, α {de(hi['alpha'], 2)}): das beste Fenster wandert mit dem Rauschen, "
        f"und die Glättung findet es von selbst. Holt-Winters liegt in {n_win} von {n_all} Reihen-Stufen-Kombinationen vor dem (im Nachhinein besten!) Wochenmittel; der Abstand beträgt {de(lo['best_k_mase'] - lo['hw_mult'], 2)} bei Rauschen {de(lo['noise'], 2)} und "
        f"{de(hi['best_k_mase'] - hi['hw_mult'], 2)} bei {de(hi['noise'], 2)} MASE-Punkten."
    )

st.markdown("---")

st.subheader("🔬 Wie empfindlich ist α - und hilft Neuschätzen?")
st.caption(f"Standardreihe, Holt-Winters multiplikativ; α von Hand auf {', '.join(de(x, 2) for x in C.ALPHA_GRID)} gesetzt. Dazu: die Parameter alle {C.REFIT_EVERY} Ursprünge auf allen bis dahin bekannten Tagen neu schätzen. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa 10 bis 30 Sekunden.")
if st.button("α durchrechnen", key="alpha_start"):
    st.session_state["alpha_on"] = True
if st.session_state.get("alpha_on"):
    ra = _alpha(C.ALPHA_GRID, C.EXP_SEEDS)
    st.plotly_chart(build_alpha(ra), width="stretch", key="alpha_chart")
    g = {r["alpha"]: r["mase"] for r in ra["rows"]}
    lo_a, hi_a = min(g), max(g)
    mid = [v for al, v in g.items() if 0.05 <= al <= 0.15]
    st.warning(
        f"**Befund:** Die Kurve hat einen breiten Boden: zwischen α = 0,05 und 0,15 schwankt die MASE nur um {de(max(mid) - min(mid), 3)}. An den Rändern steigt sie: bei α = {de(lo_a, 2)} (sehr langes Gedächtnis) liegt sie bei {de(g[lo_a], 2)}, bei α = {de(hi_a, 2)} "
        f"(der letzte Tag zählt fast allein) bei {de(g[hi_a], 2)}. Das aus den Trainingstagen geschätzte α ({de(ra['fit_alpha'], 2)}) erreicht {de(ra['fit_mase'], 3)}. Alle {ra['refit_every']} Ursprünge neu zu schätzen ändert daran nichts ({de(ra['refit_mase'], 3)}): "
        "die Parameter aus zwei Jahren Training reichen, weil sich die Reihe in ihrem Aufbau nicht ändert."
    )

st.markdown("---")

st.subheader("🔬 Was passiert an Feiertagen und Aktionen?")
st.caption(f"Standardreihe mit Ereignisstärke {', '.join(de(x, 2) for x in C.EVENT_LEVELS)}; die Ereignistage (Feiertag, Folgetag, Aktion) sind in allen Stärken dieselben. Fehler je Tagesart, Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Ereignisse durchrechnen", key="events_start"):
    st.session_state["events_on"] = True
if st.session_state.get("events_on"):
    re_ = _events(C.EVENT_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_events(re_), width="stretch", key="events_chart")
    top = re_[-1]
    after_word = "schlechter" if top["hw_mult"][1] > top["snaive_k"][1] else "besser"
    st.warning(
        f"**Befund:** Ohne Ereignisse liegt Holt-Winters fast auf dem Orakel ({de(re_[0]['hw_mult'][2], 1)} gegen {de(re_[0]['oracle'][2], 1)} Aufträge Fehler an normalen Tagen). Bei Ereignisstärke {de(top['events'], 1)} steigt der Fehler an den Ereignistagen "
        f"({pct(top['share'][0])} der Prognosetage) auf {de(top['hw_mult'][0], 1)} Aufträge, das {de(top['hw_mult'][0] / top['oracle'][0], 1)}-Fache des Orakels, und ist dabei nicht besser als beim Wochenmittel ({de(top['snaive_k'][0], 1)}). In den {C.EVENT_AFTER_DAYS} Tagen danach ist die Glättung "
        f"{after_word} als das Wochenmittel ({de(top['hw_mult'][1], 1)} gegen {de(top['snaive_k'][1], 1)}): das Ereignis steckt im Niveau, und das Gedächtnis gibt es nur langsam wieder her. An allen übrigen Tagen liegt sie klar vorn ({de(top['hw_mult'][2], 1)} gegen {de(top['snaive_k'][2], 1)})."
    )

st.markdown("---")

st.subheader("🔬 Additiv oder multiplikativ?")
st.caption(f"Standardreihe mit Wochenmuster-Stärke {', '.join(de(x, 1) for x in C.WEEKLY_LEVELS)}; Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer wenige Sekunden.")
if st.button("Wochenmuster durchrechnen", key="weekly_start"):
    st.session_state["weekly_on"] = True
if st.session_state.get("weekly_on"):
    rw = _weekly(C.WEEKLY_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_weekly(rw), width="stretch", key="weekly_chart")
    lo, hi = rw[0], rw[-1]
    st.warning(
        f"**Befund:** Das Vehikel erzeugt ein multiplikatives Muster (der Freitag ist immer x % über dem Niveau). Bei schwachem Wochenmuster ({de(lo['weekly'], 1)}) liegen additiv und multiplikativ nah beieinander ({de(lo['hw_add'], 2)} und {de(lo['hw_mult'], 2)}); "
        f"bei starkem ({de(hi['weekly'], 1)}) fällt der additive Ansatz auf {de(hi['hw_add'], 2)} zurück, der multiplikative bleibt bei {de(hi['hw_mult'], 2)}: ein fester Aufschlag in Aufträgen passt nicht mehr, sobald das Niveau wandert (Trend, Jahresmuster)."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Reihe selbst zählt** | Feiertage und Aktionen sind der Glättung unbekannt: an den Ereignistagen so schlecht wie das Wochenmittel, danach durch das verschmutzte Niveau sogar schlechter. | Dynamische Regression (Stück 4) |
| **Ein Wochenmuster genügt** | Das Jahresmuster (365 Tage) ist als Saison nicht schätzbar; die Glättung sieht es als langsame Niveauänderung und läuft ihm hinterher. | ARIMA (Stück 3), Dynamische Regression |
| **Der Bedarf ist nie null** | Bei vielen Nullen gibt das Modell negative oder verschmierte Werte aus; hier liegt der Bedarf bei 100 Aufträgen je Tag. | Croston, SBA, TSB (Stück 5) |
| **Die Parameter bleiben gültig** | Sie stammen aus zwei Jahren und ändern sich im Testjahr nicht - so ist es im Vehikel; ob das in einer echten Reihe stimmt, ist offen. | – |
| **Eine Reihe genügt** | Jedes Depot bekommt seine eigenen Parameter; ähnliche Depots teilen ihr Wissen nicht. | Globale Modelle (Boosting, Vortrainiertes Netz) |
| **Es gibt eine Punktprognose** | Das Modell nennt einen Wert; wie sicher er ist, sagt es (hier) nicht. | Prognoseintervalle (Stück 7) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Reihen. | – |
"""
)
st.caption("Die Linie: Naive Prognose → **Exponentielle Glättung** → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zustandsraum-Form (additiver Fehler).** Zustände vor Tag $t$: Niveau $\ell_{t-1}$, Trend $b_{t-1}$, Saison $s_{t-7}$. Prognose $\hat y_t = (\ell_{t-1} + \phi\, b_{t-1}) + s_{t-7}$ (multiplikativ: $\cdot\, s_{t-7}$), Fehler $e_t = y_t - \hat y_t$.

$$\ell_t = \ell_{t-1} + \phi\, b_{t-1} + \alpha\, e_t,\qquad b_t = \phi\, b_{t-1} + \beta\, e_t,\qquad s_t = s_{t-7} + \gamma\, e_t .$$

Multiplikativ: $e_t$ wird für $\ell$ und $b$ durch $s_{t-7}$ geteilt, für $s$ durch das neue Niveau $\ell_t$. Ohne Trend $b \equiv 0$ ($\beta = 0$, $\phi = 1$), ohne Saison entfällt $s$; $\phi = 1$ ist der ungedämpfte Trend. Zulässig: $0 < \alpha < 1$, $0 < \beta < \alpha$, $0 < \gamma < 1 - \alpha$.

**Gedächtnis.** Für die einfache Glättung ist $\ell_t = \alpha \sum_{i \ge 0} (1-\alpha)^i\, y_{t-i}$ plus ein verschwindender Anfangsterm: geometrisch fallende Gewichte, Halbwertszeit $\ln 0{,}5 / \ln(1-\alpha)$ Tage.

**Prognose** für den Horizont $j = 1..h$ vom Ursprung $t$ aus: $\hat y_{t+j-1} = \ell_{t-1} + (\phi + \dots + \phi^j)\, b_{t-1} + s_{t+j-1-7(\lceil j/7 \rceil)}$ (negative Werte werden auf 0 gesetzt).

**Schätzen.** $\min_{\alpha,\beta,\gamma,\phi} \sum_{t=28}^{729} e_t^2$ mit Anfangszuständen aus der Zerlegung der Trainingstage (zentriertes 7-Tage-Mittel, Verhältnisse je Wochentag, Gerade durch die ersten acht saisonbereinigten Wochen). Suche: 3.000 feste Zufallspunkte im Einheitswürfel, danach sechs
Runden lokaler Verfeinerung; deterministisch (fester Seed). **AICc** $= n \ln(\mathrm{SSE}/n) + 2k + 2k(k+1)/(n-k-1)$ mit $k$ = Glättungsparameter plus Anfangszustände. **MASE** und **Orakel** wie in der Naiven Prognose.

Implementiert in `es_ets.py` (Modell, Schätzung, Prognose), `es_forecast.py` (Rolling-Origin, Kennzahlen), `es_scenario.py` (die Reihe), `es_evaluation.py` (Analyse, fünf Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
