"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. nf_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import es_constants as C


def _model(value):
    v = str(value)
    if v not in C.ETS_MODELS:
        raise ValueError(value)
    return v


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "weekly_slider": SettingSpec("weekly", float, C.DEFAULT_WEEKLY, C.WEEKLY_MIN, C.WEEKLY_MAX),
    "yearly_slider": SettingSpec("yearly", float, C.DEFAULT_YEARLY, C.YEARLY_MIN, C.YEARLY_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "shift_slider": SettingSpec("shift", int, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "step_slider": SettingSpec("step", int, C.DEFAULT_STEP, C.STEP_MIN, C.STEP_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K_WEEKS, C.K_WEEKS_MIN, C.K_WEEKS_MAX),
    "model_select": SettingSpec("model", _model, C.DEFAULT_MODEL),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"trend": "trend_slider", "weekly": "weekly_slider", "yearly": "yearly_slider", "noise": "noise_slider", "shift": "shift_slider", "events": "events_slider", "horizon": "horizon_slider",
               "step": "step_slider", "k": "k_slider", "seed": "seed_input"}
STEPS = {"trend_slider": C.TREND_STEP, "weekly_slider": C.WEEKLY_STEP, "yearly_slider": C.YEARLY_STEP, "noise_slider": C.NOISE_STEP, "shift_slider": C.SHIFT_STEP, "events_slider": C.EVENTS_STEP}


def _p(**kw):
    base = {"trend": C.DEFAULT_TREND, "weekly": C.DEFAULT_WEEKLY, "yearly": C.DEFAULT_YEARLY, "noise": C.DEFAULT_NOISE, "shift": C.DEFAULT_SHIFT, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON,
            "step": C.DEFAULT_STEP, "k": C.DEFAULT_K_WEEKS, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Starkes Rauschen (0,4)": _p(noise=0.4),
    "Ohne Feiertage und Aktionen": _p(events=0.0),
    "Feiertage und Aktionen stark": _p(events=1.0),
    "Starkes Wochenmuster (1,5)": _p(weekly=1.5),
    "Niveausprung +30 %": _p(shift=30),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "Seed 3, 14 Tage Horizont, 352 Ursprünge im Testjahr: MASE Holt-Winters multiplikativ 0,84 (gedämpft 0,84, additiv 0,89), Wochenmittel (k = 4) 0,88, saisonal naiv 1,13, einfache Glättung 2,02, naiv 2,68; Orakel-Untergrenze 0,75. "
                    "Im Mittel über 12 Reihen: Holt-Winters multiplikativ 0,88, Wochenmittel 0,95, Orakel 0,74.",
    "Starkes Rauschen (0,4)": "Seed 3, Rauschen 0,4: das geschätzte α fällt von 0,12 auf 0,05 (langes Gedächtnis); Holt-Winters multiplikativ 0,82, Wochenmittel 0,86, saisonal naiv 1,15, einfache Glättung 1,07; Orakel 0,81 - kaum noch Luft bis zur Untergrenze.",
    "Ohne Feiertage und Aktionen": "Seed 3, ohne Feiertage und Aktionen: Holt-Winters multiplikativ 0,89 (gedämpft 0,89), Wochenmittel 0,93, saisonal naiv 1,20; Orakel 0,83. Die Glättung liegt nur 7 % über der Untergrenze.",
    "Feiertage und Aktionen stark": "Seed 3, Stärke 1,0: Holt-Winters multiplikativ 0,81, Wochenmittel 0,84, Orakel 0,62; der Abstand zur Untergrenze wächst auf 31 % (ohne Ereignisse: 7 %). Die Glättung kennt Feiertage und Aktionen nicht; die Experimente unten zeigen, wo der Fehler entsteht.",
    "Starkes Wochenmuster (1,5)": "Seed 3, Wochenmuster 1,5: Holt-Winters additiv 0,91, multiplikativ 0,83, Wochenmittel 0,87, Orakel 0,75. Ein fester Aufschlag passt nicht, sobald das Niveau wandert (Trend, Jahresmuster); ohne beides liegen additiv und multiplikativ im Mittel über 12 Reihen bei 0,74 und 0,73.",
    "Niveausprung +30 %": "Seed 3, Sprung +30 % an Tag 809: Holt-Winters multiplikativ 1,03 (geschätztes α 0,16), Wochenmittel 1,08, additiv 1,21, saisonal naiv 1,36, einfache Glättung 2,47; Orakel 0,90.",
}
