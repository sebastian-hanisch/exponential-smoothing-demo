"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import pytest

import es_constants as C
import es_evaluation as E
import es_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("trend", "trend_slider"), ("weekly", "weekly_slider"), ("yearly", "yearly_slider"), ("noise", "noise_slider"), ("shift", "shift_slider"), ("events", "events_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9


def test_standard_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall"]
    assert E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("trend_slider") == (C.TREND_MIN, C.TREND_MAX) and set(P.STEPS) == {"trend_slider", "weekly_slider", "yearly_slider", "noise_slider", "shift_slider", "events_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_model_permalink_value_is_validated():
    caster = P.SETTING_SPECS["model_select"].caster
    assert caster("holt") == "holt" and P.SETTING_SPECS["model_select"].default in C.ETS_MODELS
    with pytest.raises(ValueError):
        caster("naive")
