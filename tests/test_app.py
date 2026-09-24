"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Ursprungs-Regler, Modellwahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, fünf Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import es_constants as C
import es_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_the_verdict_metrics_and_charts():
    at = _run()
    _ok(at)
    assert len(at.metric) == 9 and len(at.get("plotly_chart")) == 5 and any("schlägt das Wochenmittel" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


@pytest.mark.parametrize("key", list(C.ETS_MODELS))
def test_every_model_in_the_detail_view_runs(key):
    at = _run(model_select=key)
    _ok(at)
    seasonal = C.ETS_MODELS[key][1] != "none"
    assert (len(at.get("plotly_chart")) == 5) == seasonal and (not any("hat keinen Saisonbaustein" in i.value for i in at.info)) == seasonal


def test_origin_slider_survives_a_shorter_test_range():
    at = _run(horizon_slider=1, origin_slider=1090)
    _ok(at)
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Reihe generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_clamped_and_validated():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["trend"] = "999"
    at.query_params["noise"] = "0.31"
    at.query_params["weekly"] = "abc"
    at.query_params["model"] = "nope"
    at.query_params["horizon"] = "0"
    at.query_params["shift"] = "-35"
    at.run()
    _ok(at)
    assert at.session_state["trend_slider"] == C.TREND_MAX and at.session_state["noise_slider"] == 0.3 and at.session_state["weekly_slider"] == C.DEFAULT_WEEKLY
    assert at.session_state["model_select"] == C.DEFAULT_MODEL and at.session_state["horizon_slider"] == C.HORIZON_MIN and at.session_state["shift_slider"] == -40


def test_permalink_model_is_taken_over():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["model"] = "holt"
    at.run()
    _ok(at)
    assert at.session_state["model_select"] == "holt"


@pytest.mark.parametrize("kw", [dict(trend_slider=C.TREND_MIN, noise_slider=C.NOISE_MAX), dict(trend_slider=C.TREND_MAX, weekly_slider=0.0, yearly_slider=0.0), dict(horizon_slider=C.HORIZON_MAX, step_slider=C.STEP_MAX),
                                dict(shift_slider=-40, k_slider=12), dict(shift_slider=40, k_slider=2, events_slider=1.0), dict(noise_slider=C.NOISE_MIN, events_slider=0.0, horizon_slider=1)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))


def test_ladder_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "K_GRID", (2, 4))
    at = _run()
    next(b for b in at.button if b.key == "ladder_start").click().run()
    _ok(at)
    assert at.session_state["ladder_on"] and any(w.value.startswith("**Befund:** Der große Sprung ist das Wochenmuster") for w in at.warning)


def test_window_alpha_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "NOISE_LEVELS", (0.04, 0.4))
    monkeypatch.setattr(C, "K_GRID", (2, 8))
    at = _run()
    next(b for b in at.button if b.key == "window_start").click().run()
    _ok(at)
    assert at.session_state["window_on"] and any("das beste Fenster wandert mit dem Rauschen" in w.value for w in at.warning)


def test_alpha_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    monkeypatch.setattr(C, "ALPHA_GRID", (0.05, 0.1, 0.15, 0.5))
    monkeypatch.setattr(C, "REFIT_EVERY", 300)
    at = _run()
    next(b for b in at.button if b.key == "alpha_start").click().run()
    _ok(at)
    assert at.session_state["alpha_on"] and any("breiten Boden" in w.value for w in at.warning)


def test_events_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    next(b for b in at.button if b.key == "events_start").click().run()
    _ok(at)
    assert at.session_state["events_on"] and any("Ereignistagen" in w.value for w in at.warning)


def test_weekly_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    next(b for b in at.button if b.key == "weekly_start").click().run()
    _ok(at)
    assert at.session_state["weekly_on"] and any("multiplikatives Muster" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{signed(" not in el.value
