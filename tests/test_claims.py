"""Jede Zahl aus README und PRESET_HELP als Test. Reihen und Schätzung sind deterministisch (fester Seed der Parametersuche); die Bänder sind trotzdem großzügiger als die Rundung, damit andere numpy-/BLAS-Versionen nicht stören."""

import numpy as np
import pytest

import es_constants as C
import es_evaluation as E
import es_presets as P


def _preset(name, keys=tuple(C.ETS_MODELS), **over):
    p = dict(P.PRESETS[name])
    p.update(over)
    return E.analyse(E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["k"], p["seed"]), keys)


def _m(a):
    return {k: v["mase"] for k, v in a.summary.items()}


def test_standard_preset_and_agreement_with_the_naive_forecast_demo():
    a = _preset("Standardfall")
    m = _m(a)
    assert len(a.origins) == 352 and a.best == "hw_mult_damped" and a.oracle["mase"] == pytest.approx(0.747, abs=0.01)
    assert m["hw_mult"] == pytest.approx(0.841, abs=0.015) and m["hw_mult_damped"] == pytest.approx(0.836, abs=0.015) and m["hw_add"] == pytest.approx(0.887, abs=0.015)
    assert m["ses"] == pytest.approx(2.02, abs=0.03) and m["holt"] == pytest.approx(2.02, abs=0.03) and m["damped"] == pytest.approx(2.02, abs=0.03)
    assert m["snaive_k"] == pytest.approx(0.878, abs=0.006) and m["snaive"] == pytest.approx(1.125, abs=0.006) and m["naive"] == pytest.approx(2.677, abs=0.006)     # dieselben Zahlen wie im Vorgänger-Stück
    assert a.fitted["hw_mult"][0].alpha == pytest.approx(0.117, abs=0.015)


def test_strong_noise_preset():
    a = _preset("Starkes Rauschen (0,4)")
    m = _m(a)
    assert m["hw_mult"] == pytest.approx(0.824, abs=0.015) and m["snaive_k"] == pytest.approx(0.856, abs=0.006) and m["snaive"] == pytest.approx(1.145, abs=0.006) and m["ses"] == pytest.approx(1.068, abs=0.03)
    assert a.oracle["mase"] == pytest.approx(0.807, abs=0.01) and a.fitted["hw_mult"][0].alpha == pytest.approx(0.048, abs=0.015) and m["hw_mult"] < m["snaive_k"]


def test_no_events_preset_is_close_to_the_floor():
    a = _preset("Ohne Feiertage und Aktionen")
    m = _m(a)
    assert m["hw_mult"] == pytest.approx(0.893, abs=0.015) and m["hw_mult_damped"] == pytest.approx(0.886, abs=0.015) and m["snaive_k"] == pytest.approx(0.927, abs=0.006) and m["snaive"] == pytest.approx(1.197, abs=0.006)
    assert a.oracle["mase"] == pytest.approx(0.833, abs=0.01) and m["hw_mult"] / a.oracle["mase"] - 1 == pytest.approx(0.07, abs=0.02)


def test_strong_events_preset():
    a = _preset("Feiertage und Aktionen stark")
    m = _m(a)
    assert m["hw_mult"] == pytest.approx(0.808, abs=0.015) and m["snaive_k"] == pytest.approx(0.844, abs=0.006) and a.oracle["mase"] == pytest.approx(0.616, abs=0.01)
    assert m["hw_mult"] / a.oracle["mase"] - 1 == pytest.approx(0.31, abs=0.03)


def test_strong_weekly_preset():
    a = _preset("Starkes Wochenmuster (1,5)")
    m = _m(a)
    assert m["hw_add"] == pytest.approx(0.908, abs=0.015) and m["hw_mult"] == pytest.approx(0.833, abs=0.015) and m["snaive_k"] == pytest.approx(0.871, abs=0.006) and a.oracle["mase"] == pytest.approx(0.746, abs=0.01)


def test_level_shift_preset():
    a = _preset("Niveausprung +30 %")
    m = _m(a)
    assert a.series.shift_day == 809 and m["hw_mult"] == pytest.approx(1.029, abs=0.02) and m["snaive_k"] == pytest.approx(1.079, abs=0.006) and m["hw_add"] == pytest.approx(1.214, abs=0.02)
    assert m["snaive"] == pytest.approx(1.359, abs=0.006) and m["ses"] == pytest.approx(2.471, abs=0.04) and a.oracle["mase"] == pytest.approx(0.902, abs=0.01) and a.fitted["hw_mult"][0].alpha == pytest.approx(0.155, abs=0.02)


@pytest.fixture(scope="module")
def twelve():
    return [E.analyse(E.Settings(seed=s)) for s in C.EXP_SEEDS]


def test_standard_over_twelve_series(twelve):
    mean = {k: float(np.mean([a.summary[k]["mase"] for a in twelve])) for k in C.METHODS}
    assert mean["ses"] == pytest.approx(2.07, abs=0.03) and mean["holt"] == pytest.approx(2.05, abs=0.03) and mean["damped"] == pytest.approx(2.04, abs=0.03)
    assert mean["hw_add"] == pytest.approx(0.95, abs=0.02) and mean["hw_mult"] == pytest.approx(0.88, abs=0.02) and mean["hw_mult_damped"] == pytest.approx(0.87, abs=0.02)
    assert mean["naive"] == pytest.approx(2.73, abs=0.03) and mean["snaive"] == pytest.approx(1.165, abs=0.02) and mean["snaive_k"] == pytest.approx(0.95, abs=0.02) and float(np.mean([a.oracle["mase"] for a in twelve])) == pytest.approx(0.74, abs=0.01)
    assert all(a.summary["hw_mult"]["mase"] < a.summary["snaive_k"]["mase"] for a in twelve) and mean["ses"] - mean["damped"] < 0.06 and mean["hw_add"] > mean["hw_mult"]


def test_fitted_parameters_over_twelve_series(twelve):
    mult = [a.fitted["hw_mult"][0] for a in twelve]
    assert float(np.mean([m.alpha for m in mult])) == pytest.approx(0.12, abs=0.015) and all(m.gamma < 0.002 for m in mult)              # das Wochenmuster wird praktisch eingefroren
    assert float(np.mean([a.fitted["hw_add"][0].gamma for a in twelve])) == pytest.approx(0.036, abs=0.02)
    assert all(a.fitted["ses"][0].alpha < 0.05 for a in twelve) and all(a.fitted["hw_mult"][0].beta < 0.01 for a in twelve)
    assert all(a.fitted["hw_mult_damped"][0].phi >= 0.8 for a in twelve) and float(np.mean([a.fitted["hw_mult_damped"][0].phi for a in twelve])) == pytest.approx(0.948, abs=0.03)


def test_the_smoothing_model_beats_the_weekly_mean_at_every_horizon(twelve):
    hw = np.mean([a.horizon_mae["hw_mult"] for a in twelve], axis=0)
    wm = np.mean([a.horizon_mae["snaive_k"] for a in twelve], axis=0)
    assert np.all(hw < wm) and hw[0] == pytest.approx(16.1, abs=0.4) and hw[13] == pytest.approx(17.4, abs=0.4) and wm[0] == pytest.approx(17.8, abs=0.4) and wm[13] == pytest.approx(18.5, abs=0.4)


def test_ladder_experiment():
    r = E.ladder_experiment()
    assert r["best_k"] == 6 and r["best_k_mase"] == pytest.approx(0.941, abs=0.01) and r["wins_vs_best_k"] == 12 and r["floor"] == pytest.approx(0.735, abs=0.01)
    assert [round(r["k_mean"][k], 2) for k in C.K_GRID] == pytest.approx([1.01, 0.96, 0.95, 0.94, 0.96, 1.04], abs=0.015)
    assert all(p.startswith("hw_mult") for p in r["picks"]) and r["aicc_gap"] < 0.01
    assert r["hw_mult_se"] < 0.05 and r["ses"] > 2 * r["hw_add"]


def test_window_alpha_experiment():
    rows = {r["noise"]: r for r in E.window_alpha_experiment()}
    assert [rows[n]["best_k"] for n in C.NOISE_LEVELS] == [2, 3, 6, 8, 12]
    assert [round(rows[n]["alpha"], 2) for n in C.NOISE_LEVELS] == pytest.approx([0.35, 0.23, 0.12, 0.07, 0.05], abs=0.02)
    assert [round(rows[n]["hw_mult"], 2) for n in C.NOISE_LEVELS] == pytest.approx([0.97, 0.91, 0.88, 0.86, 0.85], abs=0.02)
    assert all(rows[n]["wins"] == 12 for n in C.NOISE_LEVELS)
    assert rows[0.04]["best_k_mase"] - rows[0.04]["hw_mult"] == pytest.approx(0.15, abs=0.03) and rows[0.4]["best_k_mase"] - rows[0.4]["hw_mult"] == pytest.approx(0.03, abs=0.02)


def test_alpha_experiment():
    r = E.alpha_experiment()
    g = {x["alpha"]: x["mase"] for x in r["rows"]}
    assert [round(g[a], 2) for a in C.ALPHA_GRID] == pytest.approx([0.98, 0.89, 0.88, 0.88, 0.90, 0.94, 1.00], abs=0.02)
    mid = [g[0.05], g[0.09], g[0.15]]
    assert max(mid) - min(mid) < 0.02 and g[0.02] > min(mid) + 0.08 and g[0.6] > min(mid) + 0.1
    assert r["fit_alpha"] == pytest.approx(0.12, abs=0.015) and r["fit_mase"] == pytest.approx(0.880, abs=0.01) and r["refit_mase"] == pytest.approx(r["fit_mase"], abs=0.005) and r["refit_every"] == 56


def test_events_experiment():
    rows = {r["events"]: r for r in E.events_experiment()}
    assert [round(x, 2) for x in rows[1.0]["share"]] == pytest.approx([0.09, 0.27, 0.64], abs=0.01)
    lo, hi = rows[0.0], rows[1.0]
    assert lo["hw_mult"][2] == pytest.approx(14.8, abs=0.3) and lo["oracle"][2] == pytest.approx(13.9, abs=0.3) and lo["snaive_k"][2] == pytest.approx(16.3, abs=0.3)
    assert hi["hw_mult"] == pytest.approx([54.2, 20.3, 15.5], abs=0.5) and hi["snaive_k"] == pytest.approx([55.0, 18.8, 17.8], abs=0.5) and hi["oracle"] == pytest.approx([17.3, 13.7, 13.9], abs=0.3)
    assert hi["hw_mult"][0] / hi["oracle"][0] == pytest.approx(3.1, abs=0.15) and hi["hw_mult"][1] > hi["snaive_k"][1] and hi["hw_mult"][2] < hi["snaive_k"][2]
    mid = rows[0.5]
    assert mid["hw_mult"] == pytest.approx([30.0, 15.9, 15.2], abs=0.5) and mid["snaive_k"] == pytest.approx([30.4, 16.7, 16.8], abs=0.5)


def test_weekly_experiment():
    rows = {r["weekly"]: r for r in E.weekly_experiment()}
    assert [round(rows[w]["hw_add"], 2) for w in C.WEEKLY_LEVELS] == pytest.approx([0.90, 0.95, 0.98], abs=0.02) and [round(rows[w]["hw_mult"], 2) for w in C.WEEKLY_LEVELS] == pytest.approx([0.88, 0.88, 0.88], abs=0.02)


def test_additive_gap_needs_a_moving_level():
    still = [E.analyse(E._replace(E.Settings(), seed=s, weekly=1.5, trend=0, yearly=0.0), ("hw_add", "hw_mult")) for s in C.EXP_SEEDS]
    add, mult = (float(np.mean([a.summary[k]["mase"] for a in still])) for k in ("hw_add", "hw_mult"))
    assert add == pytest.approx(0.743, abs=0.02) and mult == pytest.approx(0.732, abs=0.02) and add - mult < 0.03


def test_damping_helps_a_little_at_a_long_horizon_with_a_strong_trend():
    for h, vals in ((14, (1.106, 1.098, 1.19)), (28, (1.149, 1.135, 1.228))):
        rows = [E.analyse(E._replace(E.Settings(), seed=s, horizon=h, trend=40), ("hw_mult", "hw_mult_damped")) for s in C.EXP_SEEDS]
        got = [float(np.mean([a.summary[k]["mase"] for a in rows])) for k in ("hw_mult", "hw_mult_damped", "snaive_k")]
        assert got == pytest.approx(list(vals), abs=0.03) and got[1] < got[0] < got[2]
