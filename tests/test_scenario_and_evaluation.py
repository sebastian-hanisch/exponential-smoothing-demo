"""Die Reihe (identisch zum Vorgänger-Stück), die naiven Vergleichsverfahren und die Auswertung (Analyse, fünf Experimente)."""

import numpy as np
import pytest

import es_constants as C
import es_evaluation as E
import es_forecast as F
import es_scenario as S


def test_scenario_is_the_same_vehicle_as_in_the_naive_forecast_demo():
    ser = S.generate(seed=3)
    assert ser.y[:8].tolist() == [103.0, 131.0, 117.0, 117.0, 127.0, 64.0, 46.0, 132.0] and float(ser.y.sum()) == 126867.0 and float(ser.mu.sum()) == pytest.approx(126496.28723546621)


def test_generate_is_reproducible_shaped_and_integer():
    a, b, c = S.generate(seed=4), S.generate(seed=4), S.generate(seed=5)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.n == C.N_DAYS and np.all(a.y >= 0) and np.all(a.y == np.round(a.y))


def test_events_do_not_change_the_noise_or_the_event_days():
    lo, hi = S.generate(events=0.0, seed=2), S.generate(events=1.0, seed=2)
    assert np.array_equal(lo.holiday, hi.holiday) and np.array_equal(lo.promo, hi.promo) and lo.holiday.sum() > 0 and lo.promo.sum() > 0


# --- naive Vergleichsverfahren ------------------------------------------------------------------------------------------------------------------


def _loop_baseline(method, y, t, h, k):
    hist = y[:t]
    m = len(hist)
    out = []
    for j in range(h):
        if method == "naive":
            out.append(hist[-1])
        elif method == "snaive":
            out.append(hist[m - 7 + (j % 7)])
        else:
            out.append(np.mean([hist[m - 7 * (i + 1) + (j % 7)] for i in range(k)]))
    return np.array(out)


@pytest.mark.parametrize("method", C.BASELINES)
def test_vectorised_baselines_equal_the_explicit_loop(method):
    y = S.generate(seed=6).y
    org = F.origins(len(y), 10, step=17)
    got = F.baseline_origins(method, y, org, 10, 5)
    assert got == pytest.approx(np.stack([_loop_baseline(method, y, t, 10, 5) for t in org]))


def test_baselines_by_hand():
    y = np.arange(30, dtype=float)
    assert F.baseline_origins("naive", y, [20], 3)[0].tolist() == [19, 19, 19]
    assert F.baseline_origins("snaive", y, [20], 9)[0].tolist() == [13, 14, 15, 16, 17, 18, 19, 13, 14]
    assert F.baseline_origins("snaive_k", y, [21], 2, k=2)[0].tolist() == [(14 + 7) / 2, (15 + 8) / 2]


def test_mase_scale_is_the_in_sample_seasonal_naive_mae():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70], dtype=float)
    assert F.mase_scale(y, 14) == pytest.approx(np.mean(np.abs(y[7:] - y[:7])))


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analyse_is_consistent():
    s = E.Settings(seed=2)
    a = E.analyse(s)
    assert set(a.summary) == set(C.METHODS) and set(a.fitted) == set(C.ETS_MODELS) and a.best in C.METHODS and a.best_ets in C.ETS_MODELS
    assert len(a.origins) == len(range(C.FIRST_TEST, C.N_DAYS - s.horizon + 1, s.step)) and a.errors["hw_mult"].shape == (len(a.origins), s.horizon)
    assert a.oracle["mase"] < min(v["mase"] for v in a.summary.values())
    assert a.summary["hw_mult"]["mase"] == pytest.approx(a.summary["hw_mult"]["mae"] / F.mase_scale(a.series.y, C.FIRST_TEST))


def test_analyse_with_a_subset_of_models_gives_the_same_numbers():
    full = E.analyse(E.Settings(seed=2))
    part = E.analyse(E.Settings(seed=2), ("hw_mult",))
    assert set(part.fitted) == {"hw_mult"} and part.summary["hw_mult"]["mase"] == full.summary["hw_mult"]["mase"] and part.summary["snaive_k"]["mase"] == full.summary["snaive_k"]["mase"]


def test_errors_are_forecast_minus_actual_and_first_origin_is_after_training():
    a = E.analyse(E.Settings(seed=2, horizon=3))
    t = int(a.origins[0])
    assert t == C.FIRST_TEST and a.actual[0].tolist() == a.series.y[t:t + 3].tolist()
    model, states = a.fitted["ses"]
    assert a.errors["ses"][0] + a.actual[0] == pytest.approx(np.full(3, states.level[t]))


def test_horizon_and_step_change_the_result():
    base = E.analyse(E.Settings(seed=2))
    assert len(E.analyse(E.Settings(seed=2, step=7)).origins) == pytest.approx(len(base.origins) / 7, abs=1)
    assert E.analyse(E.Settings(seed=2, k=8)).summary["snaive_k"]["mase"] != base.summary["snaive_k"]["mase"]
    assert E.analyse(E.Settings(seed=2, horizon=28)).summary["hw_mult"]["mae"] > base.summary["hw_mult"]["mae"]


def test_fits_are_shared_between_horizons_and_ks():
    E._fit_one.cache_clear()
    E.analyse(E.Settings(seed=9), ("hw_mult",))
    n = E._fit_one.cache_info().misses
    E.analyse(E.Settings(seed=9, horizon=7, k=8), ("hw_mult",))
    assert E._fit_one.cache_info().misses == n


def test_event_classes():
    ser = S.generate(seed=1)
    cls = E.event_classes(ser)
    ev = (ser.holiday + np.roll(ser.holiday, 1) + ser.promo) > 0
    assert np.array_equal(cls == 0, ev)
    day = int(np.flatnonzero(cls == 1)[0])
    assert (~ev[day]) and ev[max(0, day - C.EVENT_AFTER_DAYS):day].any() and set(np.unique(cls)) == {0, 1, 2}
    far = np.flatnonzero(cls == 2)
    assert all(not ev[d - C.EVENT_AFTER_DAYS:d + 1].any() for d in far[far > 20][:50])


def test_with_alpha_keeps_beta_and_gamma_feasible():
    a = E.analyse(E.Settings(seed=2), ("hw_mult",))
    m = E.with_alpha(a.fitted["hw_mult"][0], 0.0005)
    assert m.alpha == 0.0005 and m.beta <= 0.0005 and m.gamma == a.fitted["hw_mult"][0].gamma
    m2 = E.with_alpha(a.fitted["hw_mult"][0], 0.9999)
    assert m2.gamma <= 1 - 0.9999 + 1e-12


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------


def test_experiments_return_consistent_rows():
    r = E.ladder_experiment(seeds=(0, 1), ks=(2, 4))
    assert r["best_k"] in (2, 4) and r["best_k_mase"] == pytest.approx(r["k_mean"][r["best_k"]]) and len(r["picks"]) == 2 and 0 <= r["wins_vs_best_k"] <= 2 and r["hw_mult"] < r["ses"]
    rows = E.window_alpha_experiment(levels=(0.04, 0.3), seeds=(0, 1), ks=(2, 8))
    assert [x["noise"] for x in rows] == [0.04, 0.3] and rows[1]["floor"] > rows[0]["floor"] and rows[1]["alpha"] < rows[0]["alpha"]
    al = E.alpha_experiment(alphas=(0.05, 0.5), seeds=(0,), refit_every=200)
    assert [x["alpha"] for x in al["rows"]] == [0.05, 0.5] and al["fit_mase"] > 0 and al["refit_every"] == 200
    ev = E.events_experiment(levels=(0.0, 1.0), seeds=(0, 1))
    assert [x["events"] for x in ev] == [0.0, 1.0] and sum(ev[0]["share"]) == pytest.approx(1.0) and ev[1]["hw_mult"][0] > 2 * ev[0]["hw_mult"][0] and ev[0]["share"] == ev[1]["share"]
    wk = E.weekly_experiment(levels=(0.5, 1.5), seeds=(0, 1))
    assert [x["weekly"] for x in wk] == [0.5, 1.5] and wk[1]["hw_add"] > wk[0]["hw_add"] * 0.9


def test_refit_with_a_long_interval_equals_the_fixed_parameters():
    """Ein einziger Block: die Neuschätzung am ersten Ursprung sieht genau die Trainingstage - dieselben Parameter, dieselbe Zahl."""
    al = E.alpha_experiment(alphas=(0.1,), seeds=(0, 1), refit_every=10 ** 6)
    assert al["refit_mase"] == pytest.approx(al["fit_mase"], abs=1e-12)
