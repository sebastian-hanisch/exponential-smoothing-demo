"""Der Kern: Zustandsschritte gegen Handrechnung, Kreuzprobe gegen statsmodels (feste Parameter), Schätzung, kein Blick in die Zukunft."""

import warnings

import numpy as np
import pytest

import es_constants as C
import es_ets as ETS
import es_scenario as S


def _model(key, alpha, beta, gamma, phi, l0, b0, s0=(), n_fit=10):
    trend, season = C.ETS_MODELS[key]
    return ETS.Model(key, trend, season, alpha, beta, gamma, phi, (l0, b0, tuple(s0)), 0.0, n_fit)


def test_ses_by_hand():
    m = _model("ses", 0.5, 0.0, 0.0, 1.0, 10.0, 0.0)
    st = ETS.filter_states(m, [12, 8, 10])
    assert st.level == pytest.approx([10.0, 11.0, 9.5, 9.75]) and st.error == pytest.approx([2.0, -3.0, 0.5])
    assert ETS.forecast_origins(m, st, [3], 4)[0] == pytest.approx([9.75] * 4)


def test_holt_by_hand():
    m = _model("holt", 0.5, 0.1, 0.0, 1.0, 10.0, 1.0)
    st = ETS.filter_states(m, [12, 13])
    assert st.level == pytest.approx([10.0, 11.5, 12.8]) and st.trend == pytest.approx([1.0, 1.1, 1.14]) and st.error == pytest.approx([1.0, 0.4])
    assert ETS.forecast_origins(m, st, [2], 2)[0] == pytest.approx([12.8 + 1.14, 12.8 + 2 * 1.14])


def test_damped_trend_forecast_by_hand():
    m = _model("damped", 0.5, 0.1, 0.0, 0.9, 10.0, 1.0)
    st = ETS.filter_states(m, [12])
    # base = 10 + 0,9 * 1 = 10,9; e = 1,1; l = 10,9 + 0,55 = 11,45; b = 0,9 + 0,11 = 1,01
    assert st.level[1] == pytest.approx(11.45) and st.trend[1] == pytest.approx(1.01)
    assert ETS.forecast_origins(m, st, [1], 2)[0] == pytest.approx([11.45 + 0.9 * 1.01, 11.45 + (0.9 + 0.81) * 1.01])


def test_additive_season_by_hand():
    s0 = [10, 0, 0, 0, 0, 0, 0]
    m = _model("hw_add", 0.2, 0.0, 0.5, 1.0, 100.0, 0.0, s0)
    y = [118] + [101.6] * 6 + [120]
    st = ETS.filter_states(m, y)
    # Tag 0: yhat = 110, e = 8 -> l = 101,6, s[0] = 14; Tage 1..6: e = 0; Tag 7: yhat = 101,6 + 14 = 115,6, e = 4,4
    assert st.error[0] == pytest.approx(8.0) and st.error[7] == pytest.approx(4.4)
    assert st.level[8] == pytest.approx(101.6 + 0.2 * 4.4) and st.season[8][0] == pytest.approx(14 + 0.5 * 4.4)
    assert ETS.forecast_origins(m, st, [8], 2)[0] == pytest.approx([101.6 + 0.88, 101.6 + 0.88])


def test_multiplicative_season_by_hand():
    s0 = [1.2, 1, 1, 1, 1, 1, 1]
    m = _model("hw_mult", 0.2, 0.0, 0.5, 1.0, 100.0, 0.0, s0)
    st = ETS.filter_states(m, [132])
    # yhat = 100 * 1,2 = 120, e = 12; l = 100 + 0,2 * 12 / 1,2 = 102; s[0] = 1,2 + 0,5 * 12 / 102 (mit dem neuen Niveau)
    assert st.error[0] == pytest.approx(12.0) and st.level[1] == pytest.approx(102.0) and st.season[1][0] == pytest.approx(1.2 + 0.5 * 12 / 102)
    assert ETS.forecast_origins(m, st, [1], 1)[0] == pytest.approx([102.0 * 1.0])


def test_forecast_uses_the_latest_seasonal_value_of_each_weekday_and_clips_at_zero():
    m = _model("hw_add", 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, [-20, 0, 1, 2, 3, 4, 5])
    st = ETS.filter_states(m, [5] * 7)
    f = ETS.forecast_origins(m, st, [7], 14)[0]
    assert f == pytest.approx([0.0, 5, 6, 7, 8, 9, 10] * 2)


def test_state_at_an_origin_does_not_depend_on_later_days():
    y = S.generate(seed=1).y
    m = ETS.fit(y[:C.FIRST_TEST], "hw_mult")
    a = ETS.filter_states(m, y)
    y2 = y.copy()
    y2[900:] += 1000
    b = ETS.filter_states(m, y2)
    assert np.array_equal(a.level[:901], b.level[:901]) and np.array_equal(a.season[:901], b.season[:901]) and not np.array_equal(a.level[902:], b.level[902:])


def test_init_states_on_a_pure_weekly_pattern():
    t = np.arange(200)
    pattern = np.array(C.WEEKLY_PATTERN)
    y = 100.0 * pattern[t % 7] / pattern.mean()
    l0, b0, s0 = ETS.init_states(y, "add", "mul")
    assert l0 == pytest.approx(100.0, rel=1e-6) and b0 == pytest.approx(0.0, abs=1e-6) and np.array(s0) == pytest.approx(pattern / pattern.mean())
    l0a, b0a, s0a = ETS.init_states(y, "add", "add")
    assert sum(s0a) == pytest.approx(0.0, abs=1e-9) and np.array(s0a) == pytest.approx(100.0 * (pattern / pattern.mean() - 1.0))
    l0n, b0n, s0n = ETS.init_states(np.full(60, 42.0), "add", "none")
    assert (l0n, b0n, s0n) == (pytest.approx(42.0), pytest.approx(0.0), ())


def test_parameter_counts_and_aicc_formula():
    counts = {k: _model(k, 0.1, 0.01, 0.01, 0.95, 1.0, 0.0, [1] * 7).n_params for k in C.ETS_MODELS}
    assert counts == {"ses": 2, "holt": 4, "damped": 5, "hw_add": 11, "hw_mult": 11, "hw_mult_damped": 12}
    m = ETS.Model("hw_add", "add", "add", 0.1, 0.01, 0.01, 1.0, (1.0, 0.0, (0.0,) * 7), 1000.0, 700)
    k, n = 11, 700
    assert m.aicc == pytest.approx(n * np.log(1000.0 / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)) and m.rmse == pytest.approx(np.sqrt(1000.0 / 700))


# --- Kreuzprobe gegen statsmodels (feste Parameter und Anfangszustände -> kein Optimierer im Spiel) ------------------------------------------------


@pytest.fixture(scope="module")
def fitted_series():
    y = S.generate(seed=3).y
    return y, {k: ETS.fit(y[:C.FIRST_TEST], k) for k in C.ETS_MODELS}


@pytest.mark.parametrize("key", list(C.ETS_MODELS))
def test_filter_and_forecast_match_statsmodels_for_fixed_parameters(key, fitted_series):
    ETSModel = pytest.importorskip("statsmodels.tsa.exponential_smoothing.ets").ETSModel
    y, models = fitted_series
    m = models[key]
    kw = dict(error="add", trend=None if m.trend == "none" else "add", damped_trend=(m.trend == "damped"), seasonal=None if m.season == "none" else m.season, seasonal_periods=7 if m.season != "none" else None,
              initialization_method="known", initial_level=m.init[0])
    if m.trend != "none":
        kw["initial_trend"] = m.init[1]
    if m.season != "none":
        kw["initial_seasonal"] = np.array(m.init[2])
    vals = {"smoothing_level": m.alpha, "smoothing_trend": m.beta, "smoothing_seasonal": m.gamma, "damping_trend": m.phi}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ETSModel(y[:900], **kw)
        res = mod.smooth([vals[n] for n in mod.param_names])
        fc = np.asarray(res.forecast(14))
    st = ETS.filter_states(m, y)
    assert st.error[:900] == pytest.approx(np.asarray(res.resid), abs=1e-7)
    assert ETS.forecast_origins(m, st, [900], 14)[0] == pytest.approx(fc, abs=1e-7)


@pytest.mark.parametrize("key", list(C.ETS_MODELS))
def test_estimated_error_is_as_small_as_statsmodels_with_free_initial_states(key, fitted_series):
    ETSModel = pytest.importorskip("statsmodels.tsa.exponential_smoothing.ets").ETSModel
    y, models = fitted_series
    m = models[key]
    tr, se = m.trend, m.season
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ETSModel(y[:C.FIRST_TEST], error="add", trend=None if tr == "none" else "add", damped_trend=(tr == "damped"), seasonal=None if se == "none" else se, seasonal_periods=7 if se != "none" else None).fit(disp=False, maxiter=2000)
    sm_sse = float(np.sum(np.asarray(res.resid)[C.INIT_DAYS:] ** 2))
    assert m.sse <= 1.02 * sm_sse


# --- Schätzung ----------------------------------------------------------------------------------------------------------------------------------


def test_fit_is_deterministic_and_stable_across_search_seeds(fitted_series):
    y, models = fitted_series
    again = ETS.fit(y[:C.FIRST_TEST], "hw_mult")
    assert again == models["hw_mult"]
    other = [ETS.fit(y[:C.FIRST_TEST], "hw_mult", seed=sd).sse for sd in (1, 2)]
    assert all(abs(o - models["hw_mult"].sse) / models["hw_mult"].sse < 0.005 for o in other)


def test_fitted_parameters_are_feasible(fitted_series):
    _, models = fitted_series
    for m in models.values():
        assert 0 < m.alpha < 1 and m.beta <= m.alpha + 1e-12 and m.gamma <= 1 - m.alpha + 1e-12
        if m.trend == "damped":
            assert C.PHI_MIN <= m.phi <= C.PHI_MAX
        else:
            assert m.phi == 1.0
        if m.trend == "none":
            assert m.beta == 0.0
        if m.season == "none":
            assert m.gamma == 0.0


def test_more_components_never_fit_the_training_days_worse(fitted_series):
    _, models = fitted_series
    sse = {k: m.sse for k, m in models.items()}
    assert sse["holt"] <= sse["ses"] * 1.001 and sse["damped"] <= sse["holt"] * 1.001 and sse["hw_add"] < 0.4 * sse["ses"] and sse["hw_mult"] < sse["hw_add"] and sse["hw_mult_damped"] <= sse["hw_mult"] * 1.001


def test_noise_free_weekly_series_is_recovered():
    ser = S.generate(trend=0, weekly=1.0, yearly=0.0, noise=0.0, shift=0, events=0.0, seed=1)
    m = ETS.fit(ser.y[:C.FIRST_TEST], "hw_mult")
    assert m.rmse < 0.6                                     # nur die Rundung auf ganze Aufträge bleibt
    st = ETS.filter_states(m, ser.y)
    f = ETS.forecast_origins(m, st, [900], 14)[0]
    assert np.abs(f - ser.mu[900:914]).max() < 1.0


def test_candidates_are_evaluated_independently():
    y = S.generate(seed=2).y[:C.FIRST_TEST].astype(float)
    init = ETS.init_states(y, "add", "mul")
    params = np.array([[0.1, 0.001, 0.001, 1.0], [0.3, 0.01, 0.05, 1.0]])
    both = ETS._run(y, "add", "mul", params, init, C.INIT_DAYS)
    single = [ETS._run(y, "add", "mul", params[i:i + 1], init, C.INIT_DAYS)[0] for i in (0, 1)]
    assert both == pytest.approx(single)
