"""Orakel: die klassische Glättungsform von Holt-Winters (Niveau, Trend, Saison je als gewichtetes Mittel; Hyndman/Athanasopoulos, Kap. 8) in einer skalaren Schleife - ein anderer Rechenweg als die vektorisierte
Fehlerkorrektur-Form im Code. Umrechnung: beta = alpha * beta*, gamma = (1 - alpha) * gamma*. Dazu die Anfangszustände aus expliziten Schleifen und die Prognose, indem die Rekursion ohne Fehler iteriert wird."""

import numpy as np
import pytest

import es_constants as C
import es_ets as ETS

M = 7


def _classical(y, trend, season, a, bs, gs, phi, l0, b0, s0):
    l, b = l0, b0
    s = list(s0) if season != "none" else None
    err, level, slope, seas = [], [l], [b], [list(s) if s else None]
    for t, yt in enumerate(y):
        st = s[t % M] if s else None
        base = l + phi * b
        if season == "none":
            yhat, ln = base, a * yt + (1 - a) * base
        elif season == "add":
            yhat, ln = base + st, a * (yt - st) + (1 - a) * base
        else:
            yhat, ln = base * st, a * (yt / st) + (1 - a) * base
        err.append(yt - yhat)
        bn = bs * (ln - l) + (1 - bs) * phi * b if trend != "none" else 0.0
        if season == "add":
            s[t % M] = gs * (yt - ln) + (1 - gs) * st
        elif season == "mul":
            s[t % M] = gs * (yt / ln) + (1 - gs) * st
        l, b = ln, bn
        level.append(l); slope.append(b); seas.append(list(s) if s else None)
    return np.array(err), np.array(level), np.array(slope), seas


def _series(rng, n):
    t = np.arange(n)
    pattern = rng.uniform(0.5, 1.5, 7)
    y = (rng.uniform(20, 200) + rng.uniform(-0.1, 0.3) * t) * pattern[t % 7] * np.exp(rng.normal(0, rng.uniform(0.02, 0.3), n))
    return np.maximum(np.rint(y), 1.0)


@pytest.mark.parametrize("key", list(C.ETS_MODELS))
def test_states_errors_and_forecasts_match_the_classical_smoothing_form(key):
    trend, season = C.ETS_MODELS[key]
    rng = np.random.default_rng(171)
    for _ in range(6):
        n = int(rng.integers(60, 160))
        y = _series(rng, n)
        a = rng.uniform(0.02, 0.6)
        bs = rng.uniform(0.0, 1.0) if trend != "none" else 0.0
        gs = rng.uniform(0.0, 0.5) if season != "none" else 0.0
        phi = rng.uniform(0.8, 0.98) if trend == "damped" else 1.0
        init = ETS.init_states(y, trend, season)
        s0 = init[2]
        m = ETS.Model(key, trend, season, a, a * bs, gs * (1 - a), phi, init, 0.0, 1)
        st = ETS.filter_states(m, y)
        err, level, slope, seas = _classical(y, trend, season, a, bs, gs, phi, init[0], init[1], s0)
        assert st.error == pytest.approx(err, rel=1e-9, abs=1e-9)
        assert st.level == pytest.approx(level, rel=1e-9, abs=1e-9)
        assert st.trend == pytest.approx(slope, rel=1e-9, abs=1e-9)
        if season != "none":
            assert st.season == pytest.approx(np.array(seas), rel=1e-9, abs=1e-9)
        h = int(rng.integers(1, 22))
        for t in rng.integers(8, n, size=3):
            _, lv, sl, ss = _classical(y[:t], trend, season, a, bs, gs, phi, init[0], init[1], s0)   # nur y[:t]
            l, b, f = lv[-1], sl[-1], []
            for j in range(h):
                base = l + phi * b
                f.append(max(base if season == "none" else (base + ss[-1][(t + j) % M] if season == "add" else base * ss[-1][(t + j) % M]), 0.0))
                l, b = base, phi * b
            assert ETS.forecast_origins(m, st, [t], h)[0] == pytest.approx(np.array(f), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("season", ["none", "add", "mul"])
def test_init_states_match_explicit_loops(season):
    rng = np.random.default_rng(7)
    for trend in ("none", "add"):
        for _ in range(4):
            n = int(rng.integers(70, 200))
            y = _series(rng, n)
            l0, b0, s0 = ETS.init_states(y, trend, season)
            if season == "none":
                b = (np.mean(y[14:28]) - np.mean(y[:14])) / 14 if trend != "none" else 0.0
                assert (l0, b0) == pytest.approx((np.mean(y[:28]) - b * 14.5, b))
                continue
            rat = {d: [] for d in range(7)}
            for t in range(3, n - 3):
                mu = np.mean(y[t - 3:t + 4])
                rat[t % 7].append(y[t] - mu if season == "add" else y[t] / mu)
            sv = np.array([np.mean(rat[d]) for d in range(7)])
            sv = sv - sv.mean() if season == "add" else sv / sv.mean()
            z = np.array([y[t] - sv[t % 7] if season == "add" else y[t] / sv[t % 7] for t in range(56)])
            if trend == "none":
                ll, bb = z.mean(), 0.0
            else:
                bb, ll = np.linalg.lstsq(np.stack([np.arange(1, 57.0), np.ones(56)], 1), z, rcond=None)[0]
            assert (l0, b0) == pytest.approx((ll, bb), abs=1e-8)
            assert np.array(s0) == pytest.approx(sv, abs=1e-9)
