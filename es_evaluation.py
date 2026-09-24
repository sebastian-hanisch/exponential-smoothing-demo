"""Auswertung: Glättungsmodelle im Rolling-Origin-Vergleich auf einer Tagesaufträge-Reihe und Experimente (Bausteine, Rauschen, Ereignisse, Trend und Niveausprung, Glättungsparameter)."""

import dataclasses
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import es_constants as C
import es_ets as ETS
import es_forecast as F
import es_scenario as S


@dataclass(frozen=True)
class Settings:
    trend: int = C.DEFAULT_TREND
    weekly: float = C.DEFAULT_WEEKLY
    yearly: float = C.DEFAULT_YEARLY
    noise: float = C.DEFAULT_NOISE
    shift: int = C.DEFAULT_SHIFT
    events: float = C.DEFAULT_EVENTS
    horizon: int = C.DEFAULT_HORIZON
    step: int = C.DEFAULT_STEP
    k: int = C.DEFAULT_K_WEEKS
    seed: int = 3


@dataclass
class Analysis:
    settings: Settings
    series: S.Series
    fitted: dict           # Schlüssel -> (Modell, Zustände)
    origins: np.ndarray
    actual: np.ndarray
    errors: dict
    summary: dict
    oracle: dict           # Kennzahlen der Orakel-Prognose (wahrer Erwartungswert)
    horizon_mae: dict
    origin_mae: dict

    @property
    def best(self):
        return min(self.summary, key=lambda m: self.summary[m]["mase"])

    @property
    def best_ets(self):
        return min(self.fitted, key=lambda m: self.summary[m]["mase"])


def make_series(s, seed=None):
    return S.generate(s.trend, s.weekly, s.yearly, s.noise, s.shift, s.events, seed=s.seed if seed is None else seed)


def oracle_summary(series, org, h):
    """Kennzahlen der Orakel-Prognose: der wahre Erwartungswert als Prognose, gemessen am beobachteten Wert (untere Grenze für jedes Verfahren im Mittel)."""
    E = np.stack([series.mu[t:t + h] - series.y[t:t + h] for t in org])
    scale = F.mase_scale(series.y, C.FIRST_TEST)
    mae = float(np.abs(E).mean())
    return {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}


@lru_cache(maxsize=128)
def _series(trend, weekly, yearly, noise, shift, events, seed):
    return make_series(Settings(trend, weekly, yearly, noise, shift, events, seed=seed))


@lru_cache(maxsize=1024)
def _fit_one(trend, weekly, yearly, noise, shift, events, seed, key):
    """Ein Glättungsmodell auf den Trainingstagen der Reihe schätzen und durch die ganze Reihe fortschreiben (je Reihe und Modell einmal)."""
    series = _series(trend, weekly, yearly, noise, shift, events, seed)
    model = ETS.fit(series.y[:C.FIRST_TEST], key)
    return model, ETS.filter_states(model, series.y)


def analyse(s, keys=tuple(C.ETS_MODELS)):
    series = _series(s.trend, s.weekly, s.yearly, s.noise, s.shift, s.events, s.seed)
    fitted = {k: _fit_one(s.trend, s.weekly, s.yearly, s.noise, s.shift, s.events, s.seed, k) for k in keys}
    org, actual, errors = F.rolling_origin(series.y, fitted, s.horizon, s.step, s.k)
    return Analysis(s, series, fitted, org, actual, errors, F.summarize(errors, series.y), oracle_summary(series, org, s.horizon), F.per_horizon(errors), F.per_origin(errors))


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


def _row(res, extra=None):
    row = dict(extra or {})
    for m, v in res.items():
        row[m], row[m + "_se"] = _mean_se(v)
    return row


# --- Experiment 1: Bausteine und das Wochenmittel mit dem besten k ----------------------------------------------------------------------------


def ladder_experiment(seeds=None, base=None, ks=C.K_GRID):
    """Standardreihen: MASE aller Verfahren (Mittel über die Seeds) und des Wochenmittels für jedes k des Gitters; dazu die Modellwahl nach AICc (auf den Trainingstagen) und was sie gegenüber dem im Nachhinein besten
    Glättungsmodell kostet."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    res = {m: [] for m in C.METHODS}
    kres = {k: [] for k in ks}
    floor, picks, gaps = [], [], []
    for sd in seeds:
        a = analyse(_replace(base, seed=sd))
        for m in C.METHODS:
            res[m].append(a.summary[m]["mase"])
        floor.append(a.oracle["mase"])
        for k in ks:
            kres[k].append(analyse(_replace(base, seed=sd, k=k), ("hw_mult",)).summary["snaive_k"]["mase"])
        aic = {m: a.fitted[m][0].aicc for m in C.ETS_MODELS}
        pick = min(aic, key=aic.get)
        picks.append(pick)
        gaps.append(a.summary[pick]["mase"] - a.summary[a.best_ets]["mase"])
    k_mean = {k: float(np.mean(v)) for k, v in kres.items()}
    best_k = min(k_mean, key=k_mean.get)
    row = _row(res, {"n_seeds": len(seeds), "floor": float(np.mean(floor)), "k_mean": k_mean, "best_k": best_k, "best_k_mase": k_mean[best_k], "picks": picks, "aicc_gap": float(np.mean(gaps))})
    row["wins_vs_best_k"] = int(sum(h < k for h, k in zip(res["hw_mult"], kres[best_k])))
    return row


# --- Experiment 2: Fenster von Hand gegen geschätztes alpha --------------------------------------------------------------------------------------


def window_alpha_experiment(levels=None, seeds=None, base=None, ks=C.K_GRID):
    """Je Rauschstufe: das Wochenmittel mit dem (im Nachhinein) besten k gegen Holt-Winters multiplikativ, dessen alpha aus den Trainingstagen geschätzt wird."""
    levels = C.NOISE_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for nz in levels:
        hw, floor, alphas = [], [], []
        kres = {k: [] for k in ks}
        for sd in seeds:
            a = analyse(_replace(base, noise=nz, seed=sd), ("hw_mult",))
            hw.append(a.summary["hw_mult"]["mase"])
            floor.append(a.oracle["mase"])
            alphas.append(a.fitted["hw_mult"][0].alpha)
            for k in ks:
                kres[k].append(analyse(_replace(base, noise=nz, seed=sd, k=k), ("hw_mult",)).summary["snaive_k"]["mase"])
        k_mean = {k: float(np.mean(v)) for k, v in kres.items()}
        best_k = min(k_mean, key=k_mean.get)
        rows.append({"noise": nz, "n_seeds": len(seeds), "floor": float(np.mean(floor)), "hw_mult": float(np.mean(hw)), "best_k": best_k, "best_k_mase": k_mean[best_k], "alpha": float(np.mean(alphas)),
                     "wins": int(sum(h < b for h, b in zip(hw, kres[best_k])))})
    return rows


# --- Experiment 3: wie empfindlich ist alpha, und hilft Neuschätzen? -------------------------------------------------------------------------


def with_alpha(model, alpha):
    """Dasselbe Modell mit von Hand gesetztem alpha; beta und gamma werden auf die zulässige Grenze (beta <= alpha, gamma <= 1 - alpha) gekappt."""
    return dataclasses.replace(model, alpha=alpha, beta=min(model.beta, alpha), gamma=min(model.gamma, 1 - alpha))


def alpha_experiment(alphas=C.ALPHA_GRID, seeds=None, base=None, refit_every=C.REFIT_EVERY, key="hw_mult"):
    """Holt-Winters multiplikativ mit von Hand gesetztem alpha; dazu die geschätzten Parameter gegen Parameter, die alle refit_every Ursprünge auf allen bis dahin bekannten Tagen neu geschätzt werden."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    h = base.horizon
    grid = {a: [] for a in alphas}
    fitted_m, fitted_a, refit_m = [], [], []
    for sd in seeds:
        st = _replace(base, seed=sd)
        series = _series(st.trend, st.weekly, st.yearly, st.noise, st.shift, st.events, sd)
        model, states = _fit_one(st.trend, st.weekly, st.yearly, st.noise, st.shift, st.events, sd, key)
        org = F.origins(series.n, h, step=base.step)
        act = np.stack([series.y[t:t + h] for t in org])
        scale = F.mase_scale(series.y, C.FIRST_TEST)
        for a_ in alphas:
            m2 = with_alpha(model, a_)
            grid[a_].append(np.abs(ETS.forecast_origins(m2, ETS.filter_states(m2, series.y), org, h) - act).mean() / scale)
        fitted_m.append(np.abs(ETS.forecast_origins(model, states, org, h) - act).mean() / scale)
        fitted_a.append(model.alpha)
        f = np.zeros(act.shape)
        for i0 in range(0, len(org), refit_every):
            blk = org[i0:i0 + refit_every]
            m3 = ETS.fit(series.y[:blk[0]], key)
            f[i0:i0 + refit_every] = ETS.forecast_origins(m3, ETS.filter_states(m3, series.y), blk, h)
        refit_m.append(np.abs(f - act).mean() / scale)
    rows = [dict(alpha=a_, **dict(zip(("mase", "mase_se"), _mean_se(v)))) for a_, v in grid.items()]
    return {"rows": rows, "n_seeds": len(seeds), "fit_alpha": float(np.mean(fitted_a)), "fit_mase": float(np.mean(fitted_m)), "refit_mase": float(np.mean(refit_m)), "refit_every": refit_every}


# --- Experiment 4: Ereignisse -------------------------------------------------------------------------------------------------------------------


def event_classes(series):
    """0 = Ereignistag (Feiertag, Tag danach, Aktion), 1 = höchstens 14 Tage nach dem letzten Ereignistag, 2 = sonst."""
    ev = (series.holiday + np.roll(series.holiday, 1) + series.promo) > 0
    cls = np.full(series.n, 2)
    last = -10 ** 6
    for d in range(series.n):
        if ev[d]:
            last = d
            cls[d] = 0
        elif d - last <= C.EVENT_AFTER_DAYS:
            cls[d] = 1
    return cls


def events_experiment(levels=None, seeds=None, base=None, methods=("hw_mult", "snaive_k")):
    """Fehler (MAE in Aufträgen) an Ereignistagen, in den Tagen danach und sonst - für jede Ereignisstärke; die Ereignistage sind in allen Stärken dieselben."""
    levels = C.EVENT_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for ev in levels:
        acc = {m: [[], [], []] for m in methods}
        orc = [[], [], []]
        share = []
        for sd in seeds:
            a = analyse(_replace(base, events=ev, seed=sd), ("hw_mult",))
            days = a.origins[:, None] + np.arange(base.horizon)[None, :]
            cm = event_classes(a.series)[days]
            oe = np.abs(a.series.mu[days] - a.series.y[days])
            share.append([float(np.mean(cm == c)) for c in range(3)])
            for c in range(3):
                orc[c].append(float(oe[cm == c].mean()))
                for m in methods:
                    acc[m][c].append(float(np.abs(a.errors[m])[cm == c].mean()))
        row = {"events": ev, "n_seeds": len(seeds), "share": list(np.mean(share, axis=0)), "oracle": [float(np.mean(v)) for v in orc]}
        for m in methods:
            row[m] = [float(np.mean(v)) for v in acc[m]]
        rows.append(row)
    return rows


# --- Experiment 5: additiv oder multiplikativ ----------------------------------------------------------------------------------------------------


def weekly_experiment(levels=None, seeds=None, base=None):
    levels = C.WEEKLY_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for w in levels:
        res = {m: [] for m in ("hw_add", "hw_mult", "snaive_k")}
        floor = []
        for sd in seeds:
            a = analyse(_replace(base, weekly=w, seed=sd), ("hw_add", "hw_mult"))
            for m in res:
                res[m].append(a.summary[m]["mase"])
            floor.append(a.oracle["mase"])
        rows.append(_row(res, {"weekly": w, "n_seeds": len(seeds), "floor": float(np.mean(floor))}))
    return rows
