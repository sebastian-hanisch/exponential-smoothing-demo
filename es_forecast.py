"""Rolling-Origin-Auswertung der Glättungsmodelle und der naiven Vergleichsverfahren aus Stück 1.

Ein Ursprung t heißt: bekannt sind die Tage 0..t-1, prognostiziert werden die Tage t..t+h-1. Die Glättungsparameter stammen aus den Trainingstagen (vor dem ersten Ursprung); die Zustände laufen mit jedem neuen Tag weiter.

  naive     letzter beobachteter Tag
  snaive    derselbe Wochentag der letzten Woche
  snaive_k  Mittel desselben Wochentags über die letzten k Wochen"""

import numpy as np

import es_constants as C
import es_ets as ETS


def baseline_origins(method, y, org, h, k=C.DEFAULT_K_WEEKS):
    """Naive Prognosen für alle Ursprünge auf einmal: (Ursprünge, h). Voraussetzung: jeder Ursprung liegt mindestens 7 * k Tage hinter dem Anfang."""
    org = np.asarray(org)
    j = np.arange(h)
    if method == "naive":
        return np.repeat(y[org - 1][:, None], h, axis=1)
    if method == "snaive":
        return y[org[:, None] - 7 + (j % 7)[None, :]]
    if method == "snaive_k":
        return np.mean([y[org[:, None] - 7 * (i + 1) + (j % 7)[None, :]] for i in range(k)], axis=0)
    raise ValueError(method)


def origins(n, h, first=C.FIRST_TEST, step=C.DEFAULT_STEP):
    return np.arange(first, n - h + 1, step)


def mase_scale(y, t0):
    """Mittlerer absoluter Fehler der saisonal naiven Prognose (Periode 7) innerhalb der Trainingsdaten y[:t0] - Nenner der MASE."""
    return float(np.abs(y[7:t0] - y[:t0 - 7]).mean())


def fit_models(y, keys=tuple(C.ETS_MODELS), first=C.FIRST_TEST):
    """Parameter aller Modelle auf y[:first] schätzen und die Zustände durch die ganze Reihe fortschreiben: {Schlüssel: (Modell, Zustände)}."""
    out = {}
    for key in keys:
        model = ETS.fit(y[:first], key)
        out[key] = (model, ETS.filter_states(model, y))
    return out


def rolling_origin(y, fitted, h=C.DEFAULT_HORIZON, step=C.DEFAULT_STEP, k=C.DEFAULT_K_WEEKS, first=C.FIRST_TEST, baselines=C.BASELINES):
    """Prognosefehler (Prognose minus Ist) je Ursprung und Horizont: {Verfahren: (Ursprünge, h)}, Ursprünge und Ist-Werte (Ursprünge, h)."""
    org = origins(len(y), h, first, step)
    actual = np.stack([y[t:t + h] for t in org])
    errors = {}
    for key, (model, states) in fitted.items():
        errors[key] = ETS.forecast_origins(model, states, org, h) - actual
    for mth in baselines:
        errors[mth] = baseline_origins(mth, y, org, h, k) - actual
    return org, actual, errors


def summarize(errors, y, first=C.FIRST_TEST):
    """MAE, RMSE, ME (Verzerrung) und MASE je Verfahren über alle Ursprünge und Horizonte."""
    scale = mase_scale(y, first)
    out = {}
    for mth, E in errors.items():
        mae = float(np.abs(E).mean())
        out[mth] = {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}
    return out


def per_horizon(errors):
    return {m: np.abs(E).mean(axis=0) for m, E in errors.items()}


def per_origin(errors):
    """MAE je Ursprung (über den Horizont gemittelt)."""
    return {m: np.abs(E).mean(axis=1) for m, E in errors.items()}
