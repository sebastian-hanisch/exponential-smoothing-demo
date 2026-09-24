"""Plotly-Abbildungen der Exponentielle-Glättung-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import es_constants as C
import es_forecast as F
import es_ets as ETS

COLORS = {"ses": "#cab2d6", "holt": "#a78bc4", "damped": "#6a3d9a", "hw_add": "#f4b183", "hw_mult": "#e6550d", "hw_mult_damped": "#a63603", "naive": "#7f7f7f", "snaive": "#4c78a8", "snaive_k": "#17becf"}
SHORT = {"ses": "Einfache Glättung", "holt": "Holt", "damped": "Holt gedämpft", "hw_add": "HW additiv", "hw_mult": "HW multiplikativ", "hw_mult_damped": "HW mult. gedämpft", "naive": "Naiv", "snaive": "Saisonal naiv",
         "snaive_k": "Wochenmittel"}
ACTUAL = "#14233B"
ORACLE = "#54a24b"
WARN = "#f58518"
CLASS_NAMES = ("Ereignistag", f"bis {C.EVENT_AFTER_DAYS} Tage danach", "sonst")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def method_label(m, k):
    return f"Wochenmittel der letzten {k} Wochen" if m == "snaive_k" else C.METHOD_NAMES[m]


def build_series(a, origin, key):
    """Die ganze Reihe (drei Jahre), das Niveau des gewählten Modells (Zustand, der zu jedem Tag nur die Tage davor kennt), der Erwartungswert und der gewählte Ursprung."""
    s = a.series
    t = np.arange(s.n)
    lvl = a.fitted[key][1].level[:s.n]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s.y, mode="lines", name="Tagesaufträge", line=dict(color=ACTUAL, width=1)))
    fig.add_trace(go.Scatter(x=t, y=s.mu, mode="lines", name="Erwartungswert (ohne Rauschen)", line=dict(color=ORACLE, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=lvl, mode="lines", name=f"Niveau des Modells ({SHORT[key]})", line=dict(color=COLORS[key], width=2.5)))
    fig.add_vrect(x0=C.FIRST_TEST, x1=s.n, fillcolor="rgba(245,133,24,0.08)", line_width=0, annotation_text="Testjahr (Ursprünge)", annotation_position="top left")
    fig.add_vline(x=origin, line=dict(color=WARN, dash="dash"))
    if s.shift_day >= 0:
        fig.add_vline(x=s.shift_day, line=dict(color="#e45756", dash="dot"), annotation_text="Niveausprung", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 320)


def build_origin(a, origin, key):
    """Die 42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Erwartungswert, das gewählte Modell und drei Vergleichsverfahren."""
    s, st = a.series, a.settings
    h = st.horizon
    org = np.array([origin])
    x_hist = np.arange(origin - 42, origin)
    x_fut = np.arange(origin, origin + h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=s.y[origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=s.y[origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=2), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=s.mu[origin:origin + h], mode="lines", name="Erwartungswert", line=dict(color=ORACLE, width=2, dash="dot")))
    shown = [key] + [m for m in ("snaive_k", "ses", "naive") if m != key]
    for m in shown:
        if m in C.ETS_MODELS:
            model, states = a.fitted[m]
            f = ETS.forecast_origins(model, states, org, h)[0]
        else:
            f = F.baseline_origins(m, s.y, org, h, st.k)[0]
        fig.add_trace(go.Scatter(x=x_fut, y=f, mode="lines", name=method_label(m, st.k), line=dict(color=COLORS[m], width=3 if m == key else 1.4, dash="solid" if m == key else "dash")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.35))


def learned_weekly(a, origin, key):
    """Das vom Modell gelernte Wochenmuster am Ursprung als Faktoren um 1 (additive Werte durch das Niveau geteilt); None bei Modellen ohne Saison."""
    model, states = a.fitted[key]
    if model.season == "none":
        return None
    s = states.season[origin]
    return s if model.season == "mul" else 1.0 + s / max(states.level[origin], C.EPS)


def true_weekly(weekly):
    pattern = np.array(C.WEEKLY_PATTERN)
    pattern = pattern / pattern.mean()
    return 1.0 + weekly * (pattern - 1.0)


def build_state(a, origin, key):
    learned = learned_weekly(a, origin, key)
    if learned is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(C.WEEKDAYS), y=learned, name="vom Modell gelernt (Zustand am Ursprung)", marker=dict(color=COLORS[key])))
    fig.add_trace(go.Scatter(x=list(C.WEEKDAYS), y=true_weekly(a.settings.weekly), mode="markers", name="wahres Wochenmuster", marker=dict(color=ORACLE, size=11, symbol="diamond")))
    fig.update_yaxes(title_text="Faktor gegenüber dem Niveau", rangemode="tozero")
    return _base(fig, 300)


def build_bars(a):
    """MASE je Verfahren über alle Ursprünge und Horizonte, dazu die Orakel-Untergrenze."""
    ms = sorted(a.summary, key=lambda m: a.summary[m]["mase"])
    fig = go.Figure(go.Bar(x=[method_label(m, a.settings.k) for m in ms], y=[a.summary[m]["mase"] for m in ms], marker=dict(color=[COLORS[m] for m in ms]),
                           text=[f"{a.summary[m]['mase']:.2f}".replace(".", ",") for m in ms], textposition="outside", showlegend=False))
    fig.add_hline(y=a.oracle["mase"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel (wahrer Erwartungswert)", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="MASE 1 = saisonal naiv im Training", annotation_position="bottom right")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380)


def build_horizon(a):
    scale = F.mase_scale(a.series.y, C.FIRST_TEST)
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in a.summary:
        fig.add_trace(go.Scatter(x=xs, y=a.horizon_mae[m] / scale, mode="lines+markers", name=method_label(m, a.settings.k), line=dict(color=COLORS[m], width=2.5 if m in ("hw_mult", "snaive_k") else 1.3), marker=dict(size=4)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.35))


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_ladder(row):
    """Links: MASE der Verfahren (Fehlerbalken: Standardfehler über die Reihen), das Wochenmittel mit dem besten k. Rechts: MASE des Wochenmittels über k, dazu Holt-Winters multiplikativ."""
    order = ["naive", "ses", "holt", "damped", "snaive", "snaive_k", "hw_add", "hw_mult", "hw_mult_damped"]
    labels = [f"Wochenmittel (k = {row['best_k']})" if m == "snaive_k" else SHORT[m] for m in order]
    vals = [row["best_k_mase"] if m == "snaive_k" else row[m] for m in order]
    ses = [0.0 if m == "snaive_k" else row[m + "_se"] for m in order]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], subplot_titles=("MASE der Verfahren (Standardreihen)", "Wochenmittel: MASE über k"))
    fig.add_trace(go.Bar(x=labels, y=vals, error_y=dict(type="data", array=ses), marker=dict(color=[COLORS[m] for m in order]), text=[f"{v:.2f}".replace(".", ",") for v in vals], textposition="outside", showlegend=False), row=1, col=1)
    fig.add_hline(y=row["floor"], line=dict(color=ORACLE, dash="dot"), row=1, col=1, annotation_text="Orakel", annotation_position="top left")
    ks = list(row["k_mean"])
    fig.add_trace(go.Scatter(x=[str(k) for k in ks], y=[row["k_mean"][k] for k in ks], mode="lines+markers", name="Wochenmittel", line=dict(color=COLORS["snaive_k"], width=2), showlegend=False), row=1, col=2)
    fig.add_hline(y=row["hw_mult"], line=dict(color=COLORS["hw_mult"], dash="dash"), row=1, col=2, annotation_text="HW multiplikativ", annotation_position="bottom right")
    fig.update_xaxes(title_text="Wochen k", type="category", row=1, col=2)
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    return _base(fig, 380)


def build_window_alpha(rows):
    xs = [f"{r['noise']:.2f}".replace(".", ",") + f"<br>k = {r['best_k']} / α = " + f"{r['alpha']:.2f}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["best_k_mase"] for r in rows], mode="lines+markers", name="Wochenmittel mit dem besten k (im Nachhinein gewählt)", line=dict(color=COLORS["snaive_k"], width=2)))
    fig.add_trace(go.Scatter(x=xs, y=[r["hw_mult"] for r in rows], mode="lines+markers", name="Holt-Winters multiplikativ (α geschätzt)", line=dict(color=COLORS["hw_mult"], width=2)))
    fig.add_trace(go.Scatter(x=xs, y=[r["floor"] for r in rows], mode="lines", name="Orakel (Untergrenze)", line=dict(color=ORACLE, dash="dot", width=2)))
    fig.update_xaxes(title_text="Streuung des Rauschens (darunter: bestes k und geschätztes α)", type="category")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.4))


def build_alpha(res):
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["alpha"] for r in rows], y=[r["mase"] for r in rows], error_y=dict(type="data", array=[r["mase_se"] for r in rows]), mode="lines+markers", name="von Hand gesetztes α", line=dict(color=COLORS["hw_mult"], width=2)))
    fig.add_vline(x=res["fit_alpha"], line=dict(color=ORACLE, dash="dot"), annotation_text=f"geschätzt: α = {res['fit_alpha']:.2f}".replace(".", ","), annotation_position="top right")
    fig.update_xaxes(title_text="Glättungsparameter α des Niveaus (β und γ wie geschätzt, ggf. gekappt)", type="log", tickvals=[r["alpha"] for r in rows], ticktext=[f"{r['alpha']:.2f}".replace(".", ",") for r in rows])
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 340).update_layout(showlegend=False)


def build_events(rows):
    fig = make_subplots(rows=1, cols=len(rows), shared_yaxes=True, subplot_titles=[f"Ereignisstärke {r['events']:.2f}".replace(".", ",") for r in rows])
    for i, r in enumerate(rows, start=1):
        for name, vals, color in (("Orakel (Rauschen)", r["oracle"], ORACLE), ("Holt-Winters multiplikativ", r["hw_mult"], COLORS["hw_mult"]), ("Wochenmittel", r["snaive_k"], COLORS["snaive_k"])):
            fig.add_trace(go.Bar(x=list(CLASS_NAMES), y=vals, name=name, marker=dict(color=color), showlegend=(i == 1), text=[f"{v:.0f}" for v in vals], textposition="outside"), row=1, col=i)
    fig.update_yaxes(title_text="mittlerer absoluter Fehler (Aufträge je Tag)", rangemode="tozero", col=1)
    fig.update_layout(barmode="group")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3))


def build_weekly(rows):
    xs = [f"{r['weekly']:.1f}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for m, name in (("hw_add", "Holt-Winters additiv"), ("hw_mult", "Holt-Winters multiplikativ"), ("snaive_k", "Wochenmittel")):
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], error_y=dict(type="data", array=[r[m + "_se"] for r in rows]), mode="lines+markers", name=name, line=dict(color=COLORS[m], width=2)))
    fig.add_trace(go.Scatter(x=xs, y=[r["floor"] for r in rows], mode="lines", name="Orakel (Untergrenze)", line=dict(color=ORACLE, dash="dot", width=2)))
    fig.update_xaxes(title_text="Stärke des Wochenmusters", type="category")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))
