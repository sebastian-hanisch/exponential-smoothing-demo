"""Konstanten der Exponentielle-Glättung-Demo: Vehikel "Tagesaufträge eines Depots" (Stück 2 der Zeitreihen-Prognose-Linie, wortgleich zu Stück 1), Verfahren, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095                                       # drei Jahre täglich
FIRST_TEST = 730                                    # Ursprünge liegen im letzten Jahr; die Parameter werden auf den ersten zwei Jahren geschätzt
LEVEL = 100.0                                       # mittlere Tagesaufträge zu Beginn
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)     # Tage im Jahr (0 = 1. Januar), an denen das Depot ruht
HOLIDAY_DROP = 0.5                                  # Rückgang am Feiertag (Faktor 1 - 0,5 * Stärke)
HOLIDAY_REBOUND = 0.15                              # Nachholeffekt am Folgetag
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3

TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -20, 40, 5, 10              # Prozent je Jahr
WEEKLY_MIN, WEEKLY_MAX, WEEKLY_STEP, DEFAULT_WEEKLY = 0.0, 1.5, 0.25, 1.0
YEARLY_MIN, YEARLY_MAX, YEARLY_STEP, DEFAULT_YEARLY = 0.0, 0.5, 0.05, 0.2
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.02, 0.5, 0.02, 0.14
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = -40, 40, 10, 0             # Prozent, Niveausprung an einem zufälligen Tag im Testjahr
EVENTS_MIN, EVENTS_MAX, EVENTS_STEP, DEFAULT_EVENTS = 0.0, 1.0, 0.25, 0.5     # Stärke von Feiertagen und Aktionen
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
STEP_MIN, STEP_MAX, DEFAULT_STEP = 1, 14, 1
K_WEEKS_MIN, K_WEEKS_MAX, DEFAULT_K_WEEKS = 2, 12, 4

# --- Verfahren --------------------------------------------------------------------------------------------------------------------------------
# Exponentielle Glättung in der Zustandsraum-Schreibweise mit additivem Fehler: (Trend, Saison, gedämpft).
ETS_MODELS = {
    "ses": ("none", "none"),
    "holt": ("add", "none"),
    "damped": ("damped", "none"),
    "hw_add": ("add", "add"),
    "hw_mult": ("add", "mul"),
    "hw_mult_damped": ("damped", "mul"),
}
BASELINES = ("naive", "snaive", "snaive_k")
METHODS = tuple(ETS_MODELS) + BASELINES
METHOD_NAMES = {
    "ses": "Einfache Glättung (SES)",
    "holt": "Holt (Niveau + Trend)",
    "damped": "Holt gedämpft",
    "hw_add": "Holt-Winters additiv",
    "hw_mult": "Holt-Winters multiplikativ",
    "hw_mult_damped": "Holt-Winters multiplikativ, gedämpft",
    "naive": "Naiv (letzter Tag)",
    "snaive": "Saisonal naiv (letzte Woche)",
    "snaive_k": "Saisonal: Mittel der letzten k Wochen",
}
SEASON_PERIOD = 7
DEFAULT_MODEL = "hw_mult"                           # Modell in der Ursprungsansicht

# --- Parameterschätzung -----------------------------------------------------------------------------------------------------------------------
INIT_DAYS = 28                                      # Anfangszustände ohne Saison aus den ersten vier Wochen; die Fehlersumme zählt ab diesem Tag
INIT_WEEKS = 8                                      # mit Saison: Gerade durch die ersten acht saisonbereinigten Wochen
FIT_STAGE1 = 3000                                   # Zufallspunkte im Einheitswürfel
FIT_TOP = 6                                         # so viele Startpunkte werden verfeinert
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Experimente (feste Seeds) ----------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(12))
NOISE_LEVELS = (0.04, 0.08, 0.14, 0.24, 0.4)
TREND_LEVELS = (-20, 0, 10, 25, 40)
K_GRID = (2, 3, 4, 6, 8, 12)                        # Wochen im Wochenmittel, aus denen das (im Nachhinein) beste k gewählt wird
ALPHA_GRID = (0.02, 0.05, 0.09, 0.15, 0.25, 0.4, 0.6)
REFIT_EVERY = 56                                    # Ursprünge zwischen zwei Neuschätzungen der Parameter
EVENT_LEVELS = (0.0, 0.5, 1.0)
EVENT_AFTER_DAYS = 14                               # so viele Tage nach dem letzten Ereignistag gelten als "danach"
WEEKLY_LEVELS = (0.5, 1.0, 1.5)
