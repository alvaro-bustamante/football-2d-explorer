"""Team and player frames, block metrics and the metric registry (labels in EN/ES/PT).

Each metric returns one of:
  ("prop",   successes, attempts)   Beta-Binomial  -> shown in %
  ("rate",   count, exposure)       Gamma-Poisson  -> per match (teams) or per 90 (players)
  ("normal", value, std_error)      Normal-Normal  -> xG, positions, differences
  ("none",   value, None)           no uncertainty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, factorial
from typing import Callable

import numpy as np
import pandas as pd

from core.bayes import eb_beta_binomial, eb_gamma_poisson, eb_normal

from .i18n import lang

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
PENALTY_XG = 0.78
SHRINK = {"prop": eb_beta_binomial, "normal": eb_normal, "rate": eb_gamma_poisson}


@dataclass(frozen=True)
class Metric:
    key: str
    names: tuple            # (en, es, pt)
    better: str | None      # "high", "low" or None
    compute: Callable
    helps: tuple = ("", "", "")
    scale: float = 1.0      # e.g. 90 for "per 90" rates, 100 for percentages
    tags: tuple = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return self.names[{"en": 0, "es": 1, "pt": 2}[lang()]]

    @property
    def help(self) -> str:
        return self.helps[{"en": 0, "es": 1, "pt": 2}[lang()]]


# ----------------------------------------------------------------- xPts
def _poisson(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def expected_points(xg_for: float, xg_against: float, max_goals: int = 10) -> float:
    """Points expected from a match given both teams' xG (independent Poisson goals)."""
    if not (np.isfinite(xg_for) and np.isfinite(xg_against)):
        return np.nan
    pf = [_poisson(k, max(xg_for, 0.01)) for k in range(max_goals)]
    pa = [_poisson(k, max(xg_against, 0.01)) for k in range(max_goals)]
    win = sum(pf[i] * pa[j] for i in range(max_goals) for j in range(i))
    draw = sum(pf[i] * pa[i] for i in range(max_goals))
    return 3 * win + draw


# ------------------------------------------------------------ block metrics
def _hull_area(pts: np.ndarray) -> float:
    """Convex hull area (monotone chain + shoelace)."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def block_by_match(positions: pd.DataFrame) -> pd.DataFrame:
    """Block metrics per team and match from the starters' average positions (in metres)."""
    if positions is None or positions.empty:
        return pd.DataFrame()
    p = positions[positions["starter"].astype(bool) & (positions["position"] != "G")].copy()
    p["xm"] = p["x"] / 100 * PITCH_LENGTH
    p["ym"] = p["y"] / 100 * PITCH_WIDTH
    rows = []
    for (mid, tid), g in p.groupby(["match_id", "team_id"]):
        if len(g) < 6:
            continue
        xs, ys = np.sort(g["xm"].values), np.sort(g["ym"].values)
        d = g[g["position"] == "D"]
        rows.append({"match_id": mid, "team_id": tid,
                     "def_line": d["xm"].mean() if len(d) else xs[:4].mean(),
                     "centroid": g["xm"].mean(),
                     "length": xs[-2:].mean() - xs[:2].mean(),
                     "width": ys[-2:].mean() - ys[:2].mean(),
                     "area": _hull_area(g[["xm", "ym"]].values)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- frames
NORMAL_SERIES = {   # per-match series summarised as mean + standard error
    "xg_for": lambda M: M["expectedGoals"],
    "xg_against": lambda M: M["opp_expectedGoals"],
    "xgd": lambda M: M["expectedGoals"] - M["opp_expectedGoals"],
    "possession": lambda M: M["ballPossession"],
    "g_minus_xg": lambda M: M["gf"] - M["expectedGoals"],
    "ga_minus_xga": lambda M: M["ga"] - M["opp_expectedGoals"],
    "points": lambda M: M["points"],
    "xpts": lambda M: M["xpts"],
    "luck": lambda M: M["points"] - M["xpts"],
    "km": lambda M: M.get("kilometersCovered"),
    "sprints": lambda M: M.get("numberOfSprints"),
    "def_line": lambda M: M.get("def_line"),
    "centroid": lambda M: M.get("centroid"),
    "length": lambda M: M.get("length"),
    "width": lambda M: M.get("width"),
    "area": lambda M: M.get("area"),
}


def match_table(data: dict) -> pd.DataFrame:
    M = data["matches"].copy()
    M["points"] = np.select([M.gf > M.ga, M.gf == M.ga], [3, 1], 0)
    M["xpts"] = [expected_points(a, b) for a, b in zip(M["expectedGoals"], M["opp_expectedGoals"])]
    blocks = block_by_match(data.get("positions"))
    if not blocks.empty:
        M = M.merge(blocks, on=["match_id", "team_id"], how="left")
    return M


def team_frame(data: dict) -> pd.DataFrame:
    M = match_table(data)
    g = M.groupby("team_id")
    num = M.select_dtypes("number").drop(columns=["match_id", "round", "team_id", "opp_id"], errors="ignore")
    T = num.groupby(M["team_id"]).sum(min_count=1)
    T["n"] = g.size()
    T["padj_exposure"] = ((100 - M["ballPossession"]) / 50).groupby(M["team_id"]).sum()
    for key, fn in NORMAL_SERIES.items():
        s = fn(M)
        if s is None or pd.isna(s).all():
            T[f"{key}_mean"], T[f"{key}_se"] = np.nan, np.nan
            continue
        s = pd.Series(np.asarray(s, float), index=M.index)
        mean = s.groupby(M["team_id"]).mean()
        var = s.groupby(M["team_id"]).var(ddof=1)
        pooled = np.sqrt(np.nanmean(var)) if np.isfinite(var).any() else np.nan
        cnt = s.notna().groupby(M["team_id"]).sum()
        T[f"{key}_mean"] = mean
        T[f"{key}_se"] = pooled / np.sqrt(cnt.clip(lower=1))
    info = data["teams"].set_index("team_id") if not data["teams"].empty else pd.DataFrame(index=T.index)
    for c in ("name", "short_name", "code", "primary", "secondary", "crest_url", "penaltiesTaken",
              "ownHalfPassesTotalAgainst", "tackles", "interceptions", "fouls"):
        if c in info.columns:
            T[f"info_{c}"] = info[c].reindex(T.index)
    if "info_name" not in T or T["info_name"].isna().all():
        T["info_name"] = g["team"].first()
    T["info_short_name"] = T.get("info_short_name", T["info_name"]).fillna(T["info_name"])
    T.attrs.update(exposure="n", unit="unit_matches", per=1.0)
    return T


def player_frame(data: dict) -> pd.DataFrame:
    P = data["players"].copy()
    for col in ("minutesPlayed", "expectedGoals", "penaltiesTaken", "expectedAssists", "goals",
                "groundDuelsWon", "groundDuelsWonPercentage", "aerialDuelsWon", "aerialLost"):
        if col not in P.columns:
            P[col] = np.nan
    P = P[P["minutesPlayed"].fillna(0) > 0].set_index("player_id")
    P["m90"] = P["minutesPlayed"] / 90
    P["npxg"] = (P["expectedGoals"] - PENALTY_XG * P["penaltiesTaken"].fillna(0)).clip(lower=0)
    P["npxg_xa"] = P["npxg"] + P["expectedAssists"]
    pct = P["groundDuelsWonPercentage"].replace(0, np.nan)
    P["groundDuels"] = P["groundDuelsWon"] / (pct / 100)
    P["aerialDuels"] = P["aerialDuelsWon"] + P["aerialLost"]
    if "position" not in P.columns:
        P["position"] = "?"
    P.attrs.update(exposure="m90", unit="unit_minutes", per=90.0)
    return P


# ----------------------------------------------------------------- helpers
def _c(F, col):
    return F[col] if col in F.columns else pd.Series(np.nan, index=F.index)


def _prop(s, n):
    return lambda F: ("prop", _c(F, s) if isinstance(s, str) else s(F), _c(F, n) if isinstance(n, str) else n(F))


def _rate(col, exposure="n"):
    return lambda F: ("rate", _c(F, col) if isinstance(col, str) else col(F), _c(F, exposure))


def _normal(key):
    return lambda F: ("normal", _c(F, f"{key}_mean"), _c(F, f"{key}_se"))


def _ratio(num, den):
    """Ratio of two counts with a delta-method standard error."""
    def f(F):
        a, b = num(F), den(F)
        r = a / b.replace(0, np.nan)
        se = r * np.sqrt(1 / a.clip(lower=0.5) + 1 / b.clip(lower=0.5))
        return ("normal", r, se)
    return f


def _per90_normal(value_col, se_col):
    return lambda F: ("normal", _c(F, value_col), _c(F, se_col))


# ------------------------------------------------------------------ teams
TEAM_METRICS = [
    # overall
    Metric("xgd", ("xG difference per match", "Diferencia de xG por partido", "Diferença de xG por jogo"), "high",
           _normal("xgd"), ("xG created minus xG conceded: the best single measure of level.",
                            "xG creado menos xG concedido: el mejor indicador del nivel real.",
                            "xG criado menos xG concedido: o melhor indicador do nível real.")),
    Metric("points", ("Points per match", "Puntos por partido", "Pontos por jogo"), "high", _normal("points")),
    Metric("xpts", ("Expected points per match", "Puntos esperados por partido", "Pontos esperados por jogo"), "high",
           _normal("xpts"), ("Points expected from the xG of both teams in each match.",
                             "Puntos esperados según el xG de ambos equipos en cada partido.",
                             "Pontos esperados pelo xG de ambas as equipas em cada jogo.")),
    Metric("luck", ("Luck: points − xPts per match", "Suerte: puntos − xPts por partido", "Sorte: pontos − xPts por jogo"),
           None, _normal("luck"), ("Above 0: more points than the chances deserved.",
                                   "Por encima de 0: más puntos de los que merecían las ocasiones.",
                                   "Acima de 0: mais pontos do que as ocasiões mereciam.")),
    # attack
    Metric("xg_for", ("xG per match", "xG por partido", "xG por jogo"), "high", _normal("xg_for")),
    Metric("npxg_for", ("Non-penalty xG per match", "xG sin penaltis por partido", "xG sem penáltis por jogo"), "high",
           lambda F: ("normal", F["xg_for_mean"] - PENALTY_XG * _c(F, "info_penaltiesTaken").fillna(0) / F["n"],
                      F["xg_for_se"])),
    Metric("goals", ("Goals per match", "Goles por partido", "Golos por jogo"), "high", _rate("gf")),
    Metric("shots", ("Shots per match", "Tiros por partido", "Remates por jogo"), "high", _rate("totalShotsOnGoal")),
    Metric("xg_shot", ("xG per shot", "xG por tiro", "xG por remate"), "high",
           lambda F: ("normal", F["expectedGoals"] / F["totalShotsOnGoal"],
                      np.sqrt(F["expectedGoals"].clip(lower=0.1)) / F["totalShotsOnGoal"]),
           ("Average chance quality.", "Calidad media de las ocasiones.", "Qualidade média das ocasiões.")),
    Metric("conversion", ("Shot conversion %", "Conversión de tiros %", "Conversão de remates %"), "high",
           _prop("gf", "totalShotsOnGoal"), scale=100),
    Metric("on_target", ("Shots on target %", "Tiros a puerta %", "Remates enquadrados %"), "high",
           _prop("shotsOnGoal", "totalShotsOnGoal"), scale=100),
    Metric("g_minus_xg", ("Goals − xG per match", "Goles − xG por partido", "Golos − xG por jogo"), None,
           _normal("g_minus_xg"), ("Finishing above (positive) or below expectation: mostly luck in the short run.",
                                   "Definición por encima (positivo) o por debajo de lo esperado: a corto plazo, sobre todo suerte.",
                                   "Finalização acima (positivo) ou abaixo do esperado: a curto prazo, sobretudo sorte.")),
    Metric("big_chances", ("Big chances per match", "Ocasiones claras por partido", "Grandes oportunidades por jogo"),
           "high", _rate("bigChanceCreated")),
    Metric("box_touches", ("Touches in opp. box per match", "Toques en el área rival por partido",
                           "Toques na área adversária por jogo"), "high", _rate("touchesInOppBox")),
    # style
    Metric("possession", ("Possession %", "Posesión %", "Posse de bola %"), None, _normal("possession")),
    Metric("field_tilt", ("Field tilt %", "Field tilt (dominio territorial) %", "Field tilt (domínio territorial) %"),
           None, _prop("finalThirdEntries", lambda F: F["finalThirdEntries"] + F["opp_finalThirdEntries"]),
           ("Share of the final-third entries in the match: territorial dominance.",
            "Cuota de las entradas en el último tercio del partido: dominio territorial.",
            "Quota das entradas no último terço do jogo: domínio territorial."), scale=100),
    Metric("pass_acc", ("Pass accuracy %", "Precisión de pase %", "Precisão de passe %"), "high",
           _prop("accuratePasses", "passes"), scale=100),
    Metric("ppda", ("PPDA (approx.)", "PPDA (aprox.)", "PPDA (aprox.)"), None,
           _ratio(lambda F: _c(F, "info_ownHalfPassesTotalAgainst"),
                  lambda F: _c(F, "info_tackles") + _c(F, "info_interceptions") + _c(F, "info_fouls")),
           ("Opponent passes in their own half per defensive action. LOWER = more intense pressing. "
            "Approximation: defensive actions are counted anywhere on the pitch.",
            "Pases del rival en su campo por cada acción defensiva. MÁS BAJO = presión más intensa. "
            "Aproximación: las acciones defensivas se cuentan en todo el campo.",
            "Passes do adversário no seu meio-campo por ação defensiva. MAIS BAIXO = pressão mais intensa. "
            "Aproximação: as ações defensivas contam-se em todo o campo.")),
    # defence
    Metric("xg_against", ("xG conceded per match", "xG concedido por partido", "xG concedido por jogo"), "low",
           _normal("xg_against")),
    Metric("goals_against", ("Goals conceded per match", "Goles en contra por partido", "Golos sofridos por jogo"),
           "low", _rate("ga")),
    Metric("shots_against", ("Shots conceded per match", "Tiros concedidos por partido", "Remates concedidos por jogo"),
           "low", _rate("opp_totalShotsOnGoal")),
    Metric("xg_shot_against", ("xG per shot conceded", "xG por tiro concedido", "xG por remate concedido"), "low",
           lambda F: ("normal", F["opp_expectedGoals"] / F["opp_totalShotsOnGoal"],
                      np.sqrt(F["opp_expectedGoals"].clip(lower=0.1)) / F["opp_totalShotsOnGoal"])),
    Metric("ga_minus_xga", ("Goals conceded − xGA per match", "Goles en contra − xGA por partido",
                            "Golos sofridos − xGA por jogo"), "low", _normal("ga_minus_xga"),
           ("Below 0: the goalkeeper (or luck) saved more than expected.",
            "Por debajo de 0: el portero (o la suerte) evitó más goles de los esperados.",
            "Abaixo de 0: o guarda-redes (ou a sorte) evitou mais golos do que o esperado.")),
    Metric("save_pct", ("Save %", "Paradas %", "Defesas %"), "high",
           _prop("goalkeeperSaves", "opp_shotsOnGoal"), scale=100),
    Metric("tackles", ("Tackles per match (poss. adj.)", "Entradas por partido (aj. posesión)",
                       "Desarmes por jogo (aj. posse)"), None, _rate("totalTackle", "padj_exposure"),
           ("Adjusted to 50% possession: teams with less of the ball defend more.",
            "Ajustado a un 50% de posesión: los equipos con menos balón defienden más.",
            "Ajustado a 50% de posse: as equipas com menos bola defendem mais.")),
    Metric("interceptions", ("Interceptions per match (poss. adj.)", "Intercepciones por partido (aj. posesión)",
                             "Interceções por jogo (aj. posse)"), None, _rate("interceptionWon", "padj_exposure")),
    Metric("recoveries", ("Ball recoveries per match", "Recuperaciones por partido", "Recuperações por jogo"), "high",
           _rate("ballRecovery")),
    Metric("clearances", ("Clearances per match", "Despejes por partido", "Alívios por jogo"), None,
           _rate("totalClearance")),
    Metric("duels", ("Ground duels won %", "Duelos en el suelo ganados %", "Duelos no chão ganhos %"), "high",
           _prop("groundDuelsPercentage", "groundDuelsPercentage_total"), scale=100),
    Metric("aerials", ("Aerial duels won %", "Duelos aéreos ganados %", "Duelos aéreos ganhos %"), "high",
           _prop("aerialDuelsPercentage", "aerialDuelsPercentage_total"), scale=100),
    Metric("tackle_win", ("Tackles won %", "Entradas ganadas %", "Desarmes ganhos %"), "high",
           _prop("wonTacklePercent", "wonTacklePercent_total"), scale=100),
    Metric("errors", ("Errors leading to a shot per match", "Errores que acaban en tiro por partido",
                      "Erros que dão remate por jogo"), "low", _rate("errorsLeadToShot")),
    # block (average positions)
    Metric("def_line", ("Defensive line height (m)", "Altura de la línea defensiva (m)", "Altura da linha defensiva (m)"),
           None, _normal("def_line"), ("Average distance of the defenders from their own goal (event-based).",
                                       "Distancia media de los defensas a su portería (a partir de eventos).",
                                       "Distância média dos defesas à sua baliza (a partir de eventos).")),
    Metric("centroid", ("Team centroid height (m)", "Altura del centroide del equipo (m)", "Altura do centroide da equipa (m)"),
           None, _normal("centroid")),
    Metric("length", ("Block length (m)", "Longitud del bloque (m)", "Comprimento do bloco (m)"), None, _normal("length"),
           ("Distance between the deepest and the most advanced players: short = compact.",
            "Distancia entre los jugadores más retrasados y más adelantados: corta = compacto.",
            "Distância entre os jogadores mais recuados e mais avançados: curta = compacto.")),
    Metric("width", ("Block width (m)", "Anchura del bloque (m)", "Largura do bloco (m)"), None, _normal("width")),
    Metric("area", ("Block area (m²)", "Área del bloque (m²)", "Área do bloco (m²)"), None, _normal("area")),
    # physical
    Metric("km", ("Distance covered per match (km)", "Distancia recorrida por partido (km)", "Distância percorrida por jogo (km)"),
           None, _normal("km")),
    Metric("sprints", ("Sprints per match", "Sprints por partido", "Sprints por jogo"), None, _normal("sprints")),
]


# ---------------------------------------------------------------- players
def _p90(col, better="high", names=None, helps=("", "", "")):
    return Metric(col, names, better, _rate(col, "m90"), helps, scale=90)


PLAYER_METRICS = [
    Metric("npxg_xa", ("npxG + xA per 90", "npxG + xA por 90", "npxG + xA por 90"), "high", _rate("npxg_xa", "m90"),
           ("Non-penalty expected goals plus expected assists: total chance involvement.",
            "Goles esperados sin penaltis más asistencias esperadas: participación total en ocasiones.",
            "Golos esperados sem penáltis mais assistências esperadas: participação total em ocasiões."), scale=90),
    Metric("npxg", ("npxG per 90", "npxG por 90", "npxG por 90"), "high", _rate("npxg", "m90"), scale=90),
    _p90("expectedAssists", names=("xA per 90", "xA por 90", "xA por 90")),
    _p90("goals", names=("Goals per 90", "Goles por 90", "Golos por 90")),
    _p90("totalShots", names=("Shots per 90", "Tiros por 90", "Remates por 90")),
    Metric("g_minus_xg", ("Goals − xG per 90", "Goles − xG por 90", "Golos − xG por 90"), None,
           lambda F: ("normal", (F["goals"] - F["expectedGoals"]) / F["m90"],
                      np.sqrt(F["expectedGoals"].clip(lower=0.3)) / F["m90"]),
           ("Finishing above (positive) or below expectation. Strongly shrunk: it takes many shots to prove.",
            "Definición por encima (positivo) o por debajo de lo esperado. Muy contraída: hacen falta muchos tiros para demostrarla.",
            "Finalização acima (positivo) ou abaixo do esperado. Muito contraída: são precisos muitos remates para a provar.")),
    Metric("conversion", ("Shot conversion %", "Conversión de tiros %", "Conversão de remates %"), "high",
           _prop("goals", "totalShots"), scale=100),
    Metric("on_target", ("Shots on target %", "Tiros a puerta %", "Remates enquadrados %"), "high",
           _prop("shotsOnTarget", "totalShots"), scale=100),
    _p90("keyPasses", names=("Key passes per 90", "Pases clave por 90", "Passes decisivos por 90")),
    _p90("bigChancesCreated", names=("Big chances created per 90", "Ocasiones claras creadas por 90",
                                     "Grandes oportunidades criadas por 90")),
    _p90("successfulDribbles", names=("Successful dribbles per 90", "Regates completados por 90", "Dribles certos por 90")),
    Metric("dribble_pct", ("Dribble success %", "Éxito en el regate %", "Sucesso no drible %"), "high",
           _prop("successfulDribbles", "totalContest"), scale=100),
    _p90("touchesInOppBox", names=("Touches in opp. box per 90", "Toques en el área rival por 90",
                                   "Toques na área adversária por 90")),
    Metric("pass_acc", ("Pass accuracy %", "Precisión de pase %", "Precisão de passe %"), "high",
           _prop("accuratePasses", "totalPasses"), scale=100),
    Metric("long_acc", ("Long ball accuracy %", "Precisión en balón largo %", "Precisão em bola longa %"), "high",
           _prop("accurateLongBalls", "totalLongBalls"), scale=100),
    Metric("cross_acc", ("Cross accuracy %", "Precisión de centro %", "Precisão de cruzamento %"), "high",
           _prop("accurateCrosses", "totalCross"), scale=100),
    # defensive
    _p90("defensiveContributions", names=("Defensive actions per 90", "Acciones defensivas por 90", "Ações defensivas por 90"),
         helps=("Tackles + interceptions + clearances + blocks.", "Entradas + intercepciones + despejes + bloqueos.",
                "Desarmes + interceções + alívios + bloqueios.")),
    _p90("tackles", names=("Tackles per 90", "Entradas por 90", "Desarmes por 90")),
    Metric("tackle_win", ("Tackles won %", "Entradas ganadas %", "Desarmes ganhos %"), "high",
           _prop("tacklesWon", "tackles"), scale=100),
    _p90("interceptions", names=("Interceptions per 90", "Intercepciones por 90", "Interceções por 90")),
    _p90("ballRecovery", names=("Ball recoveries per 90", "Recuperaciones por 90", "Recuperações por 90")),
    _p90("clearances", better=None, names=("Clearances per 90", "Despejes por 90", "Alívios por 90")),
    _p90("outfielderBlocks", names=("Blocked shots per 90", "Tiros bloqueados por 90", "Remates bloqueados por 90")),
    _p90("possessionWonAttThird", names=("Possession won in final third per 90", "Balones ganados en último tercio por 90",
                                         "Bolas ganhas no último terço por 90"),
         helps=("High pressing that works.", "Presión alta que funciona.", "Pressão alta que funciona.")),
    _p90("dribbledPast", better="low", names=("Dribbled past per 90", "Veces regateado por 90", "Vezes driblado por 90")),
    Metric("aerial_pct", ("Aerial duels won %", "Duelos aéreos ganados %", "Duelos aéreos ganhos %"), "high",
           _prop("aerialDuelsWon", "aerialDuels"), scale=100),
    Metric("ground_pct", ("Ground duels won %", "Duelos en el suelo ganados %", "Duelos no chão ganhos %"), "high",
           _prop("groundDuelsWon", "groundDuels"), scale=100),
    _p90("possessionLost", better="low", names=("Possession lost per 90", "Pérdidas por 90", "Perdas de bola por 90")),
    # goalkeepers
    Metric("save_pct", ("Save %", "Paradas %", "Defesas %"), "high",
           _prop("saves", lambda F: _c(F, "saves") + _c(F, "goalsConceded")), scale=100),
    # overall / physical
    Metric("rating", ("Sofascore rating", "Valoración Sofascore", "Avaliação Sofascore"), "high",
           lambda F: ("normal", _c(F, "rating"), 0.75 / np.sqrt(_c(F, "appearances").clip(lower=1)))),
    Metric("km90", ("Distance per 90 (km)", "Distancia por 90 (km)", "Distância por 90 (km)"), None,
           lambda F: ("normal", _c(F, "kilometersCoveredPer90"), 0.8 / np.sqrt(F["m90"].clip(lower=0.5)))),
    Metric("minutes", ("Minutes played", "Minutos jugados", "Minutos jogados"), None,
           lambda F: ("none", F["minutesPlayed"], None)),
]


# ---------------------------------------------------------------- evaluate
def evaluate(metric: Metric, frame: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Observed value, adjusted (posterior) value, posterior SD and the stabilization point.

    groups: optional labels (e.g. player positions) to fit one prior per group.
    """
    kind, a, b = metric.compute(frame)
    a = pd.Series(np.asarray(a, float), index=frame.index)
    b = pd.Series(np.asarray(b, float), index=frame.index) if b is not None else None
    out = pd.DataFrame(index=frame.index, columns=["observed", "adjusted", "sd"], dtype=float)
    per = frame.attrs.get("per", 1.0) if kind == "rate" else 1.0
    stab = []
    keys = groups.reindex(frame.index).fillna("?") if groups is not None else pd.Series("all", index=frame.index)
    for _, idx in keys.groupby(keys).groups.items():
        sa = a.loc[idx]
        sb = b.loc[idx] if b is not None else None
        if kind == "prop":
            s, n = sa.values, sb.values
            with np.errstate(divide="ignore", invalid="ignore"):
                obs = np.where(n > 0, s / n, np.nan)
            post, sd, kappa = SHRINK["prop"](s, n)
            out.loc[idx, "observed"], out.loc[idx, "adjusted"], out.loc[idx, "sd"] = 100 * obs, 100 * post, 100 * sd
            stab.append(kappa)
        elif kind == "rate":
            c, e = sa.values, sb.values
            with np.errstate(divide="ignore", invalid="ignore"):
                obs = np.where(e > 0, c / e, np.nan)
            post, sd, beta = SHRINK["rate"](c, e)
            out.loc[idx, "observed"], out.loc[idx, "adjusted"], out.loc[idx, "sd"] = obs, post, sd
            stab.append(beta * per)                 # exposure units -> matches or minutes
        elif kind == "normal":
            y, se = sa.values, sb.values
            post, sd, tau2 = SHRINK["normal"](y, se)
            out.loc[idx, "observed"], out.loc[idx, "adjusted"], out.loc[idx, "sd"] = y, post, sd
            exposure = frame.loc[idx, frame.attrs.get("exposure", "n")].values.astype(float)
            v = se ** 2 * exposure
            per_unit = np.nanmedian(v) if np.isfinite(v).any() else np.nan
            unit_scale = frame.attrs.get("per", 1.0)            # players: m90 -> minutes
            stab.append(per_unit / tau2 * unit_scale if np.isfinite(tau2) and tau2 > 0 else np.nan)
        else:
            out.loc[idx, "observed"], out.loc[idx, "adjusted"], out.loc[idx, "sd"] = sa.values, sa.values, 0.0
            stab.append(np.nan)
    out.attrs["kind"] = kind
    out.attrs["stabilization"] = float(np.nanmedian(stab)) if np.isfinite(stab).any() else np.nan
    return out


def block_type(def_line_adjusted: pd.Series) -> pd.Series:
    """HIGH / MID / LOW block by league terciles of the (adjusted) defensive line height."""
    q1, q2 = def_line_adjusted.quantile([1 / 3, 2 / 3])
    return pd.Series(np.select([def_line_adjusted >= q2, def_line_adjusted >= q1], ["high", "mid"], "low"),
                     index=def_line_adjusted.index)
