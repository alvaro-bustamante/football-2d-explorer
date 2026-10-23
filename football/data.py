"""Datasets: a football-warehouse folder (or files in data/) and fictional demo data.

A dataset is four CSV files sharing a prefix, e.g. data/laliga_26-27_*.csv:

  *_matches.csv    one row per team and match: match_id, round, team_id, team, opp_id, opp,
                   home, gf, ga, then match statistics with Sofascore-style keys
                   (ballPossession, expectedGoals, totalShotsOnGoal...) and the same keys
                   prefixed opp_ for the opponent. Ratio statistics add <key>_total
                   (e.g. aerialDuelsPercentage = won, aerialDuelsPercentage_total = contested).
  *_teams.csv      team_id, name, short_name, code, primary, secondary, crest_url, season stats.
  *_players.csv    player_id, name, short_name, team_id, team, position (G/D/M/F), photo_url,
                   season totals (minutesPlayed, goals, expectedGoals, tackles...).
  *_positions.csv  average positions: match_id, team_id, player_id, name, position, x, y
                   (0-100, own goal at x=0), points (events), starter, minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")


def data_dir() -> Path:
    """Where the datasets are: the football warehouse if there is one, else data/ in this project.

    In this order: the environment variable FOOTBALL_WAREHOUSE, `football_warehouse_dir` in
    .streamlit/secrets.toml, ../football-warehouse/data/warehouse (both projects side by side), data/."""
    import os
    candidates = [os.environ.get("FOOTBALL_WAREHOUSE")]
    try:
        import streamlit as st
        candidates.append(st.secrets.get("football_warehouse_dir"))
    except Exception:
        pass
    here = Path(__file__).resolve().parents[1]
    candidates += [here.parent / "football-warehouse" / "data" / "warehouse", here / "data", DATA_DIR]
    for c in candidates:
        if c and Path(c).is_dir() and any(Path(c).glob("*_matches.csv")):
            return Path(c)
    return DATA_DIR
PARTS = ("matches", "teams", "players", "positions")
LEAGUE_NAMES = {"laliga": "LaLiga", "premier-league": "Premier League", "serie-a": "Serie A",
                "bundesliga": "Bundesliga", "ligue-1": "Ligue 1", "liga-portugal": "Liga Portugal",
                "champions-league": "UEFA Champions League",
                "liga-f": "Liga F", "uwcl": "UEFA Women's Champions League",
                "eredivisie": "Eredivisie", "pro-league": "Pro League", "super-lig": "Süper Lig",
                "premiership": "Scottish Premiership", "brasileirao": "Brasileirão", "liga-profesional": "Liga Profesional",
                "mls": "MLS", "laliga2": "LaLiga Hypermotion", "championship": "Championship", "serie-b": "Serie B",
                "2-bundesliga": "2. Bundesliga", "ligue-2": "Ligue 2", "europa-league": "UEFA Europa League",
                "conference-league": "UEFA Conference League"}
WOMEN = {"liga-f", "uwcl"}


def gender_of(prefix: str) -> str:
    return "women" if prefix.rpartition("_")[0] in WOMEN else "men"


def list_datasets(data_dir: Path | None = None) -> list[str]:
    data_dir = data_dir or globals()["data_dir"]()
    if not data_dir.exists():
        return []
    return sorted(f.name[:-len("_matches.csv")] for f in data_dir.glob("*_matches.csv"))


def dataset_label(prefix: str) -> str:
    slug, _, season = prefix.rpartition("_")
    return f"{LEAGUE_NAMES.get(slug, slug.replace('-', ' ').title())} {season.replace('-', '/')}"


def load_dataset(prefix: str, data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = data_dir or globals()["data_dir"]()
    out = {}
    for part in PARTS:
        f = data_dir / f"{prefix}_{part}.csv"
        out[part] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


# ----------------------------------------------------------------------- demo
def demo(n_teams: int = 18, n_rounds: int = 10, seed: int = 11) -> dict[str, pd.DataFrame]:
    """A fictional league with the same structure as real datasets."""
    rng = np.random.default_rng(seed)
    letters = [chr(65 + i) for i in range(n_teams)]
    att = rng.normal(1.35, 0.35, n_teams).clip(0.5)          # true xG created per match
    dfn = rng.normal(1.35, 0.30, n_teams).clip(0.5)          # true xG conceded per match
    poss = rng.normal(50, 6, n_teams)
    press = rng.normal(0, 1, n_teams)                        # pressing height (z-score)
    teams = pd.DataFrame({
        "team_id": range(1, n_teams + 1), "name": [f"Team {L} FC" for L in letters],
        "short_name": [f"Team {L}" for L in letters], "code": [f"T{L}{L}" for L in letters],
        "primary": [f"#{rng.integers(0x333333, 0xDDDDDD):06x}" for _ in letters], "secondary": "#ffffff",
        "crest_url": None})
    rows, pos_rows = [], []
    order = list(range(n_teams))
    mid = 0
    squads = {}
    roles = ["G"] + ["D"] * 4 + ["M"] * 4 + ["F"] * 2
    base_xy = {"G": [(8, 50)], "D": [(30, 15), (25, 38), (25, 62), (30, 85)],
               "M": [(45, 25), (42, 45), (48, 55), (45, 75)], "F": [(62, 40), (62, 60)]}
    for ti in range(n_teams):
        squads[ti] = [(ti * 100 + j + 1, roles[j]) for j in range(11)] + \
                     [(ti * 100 + 12 + j, r) for j, r in enumerate(["D", "M", "F", "M", "D"])]
    for rnd in range(1, n_rounds + 1):
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            mid += 1
            xa = max(0.1, rng.normal(att[a] * dfn[b] / 1.35, 0.5))
            xb = max(0.1, rng.normal(att[b] * dfn[a] / 1.35, 0.5))
            ga, gb = rng.poisson(xa), rng.poisson(xb)
            pa = float(np.clip(rng.normal(poss[a] - poss[b] + 50, 6), 25, 75))
            stats = {}
            for side, x, g, p in ((a, xa, ga, pa), (b, xb, gb, 100 - pa)):
                shots = max(int(rng.poisson(x * 9)), g)
                passes = int(rng.normal(p * 9, 40))
                stats[side] = {
                    "ballPossession": round(p), "expectedGoals": round(x, 2), "totalShotsOnGoal": shots,
                    "shotsOnGoal": min(shots, int(rng.binomial(shots, 0.35)) + g), "bigChanceCreated": rng.poisson(x * 1.6),
                    "touchesInOppBox": rng.poisson(x * 14 + 5), "finalThirdEntries": rng.poisson(p * 0.9),
                    "passes": passes, "accuratePasses": int(passes * rng.uniform(0.75, 0.9)),
                    "totalTackle": rng.poisson(16 - p / 10), "interceptionWon": rng.poisson(8),
                    "ballRecovery": rng.poisson(45), "totalClearance": rng.poisson(18 - p / 8), "fouls": rng.poisson(12),
                    "errorsLeadToShot": rng.poisson(0.4), "goalkeeperSaves": rng.poisson(2.5),
                    "kilometersCovered": round(rng.normal(108, 3), 1), "numberOfSprints": int(rng.normal(120, 15)),
                }
                for key, n in (("groundDuelsPercentage", 60), ("aerialDuelsPercentage", 25),
                               ("wonTacklePercent", 16), ("dribblesPercentage", 15)):
                    tot = int(rng.poisson(n))
                    stats[side][key], stats[side][key + "_total"] = int(rng.binomial(tot, 0.5)), tot
            for side, opp, home, gf, gag in ((a, b, True, ga, gb), (b, a, False, gb, ga)):
                r = {"match_id": mid, "round": rnd, "team_id": side + 1, "team": teams.name[side],
                     "opp_id": opp + 1, "opp": teams.name[opp], "home": home, "gf": gf, "ga": gag}
                r.update(stats[side])
                r.update({f"opp_{k}": v for k, v in stats[opp].items()})
                rows.append(r)
                # average positions: shifted by pressing height and possession
                shift = 4 * press[side] + (stats[side]["ballPossession"] - 50) * 0.15
                for pid, role in squads[side][:11]:
                    k = sum(1 for p_, r_ in squads[side][:11] if r_ == role and p_ < pid)
                    bx, by = base_xy[role][k]
                    pos_rows.append({"match_id": mid, "team_id": side + 1, "player_id": pid,
                                     "name": f"Player {letters[side]}{pid % 100}", "position": role,
                                     "x": float(np.clip(bx + (0 if role == "G" else shift) + rng.normal(0, 3), 2, 98)),
                                     "y": float(np.clip(by + rng.normal(0, 4), 2, 98)),
                                     "points": int(rng.integers(15, 80)), "starter": True, "minutes": 90})
    matches = pd.DataFrame(rows)
    # season totals per player
    prow = []
    for ti in range(n_teams):
        for pid, role in squads[ti]:
            starter = pid % 100 <= 11
            mins = int(rng.integers(500, 900) if starter else rng.integers(60, 450))
            m90 = mins / 90
            xg_rate = {"G": 0.0, "D": 0.06, "M": 0.15, "F": 0.42}[role] * rng.uniform(0.5, 1.6)
            xa_rate = {"G": 0.0, "D": 0.06, "M": 0.14, "F": 0.15}[role] * rng.uniform(0.5, 1.6)
            xg = xg_rate * m90
            shots = rng.poisson(xg * 8 + 0.2 * m90)
            passes = rng.poisson({"G": 28, "D": 55, "M": 50, "F": 22}[role] * m90)
            aer = rng.poisson({"G": 0.3, "D": 3.5, "M": 1.5, "F": 3}[role] * m90)
            grd = rng.poisson({"G": 0.3, "D": 4, "M": 6, "F": 5}[role] * m90)
            cont = rng.poisson({"G": 0, "D": 0.6, "M": 1.6, "F": 2.4}[role] * m90)
            tck = rng.poisson({"G": 0.1, "D": 2.2, "M": 2.0, "F": 0.8}[role] * m90)
            prow.append({
                "player_id": pid, "name": f"Player {letters[ti]}{pid % 100}", "short_name": f"{letters[ti]}{pid % 100}",
                "team_id": ti + 1, "team": teams.name[ti], "position": role, "photo_url": None,
                "minutesPlayed": mins, "appearances": max(1, mins // 75),
                "goals": rng.poisson(xg), "expectedGoals": round(xg, 2), "penaltiesTaken": rng.poisson(0.05 * m90) if role == "F" else 0,
                "assists": rng.poisson(xa_rate * m90), "expectedAssists": round(xa_rate * m90, 2),
                "totalShots": shots, "shotsOnTarget": rng.binomial(shots, 0.35),
                "keyPasses": rng.poisson(xa_rate * 9 * m90), "bigChancesCreated": rng.poisson(xa_rate * 2 * m90),
                "totalPasses": passes, "accuratePasses": rng.binomial(passes, rng.uniform(0.72, 0.9)),
                "totalLongBalls": rng.poisson(4 * m90), "accurateLongBalls": rng.poisson(2 * m90),
                "totalCross": rng.poisson(1 * m90), "accurateCrosses": rng.poisson(0.25 * m90),
                "totalContest": cont, "successfulDribbles": rng.binomial(cont, 0.5),
                "aerialDuelsWon": rng.binomial(aer, 0.5), "aerialLost": 0, "groundDuelsWon": rng.binomial(grd, 0.5),
                "groundDuelsWonPercentage": 50.0, "tackles": tck, "tacklesWon": rng.binomial(tck, 0.6),
                "interceptions": rng.poisson({"G": 0.1, "D": 1.3, "M": 1.0, "F": 0.3}[role] * m90),
                "clearances": rng.poisson({"G": 0.5, "D": 3.5, "M": 0.8, "F": 0.4}[role] * m90),
                "ballRecovery": rng.poisson({"G": 3, "D": 5, "M": 5.5, "F": 2.5}[role] * m90),
                "outfielderBlocks": rng.poisson({"G": 0, "D": 0.6, "M": 0.3, "F": 0.1}[role] * m90),
                "possessionWonAttThird": rng.poisson({"G": 0, "D": 0.2, "M": 0.5, "F": 0.7}[role] * m90),
                "dribbledPast": rng.poisson({"G": 0.1, "D": 0.8, "M": 0.9, "F": 0.4}[role] * m90),
                "touchesInOppBox": rng.poisson({"G": 0, "D": 0.6, "M": 1.5, "F": 5}[role] * m90),
                "defensiveContributions": 0, "possessionLost": rng.poisson(9 * m90), "rating": round(rng.normal(6.9, 0.3), 2),
                "saves": rng.poisson(3 * m90) if role == "G" else 0, "goalsConceded": rng.poisson(1.3 * m90) if role == "G" else 0,
                "kilometersCoveredPer90": round(rng.normal(10.5, 0.8), 2) if role != "G" else round(rng.normal(5.5, 0.4), 2),
            })
            prow[-1]["aerialLost"] = rng.binomial(aer, 0.5)
            prow[-1]["defensiveContributions"] = prow[-1]["tackles"] + prow[-1]["interceptions"] + \
                prow[-1]["clearances"] + prow[-1]["outfielderBlocks"]
    players = pd.DataFrame(prow)
    teams["penaltiesTaken"] = rng.poisson(1.5, n_teams)
    teams["ownHalfPassesTotalAgainst"] = rng.poisson(1500, n_teams)
    return {"matches": matches, "teams": teams, "players": players, "positions": pd.DataFrame(pos_rows)}
