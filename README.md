# Football 2D Explorer

Compare **any two football metrics** for teams or players in the big European leagues and the Champions
League, **men's and women's** (Liga F and the UEFA Women's Champions League), and see
**how much of each difference is real** thanks to Bayesian uncertainty ellipses.
Interface in **English, Spanish and Portuguese**.

## Why

Football has few events. A striker with 3 goals from 4 shots is not a 75% finisher, and a team with
two more points than another after six rounds may simply have been luckier. This app shows the observed
value, an **adjusted estimate with its uncertainty**, and answers the questions coaches actually ask:

- **Who is better, really?** Pick two teams or players and get the probability that one is better.
- **When can I trust this number?** Each metric shows how many matches, minutes or attempts it takes
  before the data weighs as much as the league average.
- **Who is lucky?** Points vs expected points (xPts), goals vs xG.
- **How does a team defend?** Average XI on the pitch, defensive line height, block length, width and
  area, and an automatic HIGH / MID / LOW block classification.

## Features

- Teams or players; ~40 team and ~30 player metrics: xG, npxG, xA, xG per shot, conversion, field tilt,
  PPDA (approx.), possession, duels, tackles and interceptions (possession-adjusted), recoveries,
  errors, block metrics, distance covered and sprints.
- **Position-specific priors** for players: a centre-back is compared with centre-backs.
- Observed or Bayes-adjusted values, optional trails between them, 50/68/95% credible ellipses for
  everyone or only the teams/players you choose.
- Names, codes, crests or face-cropped player photos as labels; quadrant badges; top-3 / bottom-3 boards.
- Automatic round label ("MATCHWEEK 9" / "JORNADA 9").

## Method

Empirical-Bayes models with a league-estimated (or position-estimated) prior:

| Metric type | Model |
|---|---|
| Counts per match / per 90 (goals, shots, tackles…) | Gamma-Poisson |
| Percentages (conversion, duels, pass accuracy…) | Beta-Binomial |
| Continuous values (xG, xPts, positions…) | Normal-Normal, with the standard error from match-to-match variation |

The stabilization point is read directly from the prior: the prior's strength in attempts
(Beta-Binomial), in matches or minutes (Gamma-Poisson), or the sample size at which sampling noise
equals the true spread (Normal-Normal). Ellipses assume independent errors on both axes.

Average positions are **event-based** (where each player acts on the ball), not tracking data: useful
to compare teams, not an exact picture of the block.

## Data

The app reads datasets from `data/` (four CSV files per league and season, see `football/data.py`) and
includes a fictional demo league. No scraped or private data is included in this repository.

## Run it locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py               Streamlit interface
core/bayes.py        Beta-Binomial, Gamma-Poisson and Normal-Normal shrinkage, ellipses, P(A > B)
core/images.py       crests and face-cropped photos (cached)
core/annotate.py     quadrant badges and leaderboards
football/metrics.py  team/player frames, block metrics, xPts and the metric registry
football/data.py     dataset format and demo data
football/i18n.py     English / Spanish / Portuguese texts
```

## Author

**Álvaro Bustamante-Sánchez, PhD** · Head of the Applied Technology & Research Lab ·
[LinkedIn](https://www.linkedin.com/in/alvaro-bustamante-163b5655) ·
[Google Scholar](https://scholar.google.com/citations?user=e_gK-2YAAAAJ)

MIT License.
