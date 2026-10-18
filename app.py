"""Football 2D Explorer: compare any two metrics for teams or players, with Bayesian uncertainty.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.annotate import BADGE_FONT, badge_fill, leaders, leaders_html
from core.bayes import ellipse_path, ellipse_scale, prob_better
from football import data as fdata
from football.i18n import LANGS, set_lang, smart_upper, t
from football.metrics import (PITCH_LENGTH, PITCH_WIDTH, PLAYER_METRICS, TEAM_METRICS, block_type,
                              evaluate, player_frame, team_frame)
from football.style import readable_color

# ─── PERSONAL HOOKS ───────────────────────────────────────────────────────────
# personal/ is excluded from GitHub (.gitignore). Set to False to hide its features
# (Instagram / LinkedIn export and the Madrid clubs shortcut).
PERSONAL_FEATURES = True
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Football 2D Explorer", page_icon="⚽", layout="wide")
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700'
            '&display=swap" rel="stylesheet">', unsafe_allow_html=True)
SMALL = 10


def personal(module: str):
    if not PERSONAL_FEATURES:
        return None
    try:
        return __import__(f"personal.{module}", fromlist=["_"])
    except ImportError:
        return None


@st.cache_data(show_spinner=False)
def load(prefix: str | None):
    return fdata.demo() if prefix is None else fdata.load_dataset(prefix)


@st.cache_data(show_spinner="…")
def prepared_images(urls: tuple, kind: str, version: int = 0) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from core.images import prepare
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(urls, pool.map(lambda u: prepare(u, kind), urls)))


def fmt_value(v: float) -> str:
    return "—" if not np.isfinite(v) else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")


# ─────────────────────────────────────────────────────────────── sidebar: data
with st.sidebar:
    lang_name = st.selectbox("🌐 Language / Idioma", list(LANGS), index=1)
    set_lang(LANGS[lang_name])
    st.title("⚽ " + t("app_title"))
    datasets = fdata.list_datasets()
    sources = ([t("saved")] if datasets else []) + [t("demo")]
    source = st.radio(t("data_source"), sources)
    if source == t("saved"):
        prefix = st.selectbox(t("dataset"), datasets, format_func=fdata.dataset_label)
        context = fdata.dataset_label(prefix) + " · Data: Sofascore"
    else:
        prefix = None
        context = t("demo")
        if not datasets:
            st.caption(t("no_data"))
    level = st.radio(t("compare"), [t("teams"), t("players")], horizontal=True)
    is_teams = level == t("teams")
    views = [t("view_2d")] + ([t("view_luck"), t("view_pitch")] if is_teams else [])
    view = st.radio(t("view"), views)

data = load(prefix)
T = team_frame(data)
team_name = T["info_name"].astype(str)
team_short = T["info_short_name"].astype(str)
team_color = pd.Series([readable_color(p, s, tid) for tid, p, s in
                        zip(T.index, T.get("info_primary", pd.Series(None, index=T.index)),
                            T.get("info_secondary", pd.Series(None, index=T.index)))], index=T.index)
games_played = int(T["n"].max()) if len(T) else 0
madrid = personal("presets")


# ───────────────────────────────────────────────────────────── pitch view
def pitch_figure() -> go.Figure:
    fig = go.Figure()
    L, W = PITCH_LENGTH, PITCH_WIDTH
    line = dict(color="rgba(255,255,255,0.35)", width=1.5)
    fig.add_shape(type="rect", x0=0, y0=0, x1=L, y1=W, line=line, fillcolor="rgba(46,125,50,0.18)", layer="below")
    fig.add_shape(type="line", x0=L / 2, y0=0, x1=L / 2, y1=W, line=line, layer="below")
    fig.add_shape(type="circle", x0=L / 2 - 9.15, y0=W / 2 - 9.15, x1=L / 2 + 9.15, y1=W / 2 + 9.15, line=line,
                  layer="below")
    for x0, x1 in ((0, 16.5), (L - 16.5, L)):
        fig.add_shape(type="rect", x0=x0, y0=W / 2 - 20.16, x1=x1, y1=W / 2 + 20.16, line=line, layer="below")
    for x0, x1 in ((0, 5.5), (L - 5.5, L)):
        fig.add_shape(type="rect", x0=x0, y0=W / 2 - 9.16, x1=x1, y1=W / 2 + 9.16, line=line, layer="below")
    fig.update_xaxes(range=[-3, L + 3], visible=False)
    fig.update_yaxes(range=[-3, W + 3], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), showlegend=True,
                      legend=dict(orientation="h", y=-0.02))
    return fig


def average_xi(team_id: int, min_starts: int) -> pd.DataFrame:
    pos = data["positions"]
    p = pos[(pos["team_id"] == team_id) & pos["starter"].astype(bool)].copy()
    p["xm"], p["ym"] = p["x"] / 100 * PITCH_LENGTH, p["y"] / 100 * PITCH_WIDTH
    g = p.groupby("player_id")
    xi = pd.DataFrame({"name": g["name"].last(), "position": g["position"].agg(lambda s: s.mode().iat[0]),
                       "starts": g.size(), "x": g["xm"].mean(), "y": g["ym"].mean(),
                       "sx": g["xm"].std(ddof=1), "sy": g["ym"].std(ddof=1)})
    return xi[xi["starts"] >= min_starts].sort_values("starts", ascending=False)


if view == t("view_pitch"):
    st.title(t("view_pitch"))
    st.caption(t("pitch_note"))
    if data["positions"].empty:
        st.info("No average positions in this dataset.")
        st.stop()
    order = team_short.sort_values().index
    c1, c2, c3 = st.columns([2, 2, 1])
    tid = c1.selectbox(t("pitch_team"), order, format_func=lambda i: team_short[i])
    tid2 = c2.selectbox(t("pitch_compare"), [None] + [i for i in order if i != tid],
                        format_func=lambda i: "—" if i is None else team_short[i])
    min_starts = c3.number_input(t("min_matches"), 1, max(games_played, 1), min(3, max(games_played, 1)))
    blocks = {m.key: evaluate(m, T) for m in TEAM_METRICS if m.key in ("def_line", "centroid", "length", "width", "area")}
    btype = block_type(blocks["def_line"]["adjusted"])
    fig = pitch_figure()
    for team_id, flip in ((tid, False), (tid2, True)):
        if team_id is None:
            continue
        xi = average_xi(team_id, min_starts)
        col = team_color[team_id]
        X = PITCH_LENGTH - xi["x"] if flip else xi["x"]
        Y = PITCH_WIDTH - xi["y"] if flip else xi["y"]
        for xx, yy, sx, sy in zip(X, Y, xi["sx"].fillna(0), xi["sy"].fillna(0)):
            ex, ey = ellipse_path(xx, yy, max(sx, 0.5), max(sy, 0.5))
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself", line=dict(color=col, width=1),
                                     fillcolor=col, opacity=0.18, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=X, y=Y, mode="markers+text", text=xi["name"], textposition="top center",
                                 marker=dict(size=13, color=col, line=dict(color="white", width=1)),
                                 textfont=dict(size=10), name=team_short[team_id],
                                 hovertext=[f"{n} · {p} · {s}" for n, p, s in zip(xi["name"], xi["position"], xi["starts"])],
                                 hoverinfo="text"))
        dl = blocks["def_line"].loc[team_id, "adjusted"]
        ce = blocks["centroid"].loc[team_id, "adjusted"]
        for val, dash, lab in ((dl, "dash", t("def_line")), (ce, "dot", t("centroid"))):
            xv = PITCH_LENGTH - val if flip else val
            fig.add_shape(type="line", x0=xv, x1=xv, y0=0, y1=PITCH_WIDTH, line=dict(color=col, width=2, dash=dash))
            fig.add_annotation(x=xv, y=PITCH_WIDTH + 2, text=f"{lab} {val:.0f} m", showarrow=False,
                               font=dict(size=SMALL, color=col))
    title = f"{team_short[tid]} · {t('block_type')} {t('block_' + btype[tid])}"
    if tid2 is not None:
        title += f"  vs  {team_short[tid2]} · {t('block_type')} {t('block_' + btype[tid2])}"
    fig.update_layout(title=title)
    st.plotly_chart(fig)
    st.subheader(t("block_table"))
    tab = pd.DataFrame({t("team"): team_short, t("block_type"): btype.map(lambda b: t("block_" + b))})
    for key, ev in blocks.items():
        m = next(m for m in TEAM_METRICS if m.key == key)
        tab[m.label] = ev["adjusted"].round(1)
    st.dataframe(tab.sort_values(tab.columns[2], ascending=False), hide_index=True)
    st.stop()


# ───────────────────────────────────────────────────────── 2D chart: frames
if is_teams:
    frame = T
    registry = TEAM_METRICS
    full_names, short_names = team_name, team_short
    codes = T.get("info_code", team_short).fillna(team_short).astype(str)
    crest_urls = T.get("info_crest_url", pd.Series(None, index=T.index))
    photo_urls = pd.Series(None, index=T.index)
    colors = list(team_color)
    team_of = pd.Series(T.index, index=T.index)
    groups = None
else:
    P = player_frame(data)
    max_min = int(P["minutesPlayed"].max()) if len(P) else 90
    with st.sidebar:
        st.divider()
        st.subheader(t("filters"))
        min_minutes = st.slider(t("min_minutes"), 0, max_min, min(450, max_min), step=45)
        pos_opts = [p for p in ("G", "D", "M", "F") if p in set(P["position"].dropna())]
        pos_sel = st.multiselect(t("positions"), pos_opts, default=[p for p in pos_opts if p != "G"],
                                 format_func=lambda p: t("pos_" + p))
        team_sel = st.multiselect(t("teams_filter"), list(team_short.sort_values().index), placeholder=t("all_teams"),
                                  format_func=lambda i: team_short.get(i, i))
        use_pos_priors = st.toggle(t("position_priors"), value=True, help=t("position_priors_help"))
    frame = P[(P["minutesPlayed"] >= min_minutes) & P["position"].isin(pos_sel)]
    if team_sel:
        frame = frame[frame["team_id"].isin(team_sel)]
    if len(frame) < 5:
        st.warning(t("too_few"))
        st.stop()
    frame = frame.copy()
    frame.attrs.update(P.attrs)
    registry = PLAYER_METRICS
    full_names = frame["name"].astype(str)
    short_names = frame.get("short_name", frame["name"]).fillna(frame["name"]).astype(str)
    codes = frame["team_id"].map(T.get("info_code", team_short)).fillna("").astype(str)
    crest_urls = frame["team_id"].map(T.get("info_crest_url", pd.Series(dtype=object)))
    photo_urls = frame.get("photo_url", pd.Series(None, index=frame.index))
    colors = [team_color.get(i, "#9AA0A6") for i in frame["team_id"]]
    team_of = frame["team_id"]
    groups = frame["position"] if use_pos_priors else None

# Only metrics with data in this dataset.
evals = {}
for m in registry:
    try:
        ev = evaluate(m, frame, groups)
    except (KeyError, ValueError, ZeroDivisionError):   # a column this dataset does not have
        continue
    if np.isfinite(ev["observed"].astype(float)).any():
        evals[m.key] = ev
registry = [m for m in registry if m.key in evals]
by_label = {m.label: m for m in registry}

# ─────────────────────────────────────────────────────────── sidebar: display
with st.sidebar:
    st.divider()
    st.subheader(t("display"))
    if is_teams:
        lab_opts = ["lab_short", "lab_full", "lab_code", "lab_crest", "lab_none"]
        label_mode = st.radio(t("labels_teams"), lab_opts, format_func=t)
    else:
        lab_opts = ["lab_short", "lab_full", "lab_photo", "lab_team_crest", "lab_none"]
        label_mode = st.radio(t("labels_players"), lab_opts, format_func=t)
    ellipse_mode = st.radio(t("ellipses"), ["ell_all", "ell_choose", "ell_none"], horizontal=True,
                            index=0 if is_teams else 1, format_func=t)
    show_ellipses = ellipse_mode != "ell_none"
    ci_level = st.select_slider(t("cred_level"), options=["50%", "68%", "95%"], value="68%",
                                disabled=not show_ellipses)
    estimate = st.radio(t("points_show"), ["observed", "adjusted"], horizontal=True, format_func=t)
    show_trails = st.checkbox(t("trails"), value=False)
    show_medians = st.checkbox(t("medians"), value=True)
    show_quadrants = st.checkbox(t("quadrants"), value=True)
    show_leaders = st.checkbox(t("leaders"), value=True)
    round_text = st.text_input(t("round_label"), f"{t('round_word')} {games_played}" if games_played else "",
                               help=t("round_help"))

labels = {"lab_short": short_names, "lab_full": full_names, "lab_code": codes}.get(label_mode, short_names)
image_urls = {"lab_crest": crest_urls, "lab_team_crest": crest_urls, "lab_photo": photo_urls}.get(label_mode)
image_kind = "photo" if label_mode == "lab_photo" else "crest"
use_images = image_urls is not None and image_urls.notna().any()
if use_images:
    from core.images import CROP_VERSION
    ready = prepared_images(tuple(sorted({u for u in image_urls if isinstance(u, str) and u})), image_kind,
                            CROP_VERSION)
    image_urls = image_urls.map(lambda u: ready.get(u) if isinstance(u, str) else None)
    use_images = image_urls.notna().any()
if image_urls is not None and not use_images:
    st.sidebar.caption(t("no_images"))

# ─────────────────────────────────────────────────────────────────── controls
st.title(t("app_title"))
round_html = (f"<span style='font-family:{BADGE_FONT};font-weight:700;letter-spacing:.12em;background:#FFB000;"
              f"color:#111;padding:2px 10px;border-radius:12px;font-size:0.85rem'>{round_text.upper()}</span> &nbsp;"
              if round_text else "")
st.markdown(f"{round_html}<span style='opacity:.7'>{t('caption', context=context)}</span>", unsafe_allow_html=True)

luck_view = view == t("view_luck")
if luck_view:
    mx, my = next(m for m in registry if m.key == "xpts"), next(m for m in registry if m.key == "points")
    inv_x = inv_y = False
    st.subheader(t("luck_title"))
    st.caption(t("luck_text"))
else:
    c1, c2 = st.columns(2)
    dx, dy = ("xg_for", "xg_against") if is_teams else ("npxg_xa", "defensiveContributions")
    keys = [m.key for m in registry]
    dx = dx if dx in keys else keys[0]
    dy = dy if dy in keys else keys[min(1, len(keys) - 1)]
    mx = by_label[c1.selectbox(t("x_axis"), list(by_label), index=keys.index(dx))]
    my = by_label[c2.selectbox(t("y_axis"), list(by_label), index=keys.index(dy))]
    inv_x = c1.checkbox(t("reverse_x"), value=mx.better == "low")
    inv_y = c2.checkbox(t("reverse_y"), value=my.better == "low")

hl_options = sorted(labels.unique())
hl_default = []
if madrid is not None and st.button("⭐ " + t("madrid")):
    mad_ids = [i for i in T.index if madrid.is_madrid(team_name[i])]
    hl_default = sorted(set(labels[team_of.isin(mad_ids).values]))
highlight = st.multiselect(t("highlight"), hl_options, default=hl_default)
ellipse_for = (st.multiselect(t("ellipses_for"), hl_options, default=highlight)
               if ellipse_mode == "ell_choose" else [])

# ──────────────────────────────────────────────────────────────────── compute
ex, ey = evals[mx.key], evals[my.key]
col = "adjusted" if estimate == "adjusted" else "observed"
x, y = ex[col], ey[col]
is_hl = labels.isin(highlight).to_numpy() if highlight else np.zeros(len(frame), bool)
want_ell = (labels.isin(ellipse_for).to_numpy() if ellipse_mode == "ell_choose"
            else np.full(len(frame), ellipse_mode == "ell_all"))
k = ellipse_scale(int(ci_level.rstrip("%")) / 100)


def side_word(metric, high_side: bool) -> tuple[str, str, str]:
    if metric.better is None:
        return ("▲", "#8A8F98", t("w_high")) if high_side else ("▼", "#8A8F98", t("w_low"))
    good = high_side if metric.better == "high" else not high_side
    return ("▲", "#2E9E44", t("w_good")) if good else ("▼", "#D64545", t("w_bad"))


quads = {}
for vert in ("top", "bottom"):
    for horiz in ("left", "right"):
        ax_, cx_, wx = side_word(mx, (horiz == "right") != inv_x)
        ay_, cy_, wy = side_word(my, (vert == "top") != inv_y)
        quads[f"{vert}_{horiz}"] = [(ax_, cx_, f"{wx} {mx.label}"), (ay_, cy_, f"{wy} {my.label}")]

TAGS = {"Best": t("tag_best"), "Worst": t("tag_worst"), "Highest": t("tag_highest"), "Lowest": t("tag_lowest")}
boards = []
if show_leaders and not luck_view:
    for m, vals in ((mx, x), (my, y)):
        fmt = "{:.2f}" if np.nanmax(np.abs(vals)) < 10 else "{:.1f}"
        b = leaders(vals, labels, m, fmt=fmt)
        if b:
            b["top_tag"], b["bottom_tag"] = TAGS[b["top_tag"]], TAGS[b["bottom_tag"]]
            boards.append(b)

# ───────────────────────────────────────────────────────────────────── figure
fig = go.Figure()
ellipses = []
if show_ellipses:
    for i in range(len(frame)):
        if not want_ell[i]:
            continue
        rx, ry = k * ex["sd"].iloc[i], k * ey["sd"].iloc[i]
        if not (np.isfinite(rx) and np.isfinite(ry)) or (rx == 0 and ry == 0):
            continue
        cx, cy = ex["adjusted"].iloc[i], ey["adjusted"].iloc[i]
        ellipses.append((cx, cy, rx, ry, colors[i]))
        px_, py_ = ellipse_path(cx, cy, rx, ry)
        fig.add_trace(go.Scatter(x=px_, y=py_, mode="lines", fill="toself", hoverinfo="skip",
                                 line=dict(width=1, color=colors[i]), fillcolor=colors[i],
                                 opacity=0.35 if is_hl[i] else 0.16, showlegend=False))
if show_trails:   # faint line from the observed value to the adjusted one (the point)
    other = "observed" if col == "adjusted" else "adjusted"
    for i in range(len(frame)):
        if want_ell[i] or is_hl[i] or len(frame) <= 30:
            fig.add_trace(go.Scatter(x=[ex[other].iloc[i], x.iloc[i]], y=[ey[other].iloc[i], y.iloc[i]],
                                     mode="lines+markers", hoverinfo="skip", showlegend=False, opacity=0.55,
                                     line=dict(color=colors[i], width=1.2),
                                     marker=dict(size=[6, 0], color="rgba(0,0,0,0)",
                                                 line=dict(color=colors[i], width=1))))
if luck_view:
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)])) - 0.1
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)])) + 0.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#888", dash="dash"),
                             hoverinfo="skip", showlegend=False))

show_text = label_mode != "lab_none" and not use_images and (is_teams or len(frame) <= 25)
text = [lab if (show_text or h) else "" for lab, h in zip(labels, is_hl)]
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers+text", text=text, textposition="top center", textfont=dict(size=11),
    marker=dict(size=[16 if h else 11 for h in is_hl], color=colors,
                opacity=[0 if use_images and isinstance(u, str) else 1
                         for u in (image_urls if use_images else [None] * len(frame))],
                line=dict(width=[2.5 if h else 1 for h in is_hl], color="#FFFFFF")),
    customdata=np.stack([ex["observed"], ex["adjusted"], ex["sd"], ey["observed"], ey["adjusted"], ey["sd"]], axis=1),
    hovertext=full_names, showlegend=False,
    hovertemplate=("<b>%{hovertext}</b><br>"
                   f"{mx.label}: %{{customdata[0]:.2f}} ({t('adj')} %{{customdata[1]:.2f}} ± %{{customdata[2]:.2f}})<br>"
                   f"{my.label}: %{{customdata[3]:.2f}} ({t('adj')} %{{customdata[4]:.2f}} ± %{{customdata[5]:.2f}})"
                   "<extra></extra>")))
if use_images:
    xr = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    yr = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    size = 0.07 if is_teams or len(frame) <= 30 else 0.045
    for xi, yi, url, h in zip(x, y, image_urls, is_hl):
        if isinstance(url, str) and url:
            s = 1.35 if h else 1.0
            fig.add_layout_image(source=url, x=xi, y=yi, xref="x", yref="y", sizex=size * xr * s,
                                 sizey=size * yr * s, xanchor="center", yanchor="middle", layer="above")
if show_medians and not luck_view:
    fig.add_vline(x=float(np.nanmedian(x)), line_dash="dot", line_color="#888")
    fig.add_hline(y=float(np.nanmedian(y)), line_dash="dot", line_color="#888")
if show_quadrants and not luck_view:
    for key, lines in quads.items():
        vert, horiz = key.split("_")
        ordered = lines if vert == "top" else list(reversed(lines))
        for j, (arrow, color, txt) in enumerate(ordered):
            fig.add_annotation(xref="paper", yref="paper", showarrow=False,
                               x=0.006 if horiz == "left" else 0.994, y=0.994 if vert == "top" else 0.006,
                               xanchor=horiz, yanchor=vert, yshift=(-1 if vert == "top" else 1) * 24 * j,
                               text=f"<b>{arrow} {smart_upper(txt)}</b>",
                               font=dict(size=SMALL + 1, color=color, family=BADGE_FONT),
                               bgcolor=badge_fill(color), bordercolor=color, borderwidth=1, borderpad=3)
bottom = 40
for j, board in enumerate(boards):
    fig.add_annotation(xref="paper", yref="paper", showarrow=False, align="left", x=0, y=-0.11 - 0.075 * j,
                       xanchor="left", yanchor="top", text=leaders_html(board), font=dict(size=SMALL))
    bottom = 150
fig.update_xaxes(title=mx.label, autorange="reversed" if inv_x else True)
fig.update_yaxes(title=my.label, autorange="reversed" if inv_y else True)
fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=bottom),
                  title=(f"{round_text.upper()} · " if round_text else "") + f"{my.label} vs {mx.label}")
st.plotly_chart(fig)

# ─────────────────────────────────────────────────────────────── Bayes panels
c1, c2 = st.columns(2)
with c1.expander("🎲 " + t("h2h"), expanded=bool(len(highlight) >= 2)):
    pick = st.multiselect(t("h2h_pick"), hl_options, default=highlight[:2], max_selections=2, key="h2h")
    if len(pick) == 2:
        ia, ib = [int(np.flatnonzero(labels.values == p)[0]) for p in pick]
        for m, ev in ((mx, ex), (my, ey)):
            if ev.attrs.get("kind") == "none":
                continue
            p = prob_better(ev["adjusted"].iloc[ia], ev["sd"].iloc[ia], ev["adjusted"].iloc[ib], ev["sd"].iloc[ib],
                            higher_is_better=m.better != "low")
            key = "h2h_line" if m.better else "h2h_line_neutral"
            a, b = (pick[0], pick[1]) if p >= 0.5 else (pick[1], pick[0])
            st.markdown(t(key, a=a, b=b, metric=m.label, p=100 * max(p, 1 - p)))
        st.caption(t("h2h_note"))
with c2.expander("⏱️ " + t("stability")):
    for m, ev in ((mx, ex), (my, ey)):
        n_star = ev.attrs.get("stabilization", np.nan)
        if ev.attrs.get("kind") == "none" or not np.isfinite(n_star):
            st.markdown(t("stab_none", metric=m.label))
        else:
            unit = t("unit_attempts") if ev.attrs.get("kind") == "prop" else t(frame.attrs.get("unit", "unit_matches"))
            st.markdown(t("stab_trials", metric=m.label, n=n_star, unit=unit))

# ──────────────────────────────────────────────────────────────────────── table
z = 1.96
table = pd.DataFrame({t("name"): full_names.values})
if not is_teams:
    table[t("team")] = frame["team_id"].map(team_short).values
    table[t("minutes")] = frame["minutesPlayed"].values
else:
    table[t("matches")] = frame["n"].values
for m, ev in ((mx, ex), (my, ey)):
    table[m.label] = ev["observed"].round(2).values
    table[f"{m.label} {t('adj')}"] = ev["adjusted"].round(2).values
    table[f"{m.label} {t('ci95')}"] = [f"{a - z * s:.2f} – {a + z * s:.2f}" if s > 0 else ""
                                       for a, s in zip(ev["adjusted"], ev["sd"])]
if luck_view:
    table = table.sort_values(table.columns[-3] if False else f"{my.label} {t('adj')}", ascending=False)
st.dataframe(table, hide_index=True)
st.download_button(t("download"), table.to_csv(index=False).encode(), "football_view.csv", "text/csv")

with st.expander(t("method")):
    st.markdown(t("method_text", level=ci_level))
    st.markdown(f"**{mx.label}**: {mx.help or '—'}  \n**{my.label}**: {my.help or '—'}")
st.caption(t("built_by"))

# ─── PERSONAL EXPORT (see hook at the top) ────────────────────────────────────
export = personal("social_export")
if export is not None:
    export.social_export_panel({
        "title": f"{my.label} vs {mx.label}", "context": context, "level": "Teams" if is_teams else "Players",
        "x_label": mx.label, "y_label": my.label, "inv_x": inv_x, "inv_y": inv_y,
        "x": np.asarray(x, float), "y": np.asarray(y, float),
        "labels": list(labels), "show_labels": label_mode != "lab_none", "highlight": list(is_hl),
        "colors": colors, "image_urls": list(image_urls) if use_images else None, "image_kind": image_kind,
        "round": round_text, "ellipses": ellipses,
        "medians": (float(np.nanmedian(x)), float(np.nanmedian(y))) if show_medians and not luck_view else None,
        "quadrants": quads if show_quadrants and not luck_view else None, "leaders": boards,
        "estimate": "Bayes-adjusted" if estimate == "adjusted" else "Observed",
        "ci_level": ci_level if show_ellipses else None,
        "note_texts": {"Observed": t("export_obs"), "Bayes-adjusted": t("export_adj"),
                       "ellipses": t("export_ell", level=ci_level)},
    })
