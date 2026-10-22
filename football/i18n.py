"""User interface texts in English, Spanish and (European) Portuguese.

t(key) returns the text in the current language; unknown keys fall back to
English and then to the key itself, so a missing translation never breaks the app.
"""
from __future__ import annotations

LANGS = {"English": "en", "Español": "es", "Português": "pt"}
_lang = "en"


def set_lang(code: str):
    global _lang
    _lang = code if code in ("en", "es", "pt") else "en"


def lang() -> str:
    return _lang


def t(key: str, **kw) -> str:
    entry = TEXTS.get(key)
    if entry is None:
        return key.format(**kw) if kw else key
    text = entry.get(_lang) or entry.get("en") or key
    return text.format(**kw) if kw else text


TEXTS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------- general UI
    "app_title": {"en": "Football 2D Explorer", "es": "Football 2D Explorer", "pt": "Football 2D Explorer"},
    "language": {"en": "Language", "es": "Idioma", "pt": "Idioma"},
    "data_source": {"en": "Data", "es": "Datos", "pt": "Dados"},
    "dataset": {"en": "League and season", "es": "Liga y temporada", "pt": "Liga e época"},
    "demo": {"en": "Demo (fictional)", "es": "Demo (ficticio)", "pt": "Demo (fictício)"},
    "saved": {"en": "Saved data (data/ folder)", "es": "Datos guardados (carpeta data/)",
              "pt": "Dados guardados (pasta data/)"},
    "no_data": {"en": "No saved data yet: showing the demo. Datasets are read from data/<league>_<season>_*.csv.",
                "es": "Aún no hay datos guardados: se muestra la demo. Los datos se leen de data/<liga>_<temporada>_*.csv.",
                "pt": "Ainda não há dados guardados: mostra-se a demo. Os dados são lidos de data/<liga>_<época>_*.csv."},
    "compare": {"en": "Compare", "es": "Comparar", "pt": "Comparar"},
    "teams": {"en": "Teams", "es": "Equipos", "pt": "Equipas"},
    "players": {"en": "Players", "es": "Jugadores", "pt": "Jogadores"},
    "view": {"en": "View", "es": "Vista", "pt": "Vista"},
    "view_2d": {"en": "2D chart", "es": "Gráfico 2D", "pt": "Gráfico 2D"},
    "view_luck": {"en": "Luck table", "es": "Tabla de la suerte", "pt": "Tabela da sorte"},
    "view_pitch": {"en": "Pitch: average XI and block", "es": "Campo: once medio y bloque",
                   "pt": "Campo: onze médio e bloco"},
    "filters": {"en": "Filters", "es": "Filtros", "pt": "Filtros"},
    "min_minutes": {"en": "Min. minutes", "es": "Minutos mínimos", "pt": "Minutos mínimos"},
    "positions": {"en": "Positions", "es": "Posiciones", "pt": "Posições"},
    "pos_G": {"en": "Goalkeepers", "es": "Porteros", "pt": "Guarda-redes"},
    "pos_D": {"en": "Defenders", "es": "Defensas", "pt": "Defesas"},
    "pos_M": {"en": "Midfielders", "es": "Centrocampistas", "pt": "Médios"},
    "pos_F": {"en": "Forwards", "es": "Delanteros", "pt": "Avançados"},
    "teams_filter": {"en": "Teams", "es": "Equipos", "pt": "Equipas"},
    "all_teams": {"en": "All teams", "es": "Todos los equipos", "pt": "Todas as equipas"},
    "position_priors": {"en": "Compare each player with his position", "es": "Comparar a cada jugador con su posición",
                        "pt": "Comparar cada jogador com a sua posição"},
    "position_priors_help": {
        "en": "Bayes pulls each player towards the average of HIS position (a centre-back is not judged against strikers).",
        "es": "Bayes acerca a cada jugador a la media de SU posición (un central no se compara con los delanteros).",
        "pt": "O Bayes aproxima cada jogador da média da SUA posição (um central não é comparado com avançados)."},
    "display": {"en": "Display", "es": "Visualización", "pt": "Visualização"},
    "labels_teams": {"en": "Team labels", "es": "Etiquetas de equipo", "pt": "Etiquetas das equipas"},
    "labels_players": {"en": "Player labels", "es": "Etiquetas de jugador", "pt": "Etiquetas dos jogadores"},
    "lab_short": {"en": "Short name", "es": "Nombre corto", "pt": "Nome curto"},
    "lab_full": {"en": "Full name", "es": "Nombre completo", "pt": "Nome completo"},
    "lab_code": {"en": "Code", "es": "Código", "pt": "Código"},
    "lab_crest": {"en": "Crest", "es": "Escudo", "pt": "Emblema"},
    "lab_photo": {"en": "Photo", "es": "Foto", "pt": "Foto"},
    "lab_team_crest": {"en": "Team crest", "es": "Escudo del equipo", "pt": "Emblema da equipa"},
    "lab_none": {"en": "None", "es": "Ninguna", "pt": "Nenhuma"},
    "ellipses": {"en": "Uncertainty ellipses", "es": "Elipses de incertidumbre", "pt": "Elipses de incerteza"},
    "ell_all": {"en": "All", "es": "Todas", "pt": "Todas"},
    "ell_choose": {"en": "Choose", "es": "Elegir", "pt": "Escolher"},
    "ell_none": {"en": "None", "es": "Ninguna", "pt": "Nenhuma"},
    "ellipses_for": {"en": "Show ellipses for", "es": "Mostrar elipses de", "pt": "Mostrar elipses de"},
    "cred_level": {"en": "Credible level", "es": "Nivel de credibilidad", "pt": "Nível de credibilidade"},
    "points_show": {"en": "Points show", "es": "Los puntos muestran", "pt": "Os pontos mostram"},
    "observed": {"en": "Observed", "es": "Observado", "pt": "Observado"},
    "adjusted": {"en": "Bayes-adjusted", "es": "Ajustado (Bayes)", "pt": "Ajustado (Bayes)"},
    "trails": {"en": "Trail to the observed value", "es": "Estela hasta el valor observado",
               "pt": "Rasto até ao valor observado"},
    "medians": {"en": "League medians", "es": "Medianas de la liga", "pt": "Medianas da liga"},
    "quadrants": {"en": "Quadrant legend", "es": "Leyenda por cuadrantes", "pt": "Legenda por quadrantes"},
    "leaders": {"en": "Top 3 / bottom 3", "es": "Top 3 / últimos 3", "pt": "Top 3 / últimos 3"},
    "round_label": {"en": "Round label", "es": "Etiqueta de jornada", "pt": "Etiqueta da jornada"},
    "round_help": {"en": "Worked out from the games played. Edit it or leave it empty to hide it.",
                   "es": "Se calcula con los partidos jugados. Edítala o déjala vacía para ocultarla.",
                   "pt": "Calculada com os jogos disputados. Edite-a ou deixe-a vazia para a ocultar."},
    "round_word": {"en": "MATCHWEEK", "es": "JORNADA", "pt": "JORNADA"},
    "gender": {"en": "Competition", "es": "Competición", "pt": "Competição"},
    "men": {"en": "Men", "es": "Masculina", "pt": "Masculina"},
    "women": {"en": "Women", "es": "Femenina", "pt": "Feminina"},
    "x_axis": {"en": "X axis", "es": "Eje X", "pt": "Eixo X"},
    "y_axis": {"en": "Y axis", "es": "Eje Y", "pt": "Eixo Y"},
    "reverse_x": {"en": "Reverse X (better to the right)", "es": "Invertir X (mejor a la derecha)",
                  "pt": "Inverter X (melhor à direita)"},
    "reverse_y": {"en": "Reverse Y (better at the top)", "es": "Invertir Y (mejor arriba)",
                  "pt": "Inverter Y (melhor em cima)"},
    "highlight": {"en": "Highlight", "es": "Destacar", "pt": "Destacar"},
    "madrid": {"en": "Madrid clubs", "es": "Clubes de Madrid", "pt": "Clubes de Madrid"},
    "no_images": {"en": "No images available: showing dots.", "es": "No hay imágenes: se muestran puntos.",
                  "pt": "Sem imagens: mostram-se pontos."},
    "too_few": {"en": "Too few players with these filters.", "es": "Muy pocos jugadores con estos filtros.",
                "pt": "Poucos jogadores com estes filtros."},
    "caption": {"en": "{context} · uncertainty from an empirical-Bayes model",
                "es": "{context} · incertidumbre con un modelo bayesiano empírico",
                "pt": "{context} · incerteza com um modelo bayesiano empírico"},
    "download": {"en": "Download this view (CSV)", "es": "Descargar esta vista (CSV)",
                 "pt": "Descarregar esta vista (CSV)"},
    "name": {"en": "Name", "es": "Nombre", "pt": "Nome"},
    "team": {"en": "Team", "es": "Equipo", "pt": "Equipa"},
    "minutes": {"en": "Minutes", "es": "Minutos", "pt": "Minutos"},
    "matches": {"en": "Matches", "es": "Partidos", "pt": "Jogos"},
    "adj": {"en": "adj.", "es": "aj.", "pt": "aj."},
    "ci95": {"en": "95% CI", "es": "IC 95%", "pt": "IC 95%"},
    "built_by": {"en": "Built by Álvaro Bustamante-Sánchez, PhD", "es": "Creado por Álvaro Bustamante-Sánchez, PhD",
                 "pt": "Criado por Álvaro Bustamante-Sánchez, PhD"},
    # ------------------------------------------------------------ Bayes panels
    "h2h": {"en": "Who is better? (probability)", "es": "¿Quién es mejor? (probabilidad)",
            "pt": "Quem é melhor? (probabilidade)"},
    "h2h_pick": {"en": "Pick two", "es": "Elige dos", "pt": "Escolha dois"},
    "h2h_line": {"en": "**{a}** is better than **{b}** in {metric} with a **{p:.0f}%** probability.",
                 "es": "**{a}** es mejor que **{b}** en {metric} con una probabilidad del **{p:.0f}%**.",
                 "pt": "**{a}** é melhor do que **{b}** em {metric} com uma probabilidade de **{p:.0f}%**."},
    "h2h_line_neutral": {"en": "**{a}** has a higher {metric} than **{b}** with a **{p:.0f}%** probability.",
                         "es": "**{a}** tiene más {metric} que **{b}** con una probabilidad del **{p:.0f}%**.",
                         "pt": "**{a}** tem mais {metric} do que **{b}** com uma probabilidade de **{p:.0f}%**."},
    "h2h_note": {"en": "Below 75% or above 25%, the data cannot really tell them apart yet.",
                 "es": "Entre el 25% y el 75%, los datos todavía no permiten separarlos.",
                 "pt": "Entre 25% e 75%, os dados ainda não permitem separá-los."},
    "stability": {"en": "When can I trust each axis?", "es": "¿Cuándo puedo fiarme de cada eje?",
                  "pt": "Quando posso confiar em cada eixo?"},
    "stab_trials": {"en": "**{metric}**: the data weighs as much as the league average after about **{n:,.0f} {unit}**.",
                    "es": "**{metric}**: los datos pesan tanto como la media de la liga a partir de unos **{n:,.0f} {unit}**.",
                    "pt": "**{metric}**: os dados pesam tanto como a média da liga a partir de cerca de **{n:,.0f} {unit}**."},
    "stab_none": {"en": "**{metric}**: no uncertainty model (context metric).",
                  "es": "**{metric}**: sin modelo de incertidumbre (métrica de contexto).",
                  "pt": "**{metric}**: sem modelo de incerteza (métrica de contexto)."},
    "unit_attempts": {"en": "attempts", "es": "intentos", "pt": "tentativas"},
    "unit_minutes": {"en": "minutes", "es": "minutos", "pt": "minutos"},
    "unit_matches": {"en": "matches", "es": "partidos", "pt": "jogos"},
    "method": {"en": "Method: what the ellipses mean", "es": "Método: qué significan las elipses",
               "pt": "Método: o que significam as elipses"},
    "method_text": {
        "en": ("**Why adjust?** Football has few events: a striker with 3 goals from 4 shots is not a 75% finisher. "
               "Each metric is estimated with an **empirical-Bayes** model whose prior is the league itself "
               "(or the player's position): Gamma-Poisson for counts per 90 or per match, Beta-Binomial for "
               "percentages and Normal-Normal for continuous values (xG, positions). The dot is the observed value; "
               "the ellipse ({level} credible region) is centred on the adjusted value. "
               "**Overlapping ellipses = the data cannot separate them yet.**"),
        "es": ("**¿Por qué ajustar?** En fútbol hay pocos eventos: un delantero con 3 goles en 4 tiros no tiene un 75% "
               "de acierto. Cada métrica se estima con un modelo **bayesiano empírico** cuyo prior es la propia liga "
               "(o la posición del jugador): Gamma-Poisson para conteos por 90 o por partido, Beta-Binomial para "
               "porcentajes y Normal-Normal para valores continuos (xG, posiciones). El punto es el valor observado; "
               "la elipse (región creíble del {level}) está centrada en el valor ajustado. "
               "**Elipses que se solapan = los datos aún no permiten separarlos.**"),
        "pt": ("**Porquê ajustar?** No futebol há poucos eventos: um avançado com 3 golos em 4 remates não tem 75% "
               "de eficácia. Cada métrica é estimada com um modelo **bayesiano empírico** cujo prior é a própria liga "
               "(ou a posição do jogador): Gamma-Poisson para contagens por 90 ou por jogo, Beta-Binomial para "
               "percentagens e Normal-Normal para valores contínuos (xG, posições). O ponto é o valor observado; "
               "a elipse (região credível de {level}) está centrada no valor ajustado. "
               "**Elipses sobrepostas = os dados ainda não permitem separá-los.**")},
    # ------------------------------------------------------------- luck table
    "luck_title": {"en": "Points vs expected points (xPts)", "es": "Puntos frente a puntos esperados (xPts)",
                   "pt": "Pontos vs pontos esperados (xPts)"},
    "luck_text": {"en": "xPts: points a team would expect from the chances (xG) created and conceded in each match. "
                        "Above the line: more points than their football deserved so far.",
                  "es": "xPts: puntos que un equipo debería esperar según las ocasiones (xG) creadas y concedidas en cada "
                        "partido. Por encima de la línea: más puntos de los que su juego merecía hasta ahora.",
                  "pt": "xPts: pontos que uma equipa deveria esperar pelas ocasiões (xG) criadas e concedidas em cada "
                        "jogo. Acima da linha: mais pontos do que o seu jogo merecia até agora."},
    "points": {"en": "Points", "es": "Puntos", "pt": "Pontos"},
    "xpts": {"en": "Expected points (xPts)", "es": "Puntos esperados (xPts)", "pt": "Pontos esperados (xPts)"},
    "luck": {"en": "Luck (points − xPts)", "es": "Suerte (puntos − xPts)", "pt": "Sorte (pontos − xPts)"},
    "goals_minus_xg": {"en": "Goals − xG", "es": "Goles − xG", "pt": "Golos − xG"},
    "ga_minus_xga": {"en": "Goals against − xGA", "es": "Goles en contra − xGA", "pt": "Golos sofridos − xGA"},
    # ------------------------------------------------------------- pitch view
    "pitch_team": {"en": "Team", "es": "Equipo", "pt": "Equipa"},
    "pitch_compare": {"en": "Compare with (optional)", "es": "Comparar con (opcional)", "pt": "Comparar com (opcional)"},
    "pitch_note": {"en": ("Average positions come from EVENTS (where each player acts on the ball), not tracking: "
                          "useful to compare teams, not the exact block. Ellipses = match-to-match variation of each "
                          "player's average position."),
                   "es": ("Las posiciones medias salen de EVENTOS (dónde actúa cada jugador con balón), no de tracking: "
                          "sirven para comparar equipos, no son el bloque exacto. Elipses = variación de la posición "
                          "media de cada jugador de un partido a otro."),
                   "pt": ("As posições médias vêm de EVENTOS (onde cada jogador atua com bola), não de tracking: "
                          "servem para comparar equipas, não são o bloco exato. Elipses = variação da posição média "
                          "de cada jogador de jogo para jogo.")},
    "def_line": {"en": "Defensive line", "es": "Línea defensiva", "pt": "Linha defensiva"},
    "luck_above": {"en": "More points than xPts", "es": "Más puntos que xPts", "pt": "Mais pontos que xPts"},
    "luck_below": {"en": "Fewer points than xPts", "es": "Menos puntos que xPts", "pt": "Menos pontos que xPts"},
    "last_round": {"en": "Last played: {r}", "es": "Última jugada: {r}", "pt": "Última jogada: {r}"},
    "centroid": {"en": "Team centroid", "es": "Centroide del equipo", "pt": "Centroide da equipa"},
    "block_type": {"en": "Block", "es": "Bloque", "pt": "Bloco"},
    "block_high": {"en": "HIGH", "es": "ALTO", "pt": "ALTO"},
    "block_mid": {"en": "MID", "es": "MEDIO", "pt": "MÉDIO"},
    "block_low": {"en": "LOW", "es": "BAJO", "pt": "BAIXO"},
    "block_table": {"en": "Block metrics (league)", "es": "Métricas de bloque (liga)", "pt": "Métricas de bloco (liga)"},
    "min_matches": {"en": "Players with at least N starts", "es": "Jugadores con al menos N titularidades",
                    "pt": "Jogadores com pelo menos N titularidades"},
    # ------------------------------------------------------- quadrant words
    "w_good": {"en": "GOOD", "es": "MEJOR", "pt": "MELHOR"},
    "w_bad": {"en": "BAD", "es": "PEOR", "pt": "PIOR"},
    "w_high": {"en": "HIGH", "es": "MÁS", "pt": "MAIS"},
    "w_low": {"en": "LOW", "es": "MENOS", "pt": "MENOS"},
    "tag_best": {"en": "Best", "es": "Mejores", "pt": "Melhores"},
    "tag_worst": {"en": "Worst", "es": "Peores", "pt": "Piores"},
    "tag_highest": {"en": "Highest", "es": "Más altos", "pt": "Mais altos"},
    "tag_lowest": {"en": "Lowest", "es": "Más bajos", "pt": "Mais baixos"},
    "export_obs": {"en": "Observed values", "es": "Valores observados", "pt": "Valores observados"},
    "export_adj": {"en": "Bayes-adjusted values", "es": "Valores ajustados (Bayes)", "pt": "Valores ajustados (Bayes)"},
    "export_ell": {"en": "ellipses: {level} credible region", "es": "elipses: región creíble del {level}",
                   "pt": "elipses: região credível de {level}"},
}


_KEEP = {"NPXG": "npxG", "XGA": "xGA", "XPTS": "xPts", "XG": "xG", "XA": "xA"}


def smart_upper(text: str) -> str:
    """Upper case that keeps football abbreviations readable (xG, npxG, xA, xGA, xPts)."""
    import re
    up = str(text).upper()
    return re.sub(r"\b(NPXG|XGA|XPTS|XG|XA)\b", lambda m: _KEEP[m.group(1)], up)
