"""Versões Plotly interativas dos gráficos da análise causal (Módulo 2).

Mesma paleta validada para daltonismo de ``plots.py``; hover em todos os
pontos, um único eixo y por gráfico, valores rotulados diretamente (a cor
nunca carrega a informação sozinha).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_DARK = "#104281"
GREEN = "#1baf7a"
POS, NEG = "#2a78d6", "#e34948"
ORDINAL_4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="sans-serif", size=13, color=INK_2),
    title_font=dict(size=15, color=INK),
    # topo folgado para o título + subtítulo não encostarem na área do gráfico
    margin=dict(l=70, r=40, t=95, b=65),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _base_axes(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                     title_standoff=16, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                     title_standoff=16, automargin=True)


# acima disso, séries de pontos são rarefeidas só para desenho (evita travar o
# navegador e o LOWESS O(n²) na dispersão); as estatísticas do Módulo 2 são
# calculadas à parte, sobre todos os dados
MAX_PLOT_POINTS = 8000


def _thin(*arrays, max_points: int = MAX_PLOT_POINTS):
    """Rarefaz arrays paralelos por passo fixo; devolve (arrays..., reduziu?)."""
    n = len(arrays[0])
    if n <= max_points:
        return (*arrays, False)
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return (*[np.asarray(a)[idx] for a in arrays], True)


def fig_ranking(scores: pd.DataFrame, top: int = 15) -> go.Figure:
    """Barras horizontais do score; a cor codifica a direção do efeito."""
    data = scores.head(top).iloc[::-1]
    colors = [POS if d > 0 else NEG if d < 0 else MUTED for d in data["direcao"]]
    hover = [
        (f"<b>{r.parametro}</b><br>score: {r.score:.0f}/100"
         f"<br>veredito: {r.veredito}<br>direção: {r.direcao_label}"
         f"<br>melhor transformação: {r.melhor_transformacao}"
         f"<br>testes significativos: {r.testes_significativos}"
         + (f"<br>efeito: {r.efeito}" if hasattr(r, "efeito") else ""))
        for r in data.itertuples()
    ]
    fig = go.Figure(go.Bar(
        x=data["score"], y=data["parametro"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.0f}" for v in data["score"]],
        textposition="outside", textfont=dict(size=11, color=INK_2),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title="Ranking dos indicadores por evidência de influência no alvo"
              "<br><sup>azul = efeito positivo · vermelho = negativo · "
              "cinza = não-monotônico/indefinido</sup>",
        xaxis=dict(title="Score de evidência (0–100)", range=[0, 108]),
        height=max(320, 36 * len(data) + 150),
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    # nomes de indicadores podem ser longos: automargin evita corte e o
    # sufixo dá um respiro entre o rótulo e as barras
    fig.update_yaxes(gridcolor=SURFACE, automargin=True, ticksuffix="  ")
    return fig


def fig_day_diagnosis(rows: pd.DataFrame, day_label: str) -> go.Figure:
    """Barras do 'score do dia' (ranking histórico × atipicidade do dia).

    Azul = empurrão esperado para cima; vermelho = para baixo; cinza =
    dentro do típico / efeito não-monotônico.
    """
    data = rows.head(10).iloc[::-1]

    def cor(push: str) -> str:
        if str(push).endswith("CIMA"):
            return POS
        if str(push).endswith("BAIXO"):
            return NEG
        return MUTED

    colors = [cor(p) for p in data["empurrão esperado"]]
    hover = [
        (f"<b>{r['indicador']}</b><br>valor no dia: {r['valor no dia']:.4g}"
         f"<br>percentil no dia: {r['percentil no dia']}"
         f"<br>desvio: {r['desvio']}<br>empurrão: {r['empurrão esperado']}"
         f"<br>score histórico: {r['score histórico']:.0f}/100"
         f"<br>versão avaliada: {r['versão avaliada']}")
        for _, r in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["score do dia"], y=data["indicador"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.0f}" for v in data["score do dia"]],
        textposition="outside", textfont=dict(size=11, color=INK_2),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Prováveis contribuintes do dia — {day_label}"
              "<br><sup>score do dia = influência histórica × atipicidade no "
              "dia · azul = empurra o alvo para cima · vermelho = para baixo "
              "· cinza = típico/não-monotônico</sup>",
        xaxis=dict(title="Score do dia (0–100)", range=[0, 108]),
        height=max(320, 36 * len(data) + 150),
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    fig.update_yaxes(gridcolor=SURFACE, automargin=True, ticksuffix="  ")
    return fig


def _err(point: pd.Series, lo: pd.Series, hi: pd.Series) -> dict:
    """Barras de erro assimétricas a partir de (estimativa, IC-, IC+)."""
    p = np.asarray(point, dtype=float)
    plus = np.nan_to_num(np.asarray(hi, dtype=float) - p, nan=0.0)
    minus = np.nan_to_num(p - np.asarray(lo, dtype=float), nan=0.0)
    return dict(
        type="data", symmetric=False,
        array=np.clip(plus, 0, None), arrayminus=np.clip(minus, 0, None),
        color=INK_2, thickness=1.4, width=5,
    )


def _label_after_err(fig: go.Figure, cats, point, lo, hi, texts) -> None:
    """Rotula cada barra DEPOIS do traço do IC (e ajusta a folga do eixo).

    O rótulo "outside" do Plotly encosta na ponta da barra — bem onde passa a
    barra de erro. Aqui o número vai para fora do IC inteiro, então nada se
    sobrepõe, e o eixo ganha margem para o texto caber.
    """
    p = np.asarray(point, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ends = np.where(p >= 0, np.fmax(p, hi), np.fmin(p, lo))
    ends = np.where(np.isfinite(ends), ends, p)
    for cat, v, end, txt in zip(cats, p, ends, texts):
        fig.add_annotation(
            x=float(end), y=cat, text=txt, showarrow=False,
            xanchor="left" if v >= 0 else "right",
            xshift=7 if v >= 0 else -7,
            font=dict(size=11, color=INK_2),
        )
    span = float(np.nanmax(np.append(ends, 0)) - np.nanmin(np.append(ends, 0)))
    folga = max(span * 0.18, 1e-9)
    fig.update_xaxes(range=[float(np.nanmin(np.append(ends, 0))) - folga * 0.4,
                            float(np.nanmax(np.append(ends, 0))) + folga])


def fig_variance_shares(var, target: str, top: int = 12) -> go.Figure:
    """Repartição da variação do alvo entre os indicadores (com IC).

    Uma barra por indicador (azul) mais a fatia cinza do que o modelo NÃO
    explica — juntas fecham 100% da variação observada.
    """
    data = var.rows.head(top + 1).iloc[::-1]
    is_resid = data["grupo"].eq("__residuo__")
    colors = [MUTED if r else BLUE for r in is_resid]
    hover = [
        (f"<b>{r['indicador']}</b><br>fatia: {r['% da variação do alvo']:.1f}%"
         + (f"<br>IC: {r['IC inferior']:.1f}% a {r['IC superior']:.1f}%"
            if np.isfinite(r["IC inferior"]) else "<br>sem IC (bootstrap desligado)")
         + (f"<br>do que o modelo explica: {r['% do que o modelo explica']:.1f}%"
            if np.isfinite(r.get("% do que o modelo explica", np.nan)) else ""))
        for _, r in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["% da variação do alvo"], y=data["indicador"], orientation="h",
        marker=dict(color=colors),
        error_x=_err(data["% da variação do alvo"],
                     data["IC inferior"], data["IC superior"]),
        hovertext=hover, hoverinfo="text",
    ))
    _label_after_err(
        fig, data["indicador"], data["% da variação do alvo"],
        data["IC inferior"], data["IC superior"],
        [f"{v:.0f}%" for v in data["% da variação do alvo"]],
    )
    r2 = var.r2_oos
    sub = (f"cada barra é a fatia da variação de {target} atribuída ao "
           f"indicador · traço horizontal = IC {int(100 * 0.95)}% · cinza = o "
           "que os dados da planilha não explicam")
    if np.isfinite(r2):
        sub += f" · R² fora da amostra = {r2:.2f}"
    fig.update_layout(
        title=f"De onde vem a variação de {target}<br><sup>{sub}</sup>",
        xaxis=dict(title="% da variação do alvo"),
        height=max(320, 36 * len(data) + 160),
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    fig.update_yaxes(gridcolor=SURFACE, automargin=True, ticksuffix="  ")
    return fig


def fig_day_waterfall(day, day_label: str, top: int = 8) -> go.Figure:
    """Cascata: do período típico ao valor observado, passo a passo.

    Cada degrau é a contribuição (Shapley) de um indicador; por construção os
    degraus somam exatamente o desvio previsto, e o último trecho mostra o
    que sobrou sem explicação.
    """
    rows = day.rows.copy()
    head = rows.head(top)
    resto = float(rows["contribuição"].iloc[top:].sum()) if len(rows) > top else 0.0

    x = ["período típico"] + [str(i) for i in head["indicador"]]
    y = [day.y_typical] + [float(v) for v in head["contribuição"]]
    measure = ["absolute"] + ["relative"] * len(head)
    if abs(resto) > 1e-12:
        x.append(f"demais ({len(rows) - top})")
        y.append(resto)
        measure.append("relative")
    # a cascata termina no PREVISTO; o observado entra como linha de
    # referência — assim o resíduo aparece como a distância entre os dois sem
    # virar mais uma barra colorida (que se leria como "mais um culpado")
    x += ["previsto pelo modelo"]
    y += [0.0]
    measure += ["total"]

    fig = go.Figure(go.Waterfall(
        orientation="v", x=x, y=y, measure=measure,
        increasing=dict(marker=dict(color=POS)),
        decreasing=dict(marker=dict(color=NEG)),
        totals=dict(marker=dict(color=MUTED)),
        connector=dict(line=dict(color=BASELINE, width=1)),
        text=[f"{v:+.4g}".replace(".", ",") if m == "relative"
              else "" for v, m in zip(y, measure)],
        textposition="outside", textfont=dict(size=11, color=INK_2),
        cliponaxis=False,
        hovertemplate="%{x}<br>%{y:.4g}<extra></extra>",
    ))
    fig.add_hline(
        y=day.y_observed, line_color=INK_2, line_width=1.5, line_dash="dash",
        annotation_text=f"observado: {day.y_observed:.4g}".replace(".", ","),
        annotation_position="top left",
        annotation_font=dict(size=11, color=INK_2),
    )

    sub = ("parte-se do período típico e soma-se a contribuição de cada "
           "indicador (azul empurra para cima, vermelho para baixo)<br>"
           "as contribuições fecham exatamente o previsto · a distância até a "
           "linha tracejada é o que os indicadores NÃO explicam")
    # com contribuições pequenas diante do nível do alvo, o eixo em zero
    # achataria os degraus a ponto de sumirem
    caminho = np.cumsum([day.y_typical] + y[1:-1])
    lo, hi = float(min(caminho.min(), day.y_observed)), \
        float(max(caminho.max(), day.y_observed))
    yaxis = dict(title=day.target)
    if lo > 0 and (hi - lo) < 0.35 * hi:
        folga = 0.25 * max(hi - lo, 1e-9)
        yaxis["range"] = [lo - folga, hi + folga]
        sub += " · atenção: o eixo não começa em zero"

    layout = {**_LAYOUT, "margin": dict(l=70, r=40, t=115, b=90)}
    fig.update_layout(
        title=f"Como se formou o resultado de {day_label}<br><sup>{sub}</sup>",
        yaxis=yaxis, height=480, showlegend=False, **layout,
    )
    _base_axes(fig)
    fig.update_xaxes(tickangle=-30, automargin=True)
    return fig


def fig_day_contributions(day, top: int = 10, conf: float = 0.95) -> go.Figure:
    """Contribuição de cada indicador ao desvio do dia, com intervalo."""
    data = day.rows.head(top).iloc[::-1]
    colors = [POS if v > 0 else NEG for v in data["contribuição"]]
    hover = [
        (f"<b>{r['indicador']}</b><br>contribuição: {r['contribuição']:.4g}"
         + (f"<br>IC: {r['IC inferior']:.4g} a {r['IC superior']:.4g}"
            if np.isfinite(r["IC inferior"]) else "<br>sem IC")
         + f"<br>{r['% do desvio']:.0f}% do desvio do dia"
         + (f"<br>valor no dia: {r['valor no dia']:.4g} "
            f"(percentil {r['percentil no dia']:.0f})"
            if np.isfinite(r["valor no dia"]) else "")
         + (f"<br>contraprova por vizinhos: {r['efeito pelos vizinhos']:.4g}"
            if np.isfinite(r["efeito pelos vizinhos"]) else
            "<br>sem períodos parecidos (extrapolação)"))
        for _, r in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["contribuição"], y=data["indicador"], orientation="h",
        marker=dict(color=colors),
        error_x=_err(data["contribuição"], data["IC inferior"],
                     data["IC superior"]),
        hovertext=hover, hoverinfo="text",
    ))
    _label_after_err(
        fig, data["indicador"], data["contribuição"], data["IC inferior"],
        data["IC superior"],
        [f"{v:.3g}".replace(".", ",") for v in data["contribuição"]],
    )
    fig.add_vline(x=0, line_color=BASELINE, line_width=1)
    fig.update_layout(
        title="Contribuição de cada indicador ao desvio do dia"
              f"<br><sup>traço horizontal = intervalo de confiança "
              f"{int(conf * 100)}% (bootstrap por blocos + erro de "
              "amostragem do Shapley) · barras que cruzam o zero não são "
              "conclusivas</sup>",
        xaxis=dict(title=f"contribuição ({day.target})"),
        height=max(320, 36 * len(data) + 160),
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    fig.update_yaxes(gridcolor=SURFACE, automargin=True, ticksuffix="  ")
    return fig


def fig_scenarios(scen, top: int = 12, conf: float = 0.95) -> go.Figure:
    """Deslocamento típico do alvo causado por cada indicador (com IC)."""
    data = scen.rows.head(top).iloc[::-1]
    hover = [
        (f"<b>{r['indicador']}</b><br>deslocamento típico: "
         f"{r['deslocamento típico']:.4g}"
         + (f"<br>IC: {r['deslocamento IC-']:.4g} a {r['deslocamento IC+']:.4g}"
            if np.isfinite(r["deslocamento IC-"]) else "<br>sem IC")
         + f"<br>maior deslocamento: {r['maior deslocamento']:.4g}"
         + f"<br>efeito no nível médio: {r['efeito no nível médio']:.4g}"
         + f"<br>fora do típico em {r['% do tempo fora do típico']:.0f}% do tempo")
        for _, r in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["deslocamento típico"], y=data["indicador"], orientation="h",
        marker=dict(color=BLUE),
        error_x=_err(data["deslocamento típico"], data["deslocamento IC-"],
                     data["deslocamento IC+"]),
        hovertext=hover, hoverinfo="text",
    ))
    _label_after_err(
        fig, data["indicador"], data["deslocamento típico"],
        data["deslocamento IC-"], data["deslocamento IC+"],
        [f"{v:.3g}".replace(".", ",") for v in data["deslocamento típico"]],
    )
    fig.update_layout(
        title=f"Quanto cada indicador desloca {scen.target}, em média, por período"
              "<br><sup>contrafactual: diferença média (em módulo) entre o que "
              "aconteceu e o que aconteceria com o indicador sempre no valor "
              f"típico · traço horizontal = IC {int(conf * 100)}%</sup>",
        xaxis=dict(title=f"deslocamento típico ({scen.target})"),
        height=max(320, 36 * len(data) + 160),
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    fig.update_yaxes(gridcolor=SURFACE, automargin=True, ticksuffix="  ")
    return fig


def fig_response_curve(curve, conf: float = 0.95) -> go.Figure:
    """Curva contrafactual: alvo previsto para cada nível do indicador."""
    r = curve.rows
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(r["valor do indicador"]) + list(r["valor do indicador"])[::-1],
        y=list(r["IC superior"]) + list(r["IC inferior"])[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.14)", mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=r["valor do indicador"], y=r["alvo previsto"], mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=8),
        customdata=r["percentil"],
        hovertemplate=(f"{curve.group}" + ": %{x:.4g} (percentil %{customdata})"
                       f"<br>{curve.target} previsto: %{{y:.4g}}<extra></extra>"),
        showlegend=False,
    ))
    if np.isfinite(curve.typical):
        fig.add_vline(x=curve.typical, line_color=BASELINE, line_width=1.5,
                      line_dash="dot", annotation_text="valor típico",
                      annotation_position="top",
                      annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(
        title=f"Se '{curve.group}' ficasse em cada nível, onde estaria "
              f"{curve.target}?"
              "<br><sup>contrafactual do modelo, mantendo os demais "
              "indicadores como estiveram · faixa sombreada = IC "
              f"{int(conf * 100)}% · só percentis observados (sem "
              "extrapolação)</sup>",
        xaxis_title=str(curve.group), yaxis_title=f"{curve.target} previsto",
        height=400, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


def fig_corr_heatmap(df: pd.DataFrame, target: str, max_cols: int = 25) -> go.Figure:
    """Mapa de calor de Spearman (divergente vermelho-cinza-azul)."""
    cols = [target] + [c for c in df.columns if c != target][: max_cols - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = df[cols].corr(method="spearman")
    labels = [str(c)[:28] for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=[[0.0, NEG], [0.5, "#f0efec"], [1.0, POS]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{y} × %{x}<br>Spearman ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(title="ρ"),
    ))
    n = len(labels)
    fig.update_layout(
        title=f"Correlação de Spearman entre indicadores e alvo ({target})",
        height=max(440, 32 * n + 180), **_LAYOUT,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def fig_lag_profile(name: str, lag_profile: dict[int, float],
                    rolling_profile: dict[int, float]) -> go.Figure:
    """Perfil de correlação por defasagem e por janela de média móvel."""
    fig = make_subplots(
        cols=2, rows=1, shared_yaxes=True, column_widths=[0.7, 0.3],
        subplot_titles=("efeito por defasagem (lag)", "médias móveis"),
        horizontal_spacing=0.06,
    )
    lags = sorted(lag_profile)
    vals = [lag_profile[k] for k in lags]
    fig.add_trace(go.Scatter(
        x=lags, y=vals, mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=6),
        hovertemplate="lag %{x}: ρ = %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    if vals:
        i = int(np.nanargmax(np.abs(vals)))
        # rótulo abaixo do ponto quando o pico está perto do topo do eixo
        text_pos = "bottom center" if vals[i] > 0.6 else "top center"
        fig.add_trace(go.Scatter(
            x=[lags[i]], y=[vals[i]], mode="markers+text",
            marker=dict(size=13, color="rgba(0,0,0,0)",
                        line=dict(width=2, color=INK)),
            text=[f"lag {lags[i]}: ρ={vals[i]:.2f}"], textposition=text_pos,
            textfont=dict(size=11, color=INK_2),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
    wins = sorted(rolling_profile)
    if wins:
        fig.add_trace(go.Bar(
            x=[str(w) for w in wins], y=[rolling_profile[w] for w in wins],
            marker=dict(color=BLUE), width=0.55,
            hovertemplate="janela %{x}: ρ = %{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
    fig.add_hline(y=0, line_color=BASELINE, line_width=1)
    fig.update_yaxes(range=[-1, 1], title_text="Spearman ρ com o alvo",
                     row=1, col=1)
    fig.update_xaxes(title_text="defasagem (períodos)", row=1, col=1)
    fig.update_xaxes(title_text="janela", row=1, col=2)
    fig.update_layout(
        title=f"{name} — quando o efeito aparece",
        height=380, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


def fig_scatter(x: pd.Series, y: pd.Series, name: str, target: str,
                transform_label: str) -> go.Figure:
    """Dispersão do indicador (melhor transformação) vs. alvo, com LOWESS."""
    m = x.notna() & y.notna()
    xa, ya = x[m].to_numpy(float), y[m].to_numpy(float)
    # rarefaz para o desenho E para o LOWESS (O(n²) — inviável em séries longas)
    xd, yd, thinned = _thin(xa, ya)
    fig = go.Figure(go.Scatter(
        x=xd, y=yd, mode="markers",
        marker=dict(size=6, color=BLUE, opacity=0.4),
        name="observações",
        hovertemplate=f"{name}: %{{x:.4g}}<br>{target}: %{{y:.4g}}"
                      "<extra></extra>",
    ))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smooth = lowess(yd, xd, frac=0.4, return_sorted=True)
        fig.add_trace(go.Scatter(
            x=smooth[:, 0], y=smooth[:, 1], mode="lines",
            line=dict(color=BLUE_DARK, width=2.5), name="tendência (LOWESS)",
            hoverinfo="skip",
        ))
    except Exception:
        pass
    sub = transform_label + (
        f" · exibindo {len(xd):,} de {len(xa):,} pontos".replace(",", ".")
        if thinned else "")
    fig.update_layout(
        title=f"Forma da relação — {name} vs. {target}"
              f"<br><sup>{sub}</sup>",
        xaxis_title=f"{name} ({transform_label})", yaxis_title=target,
        height=410, showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


def fig_quartile_box(x: pd.Series, y: pd.Series, name: str, target: str) -> go.Figure:
    """Boxplot do alvo por quartil do indicador (rampa ordinal de um matiz)."""
    m = x.notna() & y.notna()
    xa, ya = x[m], y[m]
    fig = go.Figure()
    try:
        qbins = pd.qcut(xa, 4, labels=["Q1 (baixo)", "Q2", "Q3", "Q4 (alto)"],
                        duplicates="drop")
        for cat, color in zip(qbins.cat.categories, ORDINAL_4):
            fig.add_trace(go.Box(
                y=ya[qbins == cat], name=str(cat),
                marker=dict(color=color, size=4),
                line=dict(color=INK_2, width=1.4), fillcolor=color,
                boxpoints="outliers",
                hovertemplate=f"{target}: %{{y:.4g}}<extra>{cat}</extra>",
            ))
    except ValueError:
        pass
    fig.update_layout(
        title=f"{target} por faixa de {name}"
              "<br><sup>cada caixa mostra a distribuição do alvo quando o "
              "indicador está naquela faixa de valores</sup>",
        yaxis_title=target, xaxis_title=f"quartis de {name}",
        height=410, showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


def fig_timeseries_overlay(y: pd.Series, x: pd.Series,
                           target: str, name: str) -> go.Figure:
    """Alvo e indicador padronizados (z-score) no tempo — um único eixo."""
    def z(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std and np.isfinite(std) else s * 0

    # z-score sobre a série completa; rarefaz só o que vai para a tela
    yi, yv, thinned = _thin(np.asarray(y.index, dtype=object), z(y).to_numpy(float))
    xi, xv, _ = _thin(np.asarray(x.index, dtype=object), z(x).to_numpy(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yi, y=yv, mode="lines",
        line=dict(color=BLUE, width=2), name=f"{target} (alvo)",
        hovertemplate="%{x}<br>z: %{y:.2f}<extra>" + target + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xi, y=xv, mode="lines",
        line=dict(color=GREEN, width=1.6), name=name, opacity=0.9,
        hovertemplate="%{x}<br>z: %{y:.2f}<extra>" + name + "</extra>",
    ))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1)
    sub = ("séries padronizadas (z-score) para caberem no mesmo eixo — "
           "procure movimentos conjuntos ou defasados")
    if thinned:
        sub += f" · exibindo {len(yi):,} de {len(y):,} pontos".replace(",", ".")
    fig.update_layout(
        title="Evolução temporal comparada"
              f"<br><sup>{sub}</sup>",
        yaxis_title="z-score (padronizado)",
        height=380,
        legend=dict(orientation="h", y=1.04, yanchor="bottom",
                    x=1, xanchor="right"),
        **_LAYOUT,
    )
    _base_axes(fig)
    return fig
