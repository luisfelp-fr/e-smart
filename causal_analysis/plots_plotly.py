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


def fig_ranking(scores: pd.DataFrame, top: int = 15) -> go.Figure:
    """Barras horizontais do score; a cor codifica a direção do efeito."""
    data = scores.head(top).iloc[::-1]
    colors = [POS if d > 0 else NEG if d < 0 else MUTED for d in data["direcao"]]
    hover = [
        (f"<b>{r.parametro}</b><br>score: {r.score:.0f}/100"
         f"<br>veredito: {r.veredito}<br>direção: {r.direcao_label}"
         f"<br>melhor transformação: {r.melhor_transformacao}"
         f"<br>testes significativos: {r.testes_significativos}")
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
    fig = go.Figure(go.Scatter(
        x=xa, y=ya, mode="markers",
        marker=dict(size=6, color=BLUE, opacity=0.4),
        name="observações",
        hovertemplate=f"{name}: %{{x:.4g}}<br>{target}: %{{y:.4g}}"
                      "<extra></extra>",
    ))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smooth = lowess(ya, xa, frac=0.4, return_sorted=True)
        fig.add_trace(go.Scatter(
            x=smooth[:, 0], y=smooth[:, 1], mode="lines",
            line=dict(color=BLUE_DARK, width=2.5), name="tendência (LOWESS)",
            hoverinfo="skip",
        ))
    except Exception:
        pass
    fig.update_layout(
        title=f"Forma da relação — {name} vs. {target}"
              f"<br><sup>{transform_label}</sup>",
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=z(y), mode="lines",
        line=dict(color=BLUE, width=2), name=f"{target} (alvo)",
        hovertemplate="%{x}<br>z: %{y:.2f}<extra>" + target + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x.index, y=z(x), mode="lines",
        line=dict(color=GREEN, width=1.6), name=name, opacity=0.9,
        hovertemplate="%{x}<br>z: %{y:.2f}<extra>" + name + "</extra>",
    ))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1)
    fig.update_layout(
        title="Evolução temporal comparada"
              "<br><sup>séries padronizadas (z-score) para caberem no mesmo "
              "eixo — procure movimentos conjuntos ou defasados</sup>",
        yaxis_title="z-score (padronizado)",
        height=380,
        legend=dict(orientation="h", y=1.04, yanchor="bottom",
                    x=1, xanchor="right"),
        **_LAYOUT,
    )
    _base_axes(fig)
    return fig
