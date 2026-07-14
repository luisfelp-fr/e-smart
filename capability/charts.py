"""Gráficos Plotly interativos do módulo de capabilidade.

Paleta e regras espelham o sistema validado para daltonismo já usado no
pacote de análise causal (mesmos hex): azul para os dados, vermelho apenas
para violações/fora de especificação (sempre com rótulo/hover — a cor nunca
carrega a informação sozinha), cinzas recessivos para grade e limites de
controle, um único eixo y por gráfico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .control_chart import RULE_LABELS, IMRResult
from .indices import CapabilityIndices
from .nonparametric import BoxStats, SuggestedLimits
from .pipeline import CapabilityReport
from .transforms import inverse

# --- paleta (idêntica a causal_analysis/plots.py; validada p/ daltonismo) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"       # dados
BLUE_DARK = "#104281"  # curvas ajustadas
RED = "#e34948"        # violações / fora de especificação
GREEN = "#008300"      # limites sugeridos (sempre com rótulo)
AMBER = "#eda100"      # avisos / zona marginal

_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="sans-serif", size=13, color=INK_2),
    title_font=dict(size=15, color=INK),
    margin=dict(l=60, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _base_axes(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE)


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}".replace(".", ",")


def _spec_line(fig: go.Figure, x: float, label: str, color: str = RED,
               dash: str = "solid", position: str = "top",
               row=None, col=None) -> None:
    kwargs = {}
    if row is not None:
        kwargs = dict(row=row, col=col)
    fig.add_vline(
        x=x, line_color=color, line_width=2, line_dash=dash,
        annotation_text=f"{label} = {_fmt(x, 4)}",
        annotation_position=position,
        annotation_font=dict(size=11, color=color),
        **kwargs,
    )


# ------------------------------------------------------------- histograma

def fig_capability_hist(rep: CapabilityReport) -> go.Figure:
    """Histograma com limites, curva ajustada e faixa fora de especificação."""
    x = rep.series.dropna()
    idx = rep.indices
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=x, histnorm="probability density",
        marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
        opacity=0.75, name="dados",
        hovertemplate="faixa: %{x}<br>densidade: %{y:.4f}<extra></extra>",
    ))

    # curva do modelo: normal (Caso 1) ou normal retro-convertida (Caso 2)
    grid_x, pdf = _fitted_curve(rep)
    if grid_x is not None:
        fig.add_trace(go.Scatter(
            x=grid_x, y=pdf, mode="lines",
            line=dict(color=BLUE_DARK, width=2.5),
            name="modelo ajustado",
            hovertemplate="x: %{x:.4g}<br>densidade: %{y:.4f}<extra></extra>",
        ))

    # sombreado fora de especificação
    xmin = float(min(x.min(), rep.lsl if rep.lsl is not None else x.min()))
    xmax = float(max(x.max(), rep.usl if rep.usl is not None else x.max()))
    pad = 0.06 * (xmax - xmin or 1.0)
    if rep.lsl is not None:
        fig.add_vrect(x0=xmin - pad, x1=rep.lsl, fillcolor=RED, opacity=0.07,
                      line_width=0)
        _spec_line(fig, rep.lsl, "LIE")
    if rep.usl is not None:
        fig.add_vrect(x0=rep.usl, x1=xmax + pad, fillcolor=RED, opacity=0.07,
                      line_width=0)
        _spec_line(fig, rep.usl, "LSE")

    if rep.case == 2 and np.isfinite(rep.display_median):
        _spec_line(fig, rep.display_median, "mediana", color=INK_2, dash="dot")

    sub = _indices_subtitle(idx)
    fig.update_layout(
        title=f"Capabilidade — {rep.indicator}<br><sup>{sub}</sup>",
        xaxis_title=rep.indicator, yaxis_title="densidade",
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


def _fitted_curve(rep: CapabilityReport) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Densidade do modelo na escala ORIGINAL (retro-convertida no Caso 2)."""
    idx = rep.indices
    if idx is None or not np.isfinite(idx.sigma_overall) or idx.sigma_overall <= 0:
        return None, None
    if rep.case == 1:
        mu, sd = idx.mean, idx.sigma_overall
        gx = np.linspace(mu - 4 * sd, mu + 4 * sd, 300)
        return gx, stats.norm.pdf(gx, mu, sd)
    if rep.case == 2 and rep.transform and rep.transform.best:
        best = rep.transform.best
        mu, sd = idx.mean, idx.sigma_overall  # no espaço transformado
        t_grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
        x_grid = inverse(best.name, best.params, t_grid)
        ok = np.isfinite(x_grid)
        t_grid, x_grid = t_grid[ok], x_grid[ok]
        if len(x_grid) < 10 or np.any(np.diff(x_grid) <= 0):
            return None, None
        # mudança de variável: f_X(x) = φ(t) · dt/dx
        pdf = stats.norm.pdf(t_grid, mu, sd) * np.gradient(t_grid, x_grid)
        return x_grid, pdf
    return None, None


def _indices_subtitle(idx: CapabilityIndices | None) -> str:
    if idx is None:
        return ""
    parts = []
    if np.isfinite(idx.cp):
        parts.append(f"Cp={_fmt(idx.cp)}")
    if np.isfinite(idx.cpk):
        parts.append(f"Cpk={_fmt(idx.cpk)}")
    if np.isfinite(idx.pp):
        parts.append(f"Pp={_fmt(idx.pp)}")
    if np.isfinite(idx.ppk):
        parts.append(f"Ppk={_fmt(idx.ppk)}")
    if np.isfinite(idx.ppm_total):
        parts.append(f"PPM≈{idx.ppm_total:,.0f}".replace(",", "."))
    if idx.verdict:
        parts.append(f"veredito: {idx.verdict}")
    return " · ".join(parts)


# ------------------------------------------------------------- carta I-AM

def fig_imr(rep: CapabilityReport) -> go.Figure:
    """Carta I-AM em dois painéis com causas especiais destacadas."""
    imr: IMRResult = rep.imr
    x = rep.series.dropna()
    labels = list(x.index)
    vals = x.to_numpy(dtype=float)
    mr = np.abs(np.diff(vals))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("Individuais (valores na ordem de produção)",
                        "Amplitude Móvel (|diferença entre pontos vizinhos|)"),
    )

    # ---- painel 1: individuais
    fig.add_trace(go.Scatter(
        x=labels, y=vals, mode="lines+markers",
        line=dict(color=BLUE, width=1.6), marker=dict(size=5, color=BLUE),
        name="valores",
        hovertemplate="%{x}<br>valor: %{y:.4g}<extra></extra>",
    ), row=1, col=1)

    for y, txt, dash in ((imr.center, "média", "solid"),
                         (imr.ucl, "LSC", "dash"), (imr.lcl, "LIC", "dash")):
        if np.isfinite(y):
            fig.add_hline(y=y, line_color=INK_2 if dash == "solid" else MUTED,
                          line_width=1.4, line_dash=dash, row=1, col=1,
                          annotation_text=f"{txt}={_fmt(y, 3)}",
                          annotation_font=dict(size=10, color=INK_2),
                          annotation_position="right")

    if imr.violations:
        v_labels = [lb for lb in labels if lb in imr.violations]
        v_vals = [float(x.loc[lb]) for lb in v_labels]
        v_txt = ["<br>".join(RULE_LABELS[r] for r in imr.violations[lb])
                 for lb in v_labels]
        fig.add_trace(go.Scatter(
            x=v_labels, y=v_vals, mode="markers",
            marker=dict(size=11, color=RED, symbol="circle-open",
                        line=dict(width=2.5, color=RED)),
            name="causa especial", text=v_txt,
            hovertemplate="%{x}<br>valor: %{y:.4g}<br>%{text}<extra></extra>",
        ), row=1, col=1)

    # ---- painel 2: amplitudes móveis
    if len(mr):
        mr_labels = labels[1:]
        fig.add_trace(go.Scatter(
            x=mr_labels, y=mr, mode="lines+markers",
            line=dict(color=BLUE, width=1.4), marker=dict(size=4, color=BLUE),
            name="amplitude móvel", showlegend=False,
            hovertemplate="%{x}<br>AM: %{y:.4g}<extra></extra>",
        ), row=2, col=1)
        for y, txt in ((imr.mr_bar, "AM̄"), (imr.mr_ucl, "LSC")):
            if np.isfinite(y):
                fig.add_hline(y=y, line_color=MUTED, line_width=1.2,
                              line_dash="dash" if txt == "LSC" else "solid",
                              row=2, col=1,
                              annotation_text=f"{txt}={_fmt(y, 3)}",
                              annotation_font=dict(size=10, color=INK_2),
                              annotation_position="right")
        if imr.mr_violations:
            mv = [lb for lb in mr_labels if lb in imr.mr_violations]
            mv_vals = [float(mr[mr_labels.index(lb)]) for lb in mv]
            fig.add_trace(go.Scatter(
                x=mv, y=mv_vals, mode="markers",
                marker=dict(size=10, color=RED, symbol="circle-open",
                            line=dict(width=2.5, color=RED)),
                name="AM acima do limite", showlegend=False,
                hovertemplate="%{x}<br>AM: %{y:.4g}<br>acima do limite"
                              "<extra></extra>",
            ), row=2, col=1)

    n_viol = len(imr.violations)
    status = ("processo sob controle estatístico" if imr.in_control
              else f"{n_viol} ponto(s) com causa especial")
    layout = dict(_LAYOUT)
    layout["margin"] = dict(l=60, r=110, t=60, b=50)  # rótulos LSC/LIC à direita
    fig.update_layout(
        title=f"Carta I-AM — {rep.indicator}<br><sup>{status} · "
              f"σ curto prazo = {_fmt(imr.sigma_within, 4)}</sup>",
        height=560, showlegend=True,
        legend=dict(orientation="h", y=1.10, x=1, xanchor="right"),
        **layout,
    )
    _base_axes(fig)
    return fig


# ---------------------------------------------------- gráfico de probabilidade

def fig_qqplot(rep: CapabilityReport, transformed: bool = False) -> go.Figure:
    """Gráfico de probabilidade normal (QQ) com reta de referência."""
    if transformed and rep.transformed_series is not None:
        data = rep.transformed_series.dropna()
        title_extra = " (dados transformados)"
        norm = rep.normality_final
    else:
        data = rep.series.dropna()
        title_extra = ""
        norm = rep.normality_raw

    (osm, osr), (slope, intercept, r) = stats.probplot(
        data.to_numpy(dtype=float), dist="norm"
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(size=6, color=BLUE, opacity=0.7),
        name="dados",
        hovertemplate="quantil teórico: %{x:.3f}<br>observado: %{y:.4g}"
                      "<extra></extra>",
    ))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(
        x=line_x, y=slope * line_x + intercept, mode="lines",
        line=dict(color=BLUE_DARK, width=2), name="referência normal",
        hoverinfo="skip",
    ))
    p_txt = _fmt(norm.ad_p, 3) if norm else "—"
    fig.update_layout(
        title=(f"Gráfico de probabilidade normal — {rep.indicator}{title_extra}"
               f"<br><sup>Anderson-Darling p = {p_txt} · "
               f"correlação do ajuste = {_fmt(float(r), 3)} "
               "(pontos sobre a reta ⇒ dados normais)</sup>"),
        xaxis_title="quantis teóricos da normal",
        yaxis_title="valores observados",
        showlegend=False, **_LAYOUT,
    )
    _base_axes(fig)
    return fig


# ---------------------------------------------------------------- box-plot

def fig_boxplot(rep: CapabilityReport) -> go.Figure:
    """Box-plot com quartis, limites do usuário e limites sugeridos (Caso 3)."""
    x = rep.series.dropna()
    box: BoxStats | None = rep.box
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x, name=rep.indicator, boxpoints="outliers",
        marker=dict(color=BLUE, size=4, opacity=0.6),
        line=dict(color=BLUE_DARK, width=2),
        fillcolor="rgba(42,120,214,0.25)",
        hovertemplate="valor: %{x:.4g}<extra></extra>",
    ))

    if rep.lsl is not None:
        _spec_line(fig, rep.lsl, "LIE (atual)")
    if rep.usl is not None:
        _spec_line(fig, rep.usl, "LSE (atual)")

    # limites sugeridos anotados EMBAIXO para não colidir com os atuais (topo)
    sug: SuggestedLimits | None = rep.suggested
    if sug is not None:
        if sug.suggested_lsl is not None:
            _spec_line(fig, sug.suggested_lsl, "LIE sugerido", color=GREEN,
                       dash="dash", position="bottom")
        if sug.suggested_usl is not None:
            _spec_line(fig, sug.suggested_usl, "LSE sugerido", color=GREEN,
                       dash="dash", position="bottom")

    sub = ""
    if box is not None and box.n:
        sub = (f"Q1={_fmt(box.q1, 4)} · mediana={_fmt(box.median, 4)} · "
               f"Q3={_fmt(box.q3, 4)} · IQR={_fmt(box.iqr, 4)}")
        if rep.empirical:
            sub += f" · fora dos limites atuais: {_fmt(rep.empirical.get('pct_out'), 2)}%"
    fig.update_layout(
        title=f"Box-plot — {rep.indicator}<br><sup>{sub}</sup>",
        xaxis_title=rep.indicator, yaxis=dict(showticklabels=False),
        showlegend=False, height=380, **_LAYOUT,
    )
    _base_axes(fig)
    return fig
