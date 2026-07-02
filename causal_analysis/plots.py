"""Gráficos do relatório (matplotlib -> PNG base64 embutido no HTML).

Paleta e regras seguem um sistema validado para daltonismo e contraste:
cores categóricas em ordem fixa, escala divergente azul-cinza-vermelho para
polaridade, rampa ordinal de um só matiz para quartis, e um único eixo y por
gráfico (séries sobrepostas são padronizadas em z-score, nunca eixo duplo).
"""

from __future__ import annotations

import base64
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- paleta (modo claro, validada: CVD ΔE 24.2, banda de luminância ok) ------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # ordem fixa, nunca ciclada
POS, NEG = "#2a78d6", "#e34948"  # par divergente (polaridade)
NEUTRAL_MID = "#f0efec"
ORDINAL_4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # quartis Q1..Q4
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", [NEG, NEUTRAL_MID, POS]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "figure.dpi": 130,
    }
)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_ranking(scores: pd.DataFrame, top: int = 15) -> str:
    """Barras horizontais do score; a cor codifica a direção do efeito."""
    data = scores.head(top).iloc[::-1]
    colors = [
        POS if d > 0 else NEG if d < 0 else MUTED for d in data["direcao"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.42 * len(data) + 1.2)))
    bars = ax.barh(data["parametro"], data["score"], color=colors, height=0.62)
    for bar, val in zip(bars, data["score"]):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", ha="left", fontsize=8.5, color=INK_2,
        )
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score de culpabilidade (0–100)")
    ax.set_title("Ranking dos parâmetros por evidência de influência no alvo")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=POS),
        plt.Rectangle((0, 0), 1, 1, color=NEG),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
    ]
    ax.legend(
        handles,
        ["efeito positivo", "efeito negativo", "não-monotônico / indefinido"],
        loc="lower right", fontsize=8,
    )
    return _to_b64(fig)


def fig_corr_heatmap(df: pd.DataFrame, target: str) -> str:
    """Mapa de calor de Spearman (divergente: vermelho -1, cinza 0, azul +1)."""
    cols = [target] + [c for c in df.columns if c != target]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = df[cols].corr(method="spearman")
    n = len(cols)
    fig, ax = plt.subplots(figsize=(min(10, 1.5 + 0.55 * n), min(9, 1.2 + 0.5 * n)))
    im = ax.imshow(corr.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.columns, fontsize=8)
    # rótulos diretos nas células — a cor nunca carrega o valor sozinha
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=INK if abs(v) < 0.6 else "#ffffff",
                )
    ax.grid(visible=False)
    ax.set_title(f"Correlação de Spearman entre parâmetros e alvo ({target})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=8, color=MUTED)
    cbar.outline.set_visible(False)
    return _to_b64(fig)


def fig_lag_profile(name: str, lag_profile: dict[int, float],
                    rolling_profile: dict[int, float]) -> str:
    """Perfil de correlação por defasagem e por janela de média móvel."""
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 2.7),
        gridspec_kw={"width_ratios": [3, 1.2]}, sharey=True,
    )
    lags = sorted(lag_profile)
    vals = [lag_profile[k] for k in lags]
    ax = axes[0]
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.plot(lags, vals, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    if vals:
        i_best = int(np.nanargmax(np.abs(vals)))
        ax.plot(lags[i_best], vals[i_best], "o", markersize=9,
                markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.4)
        ax.annotate(
            f"lag {lags[i_best]}: ρ={vals[i_best]:.2f}",
            (lags[i_best], vals[i_best]), textcoords="offset points",
            xytext=(8, 8), fontsize=8, color=INK_2,
        )
    ax.set_xlabel("Defasagem (períodos)")
    ax.set_ylabel("Spearman ρ com o alvo")
    ax.set_ylim(-1, 1)
    ax.set_title(f"{name} — efeito por defasagem")

    ax2 = axes[1]
    ax2.axhline(0, color=BASELINE, linewidth=0.8)
    wins = sorted(rolling_profile)
    if wins:
        ax2.bar([str(w) for w in wins], [rolling_profile[w] for w in wins],
                color=SERIES[0], width=0.55)
    ax2.set_xlabel("Janela da média móvel")
    ax2.set_ylim(-1, 1)
    ax2.set_title("médias móveis")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _to_b64(fig)


def fig_scatter(x: pd.Series, y: pd.Series, name: str, target: str,
                transform_label: str) -> str:
    """Dispersão parâmetro (melhor transformação) vs. alvo, com tendência LOWESS."""
    m = x.notna() & y.notna()
    xa, ya = x[m].to_numpy(float), y[m].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.scatter(xa, ya, s=14, color=SERIES[0], alpha=0.35, edgecolors="none")
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smooth = lowess(ya, xa, frac=0.4, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color="#104281", linewidth=2.2,
                label="tendência (LOWESS)")
        ax.legend(fontsize=8, loc="best")
    except Exception:
        pass
    ax.set_xlabel(f"{name} ({transform_label})")
    ax.set_ylabel(target)
    ax.set_title("Forma da relação")
    return _to_b64(fig)


def fig_quartile_box(x: pd.Series, y: pd.Series, name: str, target: str) -> str:
    """Boxplot do alvo por quartil do parâmetro (rampa ordinal de um matiz)."""
    m = x.notna() & y.notna()
    xa, ya = x[m], y[m]
    try:
        qbins = pd.qcut(xa, 4, labels=["Q1\n(baixo)", "Q2", "Q3", "Q4\n(alto)"],
                        duplicates="drop")
    except ValueError:
        qbins = None
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    if qbins is not None:
        labels = [str(c) for c in qbins.cat.categories]
        groups = [ya[qbins == c].to_numpy() for c in qbins.cat.categories]
        bp = ax.boxplot(
            groups, tick_labels=labels, patch_artist=True, widths=0.55,
            medianprops={"color": INK, "linewidth": 1.6},
            whiskerprops={"color": MUTED}, capprops={"color": MUTED},
            flierprops={"marker": ".", "markersize": 4,
                        "markerfacecolor": MUTED, "markeredgecolor": "none"},
        )
        for patch, color in zip(bp["boxes"], ORDINAL_4):
            patch.set_facecolor(color)
            patch.set_edgecolor(SURFACE)
    ax.set_xlabel(f"Quartis de {name}")
    ax.set_ylabel(target)
    ax.set_title("Alvo por faixa de percentil do parâmetro")
    ax.grid(axis="x", visible=False)
    return _to_b64(fig)


def fig_timeseries_overlay(dates: pd.Index, y: pd.Series, x: pd.Series,
                           target: str, name: str) -> str:
    """Alvo e parâmetro padronizados (z-score) no tempo — um único eixo."""
    def z(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std and np.isfinite(std) else s * 0

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.plot(dates, z(y), color=SERIES[0], linewidth=1.8, label=f"{target} (alvo)")
    ax.plot(dates, z(x), color=SERIES[1], linewidth=1.4, alpha=0.9, label=name)
    ax.set_ylabel("z-score (padronizado)")
    ax.set_title("Evolução temporal comparada")
    ax.legend(fontsize=8, loc="upper right", ncols=2)
    fig.autofmt_xdate(rotation=0, ha="center")
    return _to_b64(fig)
