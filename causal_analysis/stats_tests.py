"""Bateria de testes estatísticos de associação e precedência temporal.

Nenhum teste isolado prova causalidade; o conjunto (associação linear,
monotônica e não-linear + precedência temporal via Granger/lags + contraste de
percentis) forma o corpo de evidência que o módulo ``scoring`` agrega.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from .features import derived_features, feature_label

RNG_SEED = 42
DCOR_MAX_N = 600  # matriz O(n²): subamostra acima disso


# ---------------------------------------------------------------- associação

def _aligned(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    m = x.notna() & y.notna()
    return x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float)


def correlations(x: pd.Series, y: pd.Series) -> dict:
    """Pearson (linear), Spearman (monotônica) e Kendall (ordinal robusta)."""
    xa, ya = _aligned(x, y)
    out = {}
    if len(xa) < 8 or np.std(xa) == 0 or np.std(ya) == 0:
        for k in ("pearson", "spearman", "kendall"):
            out[k] = (np.nan, np.nan)
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = stats.pearsonr(xa, ya)
        sp = stats.spearmanr(xa, ya)
        kd = stats.kendalltau(xa, ya)
    out["pearson"] = (float(pr[0]), float(pr[1]))
    out["spearman"] = (float(sp[0]), float(sp[1]))
    out["kendall"] = (float(kd[0]), float(kd[1]))
    return out


def distance_correlation(x: pd.Series, y: pd.Series) -> float:
    """Correlação de distância (Székely): 0 apenas sob independência.

    Captura dependências não-lineares e não-monotônicas (ex.: forma de U)
    que Pearson/Spearman não enxergam.
    """
    xa, ya = _aligned(x, y)
    n = len(xa)
    if n < 10:
        return float("nan")
    if n > DCOR_MAX_N:
        idx = np.random.default_rng(RNG_SEED).choice(n, DCOR_MAX_N, replace=False)
        xa, ya = xa[idx], ya[idx]
    a = np.abs(xa[:, None] - xa[None, :])
    b = np.abs(ya[:, None] - ya[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0 or dcov2 <= 0:
        return 0.0
    return float(np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y)))


def mutual_information(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Informação mútua (nats) e seu equivalente em correlação [0, 1].

    Para gaussianas, MI = -½·ln(1-ρ²); invertendo, r_eq = sqrt(1-e^{-2·MI}),
    o que coloca a MI numa escala comparável às correlações.
    """
    xa, ya = _aligned(x, y)
    if len(xa) < 20:
        return float("nan"), float("nan")
    mi = float(
        mutual_info_regression(
            xa.reshape(-1, 1), ya, random_state=RNG_SEED, n_neighbors=5
        )[0]
    )
    r_eq = float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi))))
    return mi, r_eq


# ------------------------------------------------- lags e transformações

@dataclass
class TransformScan:
    """Melhor transformação temporal (lag/média móvel) de um parâmetro."""

    lag_profile: dict[int, float] = field(default_factory=dict)  # lag -> rho
    rolling_profile: dict[int, float] = field(default_factory=dict)  # janela -> rho
    best_label: str = "bruto (sem defasagem)"
    best_feature: str = ""
    best_rho: float = np.nan
    best_p: float = np.nan


def scan_transforms(
    x: pd.Series, y: pd.Series, max_lag: int, windows: list[int]
) -> TransformScan:
    """Varre lags 0..max_lag e médias móveis, guardando o perfil de Spearman.

    A transformação com maior |rho| indica *quando* (e de que forma
    acumulada) o parâmetro mais se associa ao alvo.
    """
    scan = TransformScan(best_feature=str(x.name))
    fam = derived_features(x, max_lag, windows)
    best_abs = -1.0
    for col in fam.columns:
        xa, ya = _aligned(fam[col], y)
        if len(xa) < 8 or np.std(xa) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p = stats.spearmanr(xa, ya)
        rho, p = float(rho), float(p)
        label = feature_label(col)
        if label.startswith("lag"):
            scan.lag_profile[int(label.split()[1])] = rho
        elif label.startswith("média"):
            scan.rolling_profile[int(label.split()[2])] = rho
        else:
            scan.lag_profile[0] = rho
        if np.isfinite(rho) and abs(rho) > best_abs:
            best_abs = abs(rho)
            scan.best_label = label
            scan.best_feature = col
            scan.best_rho = rho
            scan.best_p = p
    return scan


# ----------------------------------------------------- causalidade de Granger

def _make_stationary(s: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int]:
    """Diferencia a série até o teste ADF rejeitar raiz unitária (p<=0.05)."""
    cur = s.dropna()
    for d in range(max_diff + 1):
        if len(cur) < 20 or cur.nunique() <= 2:
            return cur, d
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = adfuller(cur, autolag="AIC")[1]
        except Exception:
            return cur, d
        if p <= 0.05:
            return cur, d
        cur = cur.diff().dropna()
    return cur, max_diff


def granger_causality(x: pd.Series, y: pd.Series, max_lag: int) -> dict | None:
    """Testa se o passado de x ajuda a prever y além do passado do próprio y.

    Ambas as séries são tornadas estacionárias (ADF + diferenciação) antes do
    teste — pré-requisito do Granger. Devolve o menor p-valor entre os lags e
    o lag correspondente, ou None se o teste não for aplicável.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return None
    ys, dy = _make_stationary(df["y"])
    xs, dx = _make_stationary(df["x"])
    d = max(dy, dx)
    data = pd.concat(
        [df["y"].diff(d) if d else df["y"], df["x"].diff(d) if d else df["x"]],
        axis=1,
    ).dropna()
    m = min(max_lag, max(1, (len(data) - 12) // 4), 12)
    if m < 1 or len(data) < 4 * m + 12:
        return None
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data[["y", "x"]], maxlag=m)
    except Exception:
        return None
    pvals = {lag: float(r[0]["ssr_ftest"][1]) for lag, r in res.items()}
    best_lag = min(pvals, key=pvals.get)
    return {
        "p_value": pvals[best_lag],
        "best_lag": int(best_lag),
        "diffs_applied": int(d),
        "lags_tested": int(m),
        "p_by_lag": pvals,
    }


# ------------------------------------------------------- análise de percentis

def percentile_effect(
    x: pd.Series, y: pd.Series, low_q: float = 0.25, high_q: float = 0.75
) -> dict | None:
    """Contrasta o alvo quando o parâmetro está alto (>=P75) vs. baixo (<=P25).

    - Mann-Whitney U entre os dois grupos (não assume normalidade);
    - delta de Cliff como tamanho de efeito (-1..1);
    - Kruskal-Wallis entre os quartis (detecta efeitos não-monotônicos);
    - deslocamento de mediana normalizado pelo IQR do alvo.
    """
    xa, ya = _aligned(x, y)
    if len(xa) < 24:
        return None
    lo_cut, hi_cut = np.quantile(xa, [low_q, high_q])
    y_low = ya[xa <= lo_cut]
    y_high = ya[xa >= hi_cut]
    if len(y_low) < 5 or len(y_high) < 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_stat, p_mw = stats.mannwhitneyu(y_high, y_low, alternative="two-sided")
    cliffs = float(2.0 * u_stat / (len(y_high) * len(y_low)) - 1.0)

    # quartis para Kruskal-Wallis (captura U invertido etc.)
    p_kw = np.nan
    try:
        qbins = pd.qcut(pd.Series(xa), 4, labels=False, duplicates="drop")
        groups = [ya[qbins.to_numpy() == g] for g in np.unique(qbins.dropna())]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_kw = float(stats.kruskal(*groups)[1])
    except Exception:
        pass

    iqr = float(np.subtract(*np.percentile(ya, [75, 25]))) or float(np.std(ya)) or 1.0
    med_low, med_high = float(np.median(y_low)), float(np.median(y_high))
    return {
        "p_mannwhitney": float(p_mw),
        "p_kruskal": p_kw,
        "cliffs_delta": cliffs,
        "median_low": med_low,
        "median_high": med_high,
        "median_shift_iqr": (med_high - med_low) / iqr,
        "low_cut": float(lo_cut),
        "high_cut": float(hi_cut),
    }


# --------------------------------------------------- estrutura temporal

def ljung_box(x: pd.Series, lags: int = 10) -> dict | None:
    """Teste de Ljung-Box: a série tem autocorrelação (estrutura temporal)?

    p pequeno ⇒ o passado da série carrega informação — pré-requisito para
    efeitos com defasagem (lag) fazerem sentido. Aplicado ao alvo, valida a
    própria busca por lags; aplicado a um parâmetro, indica persistência
    (o valor de agora "dura" vários períodos).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    xa = pd.Series(x).dropna().astype(float)
    if len(xa) < 20 or xa.nunique() <= 2:
        return None
    m = int(min(lags, len(xa) // 5))
    if m < 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = acorr_ljungbox(xa, lags=m, return_df=True)
    except Exception:
        return None
    # a estatística LB no lag m testa CONJUNTAMENTE os lags 1..m; o p do
    # maior lag é a decisão única correta (o mínimo entre lags, sem correção,
    # dispararia falsos positivos em ruído branco)
    p_joint = float(res["lb_pvalue"].iloc[-1])
    best = int(res["lb_pvalue"].idxmin())
    return {
        "p_value": p_joint,
        "best_lag": best,
        "lags_tested": m,
        "has_structure": p_joint <= 0.05,
    }


# ------------------------------------------------------------ múltiplos testes

def fdr_adjust(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Correção Benjamini-Hochberg dentro de uma família de testes.

    Recebe {parâmetro: p} e devolve {parâmetro: {p, p_adj, significant}}.
    """
    keys = [k for k, v in pvalues.items() if v is not None and np.isfinite(v)]
    if not keys:
        return {}
    raw = [pvalues[k] for k in keys]
    rejected, adj, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")
    return {
        k: {"p": raw[i], "p_adj": float(adj[i]), "significant": bool(rejected[i])}
        for i, k in enumerate(keys)
    }
