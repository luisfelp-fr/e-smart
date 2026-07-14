"""Orquestração: carrega dados, roda a bateria de testes e agrega scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import LoadDiagnostics, load_table
from .modeling import MLResult, ml_importance
from .scoring import score_parameters
from .stats_tests import (
    correlations,
    distance_correlation,
    fdr_adjust,
    granger_causality,
    ljung_box,
    mutual_information,
    percentile_effect,
    scan_transforms,
)

DEFAULT_WINDOWS = [3, 7, 14]


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    target: str
    params: list[str]
    diagnostics: LoadDiagnostics
    per_param: dict[str, dict] = field(default_factory=dict)
    fdr: dict[str, dict] = field(default_factory=dict)
    ml: MLResult | None = None
    scores: pd.DataFrame | None = None
    max_lag: int = 14
    windows: list[int] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    alpha: float = 0.05
    target_ljungbox: dict | None = None  # estrutura temporal do alvo


def run_analysis(
    path: str,
    target: str,
    date_col: str | None = None,
    max_lag: int = 14,
    windows: list[int] | None = None,
    alpha: float = 0.05,
    sep: str | None = None,
    sheet: int | str = 0,
    verbose: bool = True,
) -> AnalysisResult:
    """Executa a análise completa a partir de um arquivo."""
    df, diag = load_table(path, target=target, date_col=date_col, sep=sep, sheet=sheet)
    return analyze_dataframe(
        df, target, diagnostics=diag, max_lag=max_lag, windows=windows,
        alpha=alpha, verbose=verbose,
    )


def analyze_dataframe(
    df: pd.DataFrame,
    target: str,
    diagnostics: LoadDiagnostics | None = None,
    max_lag: int = 14,
    windows: list[int] | None = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> AnalysisResult:
    """Executa a análise sobre um DataFrame já carregado/alinhado.

    Usado pela interface para dados de múltiplas abas (já combinados na
    grade do alvo) ou após tratamentos escolhidos pelo usuário.
    """
    windows = windows or list(DEFAULT_WINDOWS)

    def log(msg: str) -> None:
        if verbose:
            print(f"  • {msg}")

    diag = diagnostics or LoadDiagnostics(
        n_rows_raw=len(df), n_rows_used=len(df),
        date_start=str(df.index.min()), date_end=str(df.index.max()),
    )
    df = df[df[target].notna()]
    params = [c for c in df.columns if c != target]
    if not params:
        raise ValueError("Nenhum parâmetro numérico restou além do alvo.")

    # lags maiores que a série suporta só geram NaN
    max_lag = int(min(max_lag, max(1, len(df) // 5)))
    windows = [w for w in windows if w < len(df) // 3] or [3]

    log(
        f"{len(df)} observações de {diag.date_start} a {diag.date_end}; "
        f"alvo '{target}' e {len(params)} parâmetros."
    )

    result = AnalysisResult(
        df=df, target=target, params=params, diagnostics=diag,
        max_lag=max_lag, windows=windows, alpha=alpha,
    )
    y = df[target]

    log("Testes de associação (Pearson, Spearman, Kendall, dCor, MI)...")
    for p in params:
        x = df[p]
        r: dict = correlations(x, y)
        r["dcor"] = distance_correlation(x, y)
        r["mi"], r["mi_r"] = mutual_information(x, y)
        result.per_param[p] = r

    log(f"Varredura de lags (0..{max_lag}) e médias móveis {windows}...")
    for p in params:
        scan = scan_transforms(df[p], y, max_lag, windows)
        result.per_param[p].update(
            best_label=scan.best_label,
            best_feature=scan.best_feature,
            best_rho=scan.best_rho,
            best_p=scan.best_p,
            lag_profile=scan.lag_profile,
            rolling_profile=scan.rolling_profile,
        )

    log("Estrutura temporal (Ljung-Box) e causalidade de Granger (ADF)...")
    result.target_ljungbox = ljung_box(y, lags=max_lag)
    for p in params:
        result.per_param[p]["ljungbox"] = ljung_box(df[p], lags=max_lag)
        result.per_param[p]["granger"] = granger_causality(df[p], y, max_lag)

    log("Análise por percentis (alto vs. baixo, quartis)...")
    for p in params:
        result.per_param[p]["percentile"] = percentile_effect(df[p], y)

    log("Random Forest + importância por permutação (validação temporal)...")
    result.ml = ml_importance(df, target, max_lag, windows)
    for p in params:
        result.per_param[p]["ml_importance"] = result.ml.importance.get(p, np.nan)
        result.per_param[p]["ml_top_feature"] = result.ml.top_feature.get(p, "—")
    if result.ml.skipped_reason:
        log(f"ML pulado: {result.ml.skipped_reason}")

    log("Correção de múltiplos testes (FDR Benjamini-Hochberg)...")
    pp = result.per_param
    result.fdr = {
        "Pearson": fdr_adjust({p: pp[p]["pearson"][1] for p in params}, alpha),
        "Spearman": fdr_adjust({p: pp[p]["spearman"][1] for p in params}, alpha),
        "Melhor lag/média móvel": fdr_adjust(
            {p: pp[p].get("best_p") for p in params}, alpha
        ),
        "Granger": fdr_adjust(
            {p: (pp[p]["granger"] or {}).get("p_value") for p in params}, alpha
        ),
        "Mann-Whitney (P75 vs P25)": fdr_adjust(
            {p: (pp[p]["percentile"] or {}).get("p_mannwhitney") for p in params}, alpha
        ),
        "Kruskal-Wallis (quartis)": fdr_adjust(
            {p: (pp[p]["percentile"] or {}).get("p_kruskal") for p in params}, alpha
        ),
    }

    result.scores = score_parameters(pp, result.fdr, alpha)
    log("Análise concluída.")
    return result
