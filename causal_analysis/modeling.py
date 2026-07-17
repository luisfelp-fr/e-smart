"""Importância não-linear via Random Forest com validação temporal.

A floresta enxerga todas as transformações (bruto, lags, médias móveis) de
todos os parâmetros ao mesmo tempo, capturando não-linearidades e interações.
A importância por permutação é medida somente em blocos de teste futuros
(TimeSeriesSplit), evitando o vazamento temporal de um shuffle aleatório.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from .features import base_param, build_matrix, feature_label

RNG_SEED = 42


@dataclass
class MLResult:
    r2_oos: float = np.nan  # R² médio fora da amostra
    r2_folds: list[float] = field(default_factory=list)
    n_features: int = 0
    n_obs: int = 0
    # por parâmetro-base: importância agregada e melhor transformação
    importance: dict[str, float] = field(default_factory=dict)
    importance_std: dict[str, float] = field(default_factory=dict)
    top_feature: dict[str, str] = field(default_factory=dict)
    skipped_reason: str | None = None


def _adaptive_effort(n_obs: int, n_features: int,
                     n_estimators: int) -> tuple[int, int]:
    """Escalona (n_estimators, n_repeats) pelo tamanho do problema.

    O custo da importância por permutação cresce com obs × features × árvores
    × repetições. Com muitos indicadores (⇒ muitas features) ou muitas linhas,
    reduzimos árvores e repetições para manter o tempo tratável no Streamlit,
    sem alterar a natureza do resultado (ranking por importância).
    """
    work = n_obs * n_features
    if work > 4_000_000:      # ex.: 15k linhas × 540 features
        return min(n_estimators, 120), 2
    if work > 1_500_000:
        return min(n_estimators, 200), 3
    return n_estimators, 5


def ml_importance(
    df: pd.DataFrame,
    target: str,
    max_lag: int,
    windows: list[int],
    n_splits: int = 4,
    n_estimators: int = 300,
) -> MLResult:
    """Treina RF em janelas crescentes e agrega importâncias por parâmetro."""
    res = MLResult()
    X, y, groups = build_matrix(df, target, max_lag, windows)
    res.n_features, res.n_obs = X.shape[1], len(X)
    if len(X) < 60:
        res.skipped_reason = (
            f"apenas {len(X)} observações completas após criar lags/médias "
            "móveis (mínimo: 60)"
        )
        return res

    trees, n_repeats = _adaptive_effort(len(X), X.shape[1], n_estimators)
    # problemas grandes: menos folds também (cada fold treina uma floresta)
    if len(X) * X.shape[1] > 4_000_000:
        n_splits = min(n_splits, 3)
    n_splits = min(n_splits, max(2, len(X) // 40))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    imp_acc = np.zeros(X.shape[1])
    imp_sq = np.zeros(X.shape[1])
    n_folds_used = 0

    for train_idx, test_idx in splitter.split(X):
        if len(test_idx) < 15:
            continue
        model = RandomForestRegressor(
            n_estimators=trees,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RNG_SEED,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            res.r2_folds.append(float(r2_score(y.iloc[test_idx], pred)))
            perm = permutation_importance(
                model,
                X.iloc[test_idx],
                y.iloc[test_idx],
                n_repeats=n_repeats,
                random_state=RNG_SEED,
                n_jobs=-1,
            )
        imp_acc += perm.importances_mean
        imp_sq += perm.importances_mean**2
        n_folds_used += 1

    if not n_folds_used:
        res.skipped_reason = "nenhum bloco de validação temporal com tamanho suficiente"
        return res

    res.r2_oos = float(np.mean(res.r2_folds))
    mean_imp = imp_acc / n_folds_used
    std_imp = np.sqrt(np.maximum(0.0, imp_sq / n_folds_used - mean_imp**2))
    per_feature = pd.Series(mean_imp, index=X.columns).clip(lower=0.0)
    per_feature_std = pd.Series(std_imp, index=X.columns)

    for param, cols in groups.items():
        fam = per_feature[cols]
        res.importance[param] = float(fam.sum())
        res.importance_std[param] = float(per_feature_std[cols].mean())
        res.top_feature[param] = feature_label(str(fam.idxmax())) if fam.max() > 0 else "—"

    # sanidade: toda coluna derivada deve pertencer a um parâmetro conhecido
    assert all(base_param(c) in groups for c in X.columns)
    return res
