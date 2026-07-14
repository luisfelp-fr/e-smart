"""Teste de normalidade orientado a capabilidade.

O teste decisório é o Anderson-Darling (sensível às caudas — exatamente a
região que importa para estimar frações fora dos limites). Shapiro-Wilk,
assimetria, curtose e a correlação do gráfico de probabilidade (PPCC)
entram como corroboração e para notas de "normalidade prática" em amostras
muito grandes, quando o p-valor rejeita desvios triviais.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad

ALPHA_DEFAULT = 0.05
BIG_N = 5000  # acima disso o p-valor fica hipersensível


@dataclass
class NormalityResult:
    """Resultado consolidado do teste de normalidade de uma série."""

    is_normal: bool = False
    ad_stat: float = np.nan
    ad_p: float = np.nan
    shapiro_p: float = np.nan
    skew: float = np.nan
    kurtosis: float = np.nan  # excesso de curtose (normal = 0)
    ppcc: float = np.nan  # correlação do gráfico de probabilidade
    n: int = 0
    alpha: float = ALPHA_DEFAULT
    practically_normal: bool = False
    note: str = ""


def test_normality(s: pd.Series, alpha: float = ALPHA_DEFAULT) -> NormalityResult:
    """Aplica Anderson-Darling (decisório) + corroborações a uma série."""
    res = NormalityResult(alpha=alpha)
    x = pd.Series(s).dropna().astype(float).to_numpy()
    res.n = len(x)
    if res.n < 8 or np.std(x) == 0:
        res.note = (
            "Amostra pequena ou sem variação — teste de normalidade não aplicável."
        )
        return res

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res.ad_stat, res.ad_p = (float(v) for v in normal_ad(x))
        except Exception:
            pass
        if 3 <= res.n <= 5000:
            try:
                res.shapiro_p = float(stats.shapiro(x)[1])
            except Exception:
                pass
        res.skew = float(stats.skew(x))
        res.kurtosis = float(stats.kurtosis(x))  # excesso (Fisher)
        try:
            (osm, osr), _ = stats.probplot(x, dist="norm")
            res.ppcc = float(np.corrcoef(osm, osr)[0, 1])
        except Exception:
            pass

    res.is_normal = bool(np.isfinite(res.ad_p) and res.ad_p > alpha)

    if res.n < 20:
        res.note = (
            f"Apenas {res.n} observações: o teste tem baixa potência e os "
            "índices de capabilidade serão instáveis."
        )
    elif res.n > BIG_N and not res.is_normal:
        # amostras enormes rejeitam desvios triviais; sinaliza quase-normalidade
        if res.ppcc >= 0.99 and abs(res.skew) < 0.5 and abs(res.kurtosis) < 1.0:
            res.practically_normal = True
            res.note = (
                "Com muitos dados o teste rejeita desvios mínimos; a forma da "
                "distribuição é praticamente normal (PPCC alto, assimetria e "
                "curtose baixas)."
            )
    return res
