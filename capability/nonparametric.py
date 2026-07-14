"""Caso 3 — análise não-paramétrica por box-plot e sugestão de limites.

Quando nem os dados brutos nem transformações atingem normalidade, a
capabilidade clássica (que assume o sino) deixa de valer. Este módulo
descreve a distribuição pelos quartis/percentis empíricos e sugere limites
de atuação realistas.

Racional da sugestão (correção importante): quartis descrevem o CENTRO da
distribuição; um limite garantível vive na CAUDA do lado restringido.

- bilateral: a faixa típica de operação é Q1–Q3 (50% central), mas um limite
  que o processo consiga cumprir ~99,7% do tempo são os percentis externos
  P0,135 e P99,865 (análogo não-paramétrico de µ ± 3σ);
- só limite inferior ("quanto maior, melhor"): o risco é a cauda BAIXA — o
  piso garantível é um percentil baixo (P0,135; na prática P1/P5). Q3 como
  piso reprovaria ~75% da produção; se desejado, Q3 é apenas meta;
- só limite superior ("quanto menor, melhor"): o teto garantível é um
  percentil ALTO (P99,865; na prática P95/P99).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BoxStats:
    """Resumo de box-plot e percentis empíricos de uma série."""

    n: int = 0
    minimum: float = np.nan
    q1: float = np.nan
    median: float = np.nan
    q3: float = np.nan
    maximum: float = np.nan
    iqr: float = np.nan
    whisker_low: float = np.nan   # cerca de Tukey inferior (Q1 - 1,5·IQR)
    whisker_high: float = np.nan  # cerca de Tukey superior (Q3 + 1,5·IQR)
    p0135: float = np.nan
    p1: float = np.nan
    p5: float = np.nan
    p95: float = np.nan
    p99: float = np.nan
    p99865: float = np.nan


@dataclass
class SuggestedLimits:
    """Limites de atuação sugeridos a partir dos percentis empíricos."""

    sided: str = "bilateral"        # 'bilateral' | 'superior' | 'inferior'
    suggested_lsl: float | None = None
    suggested_usl: float | None = None
    # alternativa mais conservadora/prática (P5 / P95)
    practical_lsl: float | None = None
    practical_usl: float | None = None
    coverage_pct: float = np.nan    # % dos dados dentro dos limites sugeridos
    rationale: str = ""
    notes: list[str] = field(default_factory=list)


def boxplot_summary(s: pd.Series) -> BoxStats:
    """Quartis, cercas de Tukey e percentis extremos empíricos da série."""
    res = BoxStats()
    x = pd.Series(s).dropna().astype(float)
    res.n = len(x)
    if res.n < 4:
        return res
    (res.p0135, res.p1, res.p5, res.q1, res.median,
     res.q3, res.p95, res.p99, res.p99865) = (
        float(v) for v in np.quantile(
            x, [0.00135, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.99865]
        )
    )
    res.minimum, res.maximum = float(x.min()), float(x.max())
    res.iqr = res.q3 - res.q1
    res.whisker_low = res.q1 - 1.5 * res.iqr
    res.whisker_high = res.q3 + 1.5 * res.iqr
    return res


def suggested_limits(
    s: pd.Series, lsl: float | None, usl: float | None
) -> SuggestedLimits:
    """Sugere limites realistas conforme o tipo de especificação do usuário."""
    box = boxplot_summary(s)
    x = pd.Series(s).dropna().astype(float)
    res = SuggestedLimits()
    if box.n < 10:
        res.notes.append("Poucos dados para sugerir limites com segurança.")
        return res

    if lsl is not None and usl is not None:
        res.sided = "bilateral"
        res.suggested_lsl, res.suggested_usl = box.p0135, box.p99865
        res.practical_lsl, res.practical_usl = box.p5, box.p95
        res.rationale = (
            "A faixa típica de operação (50% central dos dados) vai de "
            f"Q1 = {box.q1:.4g} a Q3 = {box.q3:.4g} (mediana {box.median:.4g}). "
            "Para limites que o processo cumpra ~99,7% do tempo, use os "
            f"percentis externos P0,135 = {box.p0135:.4g} e "
            f"P99,865 = {box.p99865:.4g}; uma alternativa prática (95% de "
            f"cobertura) é P5 = {box.p5:.4g} a P95 = {box.p95:.4g}."
        )
    elif lsl is not None:
        # só limite inferior => quanto maior, melhor => risco na cauda baixa
        res.sided = "inferior"
        res.suggested_lsl = box.p0135
        res.practical_lsl = box.p5
        res.rationale = (
            "Com limite apenas inferior (quanto maior, melhor), o risco está "
            "na cauda baixa: o piso que o processo consegue garantir é um "
            f"percentil baixo — P0,135 = {box.p0135:.4g} (~99,9% acima) ou, "
            f"na prática, P5 = {box.p5:.4g} (95% acima). "
            f"Q3 = {box.q3:.4g} serviria apenas como meta de melhoria "
            "(75% da produção ficaria abaixo dele)."
        )
    elif usl is not None:
        # só limite superior => quanto menor, melhor => risco na cauda alta
        res.sided = "superior"
        res.suggested_usl = box.p99865
        res.practical_usl = box.p95
        res.rationale = (
            "Com limite apenas superior (quanto menor, melhor), o risco está "
            "na cauda alta: o teto que o processo consegue garantir é um "
            f"percentil alto — P99,865 = {box.p99865:.4g} (~99,9% abaixo) ou, "
            f"na prática, P95 = {box.p95:.4g} (95% abaixo). "
            f"Q1 = {box.q1:.4g} serviria apenas como meta de melhoria."
        )
    else:
        res.notes.append("Nenhum limite informado — nada a sugerir.")
        return res

    inside = pd.Series(True, index=x.index)
    if res.suggested_lsl is not None:
        inside &= x >= res.suggested_lsl
    if res.suggested_usl is not None:
        inside &= x <= res.suggested_usl
    res.coverage_pct = float(100.0 * inside.mean())
    return res


def empirical_nonconformance(
    s: pd.Series, lsl: float | None, usl: float | None
) -> dict:
    """PPM fora dos limites contado diretamente nos dados (sem modelo)."""
    x = pd.Series(s).dropna().astype(float)
    if len(x) == 0:
        return {"ppm_below": np.nan, "ppm_above": np.nan, "ppm_total": np.nan,
                "pct_out": np.nan, "n": 0}
    below = float((x < lsl).mean()) if lsl is not None else 0.0
    above = float((x > usl).mean()) if usl is not None else 0.0
    return {
        "ppm_below": 1e6 * below,
        "ppm_above": 1e6 * above,
        "ppm_total": 1e6 * (below + above),
        "pct_out": 100.0 * (below + above),
        "n": int(len(x)),
    }
