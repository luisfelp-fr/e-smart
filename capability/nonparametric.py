"""Caso 3 — análise não-paramétrica por box-plot e sugestão de limites.

Quando nem os dados brutos nem transformações atingem normalidade, a
capabilidade clássica (que assume o sino) deixa de valer. Este módulo
descreve a distribuição pelos quartis/percentis empíricos e sugere limites
de atuação.

Racional da sugestão (regra definida pelo usuário do processo): os limites
de atuação são ancorados nos QUARTIS do box-plot — metas de operação
baseadas no comportamento real do indicador, não pisos/tetos garantíveis.

- só limite inferior ("quanto maior, melhor"): LIE sugerido = Q3 — operar
  acima do terceiro quartil, isto é, entre os 25% melhores valores
  históricos;
- bilateral: faixa sugerida de Q2 (mediana) a Q3 — a metade superior do
  comportamento típico do processo;
- só limite superior ("quanto menor, melhor"): LSE sugerido = Q1 (espelho
  da regra do limite inferior).

Os percentis de cauda (P0,135/P99,865 e P5/P95) continuam informados no
texto como referência de cobertura: são os valores que o processo ATUAL já
cumpre em ~99,7% / 95% do tempo.
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
        res.suggested_lsl, res.suggested_usl = box.median, box.q3
        res.practical_lsl, res.practical_usl = box.p5, box.p95
        res.rationale = (
            "Para indicador com os dois limites, a faixa de atuação sugerida "
            f"vai da mediana (Q2) = {box.median:.4g} ao terceiro quartil "
            f"(Q3) = {box.q3:.4g} — a metade superior do comportamento típico "
            "do processo (meta de operação, não piso/teto garantido). "
            "Como referência de cobertura do processo atual: "
            f"P5 = {box.p5:.4g} a P95 = {box.p95:.4g} contém 95% dos dados, e "
            f"P0,135 = {box.p0135:.4g} a P99,865 = {box.p99865:.4g}, ~99,7%."
        )
    elif lsl is not None:
        # só limite inferior => quanto maior, melhor => atuação = Q3
        res.sided = "inferior"
        res.suggested_lsl = box.q3
        res.practical_lsl = box.p5
        res.rationale = (
            "Com limite apenas inferior (quanto maior, melhor), o limite de "
            f"atuação sugerido é o terceiro quartil Q3 = {box.q3:.4g}: operar "
            "acima dele significa ficar entre os 25% melhores valores "
            "históricos do indicador (meta de operação). Como referência de "
            f"cobertura, o processo atual já garante P5 = {box.p5:.4g} "
            f"(95% dos dados acima) e P0,135 = {box.p0135:.4g} (~99,9% acima)."
        )
    elif usl is not None:
        # só limite superior => quanto menor, melhor => atuação = Q1 (espelho)
        res.sided = "superior"
        res.suggested_usl = box.q1
        res.practical_usl = box.p95
        res.rationale = (
            "Com limite apenas superior (quanto menor, melhor), o limite de "
            f"atuação sugerido é o primeiro quartil Q1 = {box.q1:.4g} "
            "(espelho da regra do limite inferior): operar abaixo dele "
            "significa ficar entre os 25% melhores valores históricos (meta "
            "de operação). Como referência de cobertura, o processo atual já "
            f"garante P95 = {box.p95:.4g} (95% dos dados abaixo) e "
            f"P99,865 = {box.p99865:.4g} (~99,9% abaixo)."
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
