"""Índices de capabilidade (Cp/Cpk/Pp/Ppk), PPM e vereditos amigáveis.

Dois desvios-padrão alimentam índices diferentes:

- σ_dentro = AM̄/d2 (curto prazo, da carta I-AM) → Cp/Cpk, o "potencial"
  do processo se ele se mantiver estável;
- σ_global = desvio amostral (ddof=1)          → Pp/Ppk, o "desempenho"
  realmente entregue no período, incluindo instabilidades.

Limites unilaterais são suportados: só LSE → Cpk = CPU; só LIE → Cpk = CPL;
Cp/Pp (que precisam dos dois lados) ficam indefinidos.

O método dos percentis (para distribuições não-normais, Caso 3) substitui
µ ± 3σ pelos percentis empíricos P0,135 / P50 / P99,865:
Ppu = (LSE − P50)/(P99,865 − P50) e Ppl = (P50 − LIE)/(P50 − P0,135).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# limiares clássicos de julgamento do Cpk
VERDICTS = [
    (1.67, "excelente", "Processo excelente: capaz com folga."),
    (1.33, "capaz", "Processo capaz de atender aos limites."),
    (1.00, "marginal", "Processo marginal: atende no limite — monitorar de perto."),
    (-np.inf, "incapaz", "Processo não capaz de atender aos limites definidos."),
]

TOOLTIPS = {
    "cp": (
        "Cp — capabilidade potencial: quantas vezes a faixa de especificação "
        "(LSE−LIE) contém a variação natural do processo (6·σ de curto prazo). "
        "Ignora o centramento: só diz se o processo CABERIA nos limites."
    ),
    "cpk": (
        "Cpk — capabilidade real de curto prazo: distância da média ao limite "
        "mais próximo, em unidades de 3·σ de curto prazo. Considera o "
        "centramento. Cpk ≥ 1,33 costuma indicar processo capaz."
    ),
    "pp": (
        "Pp — desempenho potencial: como o Cp, mas usando a variação total do "
        "período (σ global), incluindo instabilidades de longo prazo."
    ),
    "ppk": (
        "Ppk — desempenho real: como o Cpk, mas com a variação total do "
        "período. Se Cpk ≫ Ppk, o processo é instável ao longo do tempo."
    ),
    "sigma_within": (
        "σ de curto prazo — estimado pela amplitude móvel média da carta I-AM "
        "(AM̄/1,128). Reflete a variação 'ponto a ponto' do processo."
    ),
    "sigma_overall": (
        "σ global — desvio-padrão de todos os dados do período. Inclui "
        "deslocamentos e tendências, por isso costuma ser maior que o de "
        "curto prazo."
    ),
    "ppm": (
        "PPM — partes por milhão fora dos limites, estimadas pelo modelo "
        "(distribuição normal ajustada) ou contadas diretamente nos dados "
        "(empírico)."
    ),
    "descentrado": (
        "Cp bem maior que Cpk indica processo descentrado: recentralizar a "
        "média recupera capabilidade sem reduzir a variação."
    ),
    "iam": (
        "Carta I-AM — Individuais e Amplitude Móvel: verifica se o processo "
        "está sob controle estatístico. Pontos fora dos limites de controle "
        "(causas especiais) indicam eventos excepcionais que distorcem a "
        "análise de capabilidade."
    ),
    "normalidade": (
        "Teste de Anderson-Darling: verifica se os dados seguem a distribuição "
        "normal (sino). p-valor > 0,05 → não há evidência contra a "
        "normalidade e os índices clássicos valem."
    ),
    "transformacao": (
        "Quando os dados não são normais, uma transformação matemática "
        "monótona (ex.: logaritmo, Box-Cox) pode torná-los normais. Os índices "
        "são calculados no espaço transformado e os valores exibidos são "
        "convertidos de volta à escala original."
    ),
    "percentis": (
        "Método dos percentis: substitui µ ± 3σ pelos percentis 0,135% e "
        "99,865% dos próprios dados — a faixa que cobre 99,73% das "
        "observações sem assumir distribuição normal."
    ),
}


@dataclass
class CapabilityIndices:
    """Índices calculados para um par de limites (possivelmente unilateral)."""

    lsl: float | None = None
    usl: float | None = None
    mean: float = np.nan
    median: float = np.nan
    sigma_within: float = np.nan
    sigma_overall: float = np.nan
    n: int = 0
    cp: float = np.nan
    cpu: float = np.nan
    cpl: float = np.nan
    cpk: float = np.nan
    pp: float = np.nan
    ppu: float = np.nan
    ppl: float = np.nan
    ppk: float = np.nan
    ppm_below: float = np.nan   # estimativa modelo (σ global)
    ppm_above: float = np.nan
    ppm_total: float = np.nan
    obs_pct_out: float = np.nan  # % realmente observado fora dos limites
    method: str = "normal"       # 'normal' | 'percentil'
    sided: str = "bilateral"     # 'bilateral' | 'superior' | 'inferior'
    verdict: str = ""
    verdict_text: str = ""


def _sided(lsl: float | None, usl: float | None) -> str:
    if lsl is not None and usl is not None:
        return "bilateral"
    if usl is not None:
        return "superior"
    if lsl is not None:
        return "inferior"
    raise ValueError("Informe ao menos um limite (inferior ou superior).")


def capability_verdict(cpk: float) -> tuple[str, str]:
    """Classifica o índice dominante em uma palavra + frase explicativa."""
    if not np.isfinite(cpk):
        return "indefinido", "Não foi possível calcular o índice."
    for cut, word, text in VERDICTS:
        if cpk >= cut:
            return word, text
    return "incapaz", VERDICTS[-1][2]


def capability_indices(
    s: pd.Series,
    lsl: float | None,
    usl: float | None,
    sigma_within: float | None = None,
) -> CapabilityIndices:
    """Calcula Cp/Cpk/Pp/Ppk assumindo normalidade da série fornecida.

    ``sigma_within`` vem da carta I-AM (AM̄/d2); se ausente, cai no σ global
    (os índices de curto e longo prazo coincidem).
    """
    res = CapabilityIndices(lsl=lsl, usl=usl, sided=_sided(lsl, usl))
    x = pd.Series(s).dropna().astype(float)
    res.n = len(x)
    if res.n < 2:
        return res

    res.mean = float(x.mean())
    res.median = float(x.median())
    res.sigma_overall = float(x.std(ddof=1))
    res.sigma_within = (
        float(sigma_within)
        if sigma_within and np.isfinite(sigma_within) and sigma_within > 0
        else res.sigma_overall
    )
    sw, so, mu = res.sigma_within, res.sigma_overall, res.mean
    if sw <= 0 or so <= 0:
        return res

    if usl is not None:
        res.cpu = (usl - mu) / (3 * sw)
        res.ppu = (usl - mu) / (3 * so)
    if lsl is not None:
        res.cpl = (mu - lsl) / (3 * sw)
        res.ppl = (mu - lsl) / (3 * so)
    if res.sided == "bilateral":
        res.cp = (usl - lsl) / (6 * sw)
        res.pp = (usl - lsl) / (6 * so)
        res.cpk = float(min(res.cpu, res.cpl))
        res.ppk = float(min(res.ppu, res.ppl))
    elif res.sided == "superior":
        res.cpk, res.ppk = float(res.cpu), float(res.ppu)
    else:
        res.cpk, res.ppk = float(res.cpl), float(res.ppl)

    # PPM pelo modelo normal (σ global = desempenho esperado entregue)
    res.ppm_below = (
        float(1e6 * stats.norm.cdf((lsl - mu) / so)) if lsl is not None else 0.0
    )
    res.ppm_above = (
        float(1e6 * stats.norm.sf((usl - mu) / so)) if usl is not None else 0.0
    )
    res.ppm_total = res.ppm_below + res.ppm_above
    res.obs_pct_out = _observed_pct_out(x, lsl, usl)

    res.verdict, res.verdict_text = capability_verdict(res.ppk)
    return res


def percentile_capability(
    s: pd.Series, lsl: float | None, usl: float | None
) -> CapabilityIndices:
    """Método dos percentis (sem assumir distribuição) — usado no Caso 3."""
    res = CapabilityIndices(lsl=lsl, usl=usl, sided=_sided(lsl, usl),
                            method="percentil")
    x = pd.Series(s).dropna().astype(float)
    res.n = len(x)
    if res.n < 10:
        return res

    p_low, p50, p_high = np.quantile(x, [0.00135, 0.5, 0.99865])
    res.mean = float(x.mean())
    res.median = float(p50)
    res.sigma_overall = float(x.std(ddof=1))

    span = p_high - p_low
    if usl is not None and p_high > p50:
        res.ppu = float((usl - p50) / (p_high - p50))
    if lsl is not None and p50 > p_low:
        res.ppl = float((p50 - lsl) / (p50 - p_low))
    if res.sided == "bilateral":
        if span > 0:
            res.pp = float((usl - lsl) / span)
        res.ppk = float(np.nanmin([res.ppu, res.ppl]))
    elif res.sided == "superior":
        res.ppk = float(res.ppu)
    else:
        res.ppk = float(res.ppl)

    # sem modelo: PPM contado diretamente nos dados
    res.obs_pct_out = _observed_pct_out(x, lsl, usl)
    below = float((x < lsl).mean()) if lsl is not None else 0.0
    above = float((x > usl).mean()) if usl is not None else 0.0
    res.ppm_below, res.ppm_above = 1e6 * below, 1e6 * above
    res.ppm_total = res.ppm_below + res.ppm_above

    res.verdict, res.verdict_text = capability_verdict(res.ppk)
    return res


def _observed_pct_out(x: pd.Series, lsl: float | None, usl: float | None) -> float:
    out = pd.Series(False, index=x.index)
    if lsl is not None:
        out |= x < lsl
    if usl is not None:
        out |= x > usl
    return float(100.0 * out.mean())
