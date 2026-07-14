"""Carta de controle de Individuais e Amplitude Móvel (I-AM).

A carta I-AM verifica se o processo está sob controle estatístico antes da
análise de capabilidade: pontos de causa especial (excepcionalidades)
distorcem a estimativa de variação e, com isso, os índices Cp/Cpk. O desvio
de curto prazo σ_dentro = AM̄/d2 (d2 = 1,128 para n = 2) alimenta o Cp/Cpk.

Regras de causa especial implementadas (subconjunto clássico — o conjunto
completo de 8 regras gera excesso de alarmes falsos em uso rotineiro):

- R1: um ponto além de 3σ do centro (obrigatória);
- R2: nove pontos consecutivos do mesmo lado do centro (deslocamento);
- R3: seis pontos consecutivos sempre crescentes ou decrescentes (tendência);
- R5: dois de três pontos consecutivos além de 2σ, do mesmo lado;
- R6: quatro de cinco pontos consecutivos além de 1σ, do mesmo lado.

Na carta de Amplitude Móvel aplica-se apenas a R1: as amplitudes móveis são
autocorrelacionadas por construção e regras de sequência não são válidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

D2_N2 = 1.128   # E(AM)/σ para janelas de 2 observações
D4_N2 = 3.267   # limite superior da carta AM
E2_N2 = 2.66    # 3/d2

RULE_LABELS = {
    1: "R1: ponto além de 3σ",
    2: "R2: 9 pontos do mesmo lado do centro",
    3: "R3: 6 pontos em tendência crescente/decrescente",
    5: "R5: 2 de 3 pontos além de 2σ (mesmo lado)",
    6: "R6: 4 de 5 pontos além de 1σ (mesmo lado)",
}


@dataclass
class IMRResult:
    """Limites da carta I-AM e pontos que violaram regras de causa especial."""

    center: float = np.nan          # X̄ (carta de individuais)
    ucl: float = np.nan             # X̄ + 3·AM̄/d2
    lcl: float = np.nan             # X̄ - 3·AM̄/d2
    sigma_within: float = np.nan    # AM̄/d2 (curto prazo)
    mr_bar: float = np.nan          # média das amplitudes móveis
    mr_ucl: float = np.nan          # D4·AM̄
    n: int = 0
    # índice do ponto -> lista de regras violadas (carta de individuais)
    violations: dict = field(default_factory=dict)
    # índices (posicionais) das AM acima do limite (carta de amplitudes)
    mr_violations: list = field(default_factory=list)

    @property
    def out_of_control_index(self) -> list:
        """Rótulos de índice dos pontos com alguma causa especial."""
        return list(self.violations.keys())

    @property
    def in_control(self) -> bool:
        return not self.violations and not self.mr_violations


def _runs_same_side(sign: np.ndarray, run: int) -> set[int]:
    """Posições que fecham sequências de ``run`` pontos do mesmo lado (≠0)."""
    hits: set[int] = set()
    count, cur = 0, 0
    for i, s in enumerate(sign):
        if s != 0 and s == cur:
            count += 1
        else:
            cur, count = s, (1 if s != 0 else 0)
        if s != 0 and count >= run:
            hits.update(range(i - run + 1, i + 1))
    return hits


def _monotone_runs(x: np.ndarray, run: int) -> set[int]:
    """Posições em trechos com ``run`` pontos sempre subindo ou descendo."""
    hits: set[int] = set()
    if len(x) < run:
        return hits
    diff = np.sign(np.diff(x))
    count, cur = 0, 0
    for i, d in enumerate(diff):
        if d != 0 and d == cur:
            count += 1
        else:
            cur, count = d, (1 if d != 0 else 0)
        # count incrementos consecutivos => count+1 pontos monotônicos
        if d != 0 and count >= run - 1:
            hits.update(range(i - run + 2, i + 2))
    return hits


def _k_of_m_beyond(z: np.ndarray, k: int, m: int, limit: float) -> set[int]:
    """Posições em janelas de ``m`` pontos com ``k`` além de ``limit``·σ, mesmo lado."""
    hits: set[int] = set()
    n = len(z)
    for i in range(n - m + 1):
        w = z[i:i + m]
        for side in (1, -1):
            beyond = side * w > limit
            if beyond.sum() >= k:
                hits.update(i + j for j in range(m) if beyond[j])
    return hits


def imr_chart(s: pd.Series) -> IMRResult:
    """Monta a carta I-AM da série (NaN são ignorados, ordem preservada)."""
    res = IMRResult()
    x = pd.Series(s).dropna().astype(float)
    res.n = len(x)
    if res.n < 3:
        return res

    vals = x.to_numpy()
    mr = np.abs(np.diff(vals))
    res.mr_bar = float(np.mean(mr))
    res.center = float(np.mean(vals))
    res.sigma_within = res.mr_bar / D2_N2 if res.mr_bar > 0 else float(np.std(vals, ddof=1))
    res.ucl = res.center + 3.0 * res.sigma_within
    res.lcl = res.center - 3.0 * res.sigma_within
    res.mr_ucl = D4_N2 * res.mr_bar

    if res.sigma_within <= 0 or not np.isfinite(res.sigma_within):
        return res

    z = (vals - res.center) / res.sigma_within
    sign = np.sign(z)

    fired: dict[int, list[int]] = {}

    def mark(positions: set[int], rule: int) -> None:
        for p in positions:
            fired.setdefault(p, []).append(rule)

    mark({i for i, v in enumerate(z) if abs(v) > 3.0}, 1)
    mark(_runs_same_side(sign, 9), 2)
    mark(_monotone_runs(vals, 6), 3)
    mark(_k_of_m_beyond(z, 2, 3, 2.0), 5)
    mark(_k_of_m_beyond(z, 4, 5, 1.0), 6)

    # posições -> rótulos do índice original da série
    res.violations = {
        x.index[p]: sorted(rules) for p, rules in sorted(fired.items())
    }

    # carta AM: apenas R1 (ponto além de D4·AM̄); posição i refere-se a AM_i =
    # |x_i - x_{i-1}|, associada ao rótulo do ponto i (o segundo da diferença)
    res.mr_violations = [
        x.index[i + 1] for i, v in enumerate(mr) if v > res.mr_ucl
    ]
    return res


def remove_special_causes(s: pd.Series, index_labels: list) -> pd.Series:
    """Remove (vira NaN) os pontos indicados, preservando o índice."""
    out = pd.Series(s).astype(float).copy()
    labels = [i for i in index_labels if i in out.index]
    out.loc[labels] = np.nan
    return out
