"""Revisão dos limites de atuação: aderência e recomendação de novos limites.

O Módulo 1 é o balizador dos limites previamente definidos para o processo:
a partir do resultado da capabilidade (os 3 casos), este módulo conclui se os
limites atuais estão ADERENTES ao que o processo entrega ou se merecem
revisão, e recomenda novos limites na regra de cada caso:

- Caso 1 (dados normais): µ ± 3·σ global — a faixa que o processo cobre
  ~99,7% do tempo (voz do processo);
- Caso 2 (normais após transformação): µ ± 3σ no espaço transformado,
  retro-convertidos à escala original (já calculados no pipeline);
- Caso 3 (não-normais): regra dos quartis do box-plot definida pelo usuário
  do processo (Q3 para unilateral inferior, Q2–Q3 para bilateral, Q1 para
  unilateral superior).

A aderência reusa os limiares clássicos do veredito (indices.VERDICTS):
Ppk ≥ 1,33 → aderente; 1,00–1,33 → atenção; < 1,00 → revisar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SITUACAO_LABELS = {
    "aderente": "Limites aderentes ao processo",
    "atencao": "Limites no limite da capacidade — monitorar",
    "revisar": "Limites merecem revisão",
}

_METODO = {
    1: "µ ± 3σ do processo (voz do processo)",
    2: "µ ± 3σ retro-convertido da escala transformada",
    3: "quartis do box-plot (Q2–Q3 / Q3 / Q1)",
}


@dataclass
class LimitReview:
    """Conclusão da revisão dos limites de atuação de um indicador."""

    situacao: str = ""            # 'aderente' | 'atencao' | 'revisar'
    situacao_label: str = ""
    motivo: str = ""
    lsl: float | None = None      # limites atuais
    usl: float | None = None
    rec_lsl: float | None = None  # novos limites recomendados
    rec_usl: float | None = None
    metodo: str = ""


def _finite_or_none(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def review_limits(rep) -> LimitReview:
    """Avalia a aderência dos limites atuais e recomenda novos (por caso).

    ``rep``: CapabilityReport já preenchido por run_capability.
    """
    res = LimitReview(lsl=rep.lsl, usl=rep.usl)
    idx = rep.indices
    if idx is None or not np.isfinite(idx.ppk):
        res.situacao, res.situacao_label = "revisar", SITUACAO_LABELS["revisar"]
        res.motivo = (
            "Não foi possível calcular o desempenho (Ppk) contra os limites "
            "atuais — revise os limites e/ou a qualidade dos dados."
        )
        return res

    # aderência pelos limiares clássicos já usados no veredito
    if idx.ppk >= 1.33:
        res.situacao = "aderente"
    elif idx.ppk >= 1.00:
        res.situacao = "atencao"
    else:
        res.situacao = "revisar"
    res.situacao_label = SITUACAO_LABELS[res.situacao]

    pct = f"{idx.obs_pct_out:.2f}".replace(".", ",") if np.isfinite(
        idx.obs_pct_out) else "—"
    ppk = f"{idx.ppk:.2f}".replace(".", ",")
    if res.situacao == "aderente":
        res.motivo = (
            f"O processo atende aos limites atuais com folga adequada "
            f"(Ppk = {ppk} ≥ 1,33; {pct}% das medições fora). Os limites "
            "definidos estão aderentes ao que o processo entrega."
        )
    elif res.situacao == "atencao":
        res.motivo = (
            f"O processo atende aos limites atuais no limite da capacidade "
            f"(Ppk = {ppk}, entre 1,00 e 1,33; {pct}% fora). Monitorar de "
            "perto; considere os limites recomendados abaixo como referência."
        )
    else:
        res.motivo = (
            f"O processo NÃO atende aos limites atuais (Ppk = {ppk} < 1,00; "
            f"{pct}% das medições fora). Recomenda-se revisar os limites "
            "(ou atuar no processo) — sugestão abaixo."
        )

    # novos limites recomendados, na regra de cada caso; só os lados que o
    # usuário definiu (limite único continua único)
    res.metodo = _METODO.get(rep.case, "")
    if rep.case == 1:
        mu, so = idx.mean, idx.sigma_overall
        if np.isfinite(mu) and np.isfinite(so) and so > 0:
            if rep.lsl is not None:
                res.rec_lsl = float(mu - 3.0 * so)
            if rep.usl is not None:
                res.rec_usl = float(mu + 3.0 * so)
    elif rep.case == 2:
        if rep.lsl is not None:
            res.rec_lsl = _finite_or_none(rep.display_p0135)
        if rep.usl is not None:
            res.rec_usl = _finite_or_none(rep.display_p99865)
    elif rep.case == 3 and rep.suggested is not None:
        res.rec_lsl = _finite_or_none(rep.suggested.suggested_lsl)
        res.rec_usl = _finite_or_none(rep.suggested.suggested_usl)
    return res
