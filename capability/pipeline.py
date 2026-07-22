"""Orquestração da análise de capabilidade de um indicador (os 3 casos).

Caso 1 — dados normais: índices clássicos direto nos dados.
Caso 2 — normais após transformação: índices calculados NO ESPAÇO
         TRANSFORMADO (limites também transformados); a retro-conversão à
         escala original serve somente para exibição (mediana e percentis
         P0,135/P99,865 em unidades reais). Recalcular índices a partir dos
         percentis retro-convertidos daria números diferentes — a
         transformação é não-linear — e por isso não é feito.
Caso 3 — não-normais mesmo transformando: descrição por box-plot/percentis,
         PPM contado nos dados e sugestão de limites realistas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .control_chart import IMRResult, imr_chart, remove_special_causes
from .data_prep import OutlierReport, treat_missing, treat_outliers
from .indices import CapabilityIndices, capability_indices, percentile_capability
from .limit_review import LimitReview, review_limits
from .nonparametric import (
    BoxStats,
    SuggestedLimits,
    boxplot_summary,
    empirical_nonconformance,
    suggested_limits,
)
from .normality import NormalityResult, test_normality
from .transforms import TransformSearch, best_normalizing_transform, forward, inverse

CASE_LABELS = {
    1: "Caso 1 — dados normais: análise de capabilidade clássica",
    2: "Caso 2 — normalidade obtida por transformação",
    3: "Caso 3 — dados não-normais: análise por box-plot/percentis",
}


@dataclass
class CapabilityReport:
    """Resultado completo da análise de capabilidade de um indicador."""

    indicator: str = ""
    lsl: float | None = None
    usl: float | None = None
    case: int = 0
    case_label: str = ""
    # série analisada (após tratamentos), na escala original
    series: pd.Series | None = None
    outliers: OutlierReport | None = None
    imr: IMRResult | None = None
    removed_labels: list = field(default_factory=list)
    normality_raw: NormalityResult | None = None
    transform: TransformSearch | None = None
    normality_final: NormalityResult | None = None
    indices: CapabilityIndices | None = None
    # Caso 2: série e limites no espaço transformado + exibição retro-convertida
    transformed_series: pd.Series | None = None
    transformed_lsl: float | None = None
    transformed_usl: float | None = None
    display_median: float = np.nan       # T⁻¹(µ_t)
    display_p0135: float = np.nan        # T⁻¹(µ_t − 3σ_t)
    display_p99865: float = np.nan       # T⁻¹(µ_t + 3σ_t)
    # Caso 3
    box: BoxStats | None = None
    suggested: SuggestedLimits | None = None
    empirical: dict = field(default_factory=dict)
    # balizamento dos limites definidos pelo usuário (aderência + recomendação)
    limit_review: LimitReview | None = None
    notes: list[str] = field(default_factory=list)
    narrative: str = ""


def _transform_limit(name: str, params: dict, value: float | None,
                     notes: list[str], which: str) -> float | None:
    """Transforma um limite de especificação; trata limites fora do domínio."""
    if value is None:
        return None
    t = float(forward(name, params, value))
    if np.isfinite(t):
        return t
    if t == -np.inf:
        notes.append(
            f"O limite {which} ({value:g}) está abaixo do domínio da "
            "transformação — dentro do modelo, o processo nunca o viola."
        )
        return None
    notes.append(
        f"O limite {which} ({value:g}) não pôde ser transformado e foi "
        "ignorado no cálculo dos índices."
    )
    return None


def run_capability(
    s: pd.Series,
    indicator: str,
    lsl: float | None = None,
    usl: float | None = None,
    outlier_method: str = "nenhum",
    outlier_k: float | None = None,
    missing_method: str = "nenhum",
    remove_labels: list | None = None,
    alpha: float = 0.05,
) -> CapabilityReport:
    """Executa o fluxo completo de capabilidade para uma série.

    ``remove_labels``: rótulos de índice de causas especiais que o usuário
    optou por excluir (a carta é recalculada uma vez após a remoção).
    """
    if lsl is None and usl is None:
        raise ValueError("Informe ao menos um limite (inferior ou superior).")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("O limite inferior deve ser menor que o superior.")

    rep = CapabilityReport(indicator=indicator, lsl=lsl, usl=usl)

    # 1) tratamentos opcionais (escolha explícita do usuário)
    x, rep.outliers = treat_outliers(s, method=outlier_method, k=outlier_k)
    if rep.outliers.n_removed:
        rep.notes.append(
            f"{rep.outliers.n_removed} outlier(s) removido(s) pelo método "
            f"{rep.outliers.method} (limiar {rep.outliers.threshold:g})."
        )
    x, miss_notes = treat_missing(x, method=missing_method)
    rep.notes.extend(miss_notes)

    # 2) causas especiais escolhidas para remoção (recalcula a carta uma vez)
    if remove_labels:
        rep.removed_labels = list(remove_labels)
        x = remove_special_causes(x, rep.removed_labels)
        rep.notes.append(
            f"{len(rep.removed_labels)} ponto(s) de causa especial removido(s) "
            "a pedido do usuário; limites de controle recalculados."
        )

    rep.series = x
    rep.imr = imr_chart(x)
    if rep.imr.violations:
        rep.notes.append(
            f"A carta I-AM sinaliza {len(rep.imr.violations)} ponto(s) com "
            "causa especial. Causas excepcionais podem deturpar a análise de "
            "capabilidade — avalie removê-las (com justificativa de processo)."
        )

    rep.box = boxplot_summary(x)

    # 3) normalidade dos dados brutos (pós-tratamento)
    rep.normality_raw = test_normality(x, alpha=alpha)
    if rep.normality_raw.note:
        rep.notes.append(rep.normality_raw.note)

    if rep.normality_raw.is_normal or rep.normality_raw.practically_normal:
        rep.case = 1
        rep.normality_final = rep.normality_raw
        rep.indices = capability_indices(
            x, lsl, usl, sigma_within=rep.imr.sigma_within
        )
    else:
        # 4) transformações normalizadoras
        rep.transform = best_normalizing_transform(x, alpha=alpha)
        if rep.transform.note:
            rep.notes.append(rep.transform.note)
        best = rep.transform.best
        if best is not None:
            rep.case = 2
            rep.normality_final = best.achieved
            t_series = pd.Series(
                forward(best.name, best.params, x.to_numpy(dtype=float)),
                index=x.index, name=f"{indicator} (transformado)",
            )
            rep.transformed_series = t_series
            rep.transformed_lsl = _transform_limit(
                best.name, best.params, lsl, rep.notes, "inferior")
            rep.transformed_usl = _transform_limit(
                best.name, best.params, usl, rep.notes, "superior")
            imr_t = imr_chart(t_series)
            rep.indices = capability_indices(
                t_series, rep.transformed_lsl, rep.transformed_usl,
                sigma_within=imr_t.sigma_within,
            )
            # exibição em unidades reais: T⁻¹ da mediana e da faixa µ ± 3σ
            mu_t = rep.indices.mean
            so_t = rep.indices.sigma_overall
            if np.isfinite(mu_t) and np.isfinite(so_t):
                lo, mid, hi = inverse(
                    best.name, best.params,
                    np.array([mu_t - 3 * so_t, mu_t, mu_t + 3 * so_t]),
                )
                rep.display_p0135 = float(lo)
                rep.display_median = float(mid)
                rep.display_p99865 = float(hi)
        else:
            rep.case = 3
            rep.suggested = suggested_limits(x, lsl, usl)
            rep.empirical = empirical_nonconformance(x, lsl, usl)
            rep.indices = percentile_capability(x, lsl, usl)

    rep.case_label = CASE_LABELS.get(rep.case, "")
    rep.limit_review = review_limits(rep)
    rep.narrative = _narrative(rep)
    return rep


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}".replace(".", ",")


def _narrative(rep: CapabilityReport) -> str:
    """Resumo em linguagem simples do resultado, para leigos em estatística."""
    parts: list[str] = []
    lim = []
    if rep.lsl is not None:
        lim.append(f"inferior {_fmt(rep.lsl, 4)}")
    if rep.usl is not None:
        lim.append(f"superior {_fmt(rep.usl, 4)}")
    parts.append(
        f"Indicador '{rep.indicator}' avaliado contra o limite "
        + " e ".join(lim) + "."
    )

    if rep.imr is not None and rep.imr.violations:
        parts.append(
            f"A carta de controle identificou {len(rep.imr.violations)} "
            "ponto(s) fora do padrão (causas excepcionais), que podem "
            "distorcer o resultado."
        )
    elif rep.imr is not None and rep.imr.n:
        parts.append("O processo aparenta estar estável (sem causas especiais).")

    if rep.case == 1:
        parts.append("Os dados seguem a distribuição normal.")
    elif rep.case == 2 and rep.transform and rep.transform.best:
        parts.append(
            "Os dados não eram normais, mas a transformação "
            f"{rep.transform.best.label} os normalizou; os índices foram "
            "calculados na escala transformada e os valores exibidos foram "
            "convertidos de volta à escala original."
        )
    elif rep.case == 3:
        parts.append(
            "Os dados não seguem a normal nem após transformações; a análise "
            "usou os percentis reais dos dados (box-plot)."
        )

    idx = rep.indices
    if idx is not None and np.isfinite(idx.ppk):
        pct_in = 100.0 - idx.obs_pct_out if np.isfinite(idx.obs_pct_out) else np.nan
        parts.append(
            f"Resultado: processo {idx.verdict.upper()} "
            f"(Ppk = {_fmt(idx.ppk)}; {_fmt(pct_in, 1)}% das medições dentro "
            "dos limites). " + idx.verdict_text
        )
    if rep.case == 3 and rep.suggested is not None and rep.suggested.rationale:
        parts.append(rep.suggested.rationale)
    lr = rep.limit_review
    if lr is not None and lr.situacao:
        rec = []
        if lr.rec_lsl is not None:
            rec.append(f"inferior {_fmt(lr.rec_lsl, 4)}")
        if lr.rec_usl is not None:
            rec.append(f"superior {_fmt(lr.rec_usl, 4)}")
        frase = f"Revisão dos limites: {lr.situacao_label.lower()}."
        if rec and lr.situacao != "aderente":
            frase += (" Novos limites recomendados: " + " e ".join(rec)
                      + f" ({lr.metodo}).")
        elif rec:
            frase += (" Referência do processo: " + " e ".join(rec)
                      + f" ({lr.metodo}).")
        parts.append(frase)
    return " ".join(parts)
