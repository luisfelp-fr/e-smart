"""Relatório gerencial em linguagem simples a partir do ranking causal.

Traduz a saída técnica (scores, vereditos, transformações, métricas com
sufixos) em frases que um leitor sem formação estatística entende: quem
impactou o alvo, em que direção, com que atraso e com que confiança —
sem rótulos crípticos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aggregation import METRIC_FRIENDLY, base_indicator, metric_of
from .pipeline import AnalysisResult

_DIRECTION_PHRASES = {
    "positiva": "quando {nome} sobe, {alvo} tende a SUBIR",
    "negativa": "quando {nome} sobe, {alvo} tende a CAIR",
    "não-monotônica": (
        "{nome} afeta {alvo} de forma não-linear — existe uma faixa ideal; "
        "tanto valores muito altos quanto muito baixos pioram o resultado"
    ),
    "indefinida": "a direção do efeito de {nome} sobre {alvo} não ficou clara",
}

_CONFIDENCE_PHRASES = {
    "Alta": "evidência forte (vários testes independentes concordam)",
    "Média": "evidência moderada (mais de um teste aponta o efeito)",
    "Baixa": "evidência inicial (apenas um teste significativo)",
    "Nenhuma": "sem confirmação estatística",
}


@dataclass
class ManagerialReport:
    """Relatório gerencial pronto para exibição/exportação."""

    headline: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    ranking_table: pd.DataFrame | None = None  # colunas amigáveis


def _friendly_name(column: str) -> str:
    """"forno: temp (P90)" -> "o teto típico (percentil 10..) de temp"."""
    base = base_indicator(column)
    metric = metric_of(column)
    if metric is None:
        return f"'{base}'"
    desc = METRIC_FRIENDLY.get(metric)
    if desc is None:
        return f"'{base}' ({metric})"
    return f"'{base}' — {desc}"


def _lag_phrase(transform_label: str) -> str:
    if transform_label.startswith("lag"):
        k = transform_label.split()[1]
        return (
            f"o efeito aparece cerca de {k} período(s) DEPOIS da variação "
            "(efeito com atraso)"
        )
    if transform_label.startswith("média móvel"):
        w = transform_label.split()[2]
        return (
            f"o que importa é o acumulado de ~{w} período(s), não o valor "
            "instantâneo (efeito de permanência)"
        )
    return "o efeito é imediato (mesmo período)"


def build_managerial_report(result: AnalysisResult, top: int = 8) -> ManagerialReport:
    """Gera o relatório gerencial a partir do resultado da análise causal."""
    rep = ManagerialReport()
    scores = result.scores
    alvo = result.target
    if scores is None or scores.empty:
        rep.headline = "Nenhum resultado disponível."
        return rep

    relevant = scores[scores["veredito"].str.contains("Culpado")]
    n_rel = len(relevant)
    if n_rel == 0:
        rep.headline = (
            f"Nenhuma variável analisada mostrou influência clara sobre "
            f"'{alvo}' neste conjunto de dados."
        )
        rep.summary = (
            "Isso pode significar que os fatores decisivos não estão entre os "
            "dados coletados, que a janela de tempo analisada é curta, ou que "
            "o alvo é dominado por variação aleatória. Sugestões: ampliar o "
            "período coletado, incluir outras variáveis de processo ou revisar "
            "a granularidade dos dados."
        )
    else:
        lideres = ", ".join(
            f"'{base_indicator(p)}'" for p in
            dict.fromkeys(relevant.head(3)["parametro"].map(base_indicator))
        )
        rep.headline = (
            f"{n_rel} fator(es) apresentaram influência relevante sobre "
            f"'{alvo}'. Principais: {lideres}."
        )
        rep.summary = (
            f"A análise combinou vários métodos estatísticos independentes "
            f"(correlações, efeitos com atraso, precedência temporal e um "
            f"modelo preditivo) sobre {result.diagnostics.n_rows_used} "
            f"observações. O ranking abaixo ordena os fatores pela força "
            f"conjunta dessas evidências (score 0–100)."
        )

    for _, row in scores.head(top).iterrows():
        if "Culpado" not in str(row["veredito"]):
            continue
        nome = _friendly_name(row["parametro"])
        direcao = _DIRECTION_PHRASES.get(
            row["direcao_label"], _DIRECTION_PHRASES["indefinida"]
        ).format(nome=nome, alvo=f"'{alvo}'")
        frase = (
            f"{direcao.capitalize()}. "
            f"{_lag_phrase(str(row['melhor_transformacao'])).capitalize()}. "
            f"Confiança: {_CONFIDENCE_PHRASES.get(row['confianca'], row['confianca'])} "
            f"(score {row['score']:.0f}/100)."
        )
        rep.findings.append(frase)

    # cautelas padrão + específicas
    rep.cautions.append(
        "Correlação não é prova definitiva de causa: use este ranking para "
        "priorizar hipóteses e confirme com testes controlados no processo."
    )
    if result.target_ljungbox and not result.target_ljungbox.get("has_structure"):
        rep.cautions.append(
            f"O alvo '{alvo}' não mostrou estrutura temporal relevante "
            "(teste de Ljung-Box): efeitos com atraso (lag) devem ser lidos "
            "com cautela extra."
        )
    grupos = scores.head(top)["parametro"].map(base_indicator)
    if grupos.duplicated().any():
        rep.cautions.append(
            "Várias métricas do mesmo indicador aparecem no topo (ex.: máximo "
            "e P90 da mesma variável) — elas descrevem o MESMO fenômeno físico "
            "e dividem a 'culpa' entre si."
        )

    # tabela amigável
    tab = scores.head(top).copy()
    tab["indicador"] = tab["parametro"].map(base_indicator)
    tab["o que foi medido"] = tab["parametro"].map(
        lambda c: METRIC_FRIENDLY.get(metric_of(c) or "", "valor no período")
    )
    tab["como impacta"] = tab["direcao_label"].map({
        "positiva": "sobe junto",
        "negativa": "sobe → alvo cai",
        "não-monotônica": "existe faixa ideal",
        "indefinida": "direção incerta",
    })
    tab["quando impacta"] = tab["melhor_transformacao"].map(_lag_phrase)
    rep.ranking_table = tab[[
        "indicador", "o que foi medido", "score", "veredito",
        "como impacta", "quando impacta", "confianca",
    ]].rename(columns={"confianca": "confiança"})
    return rep
