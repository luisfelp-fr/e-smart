"""Relatório gerencial em linguagem simples a partir do ranking causal.

Traduz a saída técnica em frases que um leitor sem formação estatística
entende: quem impactou o alvo, em que direção, com que atraso e com que
confiança.

O vocabulário compartilhado com o diagnóstico do dia — nome amigável das
métricas, atraso em tempo real, enumerações — vive em ``phrasing``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aggregation import METRIC_FRIENDLY, base_indicator, fmt_timedelta_br, metric_of
from .phrasing import (
    cadencia,
    friendly_name,
    lag_phrase,
    lag_short,
    lista,
    numero,
    time_step,
    uc_first,
)
from .pipeline import AnalysisResult
from .scoring import ML_MIN_R2

_DIRECTION_PHRASES = {
    "positiva": "quando {nome} sobe, {alvo} tende a subir",
    "negativa": "quando {nome} sobe, {alvo} tende a cair",
    "não-monotônica": (
        "{nome} tem uma faixa ideal: tanto valores muito altos quanto muito "
        "baixos pioram {alvo}"
    ),
    "indefinida": "{nome} influencia {alvo}, mas a direção do efeito não ficou clara",
}

# Compactas de propósito: a explicação da escala aparece UMA vez, no resumo.
# Repetir "com vários testes independentes apontando na mesma direção" ao fim
# de cada achado é o que tornava a leitura cansativa.
_CONFIDENCE_PHRASES = {
    "Alta": "A evidência é forte",
    "Média": "A evidência é moderada",
    "Baixa": "A evidência é inicial",
    "Nenhuma": "Não há confirmação estatística",
}

# nomes privados mantidos para quem já importava daqui
_fmt_timedelta_br = fmt_timedelta_br
_friendly_name = friendly_name
_time_step = time_step
_lag_phrase = lag_phrase
_lag_short = lag_short
_cadencia = cadencia
_uc_first = uc_first


@dataclass
class ManagerialReport:
    """Relatório gerencial pronto para exibição/exportação."""

    headline: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    ranking_table: pd.DataFrame | None = None  # colunas amigáveis


def build_managerial_report(result: AnalysisResult, top: int = 8) -> ManagerialReport:
    """Gera o relatório gerencial a partir do resultado da análise causal."""
    rep = ManagerialReport()
    scores = result.scores
    alvo = result.target
    if scores is None or scores.empty:
        rep.headline = "Nenhum resultado disponível."
        return rep
    step, step_txt = time_step(result.df.index)

    relevant = scores[scores["veredito"].str.contains("Culpado")]
    n_rel = len(relevant)
    if n_rel == 0:
        rep.headline = (
            f"Nenhuma variável analisada mostrou influência clara sobre "
            f"{alvo} neste conjunto de dados."
        )
        rep.summary = (
            "Isso pode significar que os fatores decisivos não estão entre os "
            "dados coletados, que a janela de tempo analisada é curta, ou que "
            "o alvo é dominado por variação aleatória. Vale ampliar o período "
            "coletado, incluir outras variáveis de processo ou revisar a "
            "granularidade dos dados."
        )
    else:
        nomes = list(dict.fromkeys(relevant.head(3)["parametro"].map(base_indicator)))
        lideres = lista(nomes)
        if n_rel == 1:
            rep.headline = (
                f"Um fator mostrou influência relevante sobre {alvo}: {lideres}."
            )
        elif n_rel <= len(nomes):
            rep.headline = (
                f"{uc_first(numero(n_rel))} fatores mostraram influência "
                f"relevante sobre {alvo}: {lideres}."
            )
        else:
            rep.headline = (
                f"{uc_first(numero(n_rel))} fatores mostraram influência "
                f"relevante sobre {alvo}. Os principais são {lideres}."
            )
        escala = f", uma {cadencia(step_txt)}" if step_txt else ""
        rep.summary = (
            f"A análise cruzou correlações, efeitos com atraso, precedência "
            f"temporal e um modelo preditivo sobre "
            f"{result.diagnostics.n_rows_used} medições{escala}. O ranking "
            f"ordena os fatores pela força conjunta dessas evidências, numa "
            f"escala de 0 a 100. A evidência é forte quando vários testes "
            f"independentes concordam, moderada quando mais de um aponta o "
            f"efeito, e inicial quando apenas um a sustenta."
        )
    if not step_txt and n_rel:
        rep.summary += (
            " A planilha não tem coluna de data e hora, então os atrasos "
            "abaixo aparecem em número de medições."
        )

    for _, row in scores.head(top).iterrows():
        if "Culpado" not in str(row["veredito"]):
            continue
        nome = friendly_name(row["parametro"])
        direcao = _DIRECTION_PHRASES.get(
            row["direcao_label"], _DIRECTION_PHRASES["indefinida"]
        ).format(nome=nome, alvo=alvo)
        quando = lag_phrase(str(row["melhor_transformacao"]), step)
        confianca = _CONFIDENCE_PHRASES.get(row["confianca"], row["confianca"])

        # Uma frase para o efeito, outra para a evidência. Antes eram três
        # fragmentos empilhados, cada um terminando em parênteses de jargão.
        frase = (
            f"{uc_first(direcao)}, e {quando}. "
            f"{confianca} e o score fica em {row['score']:.0f} de 100."
        )

        efeito = str(row.get("efeito", "—"))
        if efeito.startswith("indireto"):
            via = efeito.split("via ")[-1].rstrip(")") if "via" in efeito else ""
            frase += (
                " Ao descontar os outros indicadores do topo a associação "
                f"enfraquece, o que sugere que o efeito passa por "
                f"{base_indicator(via)}."
                if via else
                " Ao descontar os outros indicadores do topo a associação "
                "enfraquece, o que sugere um efeito indireto."
            )
        rep.findings.append(frase)

    # cautelas padrão + específicas
    rep.cautions.append(
        "Correlação não é prova definitiva de causa: use este ranking para "
        "priorizar hipóteses e confirme com testes controlados no processo."
    )
    rep.cautions.append(
        "O ranking testa a melhor entre várias defasagens e médias móveis de "
        "cada indicador, e as linhas de evidência se sobrepõem entre si. Isso "
        "favorece encontrar associação, então leia a ordem dos indicadores e "
        "não o valor absoluto do score."
    )
    if result.ml and not (
        np.isfinite(result.ml.r2_oos) and result.ml.r2_oos >= ML_MIN_R2
    ):
        r2_txt = (f"{result.ml.r2_oos:.2f}".replace(".", ",")
                  if np.isfinite(result.ml.r2_oos) else "indisponível")
        rep.cautions.append(
            f"O modelo preditivo não conseguiu prever {alvo} em dados futuros, "
            f"com R² fora da amostra de {r2_txt}. A importância dele foi "
            "desconsiderada no score, que se apoia apenas nos testes "
            "estatísticos."
        )
    if result.target_ljungbox and not result.target_ljungbox.get("has_structure"):
        rep.cautions.append(
            f"O alvo {alvo} não mostrou estrutura temporal relevante no teste "
            "de Ljung-Box, então os efeitos com atraso devem ser lidos com "
            "cautela extra."
        )
    grupos = scores.head(top)["parametro"].map(base_indicator)
    if grupos.duplicated().any():
        rep.cautions.append(
            "Várias métricas do mesmo indicador aparecem no topo, como o pico "
            "e o teto da mesma variável. Elas descrevem o mesmo fenômeno "
            "físico e dividem a culpa entre si."
        )

    # tabela amigável
    tab = scores.head(top).copy()
    tab["indicador"] = tab["parametro"].map(base_indicator)
    tab["o que foi medido"] = tab["parametro"].map(
        lambda c: METRIC_FRIENDLY.get(metric_of(c) or "", "valor da medição")
    )
    tab["como impacta"] = tab["direcao_label"].map({
        "positiva": "sobe junto",
        "negativa": "sobe e o alvo cai",
        "não-monotônica": "existe faixa ideal",
        "indefinida": "direção incerta",
    })
    tab["quando impacta"] = tab["melhor_transformacao"].map(
        lambda label: lag_short(str(label), step)
    )
    cols = ["indicador", "o que foi medido", "score", "veredito",
            "como impacta", "quando impacta", "confianca"]
    if "efeito" in tab.columns:
        cols.append("efeito")
        if tab["efeito"].astype(str).str.startswith("indireto").any():
            rep.cautions.append(
                "Quando a coluna de efeito aponta um indicador, é sinal de "
                "que a associação com o alvo enfraquece ao descontar aquele "
                "indicador. Vale investigá-lo primeiro."
            )
    rep.ranking_table = tab[cols].rename(columns={"confianca": "confiança"})
    return rep
