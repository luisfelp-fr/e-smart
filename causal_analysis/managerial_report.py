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

from .aggregation import (
    METRIC_FRIENDLY,
    base_indicator,
    fmt_timedelta_br,
    metric_of,
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

# Compactas de propósito: a explicação da escala aparece UMA vez no resumo.
# Repetir "com vários testes independentes apontando na mesma direção" ao fim
# de cada achado é o que tornava a leitura cansativa.
_CONFIDENCE_PHRASES = {
    "Alta": "A evidência é forte",
    "Média": "A evidência é moderada",
    "Baixa": "A evidência é inicial",
    "Nenhuma": "Não há confirmação estatística",
}

# Como cada métrica derivada aparece dentro de uma frase. As descrições de
# METRIC_FRIENDLY são boas para tabela, mas travam a leitura no meio de um
# período ("'temp' — valor central na janela (robusto a picos) sobe"); aqui a
# forma é um sujeito que encaixa direto: "o valor central de temp sobe".
_METRIC_PROSE = {
    "média": "a média de {base}",
    "mediana": "o valor central de {base}",
    "mínimo": "o mínimo de {base}",
    "máximo": "o pico de {base}",
    "desvio": "a instabilidade de {base}",
    "P10": "o piso de {base}",
    "P90": "o teto de {base}",
    "% tempo>Q3": "o tempo que {base} passa na faixa alta",
    "% tempo<Q1": "o tempo que {base} passa na faixa baixa",
}


_NUMEROS = {1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco",
            6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez"}


def _numero(n: int) -> str:
    """Números pequenos por extenso — "três fatores" lê melhor que "3 fator(es)"."""
    return _NUMEROS.get(n, str(n))


def _lista(itens: list[str]) -> str:
    """Enumeração com "e" antes do último, como se escreve de verdade."""
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return f"{', '.join(itens[:-1])} e {itens[-1]}"


def _uc_first(texto: str) -> str:
    """Maiúscula só na primeira letra.

    ``str.capitalize()`` minusculiza todo o resto — um indicador chamado
    ``Temp_Forno`` virava ``temp_forno`` no texto do relatório.
    """
    return texto[:1].upper() + texto[1:] if texto else texto


@dataclass
class ManagerialReport:
    """Relatório gerencial pronto para exibição/exportação."""

    headline: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    ranking_table: pd.DataFrame | None = None  # colunas amigáveis


def _friendly_name(column: str) -> str:
    """"forno: temp (P90)" -> "o teto de temp" — sujeito pronto para a frase."""
    base = base_indicator(column)
    metric = metric_of(column)
    if metric is None:
        return base
    prose = _METRIC_PROSE.get(metric)
    if prose is None:
        return f"{base} na métrica {metric}"
    return prose.format(base=base)


# formatação humanizada de durações centralizada em aggregation.fmt_timedelta_br
_fmt_timedelta_br = fmt_timedelta_br


def _time_step(index) -> tuple[pd.Timedelta | None, str]:
    """Passo de tempo típico da série (mediana das diferenças do índice).

    Devolve (None, "") quando os dados não têm coluna de data/hora — nesse
    caso "1 período" só pode significar "1 linha da planilha".
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        diffs = pd.Series(index).diff().dropna()
        if len(diffs):
            step = diffs.median()
            if step > pd.Timedelta(0):
                return step, _fmt_timedelta_br(step)
    return None, ""


def _span(k_txt: str, step: pd.Timedelta | None) -> str:
    """Duração de ``k`` passos em tempo real; "k medições" sem data/hora.

    A escala de tempo é o que o leitor precisa: "o efeito aparece 3 horas
    depois" é acionável, "3 períodos depois" obriga a converter de cabeça.
    Sem coluna de data/hora não dá para saber a duração, e aí a unidade
    honesta é a medição.
    """
    try:
        k = int(k_txt)
    except (TypeError, ValueError):
        return k_txt
    if step is None:
        return f"{k} medições" if k != 1 else "1 medição"
    return _fmt_timedelta_br(step * k)


def _cadencia(step_txt: str) -> str:
    """Frequência das medições sem o "1" solto ("por dia", "a cada 4 horas")."""
    if step_txt.startswith("1 "):
        return f"por {step_txt[2:]}"
    return f"a cada {step_txt}"


def _lag_phrase(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Quando o efeito aparece, em tempo real e sem jargão."""
    if transform_label.startswith("lag"):
        return (
            f"o efeito aparece cerca de {_span(transform_label.split()[1], step)} "
            "depois da variação"
        )
    if transform_label.startswith("média móvel"):
        # "de" e não "das últimas": concorda com singular e plural sem
        # remendo ("o acumulado de 1 semana", "o acumulado de 7 horas")
        return (
            f"o que pesa é o acumulado de {_span(transform_label.split()[2], step)}, "
            "não o valor do momento"
        )
    return "o efeito é imediato"


def _lag_short(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Mesma informação em forma de célula de tabela: "3 horas depois"."""
    if transform_label.startswith("lag"):
        return f"{_span(transform_label.split()[1], step)} depois"
    if transform_label.startswith("média móvel"):
        return f"acumulado de {_span(transform_label.split()[2], step)}"
    return "imediato"


def build_managerial_report(result: AnalysisResult, top: int = 8) -> ManagerialReport:
    """Gera o relatório gerencial a partir do resultado da análise causal."""
    rep = ManagerialReport()
    scores = result.scores
    alvo = result.target
    if scores is None or scores.empty:
        rep.headline = "Nenhum resultado disponível."
        return rep
    step, step_txt = _time_step(result.df.index)

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
        nomes = list(dict.fromkeys(relevant.head(3)["parametro"].map(base_indicator)))
        lideres = _lista(nomes)
        if n_rel == 1:
            rep.headline = f"Um fator mostrou influência relevante sobre {alvo}: {lideres}."
        elif n_rel <= len(nomes):
            rep.headline = (
                f"{_uc_first(_numero(n_rel))} fatores mostraram influência "
                f"relevante sobre {alvo}: {lideres}."
            )
        else:
            rep.headline = (
                f"{_uc_first(_numero(n_rel))} fatores mostraram influência "
                f"relevante sobre {alvo}. Os principais são {lideres}."
            )
        escala = f", uma {_cadencia(step_txt)}" if step_txt else ""
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
        nome = _friendly_name(row["parametro"])
        direcao = _DIRECTION_PHRASES.get(
            row["direcao_label"], _DIRECTION_PHRASES["indefinida"]
        ).format(nome=nome, alvo=alvo)
        quando = _lag_phrase(str(row["melhor_transformacao"]), step)
        confianca = _CONFIDENCE_PHRASES.get(row["confianca"], row["confianca"])

        # Uma frase para o efeito, outra para a evidência. Antes eram três
        # fragmentos empilhados, cada um terminando em parênteses de jargão.
        frase = (
            f"{_uc_first(direcao)}, e {quando}. "
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
            f"O modelo preditivo não conseguiu prever '{alvo}' em dados "
            f"futuros (R² fora da amostra: {r2_txt}) — a importância dele foi "
            "DESCONSIDERADA no score. O ranking se apoia apenas nos testes "
            "estatísticos."
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
            "Várias métricas do mesmo indicador aparecem no topo, como o pico "
            "e o teto da mesma variável. Elas descrevem o mesmo fenômeno "
            "físico e dividem a culpa entre si."
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
    tab["quando impacta"] = tab["melhor_transformacao"].map(
        lambda label: _lag_short(str(label), step)
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
