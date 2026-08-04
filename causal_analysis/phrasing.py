"""Vocabulário compartilhado dos textos em linguagem simples.

O ranking (``managerial_report``) e o diagnóstico do dia (``day_diagnosis``)
descrevem as mesmas coisas — um indicador, uma métrica derivada, um atraso —
e precisam descrevê-las do mesmo jeito. Com as frases duplicadas nos dois
módulos, uma melhoria de texto num deles deixava o outro para trás.

Duas regras valem para tudo aqui:

- **Atraso em tempo real, nunca em "períodos".** "o efeito aparece 3 horas
  depois" é acionável; "3 períodos depois" obriga o leitor a converter de
  cabeça. Sem coluna de data/hora não há duração a inventar, e aí a unidade
  honesta é a medição.
- **Texto corrido.** Nada de travessão cortando a frase nem parêntese
  guardando jargão; o que precisa ser dito entra na frase.
"""

from __future__ import annotations

import pandas as pd

from .aggregation import base_indicator, fmt_timedelta_br, metric_of

# Como cada métrica derivada aparece DENTRO de uma frase. As descrições de
# METRIC_FRIENDLY são boas para cabeçalho de tabela, mas travam a leitura no
# meio de um período ("'temp' — valor central na janela (robusto a picos)
# sobe"); aqui a forma é um sujeito que encaixa direto.
METRIC_PROSE = {
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


def numero(n: int) -> str:
    """Números pequenos por extenso — "três fatores" lê melhor que "3 fator(es)"."""
    return _NUMEROS.get(n, str(n))


def lista(itens: list[str]) -> str:
    """Enumeração com "e" antes do último, como se escreve de verdade."""
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return f"{', '.join(itens[:-1])} e {itens[-1]}"


def uc_first(texto: str) -> str:
    """Maiúscula só na primeira letra.

    ``str.capitalize()`` minusculiza todo o resto — um indicador chamado
    ``Temp_Forno`` virava ``temp_forno`` no texto.
    """
    return texto[:1].upper() + texto[1:] if texto else texto


def friendly_name(column: str) -> str:
    """"forno: temp (P90)" -> "o teto de temp" — sujeito pronto para a frase."""
    base = base_indicator(column)
    metric = metric_of(column)
    if metric is None:
        return base
    prose = METRIC_PROSE.get(metric)
    if prose is None:
        return f"{base} na métrica {metric}"
    return prose.format(base=base)


def time_step(index) -> tuple[pd.Timedelta | None, str]:
    """Passo de tempo típico da série (mediana das diferenças do índice).

    Devolve (None, "") quando os dados não têm coluna de data/hora — nesse
    caso um passo só pode significar "uma linha da planilha".
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        diffs = pd.Series(index).diff().dropna()
        if len(diffs):
            step = diffs.median()
            if step > pd.Timedelta(0):
                return step, fmt_timedelta_br(step)
    return None, ""


def span(k_txt: str, step: pd.Timedelta | None) -> str:
    """Duração de ``k`` passos em tempo real; "k medições" sem data/hora."""
    try:
        k = int(k_txt)
    except (TypeError, ValueError):
        return str(k_txt)
    if step is None:
        return f"{k} medições" if k != 1 else "1 medição"
    return fmt_timedelta_br(step * k)


def cadencia(step_txt: str) -> str:
    """Frequência das medições sem o "1" solto ("por dia", "a cada 4 horas")."""
    if step_txt.startswith("1 "):
        return f"por {step_txt[2:]}"
    return f"a cada {step_txt}"


def lag_phrase(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Quando o efeito aparece, em tempo real e sem jargão."""
    if transform_label.startswith("lag"):
        return (
            f"o efeito aparece cerca de {span(transform_label.split()[1], step)} "
            "depois da variação"
        )
    if transform_label.startswith("média móvel"):
        # "de" e não "das últimas": concorda com singular e plural sem
        # remendo ("o acumulado de 1 semana", "o acumulado de 7 horas")
        return (
            f"o que pesa é o acumulado de {span(transform_label.split()[2], step)}, "
            "não o valor do momento"
        )
    return "o efeito é imediato"


def lag_short(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Mesma informação em forma de célula de tabela: "3 horas depois"."""
    if transform_label.startswith("lag"):
        return f"{span(transform_label.split()[1], step)} depois"
    if transform_label.startswith("média móvel"):
        return f"acumulado de {span(transform_label.split()[2], step)}"
    return "imediato"


def measured_short(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Em que versão o valor foi lido, para célula de tabela: "5 dias antes"."""
    if transform_label.startswith("lag"):
        return f"{span(transform_label.split()[1], step)} antes"
    if transform_label.startswith("média móvel"):
        return f"acumulado de {span(transform_label.split()[2], step)}"
    return "valor do dia"


def measured_on(transform_label: str, step: pd.Timedelta | None = None) -> str:
    """Em que versão temporal o indicador foi lido, para encaixar na frase.

    Devolve "" para o valor bruto: dizer "medido no valor do próprio dia" só
    ocupa espaço quando não há transformação nenhuma.
    """
    if transform_label.startswith("lag"):
        return f", medido {span(transform_label.split()[1], step)} antes"
    if transform_label.startswith("média móvel"):
        return f", medido no acumulado de {span(transform_label.split()[2], step)}"
    return ""
