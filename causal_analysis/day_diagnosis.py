"""Diagnóstico do dia: o que provavelmente impactou o alvo num dia específico.

Com uma única observação do alvo não existe análise causal possível; o que é
estatisticamente honesto é uma LEITURA DO DIA CONTRA O MODELO HISTÓRICO:

    provável contribuinte do dia = indicador que HISTORICAMENTE move o alvo
    (score do ranking do Módulo 2) E esteve ATÍPICO naquele dia (percentil
    do valor do dia dentro do próprio histórico).

Cada indicador do topo do ranking é avaliado na sua MELHOR transformação
temporal (ex.: "lag 3" usa o valor de 3 períodos antes; "média móvel 7" usa
o acumulado da semana) — a mesma versão que o ranking considerou relevante.
O resultado é um indício priorizado para investigação, não prova causal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aggregation import base_indicator
from .features import feature_label, single_feature
from .phrasing import friendly_name, measured_on, measured_short, time_step
from .pipeline import AnalysisResult

# |2·percentil − 1|: 0 = mediana, 1 = extremo
ATYPICAL_STRONG = 0.80   # fora de P10–P90
ATYPICAL_MILD = 0.50     # fora de P25–P75


@dataclass
class DayDiagnosis:
    """Resultado do diagnóstico de um dia/período da grade do alvo."""

    label: object = None            # rótulo do dia na grade do alvo
    target: str = ""
    target_value: float = np.nan
    target_pct: float = np.nan      # percentil do alvo no histórico (0-100)
    n_history: int = 0
    rows: pd.DataFrame | None = None  # tabela amigável dos contribuintes
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}g}".replace(".", ",")


def _friendly(param: str) -> str:
    """Sujeito pronto para a frase: "o teto de temp", não "'temp' — ...".

    O vocabulário é o mesmo do ranking; ver ``phrasing.friendly_name``.
    """
    return friendly_name(param)


def _uses_future(index: pd.Index, label) -> bool:
    """O dia diagnosticado tem dias posteriores na base?

    Se tem, o percentil do dia e o ranking histórico foram calculados com
    dados que ainda não existiam naquela data — vazamento de informação
    futura. Não é erro de cálculo, é limite de interpretação, e quem lê o
    resultado precisa saber.
    """
    try:
        pos = index.get_loc(label)
    except (KeyError, TypeError):
        return False
    if not isinstance(pos, (int, np.integer)):
        return False  # rótulo duplicado/fatiado: não dá para afirmar
    return int(pos) < len(index) - 1


def _percentile_of(series: pd.Series, value: float) -> float:
    """Percentil empírico (0-100) de ``value`` dentro da série histórica."""
    x = series.dropna().to_numpy(dtype=float)
    if len(x) == 0 or not np.isfinite(value):
        return np.nan
    return float(100.0 * (x <= value).mean())


def _deviation_label(pct: float) -> str:
    if not np.isfinite(pct):
        return "sem dados"
    if pct >= 90:
        return "muito acima do típico"
    if pct >= 75:
        return "acima do típico"
    if pct <= 10:
        return "muito abaixo do típico"
    if pct <= 25:
        return "abaixo do típico"
    return "dentro do típico"


def diagnose_day(result: AnalysisResult, label, top: int = 10) -> DayDiagnosis:
    """Diagnostica o dia ``label`` (um rótulo do índice de result.df)."""
    diag = DayDiagnosis(label=label, target=result.target)
    df = result.df
    scores = result.scores
    if scores is None or scores.empty or label not in df.index:
        diag.cautions.append("Dia indisponível na grade analisada.")
        return diag

    # passo de tempo da grade: é o que permite citar "3 horas antes" em vez
    # de "lag 3" na versão temporal avaliada de cada indicador
    step, _ = time_step(df.index)

    y = df[result.target]
    diag.n_history = int(y.notna().sum())
    diag.target_value = float(y.loc[label]) if np.isfinite(
        y.loc[label]) else np.nan
    diag.target_pct = _percentile_of(y, diag.target_value)

    if diag.n_history < 30:
        diag.cautions.append(
            f"Histórico curto ({diag.n_history} períodos): percentis pouco "
            "estáveis — leia como indicação aproximada."
        )

    linhas = []
    for _, row in scores.head(top).iterrows():
        param = row["parametro"]
        if param == result.target:
            continue
        r = result.per_param.get(param, {})
        best_feat = r.get("best_feature") or param
        transform = str(row.get("melhor_transformacao", "bruto"))
        # avalia o indicador na MESMA versão temporal que o ranking usou
        if best_feat != param:
            serie = single_feature(df[param], best_feat)
            if serie is None:
                serie = df[param]
        else:
            serie = df[param]
        if label not in serie.index:
            continue
        valor = serie.loc[label]
        lido_em = measured_short(feature_label(best_feat), step)
        if not np.isfinite(valor):
            linhas.append({
                "indicador": _friendly(param), "parametro": param,
                "valor no dia": np.nan, "percentil no dia": np.nan,
                "valor típico": float(np.nanmedian(serie)),
                "desvio": "sem dados no dia", "empurrão esperado": "—",
                "score histórico": float(row["score"]), "score do dia": 0.0,
                "versão avaliada": lido_em, "transformacao": feature_label(best_feat),
            })
            continue
        pct = _percentile_of(serie, float(valor))
        atip = abs(2.0 * pct / 100.0 - 1.0) if np.isfinite(pct) else 0.0
        dev_side = 1 if pct >= 50 else -1
        direcao = int(row.get("direcao", 0))
        if atip < ATYPICAL_MILD:
            push = "—"
        elif direcao == 0:
            push = "fora da faixa habitual"
        else:
            push = ("empurra para cima" if direcao * dev_side > 0
                    else "empurra para baixo")
        linhas.append({
            "indicador": _friendly(param), "parametro": param,
            "valor no dia": float(valor), "percentil no dia": round(pct, 1),
            "valor típico": float(np.nanmedian(serie)),
            "desvio": _deviation_label(pct), "empurrão esperado": push,
            "score histórico": float(row["score"]),
            "score do dia": round(float(row["score"]) * atip, 1),
            "versão avaliada": lido_em, "transformacao": feature_label(best_feat),
        })

    tab = pd.DataFrame(linhas)
    if not tab.empty:
        tab = tab.sort_values("score do dia", ascending=False).reset_index(
            drop=True)
        tab.index = tab.index + 1
    diag.rows = tab

    # ---- frases gerenciais -------------------------------------------------
    alvo = result.target
    if np.isfinite(diag.target_pct):
        diag.findings.append(
            f"Neste dia {alvo} valeu {_fmt(diag.target_value)}, "
            f"{_deviation_label(diag.target_pct)}: "
            f"{diag.target_pct:.0f}% dos dias do histórico ficaram abaixo "
            f"desse valor."
        )
    contribuintes = tab[(tab["score do dia"] >= 15)
                        & (tab["empurrão esperado"] != "—")] if not tab.empty \
        else pd.DataFrame()
    if contribuintes.empty:
        diag.findings.append(
            "Nenhum dos fatores historicamente relevantes saiu do padrão "
            "neste dia. O resultado pode vir da variação comum do processo ou "
            "de fatores que não estão na planilha."
        )
    else:
        for _, c in contribuintes.head(5).iterrows():
            push = str(c["empurrão esperado"])
            if push.startswith("empurra"):
                lado = "cima" if push.endswith("cima") else "baixo"
                # "o resultado" e não o nome do alvo de novo: ele já apareceu
                # na mesma frase, e repetir trava a leitura
                conclusao = (
                    f"é provável contribuinte para empurrar o resultado para {lado}"
                )
            else:
                conclusao = (
                    "é provável contribuinte, por ter saído da faixa habitual"
                )
            # A frase começa com "No dia" e não com o nome do indicador: o
            # nome precisaria de maiúscula inicial, e maiúscula em nome de
            # coluna é corrupção de dado — 'driver' viraria 'Driver'.
            # Duas frases: o que aconteceu no dia, e por que isso importa.
            diag.findings.append(
                f"No dia, {c['indicador']} ficou em "
                f"{_fmt(c['valor no dia'])}, {c['desvio']}"
                f"{measured_on(str(c['transformacao']), step)}. "
                f"Historicamente move {alvo} com score "
                f"{c['score histórico']:.0f} de 100, então {conclusao}."
            )
    diag.cautions.append(
        "O diagnóstico cruza o ranking histórico com o quanto cada indicador "
        "fugiu do normal neste dia. É um indício priorizado para "
        "investigação, não prova de causa."
    )
    diag.cautions.append(
        "O score do dia multiplica a relevância histórica pela atipicidade, e "
        "não mede quanto cada indicador contribuiu para o resultado. Ele não "
        "calcula o que teria acontecido se o indicador estivesse normal, não "
        "reparte a variação do alvo entre os indicadores e não traz intervalo "
        "de confiança. Responde o que investigar primeiro, não quem causou."
    )
    if _uses_future(result.df.index, label):
        diag.cautions.append(
            "Este dia não é o último da base. Tanto o ranking histórico "
            "quanto o percentil do dia foram calculados sobre a série "
            "inteira, incluindo dias posteriores a ele, então o diagnóstico "
            "usa informação que não existia naquela data. Para reconstituir "
            "uma decisão tomada no dia, refaça a análise com a base cortada "
            "ali."
        )
    return diag
