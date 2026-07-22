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

from .aggregation import METRIC_FRIENDLY, base_indicator, metric_of
from .features import derived_features, feature_label
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
    base = base_indicator(param)
    metric = metric_of(param)
    if metric is None:
        return f"'{base}'"
    return f"'{base}' — {METRIC_FRIENDLY.get(metric, metric)}"


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
            fam = derived_features(df[param], result.max_lag, result.windows)
            serie = fam[best_feat] if best_feat in fam.columns else df[param]
        else:
            serie = df[param]
        if label not in serie.index:
            continue
        valor = serie.loc[label]
        if not np.isfinite(valor):
            linhas.append({
                "indicador": _friendly(param), "parametro": param,
                "valor no dia": np.nan, "percentil no dia": np.nan,
                "típico (mediana)": float(np.nanmedian(serie)),
                "desvio": "sem dados no dia", "empurrão esperado": "—",
                "score histórico": float(row["score"]), "score do dia": 0.0,
                "versão avaliada": feature_label(best_feat),
            })
            continue
        pct = _percentile_of(serie, float(valor))
        atip = abs(2.0 * pct / 100.0 - 1.0) if np.isfinite(pct) else 0.0
        dev_side = 1 if pct >= 50 else -1
        direcao = int(row.get("direcao", 0))
        if direcao == 0:
            push = ("fora da faixa habitual (efeito não-monotônico)"
                    if atip >= ATYPICAL_MILD else "—")
        else:
            sobe = direcao * dev_side > 0
            push = ("alvo para CIMA" if sobe else "alvo para BAIXO") \
                if atip >= ATYPICAL_MILD else "—"
        linhas.append({
            "indicador": _friendly(param), "parametro": param,
            "valor no dia": float(valor), "percentil no dia": round(pct, 1),
            "típico (mediana)": float(np.nanmedian(serie)),
            "desvio": _deviation_label(pct), "empurrão esperado": push,
            "score histórico": float(row["score"]),
            "score do dia": round(float(row["score"]) * atip, 1),
            "versão avaliada": feature_label(best_feat),
        })

    tab = pd.DataFrame(linhas)
    if not tab.empty:
        tab = tab.sort_values("score do dia", ascending=False).reset_index(
            drop=True)
        tab.index = tab.index + 1
    diag.rows = tab

    # ---- frases gerenciais -------------------------------------------------
    alvo_txt = _deviation_label(diag.target_pct)
    if np.isfinite(diag.target_pct):
        diag.findings.append(
            f"O alvo '{result.target}' neste dia valeu "
            f"{_fmt(diag.target_value)} — percentil {diag.target_pct:.0f} do "
            f"histórico ({alvo_txt})."
        )
    contribuintes = tab[(tab["score do dia"] >= 15)
                        & (tab["empurrão esperado"] != "—")] if not tab.empty \
        else pd.DataFrame()
    if contribuintes.empty:
        diag.findings.append(
            "Nenhum dos fatores historicamente relevantes esteve fora do "
            "padrão neste dia — o resultado pode vir de variação comum do "
            "processo ou de fatores não medidos."
        )
    else:
        for _, c in contribuintes.head(5).iterrows():
            extra = ""
            if str(c["versão avaliada"]).startswith(("lag", "média")):
                extra = f" (versão avaliada: {c['versão avaliada']})"
            push = str(c["empurrão esperado"])
            if push.startswith("alvo"):
                conclusao = f"provável contribuinte para empurrar o {push}"
            else:
                conclusao = f"provável contribuinte — esteve {push}"
            diag.findings.append(
                f"{c['indicador']}: {_fmt(c['valor no dia'])} no dia — "
                f"percentil {c['percentil no dia']:.0f} do histórico, "
                f"{c['desvio']}{extra}. Historicamente move o alvo "
                f"(score {c['score histórico']:.0f}/100) → {conclusao}."
            )
    diag.cautions.append(
        "O diagnóstico do dia cruza o ranking histórico com a atipicidade do "
        "dia — é um indício priorizado para investigação, não prova causal."
    )
    return diag
