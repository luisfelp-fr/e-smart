"""Alinhamento de séries com granularidades distintas à grade do alvo.

Dados industriais chegam em frequências diferentes (segundos, minutos,
turnos...). Para correlacionar tudo com o alvo, cada série mais FINA é
agregada em janelas que terminam em cada carimbo de tempo do alvo — e a
média sozinha não basta: um pico curto de temperatura ou o tempo passado
em faixa ruim somem na média. Por isso cada coluna fina gera uma família
de métricas por janela:

- média, mediana, mínimo, máximo, desvio-padrão;
- percentis 10 e 90 (comportamento das caudas);
- % do tempo acima do Q3 global e abaixo do Q1 global da própria série
  ("tempo em faixa alta/baixa" — captura permanência em região ruim).

Séries mais GROSSAS que o alvo são propagadas (forward-fill) com aviso.
As colunas derivadas entram na análise causal como parâmetros normais, e o
efeito de defasagem (lag) de cada uma é varrido pelo pipeline existente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# sufixos das métricas derivadas (nomes curtos e legíveis no ranking)
METRIC_LABELS = {
    "media": "média",
    "mediana": "mediana",
    "min": "mínimo",
    "max": "máximo",
    "desvio": "desvio",
    "p10": "P10",
    "p90": "P90",
    "pct_acima_q3": "% tempo>Q3",
    "pct_abaixo_q1": "% tempo<Q1",
}

# descrições amigáveis para tooltips/relatório gerencial
METRIC_FRIENDLY = {
    "média": "valor médio na janela",
    "mediana": "valor central na janela (robusto a picos)",
    "mínimo": "menor valor atingido na janela",
    "máximo": "maior valor atingido na janela (picos)",
    "desvio": "instabilidade dentro da janela",
    "P10": "piso típico na janela (percentil 10)",
    "P90": "teto típico na janela (percentil 90)",
    "% tempo>Q3": "fração do tempo em faixa alta (acima do quartil superior)",
    "% tempo<Q1": "fração do tempo em faixa baixa (abaixo do quartil inferior)",
}


@dataclass
class AlignmentInfo:
    """Diagnóstico do alinhamento das abas à grade do alvo."""

    target_sheet: str = ""
    target_step: str = ""
    sheets: dict[str, str] = field(default_factory=dict)  # aba -> como foi tratada
    notes: list[str] = field(default_factory=list)


def infer_step(index: pd.Index) -> pd.Timedelta | float | None:
    """Passo mediano do índice (Timedelta para datas, float para sequência)."""
    if len(index) < 3:
        return None
    s = pd.Series(index)
    try:
        return s.diff().median()
    except TypeError:
        return None


def fmt_timedelta_br(td) -> str:
    """Duração em linguagem natural pt-BR ("12 horas", "3,5 dias")."""
    secs = float(pd.Timedelta(td).total_seconds())
    units = [
        (604800.0, ("semana", "semanas")),
        (86400.0, ("dia", "dias")),
        (3600.0, ("hora", "horas")),
        (60.0, ("minuto", "minutos")),
        (1.0, ("segundo", "segundos")),
    ]

    def fmt(v: float, sing: str, plur: str) -> str:
        txt = f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",")
        return f"{txt} {sing if abs(v - 1.0) < 1e-9 else plur}"

    # prefere a maior unidade que expressa a duração sem arredondar
    # (28 horas fica "28 horas", não "1,2 dias"); senão, aproxima na maior
    for size, (sing, plur) in units:
        v = secs / size
        if 1.0 <= v < 1000.0 and abs(v - round(v, 1)) < 1e-9:
            return fmt(v, sing, plur)
    for size, (sing, plur) in units:
        v = secs / size
        if v >= 1.0:
            return fmt(v, sing, plur)
    return "menos de 1 segundo"


def fmt_step(step) -> str:
    """Passo do índice em linguagem natural ('4 horas', '2 linhas')."""
    if step is None:
        return "desconhecido"
    if isinstance(step, pd.Timedelta):
        return fmt_timedelta_br(step)
    v = f"{float(step):g}".replace(".", ",")
    return f"{v} linha" if float(step) == 1 else f"{v} linhas"


def reduce_to_scale(
    df: pd.DataFrame, max_rows: int = 15000
) -> tuple[pd.DataFrame, str | None]:
    """Agrega linhas consecutivas pela média até caber em ``max_rows``.

    Séries muito longas (ex.: minuto a minuto por meses) tornam o Módulo 2
    lento e podem estourar a memória. Em vez de subamostrar (o que destruiria
    a estrutura de defasagens), agregamos blocos de ``fator`` linhas
    consecutivas — o equivalente a reamostrar para uma grade mais grossa, o
    que PRESERVA os efeitos com atraso/permanência, apenas numa escala de
    tempo maior. As etapas do módulo já traduzem os lags para essa escala.

    Devolve (df reduzido, nota explicativa) ou (df original, None).
    """
    n = len(df)
    if n <= max_rows or n < 2:
        return df, None
    factor = int(np.ceil(n / max_rows))
    if factor < 2:
        return df, None
    groups = np.arange(n) // factor
    reduced = df.groupby(groups).mean(numeric_only=True)
    # índice representativo: o primeiro rótulo de cada bloco (mantém o tipo e,
    # para datas, faz o passo virar ~fator × passo original)
    reduced.index = df.index[::factor][: len(reduced)]
    reduced.index.name = df.index.name
    old_step = infer_step(df.index)
    new_step = infer_step(reduced.index)
    detalhe = ""
    if isinstance(old_step, pd.Timedelta) and isinstance(new_step, pd.Timedelta):
        detalhe = (f" — o passo passou de ~{fmt_step(old_step)} para "
                   f"~{fmt_step(new_step)}")
    nota = (
        f"Volume alto: {n:,} linhas foram agregadas pela média em blocos de "
        f"{factor} para {len(reduced):,} linhas{detalhe}. Isso mantém os "
        "efeitos com atraso/permanência (numa escala de tempo maior) e evita "
        "lentidão/estouro de memória. Reduza este limite para acelerar mais, "
        "ou aumente-o para mais detalhe."
    ).replace(",", ".")
    return reduced, nota


def aggregate_to_grid(
    fine: pd.DataFrame, grid: pd.Index, prefix: str = ""
) -> pd.DataFrame:
    """Agrega um DataFrame fino em janelas (t_{i-1}, t_i] da grade do alvo.

    Cada coluna vira uma família ``coluna (métrica)``; os limiares Q1/Q3 do
    "% do tempo em faixa" são os quartis GLOBAIS da própria coluna fina.
    """
    grid = pd.Index(grid).sort_values()
    step = infer_step(grid)
    if step is None:
        raise ValueError("Grade do alvo precisa de ao menos 3 pontos.")
    edges = [grid[0] - step] + list(grid)
    binned = pd.cut(fine.index, bins=pd.Index(edges), labels=grid)

    out: dict[str, pd.Series] = {}
    for col in fine.columns:
        x = fine[col].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        g = x.groupby(binned, observed=False)
        name = f"{prefix}{col}"
        out[f"{name} (média)"] = g.mean()
        out[f"{name} (mediana)"] = g.median()
        out[f"{name} (mínimo)"] = g.min()
        out[f"{name} (máximo)"] = g.max()
        out[f"{name} (desvio)"] = g.std()
        out[f"{name} (P10)"] = g.quantile(0.10)
        out[f"{name} (P90)"] = g.quantile(0.90)
        out[f"{name} (% tempo>Q3)"] = g.apply(
            lambda v: 100.0 * (v > q3).mean() if v.notna().any() else np.nan
        )
        out[f"{name} (% tempo<Q1)"] = g.apply(
            lambda v: 100.0 * (v < q1).mean() if v.notna().any() else np.nan
        )
    agg = pd.DataFrame(out)
    agg.index = pd.Index(agg.index, name=grid.name)
    return agg.reindex(grid)


def align_sheets(
    sheets: dict[str, pd.DataFrame],
    target: str,
    max_cols: int = 400,
) -> tuple[pd.DataFrame, AlignmentInfo]:
    """Combina abas de granularidades distintas na grade temporal do alvo.

    ``sheets``: {nome da aba: DataFrame já numérico e indexado por tempo}.
    A aba que contém a coluna-alvo define a grade. Abas mais finas são
    agregadas em famílias de métricas; abas mais grossas (ou de passo
    similar) são reamostradas por forward-fill, com aviso.
    """
    info = AlignmentInfo()
    target_sheet = next(
        (name for name, df in sheets.items() if target in df.columns), None
    )
    if target_sheet is None:
        raise ValueError(
            f"A coluna-alvo '{target}' não foi encontrada em nenhuma aba."
        )
    info.target_sheet = target_sheet
    base = sheets[target_sheet].copy()
    grid = base.index
    step_t = infer_step(grid)
    info.target_step = str(step_t)

    combined = [base]
    for name, df in sheets.items():
        if name == target_sheet:
            info.sheets[name] = "aba do alvo (grade de referência)"
            continue
        # evita colisão de nomes entre abas
        dup = [c for c in df.columns if c in base.columns]
        prefix = f"{name}: " if dup else ""
        step_s = infer_step(df.index)
        if not _comparable(step_t, step_s):
            info.sheets[name] = "ignorada (eixo de tempo incompatível com o do alvo)"
            info.notes.append(
                f"Aba '{name}' ignorada: o tipo do eixo temporal não é "
                "comparável ao da aba do alvo (ex.: datas vs. sequência)."
            )
            continue
        if step_s is not None and step_t is not None and step_s < step_t * 0.75:
            agg = aggregate_to_grid(df, grid, prefix=prefix)
            info.sheets[name] = (
                f"mais fina (passo {fmt_step(step_s)}) — agregada em famílias "
                "de métricas"
            )
            combined.append(agg)
        else:
            coarse = df.copy()
            if prefix:
                coarse.columns = [f"{prefix}{c}" for c in coarse.columns]
            coarse = coarse.reindex(grid.union(coarse.index)).sort_index()
            coarse = coarse.ffill().reindex(grid)
            info.sheets[name] = (
                f"passo {fmt_step(step_s)} ≥ passo do alvo — propagada "
                "(forward-fill)"
            )
            info.notes.append(
                f"Aba '{name}' tem granularidade igual/mais grossa que o alvo; "
                "valores propagados até a próxima observação."
            )
            combined.append(coarse)

    out = pd.concat(combined, axis=1)
    if out.shape[1] > max_cols:
        info.notes.append(
            f"{out.shape[1]} colunas geradas; mantidas as {max_cols} com mais "
            "dados válidos para viabilizar a análise."
        )
        keep = out.notna().sum().sort_values(ascending=False).index[:max_cols]
        keep = [c for c in out.columns if c in set(keep) or c == target]
        out = out[keep]
    return out, info


def _comparable(a, b) -> bool:
    if a is None or b is None:
        return True
    return isinstance(a, pd.Timedelta) == isinstance(b, pd.Timedelta)


def base_indicator(column: str) -> str:
    """Nome do indicador original a partir da coluna derivada.

    "forno: temp (máximo)" -> "temp"; "temp (P90)" -> "temp"; "temp" -> "temp".
    """
    name = column
    if ": " in name:
        name = name.split(": ", 1)[1]
    if name.endswith(")") and " (" in name:
        name = name[: name.rfind(" (")]
    return name


def metric_of(column: str) -> str | None:
    """Métrica da coluna derivada ("temp (P90)" -> "P90"), ou None se bruta."""
    if column.endswith(")") and " (" in column:
        return column[column.rfind(" (") + 2 : -1]
    return None
