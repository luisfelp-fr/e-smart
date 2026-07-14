"""Carregamento, limpeza e diagnóstico da tabela de entrada.

Regras de entrada: a primeira coluna (ou a indicada em ``date_col``) contém
datas; todas as demais colunas são parâmetros numéricos. Formatos brasileiros
(dd/mm/aaaa, decimal com vírgula) são detectados automaticamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.parsing import EXCEL_EXTS, parse_dates, read_raw, to_numeric  # noqa: F401


@dataclass
class LoadDiagnostics:
    """Resumo do que foi feito com os dados na carga."""

    n_rows_raw: int = 0
    n_rows_used: int = 0
    date_col: str = ""
    freq: str | None = None
    date_start: str = ""
    date_end: str = ""
    duplicated_dates: int = 0
    missing_pct: dict[str, float] = field(default_factory=dict)
    interpolated: dict[str, int] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# As primitivas de leitura/parsing vivem em shared.parsing e são reusadas
# pelos demais módulos; os aliases privados preservam a API interna antiga.
_read_raw = read_raw
_parse_dates = parse_dates
_to_numeric = to_numeric


def load_table(
    path: str,
    target: str,
    date_col: str | None = None,
    sep: str | None = None,
    sheet: int | str = 0,
    interpolate_limit: int = 3,
    min_valid_pct: float = 0.5,
) -> tuple[pd.DataFrame, LoadDiagnostics]:
    """Carrega a tabela e devolve (DataFrame indexado por data, diagnóstico).

    Parâmetros com menos de ``min_valid_pct`` de valores válidos são
    descartados; lacunas de até ``interpolate_limit`` pontos consecutivos são
    interpoladas linearmente (o alvo nunca é interpolado).
    """
    diag = LoadDiagnostics()
    df = _read_raw(path, sep=sep, sheet=sheet)
    diag.n_rows_raw = len(df)
    if df.shape[1] < 2:
        raise ValueError(
            "A tabela precisa de ao menos 2 colunas: datas na primeira e "
            "parâmetros nas demais."
        )

    date_col = date_col or df.columns[0]
    if date_col not in df.columns:
        raise ValueError(f"Coluna de datas '{date_col}' não encontrada na tabela.")
    diag.date_col = str(date_col)

    dates, dayfirst = _parse_dates(df[date_col])
    if dates.notna().sum() < len(df) * 0.5:
        raise ValueError(
            f"Não foi possível interpretar a coluna '{date_col}' como datas."
        )
    if not dayfirst:
        diag.notes.append("Datas interpretadas no formato mês/dia (mm/dd).")

    bad_dates = int(dates.isna().sum())
    if bad_dates:
        diag.notes.append(f"{bad_dates} linha(s) com data inválida foram removidas.")

    df = df.drop(columns=[date_col])
    df.index = dates
    df = df[df.index.notna()].sort_index()

    dup = int(df.index.duplicated().sum())
    if dup:
        diag.duplicated_dates = dup
        df = df.groupby(level=0).mean(numeric_only=False)
        diag.notes.append(
            f"{dup} data(s) duplicada(s) foram agregadas pela média."
        )

    if target not in df.columns:
        raise ValueError(
            f"Coluna-alvo '{target}' não encontrada. Colunas disponíveis: "
            f"{', '.join(map(str, df.columns))}"
        )

    # conversão numérica coluna a coluna
    for col in list(df.columns):
        df[col] = _to_numeric(df[col])
        valid = df[col].notna().mean()
        diag.missing_pct[str(col)] = round(100 * (1 - valid), 1)
        if valid < min_valid_pct and col != target:
            df = df.drop(columns=[col])
            diag.dropped_columns.append(str(col))
    if diag.dropped_columns:
        diag.notes.append(
            "Colunas descartadas por excesso de valores ausentes/não numéricos: "
            + ", ".join(diag.dropped_columns)
        )
    if df[target].notna().sum() < 20:
        raise ValueError(
            f"A coluna-alvo '{target}' tem menos de 20 valores numéricos válidos."
        )

    # parâmetros constantes não carregam informação
    for col in [c for c in df.columns if c != target]:
        if df[col].nunique(dropna=True) <= 1:
            df = df.drop(columns=[col])
            diag.dropped_columns.append(str(col))
            diag.notes.append(f"Coluna constante descartada: {col}")

    # interpolação curta apenas nos parâmetros (nunca no alvo)
    params = [c for c in df.columns if c != target]
    if interpolate_limit > 0:
        for col in params:
            before = int(df[col].isna().sum())
            df[col] = df[col].interpolate(
                method="linear", limit=interpolate_limit, limit_area="inside"
            )
            filled = before - int(df[col].isna().sum())
            if filled:
                diag.interpolated[str(col)] = filled

    df = df[df[target].notna()]
    diag.n_rows_used = len(df)
    diag.freq = pd.infer_freq(df.index)
    if diag.freq is None and len(df) > 2:
        median_step = df.index.to_series().diff().median()
        diag.notes.append(
            f"Frequência irregular; passo mediano entre observações: {median_step}."
        )
    diag.date_start = str(df.index.min().date())
    diag.date_end = str(df.index.max().date())
    return df, diag
