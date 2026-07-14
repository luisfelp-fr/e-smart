"""Primitivas de leitura e conversão de dados brasileiros (CSV/Excel).

Concentra a detecção automática de separador (',', ';', tab), datas em
formato brasileiro (dd/mm/aaaa) e números com vírgula decimal / ponto de
milhar, para que todos os módulos leiam arquivos da mesma forma.
"""

from __future__ import annotations

import os

import pandas as pd

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".ods"}


def read_raw(path: str, sep: str | None = None, sheet: int | str = 0) -> pd.DataFrame:
    """Lê CSV ou Excel; para CSV, sep=None deixa o pandas detectar ',', ';' ou tab."""
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXTS:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, sep=sep, engine="python")


def read_all_sheets(path: str, sep: str | None = None) -> dict[str, pd.DataFrame]:
    """Lê todas as abas de um Excel (ou o CSV inteiro como aba única).

    Devolve {nome da aba: DataFrame bruto}. Para CSV o nome da aba é o nome
    do arquivo sem extensão.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXTS:
        sheets = pd.read_excel(path, sheet_name=None)
        return {str(k): v for k, v in sheets.items()}
    name = os.path.splitext(os.path.basename(path))[0]
    return {name: pd.read_csv(path, sep=sep, engine="python")}


def parse_dates(series: pd.Series) -> tuple[pd.Series, bool]:
    """Converte a coluna de datas testando dd/mm e mm/dd; fica com o que parseia mais."""
    as_str = series.astype(str).str.strip()
    day_first = pd.to_datetime(as_str, errors="coerce", dayfirst=True, format="mixed")
    month_first = pd.to_datetime(as_str, errors="coerce", dayfirst=False, format="mixed")
    # em empate (datas não ambíguas) preferimos dd/mm, padrão brasileiro
    if month_first.notna().sum() > day_first.notna().sum():
        return month_first, False
    return day_first, True


def detect_time_column(
    df: pd.DataFrame, date_col: str | None = None
) -> tuple[pd.Series | None, str | None, bool]:
    """Detecta a coluna temporal (a indicada ou a primeira) com guarda numérica.

    Números ("10,5", "42") também "parseiam" como datas; uma coluna só é
    aceita como tempo se parsear MELHOR como data do que como número.
    Devolve (datas ou None, nome da coluna, dayfirst).
    """
    candidate = date_col or (df.columns[0] if len(df.columns) else None)
    if candidate is None or candidate not in df.columns:
        return None, None, True
    col = df[candidate]
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.Series(col), str(candidate), True
    parsed, dayfirst = parse_dates(col)
    if to_numeric(col).notna().sum() >= parsed.notna().sum():
        return None, None, dayfirst
    if parsed.notna().sum() >= len(df) * 0.5:
        return parsed, str(candidate), dayfirst
    return None, None, dayfirst


def to_numeric(series: pd.Series) -> pd.Series:
    """Converte para número aceitando decimal com vírgula e milhar com ponto."""
    direct = pd.to_numeric(series, errors="coerce")
    if series.dtype.kind in "ifu" or direct.notna().sum() >= series.notna().sum():
        return direct
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)  # separador de milhar
        .str.replace(",", ".", regex=False)  # decimal brasileiro
    )
    br = pd.to_numeric(cleaned, errors="coerce")
    return br if br.notna().sum() > direct.notna().sum() else direct
