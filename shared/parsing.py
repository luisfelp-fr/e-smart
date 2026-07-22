"""Primitivas de leitura e conversão de dados brasileiros (CSV/Excel).

Concentra a detecção automática de separador (',', ';', tab), datas em
formato brasileiro (dd/mm/aaaa) e números com vírgula decimal / ponto de
milhar, para que todos os módulos leiam arquivos da mesma forma.
"""

from __future__ import annotations

import os

import pandas as pd

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".ods"}

# leitor que o pandas usa por extensão — para a mensagem de erro amigável
_ENGINE_BY_EXT = {".xls": "xlrd", ".ods": "odfpy"}


def _read_excel(path: str, sheet_name):
    """pd.read_excel com erro claro quando falta o leitor do formato."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError as e:
        ext = os.path.splitext(path)[1].lower()
        pkg = _ENGINE_BY_EXT.get(ext, "openpyxl")
        raise ValueError(
            f"Este ambiente não tem o leitor de arquivos {ext} instalado "
            f"(pacote '{pkg}'). Instale com `pip install {pkg}` ou salve a "
            "planilha como .xlsx/.csv e envie novamente."
        ) from e


def _detect_sep(path: str) -> str | None:
    """Detecta ';', tab ou ',' com robustez a decimal com vírgula.

    O sniffer do pandas confunde a vírgula DECIMAL ("10,5") com separador em
    arquivos de uma coluna só. Aqui cada candidato é testado com csv.reader
    (que respeita campos entre aspas): só é aceito se produzir contagem de
    campos CONSTANTE e >= 2 nas primeiras linhas; ';' e tab têm prioridade
    sobre ','. Sem candidato consistente, o arquivo é lido como coluna única.
    """
    import csv

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\r\n") for _, ln in zip(range(50), f)
                     if ln.strip()]
    except OSError:
        return None
    if not lines:
        return None
    for cand in (";", "\t", ","):
        try:
            rows = [r for r in csv.reader(lines, delimiter=cand) if r]
        except csv.Error:
            continue
        counts = {len(r) for r in rows}
        if len(counts) == 1 and counts.pop() >= 2:
            return cand
    return ";"  # coluna única: qualquer separador ausente serve


def read_raw(path: str, sep: str | None = None, sheet: int | str = 0) -> pd.DataFrame:
    """Lê CSV ou Excel; para CSV, sep=None detecta ';', tab ou ','."""
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXTS:
        return _read_excel(path, sheet_name=sheet)
    if sep is None:
        sep = _detect_sep(path)
    return pd.read_csv(path, sep=sep, engine="python")


def read_all_sheets(path: str, sep: str | None = None) -> dict[str, pd.DataFrame]:
    """Lê todas as abas de um Excel (ou o CSV inteiro como aba única).

    Devolve {nome da aba: DataFrame bruto}. Para CSV o nome da aba é o nome
    do arquivo sem extensão.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXTS:
        sheets = _read_excel(path, sheet_name=None)
        return {str(k): v for k, v in sheets.items()}
    name = os.path.splitext(os.path.basename(path))[0]
    if sep is None:
        sep = _detect_sep(path)
    return {name: pd.read_csv(path, sep=sep, engine="python")}


def datelike_columns(df: pd.DataFrame, exclude: tuple = ()) -> list[str]:
    """Colunas que parecem conter datas (para avisar quando fora da 1ª posição).

    Uma coluna conta se não é numérica e ao menos metade dos valores parseia
    como data — mesmo critério do detector oficial, que só olha a 1ª coluna.
    """
    hits: list[str] = []
    for c in df.columns:
        if str(c) in {str(e) for e in exclude}:
            continue
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            hits.append(str(c))
            continue
        if getattr(col.dtype, "kind", "O") in "ifu":
            continue
        parsed, _ = parse_dates(col)
        if to_numeric(col).notna().sum() >= parsed.notna().sum():
            continue
        if parsed.notna().sum() >= len(col) * 0.5:
            hits.append(str(c))
    return hits


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
    """Converte para número aceitando decimal com vírgula, milhar e sufixo %.

    Três tentativas, fica a que mais converte (empate favorece a primeira):
    1. direta ("12.5", 42);
    2. direta sem o sufixo '%' ("85.3%" -> 85.3 — o valor é mantido na
       escala 0-100, sem dividir por 100);
    3. formato brasileiro sem '%' ("1.234,5" -> 1234.5; "85,3%" -> 85.3).
    """
    direct = pd.to_numeric(series, errors="coerce")
    if series.dtype.kind in "ifu" or direct.notna().sum() >= series.notna().sum():
        return direct
    as_str = series.astype(str).str.strip().str.rstrip("%").str.strip()
    no_pct = pd.to_numeric(as_str, errors="coerce")
    cleaned = (
        as_str
        .str.replace(".", "", regex=False)  # separador de milhar
        .str.replace(",", ".", regex=False)  # decimal brasileiro
    )
    br = pd.to_numeric(cleaned, errors="coerce")
    best = direct
    for cand in (no_pct, br):
        if cand.notna().sum() > best.notna().sum():
            best = cand
    return best
