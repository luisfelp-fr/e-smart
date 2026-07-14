"""Carga de planilhas com múltiplas abas em granularidades distintas.

Cada aba é lida, tem sua coluna temporal detectada (ou recebe sequência
1..N) e suas colunas convertidas para número (decimal com vírgula ok).
A aba que contém o alvo define a grade temporal; as demais são alinhadas
a ela (agregação em famílias de métricas quando mais finas, forward-fill
quando mais grossas) — ver ``causal_analysis.aggregation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from causal_analysis.aggregation import AlignmentInfo, align_sheets
from shared.parsing import detect_time_column, read_all_sheets, to_numeric


@dataclass
class SheetInfo:
    """Diagnóstico da leitura de uma aba."""

    name: str = ""
    n_rows: int = 0
    has_dates: bool = False
    date_col: str | None = None
    columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_sheet_frame(
    raw: pd.DataFrame, name: str, date_col: str | None = None
) -> tuple[pd.DataFrame, SheetInfo]:
    """Indexa uma aba pelo tempo (ou sequência 1..N) e converte para número."""
    info = SheetInfo(name=name, n_rows=len(raw))
    df = raw.copy()
    dates, found_col, dayfirst = detect_time_column(df, date_col)
    if dates is not None:
        info.has_dates, info.date_col = True, found_col
        if not dayfirst:
            info.notes.append("Datas interpretadas no formato mês/dia (mm/dd).")
        df = df.drop(columns=[found_col])
        df.index = pd.Series(dates)
        df = df[df.index.notna()].sort_index()
        dup = int(df.index.duplicated().sum())
        if dup:
            df = df.groupby(level=0).mean(numeric_only=False)
            info.notes.append(f"{dup} data(s) duplicada(s) agregadas pela média.")
    else:
        df.index = pd.RangeIndex(1, len(df) + 1, name="ordem")
        info.notes.append(
            f"Aba '{name}' sem coluna de tempo; usando sequência 1..N."
        )

    for col in list(df.columns):
        df[col] = to_numeric(df[col])
        if df[col].notna().sum() == 0:
            df = df.drop(columns=[col])
            info.notes.append(f"Coluna sem valores numéricos descartada: {col}")
    info.columns = [str(c) for c in df.columns]
    return df, info


def load_workbook(
    path: str, sep: str | None = None
) -> tuple[dict[str, pd.DataFrame], list[SheetInfo]]:
    """Lê todas as abas (Excel) ou o CSV como aba única, já indexadas."""
    raws = read_all_sheets(path, sep=sep)
    frames: dict[str, pd.DataFrame] = {}
    infos: list[SheetInfo] = []
    for name, raw in raws.items():
        if raw.empty or raw.shape[1] == 0:
            infos.append(SheetInfo(name=name, notes=[f"Aba '{name}' vazia."]))
            continue
        df, info = load_sheet_frame(raw, name)
        if df.shape[1] > 0:
            frames[name] = df
        infos.append(info)
    if not frames:
        raise ValueError("Nenhuma aba com dados numéricos encontrada.")
    return frames, infos


def prepare_analysis_frame(
    path: str, target: str, sep: str | None = None
) -> tuple[pd.DataFrame, AlignmentInfo, list[SheetInfo]]:
    """Fluxo completo: lê todas as abas e devolve o DataFrame alinhado ao alvo."""
    frames, infos = load_workbook(path, sep=sep)
    if len(frames) == 1:
        name, df = next(iter(frames.items()))
        if target not in df.columns:
            raise ValueError(
                f"Coluna-alvo '{target}' não encontrada. Colunas: "
                f"{', '.join(df.columns.astype(str))}"
            )
        info = AlignmentInfo(target_sheet=name)
        info.sheets[name] = "aba única"
        return df, info, infos
    combined, info = align_sheets(frames, target)
    return combined, info, infos
