"""
helpers.py — Utilitários centrais da aba Analytics
==================================================

Reúne funções de:
  * coerção numérica com suporte a decimal brasileiro;
  * detecção automática do tipo de cada coluna (numérica, categórica,
    data/hora, texto);
  * acesso rápido aos subconjuntos de colunas;
  * adição de análises ao relatório unificado do app.

A leitura de arquivos fica fora daqui: a base vem do upload global da barra
lateral (``shared.parsing.read_all_sheets`` via ``app/data_store.py``) — a aba
Analytics só recebe a aba crua e classifica os tipos de coluna.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

# Fuso de Brasília para os carimbos de data/hora do relatório. Importante porque
# servidores (ex.: Streamlit Community Cloud) rodam em UTC — sem isso o relatório
# mostraria o horário errado. Usa a base de fusos (DST histórico) quando
# disponível e cai para o offset fixo −03:00 caso o tzdata não exista.
try:
    from zoneinfo import ZoneInfo
    _BR_TZ = ZoneInfo("America/Sao_Paulo")
except Exception:  # pragma: no cover - ambiente sem tzdata
    _BR_TZ = timezone(timedelta(hours=-3))

try:  # streamlit é opcional para permitir testes/compilação isolados
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False


# --------------------------------------------------------------------------- #
# 1. Coerção numérica (formato BR)
# --------------------------------------------------------------------------- #
def coerce_numeric_br(series: pd.Series) -> pd.Series:
    """Tenta converter uma série de texto para número tratando o formato BR.

    Trata separador de milhar ``.`` e vírgula decimal ``,`` (ex.: ``"1.234,56"``
    vira ``1234.56``). Se a conversão falhar para a maioria dos valores, retorna
    a série original (assume que não é numérica).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    raw = series.astype(str).str.strip()
    cleaned = (
        raw.str.replace(".", "", regex=False)   # separador de milhar
        .str.replace(",", ".", regex=False)      # vírgula decimal -> ponto
    )
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Só aceita como numérica se converteu a grande maioria dos não-nulos.
    non_null = series.notna().sum()
    if non_null > 0 and converted.notna().sum() >= 0.8 * non_null:
        return converted
    return series


# --------------------------------------------------------------------------- #
# 2. Detecção de tipos de coluna
# --------------------------------------------------------------------------- #
@dataclass
class ColumnTypes:
    """Resultado da classificação automática das colunas."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def kind_of(self, col: str) -> str:
        if col in self.numeric:
            return "Numérica"
        if col in self.datetime:
            return "Data/Hora"
        if col in self.categorical:
            return "Categórica"
        return "Texto"


def _looks_datetime(series: pd.Series) -> bool:
    """Heurística: a série parece ser data/hora?"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= 0.8


def detect_column_types(
    df: pd.DataFrame, max_categorical_unique: int = 20
) -> tuple[pd.DataFrame, ColumnTypes]:
    """Classifica cada coluna e devolve ``(df_convertido, ColumnTypes)``.

    Aplica conversões *in-place* numa cópia: datas viram ``datetime64`` e
    números em texto BR viram ``float``. A classificação categórica usa baixa
    cardinalidade como critério.
    """
    out = df.copy()
    types = ColumnTypes()

    for col in out.columns:
        series = out[col]

        # 1) Já é datetime64? (caso o arquivo já traga o tipo)
        if pd.api.types.is_datetime64_any_dtype(series):
            types.datetime.append(col)
            continue

        # 2) Numérica primeiro (inclui texto BR convertível). Evita que números
        #    com vírgula decimal — ex.: "50,23" — sejam confundidos com datas.
        converted = coerce_numeric_br(series)
        if pd.api.types.is_numeric_dtype(converted):
            out[col] = converted
            types.numeric.append(col)
            continue

        # 3) Data/hora (apenas para colunas não numéricas)
        if _looks_datetime(series):
            out[col] = pd.to_datetime(series, errors="coerce", dayfirst=True)
            types.datetime.append(col)
            continue

        # 4) Categórica vs. texto livre, por cardinalidade
        n_unique = series.nunique(dropna=True)
        if n_unique <= max_categorical_unique:
            types.categorical.append(col)
        else:
            types.text.append(col)

    return out, types


# --------------------------------------------------------------------------- #
# 3. Acesso rápido a subconjuntos de colunas
# --------------------------------------------------------------------------- #
def numeric_cols(state) -> list[str]:
    return list(state.get("col_types").numeric) if state.get("col_types") else []


def categorical_cols(state) -> list[str]:
    return list(state.get("col_types").categorical) if state.get("col_types") else []


def datetime_cols(state) -> list[str]:
    return list(state.get("col_types").datetime) if state.get("col_types") else []


# --------------------------------------------------------------------------- #
# 4. Session state
# --------------------------------------------------------------------------- #
def get_df():
    """Atalho seguro para o DataFrame atual da aba Analytics (ou None)."""
    if not _HAS_ST:
        return None
    return st.session_state.get("ax_df")


def has_data() -> bool:
    df = get_df()
    return df is not None and not df.empty


def add_to_report(item: dict[str, Any]) -> None:
    """Adiciona a análise ao relatório unificado do app (página 📄 Relatório).

    O item no formato do Argus é convertido para o esquema do relatório do
    e-smart e deduplicado por ``id`` (mesma análise com as mesmas variáveis e
    parâmetros entra uma vez só).
    """
    if not _HAS_ST:
        return
    from analytics.report_adapter import adapt_report_item

    from shared.limits import MAX_REPORT_ITEMS

    adapted = adapt_report_item(item)
    items = st.session_state.setdefault("report_items", [])
    if any(existing["id"] == adapted["id"] for existing in items):
        st.info("✓ Esta análise já está no relatório.", icon="📄")
        return
    if len(items) >= MAX_REPORT_ITEMS:
        st.warning(
            f"O relatório atingiu o limite de {MAX_REPORT_ITEMS} análises. "
            "Baixe o PDF e limpe a lista para continuar adicionando.",
            icon="📄",
        )
        return
    items.append(adapted)
    st.success("Análise adicionada! Veja e baixe na página **📄 Relatório**.")


def make_report_item(
    name: str,
    variables: dict[str, Any],
    params: dict[str, Any],
    interpretation: str,
    figures: list | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Cria o dicionário padronizado de uma análise para o relatório."""
    return {
        "name": name,
        "variables": variables or {},
        "params": params or {},
        "interpretation": interpretation or "",
        "figures": figures or [],
        "tables": tables or {},
        "timestamp": timestamp or "",
    }


def now_str() -> str:
    """Data/hora atual no fuso de **Brasília**, formatada (pt-BR).

    Independe do fuso do servidor onde o app roda (ex.: UTC na nuvem).
    """
    return datetime.now(_BR_TZ).strftime("%d/%m/%Y %H:%M:%S")


# --------------------------------------------------------------------------- #
# 5. Diversos
# --------------------------------------------------------------------------- #
def df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Serializa um dicionário {nome_aba: DataFrame} em bytes de um .xlsx."""
    from shared.safety import neutralize_formulas

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            safe = str(sheet_name)[:31]  # limite de nome de aba do Excel
            if isinstance(data, pd.DataFrame):
                # conteúdo vindo da planilha do usuário: célula começando com
                # "=" seria executada como fórmula por quem abrisse o arquivo
                neutralize_formulas(data).to_excel(
                    writer, sheet_name=safe, index=True)
            else:
                pd.DataFrame({"info": [str(data)]}).to_excel(
                    writer, sheet_name=safe, index=False
                )
    return buffer.getvalue()


def safe_dropna_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Alinha duas séries removendo pares com NaN. Retorna arrays numpy."""
    pair = pd.concat([x, y], axis=1).dropna()
    return pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
