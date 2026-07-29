"""Base de dados compartilhada entre as páginas do app.

A planilha enviada na barra lateral é lida uma única vez (cache por caminho
de arquivo) e alimenta todos os módulos: cada página só precisa selecionar a
aba e o indicador, sem re-parsear o arquivo a cada rerun.

Os ``max_entries`` são propositalmente baixos: o app roda no Streamlit
Community Cloud (~2,7 GB de RAM) e o cache não deve reter várias planilhas.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.utils.helpers import ColumnTypes, detect_column_types
from capability.data_prep import PrepDiagnostics, load_indicator_table
from shared.parsing import read_all_sheets


@st.cache_data(show_spinner=False, max_entries=2)
def get_all_sheets(file_path: str) -> dict[str, pd.DataFrame]:
    """Todas as abas cruas da planilha — uma leitura para o app inteiro."""
    return read_all_sheets(file_path)


def sheet_names(file_path: str) -> list[str]:
    return list(get_all_sheets(file_path))


@st.cache_data(show_spinner=False, max_entries=8)
def get_indicator_table(
    file_path: str, sheet: str
) -> tuple[pd.DataFrame, PrepDiagnostics]:
    """Tabela numérica/indexada usada pelo Módulo 1 (cacheada por aba)."""
    return load_indicator_table(file_path, sheet=sheet)


@st.cache_data(show_spinner=False, max_entries=4)
def get_analytics_base(
    file_path: str, sheet: str
) -> tuple[pd.DataFrame, ColumnTypes]:
    """Aba crua com tipos detectados (numérica/categórica/data/texto).

    A aba Analytics precisa das colunas categóricas e de data que o
    ``shared.io_loader`` descarta — por isso parte da aba crua.
    """
    return detect_column_types(get_all_sheets(file_path)[sheet])
