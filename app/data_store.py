"""Base de dados compartilhada entre as páginas do app.

A planilha enviada na barra lateral é lida uma única vez (cache por caminho
de arquivo) e alimenta todos os módulos: cada página só precisa selecionar a
aba e o indicador, sem re-parsear o arquivo a cada rerun.

Os caches do Streamlit são GLOBAIS ao processo, não por sessão: o
``max_entries`` é dividido entre todo mundo que estiver usando o app. Os
padrões baixos foram dimensionados para o Streamlit Community Cloud (~2,7 GB
de RAM) com um usuário — com várias pessoas e planilhas diferentes eles viram
o gargalo, porque quase toda leitura passa a ser cache miss e o arquivo é
reparseado do zero.

Por isso agora são ajustáveis por ambiente (``ESMART_CACHE_*``): a mesma
imagem serve do plano gratuito a uma VM corporativa. Regra prática: com N
pessoas usando arquivos distintos, ``ESMART_CACHE_SHEETS`` perto de N evita
o reparse constante — ao custo de reter N planilhas em memória.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from analytics.utils.helpers import ColumnTypes, detect_column_types
from capability.data_prep import PrepDiagnostics, load_indicator_table
from shared.parsing import read_all_sheets


def _cap(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


CACHE_SHEETS = _cap("ESMART_CACHE_SHEETS", 2)
CACHE_INDICATOR = _cap("ESMART_CACHE_INDICATOR", 8)
CACHE_ANALYTICS = _cap("ESMART_CACHE_ANALYTICS", 4)


@st.cache_data(show_spinner=False, max_entries=CACHE_SHEETS)
def get_all_sheets(file_path: str) -> dict[str, pd.DataFrame]:
    """Todas as abas cruas da planilha — uma leitura para o app inteiro."""
    return read_all_sheets(file_path)


def sheet_names(file_path: str) -> list[str]:
    return list(get_all_sheets(file_path))


@st.cache_data(show_spinner=False, max_entries=CACHE_INDICATOR)
def get_indicator_table(
    file_path: str, sheet: str
) -> tuple[pd.DataFrame, PrepDiagnostics]:
    """Tabela numérica/indexada usada pelo Módulo 1 (cacheada por aba)."""
    return load_indicator_table(file_path, sheet=sheet)


@st.cache_data(show_spinner=False, max_entries=CACHE_ANALYTICS)
def get_analytics_base(
    file_path: str, sheet: str
) -> tuple[pd.DataFrame, ColumnTypes]:
    """Aba crua com tipos detectados (numérica/categórica/data/texto).

    A aba Analytics precisa das colunas categóricas e de data que o
    ``shared.io_loader`` descarta — por isso parte da aba crua.
    """
    return detect_column_types(get_all_sheets(file_path)[sheet])
