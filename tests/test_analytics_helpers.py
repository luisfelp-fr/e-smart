"""Testes da detecção de tipos de coluna da aba Analytics (formato BR)."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics.utils.helpers import (
    ColumnTypes,
    coerce_numeric_br,
    detect_column_types,
)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DataHora": ["01/02/2026 08:00", "01/02/2026 09:00",
                         "02/02/2026 08:00", "03/02/2026 08:00"],
            "Vazao": ["1.234,56", "1.240,10", "1.250,00", "1.238,75"],
            "Pressao": [3.1, 3.0, 2.9, 3.2],
            "Turno": ["A", "B", "A", "C"],
            "Observacao": [
                "parada programada para manutenção da bomba",
                "operação normal sem eventos relevantes",
                "ajuste fino no setpoint da torre",
                "troca de operador no meio do turno",
            ],
        }
    )


def test_detect_column_types_classifica_e_converte():
    df, types = detect_column_types(_fixture(), max_categorical_unique=3)
    assert isinstance(types, ColumnTypes)
    assert types.numeric == ["Vazao", "Pressao"]
    assert types.datetime == ["DataHora"]
    assert types.categorical == ["Turno"]
    assert types.text == ["Observacao"]
    # conversões aplicadas na cópia devolvida
    assert pd.api.types.is_datetime64_any_dtype(df["DataHora"])
    assert df["DataHora"].iloc[0].day == 1 and df["DataHora"].iloc[0].month == 2
    assert df["Vazao"].iloc[0] == 1234.56  # "1.234,56" -> float
    assert types.kind_of("Vazao") == "Numérica"
    assert types.kind_of("Turno") == "Categórica"


def test_detect_column_types_nao_altera_original():
    original = _fixture()
    before = original.copy(deep=True)
    detect_column_types(original)
    pd.testing.assert_frame_equal(original, before)


def test_coerce_numeric_br_mantem_texto_nao_numerico():
    s = pd.Series(["abc", "def", "12,5", "ghi"])
    out = coerce_numeric_br(s)
    # maioria não converte -> série original preservada
    assert out.equals(s)


def test_coerce_numeric_br_converte_percentuais_br():
    s = pd.Series(["10,5", "1.000,25", "3,14"])
    out = coerce_numeric_br(s)
    assert list(out) == [10.5, 1000.25, 3.14]
