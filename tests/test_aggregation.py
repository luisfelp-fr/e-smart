"""Testes do alinhamento multi-aba, métricas de agregação e relatório gerencial."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_analysis.aggregation import (
    aggregate_to_grid,
    align_sheets,
    base_indicator,
    metric_of,
)
from causal_analysis.managerial_report import build_managerial_report
from causal_analysis.pipeline import analyze_dataframe
from causal_analysis.stats_tests import ljung_box

RNG = np.random.default_rng(5)


def test_agregacao_gera_familia_de_metricas():
    horas = pd.date_range("2024-01-01", periods=72, freq="h")
    fina = pd.DataFrame({"temp": RNG.normal(80, 2, 72)}, index=horas)
    dias = pd.date_range("2024-01-01", periods=3, freq="D") + pd.Timedelta("23h")
    agg = aggregate_to_grid(fina, dias)
    assert "temp (máximo)" in agg.columns
    assert "temp (% tempo>Q3)" in agg.columns
    assert len(agg) == 3
    # máximo da janela >= média da janela
    assert (agg["temp (máximo)"] >= agg["temp (média)"]).all()
    # frações de tempo em [0, 100]
    assert agg["temp (% tempo>Q3)"].between(0, 100).all()


def test_align_sheets_fina_e_grossa():
    dias = pd.date_range("2024-01-01", periods=30, freq="D")
    alvo = pd.DataFrame({"rendimento": RNG.normal(60, 3, 30)}, index=dias)
    horas = pd.date_range("2024-01-01", periods=30 * 24, freq="h")
    fina = pd.DataFrame({"vazao": RNG.normal(100, 5, len(horas))}, index=horas)
    semanas = pd.date_range("2024-01-01", periods=5, freq="7D")
    grossa = pd.DataFrame({"dosagem": RNG.normal(10, 1, 5)}, index=semanas)

    combinado, info = align_sheets(
        {"diario": alvo, "horario": fina, "semanal": grossa}, "rendimento"
    )
    assert info.target_sheet == "diario"
    assert "vazao (P90)" in combinado.columns          # fina -> agregada
    assert "dosagem" in combinado.columns              # grossa -> propagada
    assert len(combinado) == 30
    # forward-fill da grossa não deve ter buracos após a primeira semana
    assert combinado["dosagem"].iloc[7:].notna().all()


def test_nomes_de_colunas_derivadas():
    assert base_indicator("forno: temp (máximo)") == "temp"
    assert base_indicator("temp (P90)") == "temp"
    assert base_indicator("temp") == "temp"
    assert metric_of("temp (P90)") == "P90"
    assert metric_of("temp") is None


def test_ljung_box_detecta_estrutura():
    n = 300
    ar = np.zeros(n)
    for i in range(1, n):  # série autocorrelacionada
        ar[i] = 0.8 * ar[i - 1] + RNG.normal()
    com = ljung_box(pd.Series(ar))
    sem = ljung_box(pd.Series(RNG.normal(0, 1, n)))
    assert com["has_structure"]
    assert not sem["has_structure"]


def test_relatorio_gerencial_mastigado():
    # mecanismo claro: x alto derruba o alvo
    n = 200
    x = RNG.normal(50, 5, n)
    alvo = 100 - 1.5 * x + RNG.normal(0, 3, n)
    df = pd.DataFrame(
        {"alvo": alvo, "causa": x, "ruido": RNG.normal(0, 1, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    res = analyze_dataframe(df, "alvo", max_lag=5, verbose=False)
    rep = build_managerial_report(res)
    assert "causa" in rep.headline
    assert rep.findings, "deveria haver achados"
    # linguagem simples, sem jargão bruto
    texto = " ".join(rep.findings).lower()
    assert "tende a cair" in texto
    assert rep.ranking_table is not None
    assert "como impacta" in rep.ranking_table.columns
    assert rep.cautions  # sempre há a cautela padrão


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
