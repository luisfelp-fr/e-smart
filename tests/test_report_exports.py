"""Testes da exportação do relatório (PDF) e da leitura gerencial (lags)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_analysis.managerial_report import (
    _fmt_timedelta_br,
    _lag_phrase,
    _time_step,
)
from shared.pdf_export import _latin1, build_pdf


def _item(with_figure: bool = False) -> dict:
    tabela = pd.DataFrame(
        {"métrica": ["Cpk", "PPM fora"], "valor": [1.234, 42.0]}
    ).set_index("métrica")
    item = {
        "id": "abc123",
        "module": "Módulo 1 — Capabilidade",
        "title": "Capabilidade de 'vazão'",
        "texts": ["Processo capaz (σ = 0,5 · Ppk ≈ 1,2)."],
        "tables": {"Resumo dos índices": tabela},
        "figures": {},
    }
    if with_figure:
        import plotly.graph_objects as go

        item["figures"]["exemplo"] = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    return item


def test_build_pdf_gera_arquivo_valido():
    pdf = build_pdf([_item(), _item(with_figure=True)])
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_latin1_substitui_simbolos():
    assert "sigma" in _latin1("σ")
    assert _latin1("Ppk ≈ 1,2") == "Ppk ~ 1,2"
    assert "é" in _latin1("relatório é ótimo")  # acentos preservados
    assert "😀" not in _latin1("ok 😀")


# ------------------------------------------- tradução de lags para tempo real

def test_time_step_com_datas_e_sem_datas():
    idx = pd.date_range("2024-01-01", periods=10, freq="4h")
    step, txt = _time_step(idx)
    assert step == pd.Timedelta(hours=4)
    assert txt == "4 horas"
    step2, txt2 = _time_step(pd.RangeIndex(1, 11))
    assert step2 is None and txt2 == ""


def test_fmt_timedelta_br():
    assert _fmt_timedelta_br(pd.Timedelta(hours=12)) == "12 horas"
    assert _fmt_timedelta_br(pd.Timedelta(hours=1)) == "1 hora"
    assert _fmt_timedelta_br(pd.Timedelta(days=3, hours=12)) == "3,5 dias"
    assert _fmt_timedelta_br(pd.Timedelta(minutes=30)) == "30 minutos"
    assert _fmt_timedelta_br(pd.Timedelta(days=14)) == "2 semanas"


def test_reduce_to_scale_preserva_estrutura():
    from causal_analysis.aggregation import infer_step, reduce_to_scale

    idx = pd.date_range("2024-01-01", periods=50000, freq="min")
    df = pd.DataFrame({"a": np.arange(50000.0), "alvo": np.arange(50000.0)},
                      index=idx)
    red, nota = reduce_to_scale(df, max_rows=10000)
    assert len(red) <= 10000
    assert list(red.columns) == ["a", "alvo"]           # colunas preservadas
    assert infer_step(red.index) > pd.Timedelta(minutes=1)  # grade mais grossa
    assert nota and "agregadas" in nota
    # abaixo do teto: sem alteração
    red2, nota2 = reduce_to_scale(df.head(3000), max_rows=10000)
    assert len(red2) == 3000 and nota2 is None


def test_thin_rarefaz_mantendo_extremos():
    from capability.charts import _thin

    x = np.arange(100000.0)
    xd, reduziu = _thin(x, max_points=8000)
    assert reduziu and len(xd) <= 8001
    assert xd[0] == 0.0 and xd[-1] == 99999.0     # 1º e último preservados
    xs, reduziu2 = _thin(np.arange(500.0), max_points=8000)
    assert not reduziu2 and len(xs) == 500


def test_lag_phrase_traduzido_para_escala_de_tempo():
    step = pd.Timedelta(hours=4)
    frase = _lag_phrase("lag 3", step)
    assert "3 período(s)" in frase
    assert "12 horas" in frase
    frase_mm = _lag_phrase("média móvel 7", step)
    assert "28 horas" in frase_mm  # 7 janelas de 4h
    # sem datas: nenhuma tradução é inventada
    assert "≈" not in _lag_phrase("lag 3", None)
    assert _lag_phrase("bruto (sem defasagem)", step).startswith(
        "o efeito é imediato"
    )
