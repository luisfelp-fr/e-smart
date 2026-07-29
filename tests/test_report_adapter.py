"""Testes do adapter que leva itens da aba Analytics ao relatório unificado."""

from __future__ import annotations

import os
import sys

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))

from analytics.report_adapter import adapt_report_item, stable_id


def _argus_item(timestamp: str = "01/01/2026 10:00:00", metodo: str = "IQR") -> dict:
    import plotly.graph_objects as go

    return {
        "name": "Análise de Outliers",
        "variables": {"variável": "Vazao"},
        "params": {"método": metodo, "outliers": 3},
        "interpretation": "Foram detectados 3 outliers (2,1% dos dados).",
        "figures": [
            go.Figure(go.Box(y=[1, 2, 3, 100])),
            go.Figure(go.Scatter(x=[1, 2], y=[3, 4])),
        ],
        "tables": {
            "Outliers detectados": pd.DataFrame({"Linha": [5, 9], "Valor": [100, 98]})
        },
        "timestamp": timestamp,
    }


def test_adapta_para_esquema_do_relatorio():
    item = adapt_report_item(_argus_item())
    assert set(item) == {"id", "module", "title", "texts", "tables", "figures", "meta"}
    assert item["module"] == "Analytics"
    assert item["title"] == "Análise de Outliers — Vazao"
    # figuras: lista -> dict nomeado
    assert list(item["figures"]) == [
        "Análise de Outliers — figura 1",
        "Análise de Outliers — figura 2",
    ]
    # textos: interpretação + variáveis + parâmetros + carimbo
    assert item["texts"][0].startswith("Foram detectados")
    assert any("variável: Vazao" in t for t in item["texts"])
    assert any("método: IQR" in t for t in item["texts"])
    assert any("Gerado em" in t for t in item["texts"])
    assert isinstance(item["tables"]["Outliers detectados"], pd.DataFrame)
    assert item["meta"]["analytics"]["params"]["método"] == "IQR"


def test_id_estavel_ignora_timestamp_e_distingue_parametros():
    a = stable_id(_argus_item(timestamp="01/01/2026 10:00:00"))
    b = stable_id(_argus_item(timestamp="02/02/2026 23:59:59"))
    c = stable_id(_argus_item(metodo="Z-score"))
    assert a == b            # repetir a mesma análise não duplica
    assert a != c            # configuração diferente = item novo
    assert a.startswith("ax")


def test_item_adaptado_passa_pelo_relatorio_html_e_pdf():
    from report_builder import build_html
    from shared.pdf_export import build_pdf

    adapted = adapt_report_item(_argus_item())
    html = build_html([adapted])
    assert "Análise de Outliers" in html
    assert "Analytics" in html
    pdf = build_pdf([adapted])
    assert isinstance(pdf, bytes) and pdf.startswith(b"%PDF")


def test_variaveis_em_lista_formatadas_no_titulo():
    item = adapt_report_item(
        {"name": "Estatística Descritiva",
         "variables": {"variáveis": ["Vazao", "Pressao"]},
         "params": {}, "interpretation": "x", "figures": [], "tables": {},
         "timestamp": ""}
    )
    assert item["title"] == "Estatística Descritiva — Vazao, Pressao"
    assert any("variáveis: Vazao, Pressao" in t for t in item["texts"])


def test_item_minimo_sem_figuras_nem_tabelas():
    minimal = adapt_report_item(
        {"name": "Estatística Descritiva", "variables": {}, "params": {},
         "interpretation": "", "figures": [], "tables": {}, "timestamp": ""}
    )
    assert minimal["texts"] == []
    assert minimal["figures"] == {} and minimal["tables"] == {}
    assert minimal["title"] == "Estatística Descritiva"
