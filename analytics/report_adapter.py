"""Adapta itens de análise da aba Analytics para o relatório do e-smart.

A aba Analytics (port do Argus Analytics) monta itens no formato
``{name, variables, params, interpretation, figures: list, tables, timestamp}``;
a página 📄 Relatório e o PDF esperam
``{id, module, title, texts: [str], tables: {nome: DataFrame},
figures: {nome: go.Figure}, meta}`` (ver ``shared/pdf_export.build_pdf``).

Puro (sem Streamlit) para ser testável isoladamente.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_id(item: dict[str, Any]) -> str:
    """Id determinístico por análise + variáveis + parâmetros (sem timestamp).

    Assim o dedupe da página Relatório funciona: repetir a mesma análise com a
    mesma configuração não duplica o item, mas mudar variável/parâmetro gera um
    item novo.
    """
    payload = "|".join([
        "ax",
        str(item.get("name", "")),
        repr(sorted((str(k), str(v)) for k, v in (item.get("variables") or {}).items())),
        repr(sorted((str(k), str(v)) for k, v in (item.get("params") or {}).items())),
    ])
    return "ax" + hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def adapt_report_item(item: dict[str, Any]) -> dict[str, Any]:
    """Converte um item do formato Argus para o esquema do relatório do e-smart."""
    variables = dict(item.get("variables") or {})
    params = dict(item.get("params") or {})

    texts = [
        t for t in (
            str(item.get("interpretation") or ""),
            "Variáveis: " + "; ".join(f"{k}: {v}" for k, v in variables.items())
            if variables else "",
            "Parâmetros: " + "; ".join(f"{k}: {v}" for k, v in params.items())
            if params else "",
            f"Gerado em {item['timestamp']}." if item.get("timestamp") else "",
        ) if t
    ]

    name = str(item.get("name", "Análise"))
    figures = {
        f"{name} — figura {i + 1}": fig
        for i, fig in enumerate(item.get("figures") or [])
    }

    title = name
    if variables:
        title += " — " + ", ".join(str(v) for v in variables.values())

    return {
        "id": stable_id(item),
        "module": "Analytics",
        "title": title,
        "texts": texts,
        "tables": dict(item.get("tables") or {}),
        "figures": figures,
        "meta": {
            "analytics": {
                "variables": variables,
                "params": params,
                "timestamp": str(item.get("timestamp") or ""),
            }
        },
    }
