"""Exportação do relatório para Excel (.xlsx).

Cada análise marcada vira uma aba: textos no topo, tabelas na sequência e
figuras como imagem PNG (quando a conversão Plotly→PNG está disponível no
ambiente; caso contrário as tabelas/textos seguem normalmente).
"""

from __future__ import annotations

import io
import re

import pandas as pd


def _png_of(fig) -> bytes | None:
    """Converte figura Plotly em PNG; None se o ambiente não suportar."""
    try:
        return fig.to_image(format="png", width=980, height=560, scale=2)
    except Exception:
        return None


def _safe_sheet_name(title: str, used: set[str]) -> str:
    name = re.sub(r"[\\/*?:\[\]]", " ", title).strip()[:28] or "analise"
    base, i = name, 2
    while name.lower() in used:
        name = f"{base[:25]} {i}"
        i += 1
    used.add(name.lower())
    return name


def build_excel(items: list[dict]) -> bytes:
    """Monta o arquivo .xlsx do relatório a partir dos itens marcados.

    Cada item: {id, module, title, texts: [str], tables: {nome: DataFrame},
    figures: {nome: go.Figure}}.
    """
    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # capa com o sumário
        capa = pd.DataFrame({
            "Análise": [it["title"] for it in items],
            "Módulo": [it["module"] for it in items],
        })
        capa.index = capa.index + 1
        capa.to_excel(writer, sheet_name="Sumário", index_label="#")
        ws = writer.sheets["Sumário"]
        ws.column_dimensions["B"].width = 60
        ws.column_dimensions["C"].width = 28

        for it in items:
            sheet = _safe_sheet_name(it["title"], used)
            row = 0
            # textos
            texts = [it["title"], ""] + list(it.get("texts", []))
            pd.DataFrame({"": texts}).to_excel(
                writer, sheet_name=sheet, index=False, header=False, startrow=row
            )
            row += len(texts) + 2
            ws = writer.sheets[sheet]
            ws.column_dimensions["A"].width = 110
            # tabelas
            for name, table in it.get("tables", {}).items():
                pd.DataFrame({"": [f"▸ {name}"]}).to_excel(
                    writer, sheet_name=sheet, index=False, header=False,
                    startrow=row,
                )
                row += 1
                table.to_excel(writer, sheet_name=sheet, startrow=row)
                row += len(table) + 3
            # figuras (melhor esforço)
            for name, fig in it.get("figures", {}).items():
                png = _png_of(fig)
                if png is None:
                    continue
                try:
                    from openpyxl.drawing.image import Image as XLImage

                    img = XLImage(io.BytesIO(png))
                    img.width, img.height = 735, 420
                    ws.add_image(img, f"A{row + 2}")
                    row += 24
                except Exception:
                    continue
    return buf.getvalue()
