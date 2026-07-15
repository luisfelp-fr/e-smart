"""Exportação do relatório para PDF (.pdf).

Cada análise marcada vira uma seção: textos no topo, tabelas na sequência e
figuras como imagem PNG (quando a conversão Plotly→PNG está disponível no
ambiente; caso contrário textos e tabelas seguem normalmente).

As fontes nativas do PDF são latin-1: símbolos fora dela (σ, ≈, emojis) são
substituídos por equivalentes em texto antes da escrita.
"""

from __future__ import annotations

import datetime as dt
import io
import unicodedata

import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace

INK = (11, 11, 11)
INK_2 = (82, 81, 78)
GRID = (225, 224, 217)
HEADER_BG = (244, 243, 239)

_REPLACEMENTS = {
    "≈": "~", "≥": ">=", "≤": "<=", "·": " - ", "—": "-", "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "×": "x",
    "⇒": "=>", "→": "->", "σ": "sigma", "µ": "mu", "α": "alfa",
    "ρ": "rho", "λ": "lambda", "‰": " por mil", "″": '"',
}


def _latin1(text) -> str:
    """Adapta o texto às fontes nativas do PDF (latin-1)."""
    t = unicodedata.normalize("NFC", str(text))
    for k, v in _REPLACEMENTS.items():
        t = t.replace(k, v)
    # o que restar fora da latin-1 (emojis, acentos combinantes) é descartado
    return t.encode("latin-1", "ignore").decode("latin-1")


def _png_of(fig) -> bytes | None:
    """Converte figura Plotly em PNG; None se o ambiente não suportar."""
    try:
        return fig.to_image(format="png", width=980, height=560, scale=2)
    except Exception:
        return None


def _fmt_cell(v) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}".replace(".", ",")
    return str(v)


class _ReportPDF(FPDF):
    def header(self):  # noqa: D102 - cabeçalho padrão do fpdf2
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*INK_2)
        self.cell(0, 6, _latin1("Relatório de análise de indicadores"),
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def footer(self):  # noqa: D102 - rodapé com numeração
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*INK_2)
        self.cell(0, 6, f"{self.page_no()}/{{nb}}", align="C")


def _render_table(pdf: _ReportPDF, name: str, df: pd.DataFrame) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, _latin1(name), new_x="LMARGIN", new_y="NEXT")

    data = df if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    headers = [_latin1(c) for c in data.columns]
    font_size = 8.0 if len(headers) <= 6 else 6.5
    pdf.set_font("Helvetica", "", font_size)
    heading = FontFace(emphasis="BOLD", color=INK_2, fill_color=HEADER_BG)
    with pdf.table(
        headings_style=heading, text_align="LEFT",
        borders_layout="HORIZONTAL_LINES", line_height=font_size * 0.55,
        padding=1.2,
    ) as table:
        head = table.row()
        for h in headers:
            head.cell(h)
        for _, r in data.iterrows():
            row = table.row()
            for v in r:
                row.cell(_latin1(_fmt_cell(v)))
    pdf.ln(4)


def build_pdf(items: list[dict]) -> bytes:
    """Monta o arquivo .pdf do relatório a partir dos itens marcados.

    Cada item: {id, module, title, texts: [str], tables: {nome: DataFrame},
    figures: {nome: go.Figure}}.
    """
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # capa: título, data e sumário
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.cell(0, 11, _latin1("Relatório de análise de indicadores"),
             new_x="LMARGIN", new_y="NEXT")
    stamp = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*INK_2)
    pdf.cell(0, 6, _latin1(f"Gerado em {stamp} - {len(items)} análise(s) "
                           "selecionada(s)"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, _latin1("Sumário"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*INK_2)
    for i, it in enumerate(items, 1):
        pdf.multi_cell(0, 6, _latin1(f"{i}. {it['title']}  ({it['module']})"),
                       new_x="LMARGIN", new_y="NEXT")

    for i, it in enumerate(items, 1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 8, _latin1(f"{i}. {it['title']}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*INK_2)
        pdf.cell(0, 5, _latin1(it["module"]), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*INK)
        for text in it.get("texts", []):
            pdf.multi_cell(0, 5.4, _latin1(text),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        if it.get("texts"):
            pdf.ln(2)

        for name, table in it.get("tables", {}).items():
            if table is None or len(table) == 0:
                continue
            _render_table(pdf, name, table)

        # figuras em página inteira quando não couberem no espaço restante
        img_w = pdf.epw
        img_h = img_w * 560.0 / 980.0
        for name, fig in it.get("figures", {}).items():
            png = _png_of(fig)
            if png is None:
                continue
            if pdf.get_y() + img_h + 8 > pdf.page_break_trigger:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*INK_2)
            pdf.cell(0, 6, _latin1(name), new_x="LMARGIN", new_y="NEXT")
            pdf.image(io.BytesIO(png), w=img_w)
            pdf.ln(3)

    return bytes(pdf.output())
