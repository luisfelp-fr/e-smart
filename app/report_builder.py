"""Página do relatório: coleta de análises marcadas, preview HTML e download."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from shared.excel_export import build_excel

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
       background: #fcfcfb; color: #0b0b0b; margin: 0; padding: 28px;
       line-height: 1.55; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 34px 0 8px; border-bottom: 2px solid #e1e0d9;
     padding-bottom: 6px; }
.sub { color: #52514e; font-size: 13px; margin-bottom: 24px; }
.item-mod { display: inline-block; background: #eef4fc; color: #104281;
            border-radius: 4px; padding: 1px 8px; font-size: 12px;
            margin-left: 8px; }
p.texto { color: #22211f; font-size: 14px; margin: 8px 0; }
table { border-collapse: collapse; font-size: 13px; margin: 12px 0; }
th, td { border: 1px solid #e1e0d9; padding: 5px 10px; text-align: left; }
th { background: #f4f3ef; color: #52514e; font-weight: 600; }
.fig { margin: 14px 0; }
"""


def build_html(items: list[dict]) -> str:
    """Gera o HTML autocontido do relatório (figuras Plotly interativas)."""
    stamp = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    parts = [
        f"<style>{_CSS}</style>",
        "<h1>Relatório de análise de indicadores</h1>",
        f"<div class='sub'>Gerado em {stamp} · {len(items)} análise(s) "
        "selecionada(s)</div>",
    ]
    first_fig = True
    for it in items:
        parts.append(
            f"<h2>{it['title']}<span class='item-mod'>{it['module']}</span></h2>"
        )
        for text in it.get("texts", []):
            parts.append(f"<p class='texto'>{text}</p>")
        for name, table in it.get("tables", {}).items():
            parts.append(f"<p class='texto'><b>{name}</b></p>")
            parts.append(table.to_html(border=0, na_rep="—",
                                       float_format=lambda v: f"{v:.3f}"))
        for _, fig in it.get("figures", {}).items():
            include = True if first_fig else False
            parts.append("<div class='fig'>")
            parts.append(fig.to_html(
                full_html=False, include_plotlyjs=include,
                config={"displayModeBar": False},
            ))
            parts.append("</div>")
            first_fig = False
    return "\n".join(parts)


def render_report_page() -> None:
    st.header("📄 Relatório")
    st.caption(
        "Reúna aqui as análises marcadas com **Adicionar ao relatório** nas "
        "outras páginas. Visualize o resultado e baixe em Excel."
    )
    items: list[dict] = st.session_state.get("report_items", [])
    if not items:
        st.info(
            "O relatório está vazio. Rode uma análise no Módulo 1 ou 2 e "
            "clique em **📄 Adicionar ao relatório**.",
            icon="📭",
        )
        return

    st.subheader(f"Análises selecionadas ({len(items)})")
    for i, it in enumerate(items):
        c1, c2 = st.columns([10, 1])
        c1.markdown(f"**{i + 1}. {it['title']}**  \n:gray[{it['module']}]")
        if c2.button("🗑️", key=f"rm_{it['id']}", help="Remover do relatório"):
            items.pop(i)
            st.rerun()

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 3])
    if c1.button("👁️ Gerar preview HTML", type="primary"):
        st.session_state["report_preview"] = build_html(items)
    with c2:
        with st.spinner(""):
            xlsx = build_excel(items)
        st.download_button(
            "⬇️ Baixar em Excel (.xlsx)", data=xlsx,
            file_name="relatorio_indicadores.xlsx",
            mime="application/vnd.openxmlformats-officedocument"
                 ".spreadsheetml.sheet",
        )
    if c3.button("Limpar relatório"):
        st.session_state["report_items"] = []
        st.session_state.pop("report_preview", None)
        st.rerun()

    html = st.session_state.get("report_preview")
    if html:
        st.divider()
        st.subheader("Preview do relatório")
        st.components.v1.html(html, height=900, scrolling=True)
        st.download_button(
            "⬇️ Baixar HTML", data=html.encode("utf-8"),
            file_name="relatorio_indicadores.html", mime="text/html",
        )
