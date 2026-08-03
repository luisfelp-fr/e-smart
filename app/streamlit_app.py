"""Analisador de indicadores de processo — aplicativo Streamlit.

Módulo 1: análise de capabilidade (o indicador atende aos limites?).
Módulo 2: análise de influência (o que impacta a variável alvo?).
Relatório: reúna análises marcadas, visualize e baixe em PDF.
"""

from __future__ import annotations

import os
import sys

# o app importa os pacotes do repositório (capability, causal_analysis, shared)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

from page_analytics import render_analytics  # noqa: E402
from page_module1 import render_module1  # noqa: E402
from page_module2 import render_module2  # noqa: E402
from report_builder import render_report_page  # noqa: E402
from shared.limits import limits_summary, queue_note  # noqa: E402
from ui_components import save_upload  # noqa: E402

st.set_page_config(
    page_title="Analisador de Indicadores de Processo",
    page_icon="📈",
    layout="wide",
)


def main() -> None:
    with st.sidebar:
        st.title("📈 Analisador de Indicadores")
        st.caption("Análise estatística de indicadores de processo")

        uploaded = st.file_uploader(
            "Planilha de dados (CSV ou Excel)",
            type=["csv", "txt", "xlsx", "xls", "xlsm", "ods"],
            help="Primeira coluna com data/hora (opcional — sem ela, usamos "
                 "a ordem das linhas). Números com vírgula decimal são "
                 "aceitos. Excel pode ter várias abas, cada uma com uma "
                 "série temporal diferente.",
        )
        # A planilha carregada vira a base compartilhada de todos os módulos:
        # o caminho fica na sessão e cada página só seleciona aba e indicador.
        if uploaded:
            file_path = save_upload(uploaded)
            if st.session_state.get("file_path") != file_path:
                # arquivo novo: descarta a base derivada da aba Analytics
                for k in [k for k in st.session_state if str(k).startswith("ax_")]:
                    st.session_state.pop(k)
            st.session_state["file_path"] = file_path
            st.session_state["uploaded_name"] = uploaded.name
            st.success(f"✓ {uploaded.name}")
        file_path = st.session_state.get("file_path")

        st.divider()
        page = st.radio(
            "Navegação",
            ["📊 Módulo 1 — Capabilidade",
             "🔎 Módulo 2 — Influência no alvo",
             "📈 Analytics",
             "📄 Relatório"],
            key="nav",
            label_visibility="collapsed",
        )
        n_report = len(st.session_state.get("report_items", []))
        if n_report:
            st.caption(f"📑 {n_report} análise(s) no relatório")
        st.divider()
        fila = queue_note()
        if fila:
            st.info(f"⏳ Análises em andamento — {fila}.", icon="⏳")
        with st.expander("ℹ️ Como usar"):
            st.markdown(
                "1. **Carregue** sua planilha acima — ela vira a base de "
                "todos os módulos; em cada um você só escolhe a aba e o "
                "indicador.\n"
                "2. **Módulo 1**: escolha um indicador e informe os limites "
                "de atuação para verificar se o processo é capaz.\n"
                "3. **Módulo 2**: escolha a variável alvo para descobrir "
                "quais indicadores mais a impactam.\n"
                "4. **Analytics**: análises estatísticas guiadas "
                "(descritiva, correlação, regressão, CEP, outliers etc.).\n"
                "5. Em cada análise, clique em **Adicionar ao relatório** e "
                "baixe tudo em PDF na página Relatório."
            )
        st.caption(f":gray[Capacidade: {limits_summary()}]")

    if page.startswith("📊"):
        render_module1(file_path)
    elif page.startswith("🔎"):
        render_module2(file_path)
    elif page.startswith("📈"):
        render_analytics(file_path)
    else:
        render_report_page()


main()
