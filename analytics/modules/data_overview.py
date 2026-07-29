"""
Visão Geral dos Dados
=====================

Mostra a prévia da base ativa, o diagnóstico automático dos tipos de coluna e
métricas-resumo. A base vem do upload global da barra lateral — troque a aba
da planilha no seletor no topo da página Analytics.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(state) -> None:
    st.info(
        "📂 A base analisada aqui é a planilha carregada na **barra lateral** "
        "(a mesma dos Módulos 1 e 2). Use o seletor **Aba da planilha** acima "
        "para trocar de aba; os tipos de coluna são identificados "
        "automaticamente."
    )

    df = state.get("df")
    if df is None:
        st.warning("Nenhuma base carregada. Envie uma planilha na barra lateral.")
        return

    st.success(f"✅ Base atual: **{state.get('file_name') or 'arquivo'}**")
    _show_overview(df, state["col_types"])


def _show_overview(df: pd.DataFrame, col_types) -> None:
    tabs = st.tabs(["👁️ Prévia", "🔎 Tipos de Coluna", "📐 Resumo"])

    # --- Prévia ---------------------------------------------------------- #
    with tabs[0]:
        st.markdown("**Primeiras linhas da base:**")
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Exibindo até 50 de {len(df)} linhas.")

    # --- Tipos ----------------------------------------------------------- #
    with tabs[1]:
        st.markdown("Classificação automática de cada coluna:")
        type_rows = []
        for col in df.columns:
            type_rows.append({
                "Coluna": col,
                "Tipo detectado": col_types.kind_of(col),
                "Valores únicos": int(df[col].nunique(dropna=True)),
                "Ausentes": int(df[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)
        with st.expander("ℹ️ Como os tipos são identificados?"):
            st.markdown(
                "- **Numérica**: valores numéricos (inclusive com vírgula decimal BR).\n"
                "- **Data/Hora**: colunas que o sistema conseguiu converter em datas.\n"
                "- **Categórica**: texto com poucas categorias distintas (≤ 20).\n"
                "- **Texto**: texto livre com muitas categorias diferentes."
            )

    # --- Resumo ---------------------------------------------------------- #
    with tabs[2]:
        n_missing = int(df.isna().sum().sum())
        n_dup = int(df.duplicated().sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("🔢 Linhas", f"{len(df):,}".replace(",", "."))
        c2.metric("📊 Colunas", df.shape[1])
        c3.metric("❓ Dados ausentes", f"{n_missing:,}".replace(",", "."))
        c4, c5, c6 = st.columns(3)
        c4.metric("🔵 Colunas numéricas", len(col_types.numeric))
        c5.metric("🏷️ Colunas categóricas", len(col_types.categorical))
        c6.metric("👯 Linhas duplicadas", n_dup)

        c7, c8 = st.columns(2)
        c7.metric("📅 Colunas data/hora", len(col_types.datetime))
        c8.metric("📝 Colunas de texto", len(col_types.text))

        if n_missing > 0 or n_dup > 0:
            st.warning(
                "⚠️ A base tem dados ausentes e/ou duplicados. Recomendamos abrir "
                "**Qualidade dos Dados** para avaliar antes das análises."
            )
        else:
            st.success("✅ A base não apresenta dados ausentes nem duplicados óbvios.")
