"""Página 📈 Analytics — análises estatísticas guiadas (port do Argus Analytics).

Home com cards + telas de análise roteadas por ``st.session_state["ax_page"]``.
A base é a planilha do upload global da barra lateral: aqui o usuário só
seleciona a aba (e, dentro de cada análise, as variáveis). O relatório é
unificado com a página 📄 Relatório do app.
"""

from __future__ import annotations

import streamlit as st

from analytics.modules import (
    capability,
    control_charts,
    correlation,
    data_overview,
    data_quality,
    descriptive_stats,
    distribution,
    group_comparison,
    lag_analysis,
    normality_tests,
    outliers,
    regression,
    time_series,
)
from analytics.utils.helpers import has_data
from data_store import get_analytics_base, sheet_names

# Catálogo de análises: chave -> (rótulo, ícone, descrição, função render, precisa_dados)
ANALYSES: dict[str, dict] = {
    "data_overview": {
        "label": "Visão Geral dos Dados",
        "icon": "📋",
        "desc": "Veja a prévia da base e o diagnóstico automático dos tipos de coluna.",
        "render": data_overview.render,
        "needs_data": True,
    },
    "outliers": {
        "label": "Outliers",
        "icon": "🚨",
        "desc": "Identifique valores muito diferentes do padrão (IQR, Z-score e MAD).",
        "render": outliers.render,
        "needs_data": True,
    },
    "data_quality": {
        "label": "Qualidade dos Dados",
        "icon": "🧹",
        "desc": "Verifique valores ausentes, duplicidades, colunas constantes e problemas de formatação.",
        "render": data_quality.render,
        "needs_data": True,
    },
    "descriptive": {
        "label": "Estatística Descritiva",
        "icon": "📊",
        "desc": "Resuma o comportamento das variáveis: média, mediana, dispersão e limites.",
        "render": descriptive_stats.render,
        "needs_data": True,
    },
    "distribution": {
        "label": "Distribuição dos Dados",
        "icon": "📈",
        "desc": "Histograma, boxplot e densidade para ver concentração, assimetria e outliers.",
        "render": distribution.render,
        "needs_data": True,
    },
    "normality": {
        "label": "Teste de Normalidade",
        "icon": "🔔",
        "desc": "Descubra se uma variável segue distribuição normal (Shapiro-Wilk e Anderson-Darling).",
        "render": normality_tests.render,
        "needs_data": True,
    },
    "correlation": {
        "label": "Correlação",
        "icon": "🔗",
        "desc": "Verifique se duas ou mais variáveis se movimentam juntas. Útil para relações entre indicadores.",
        "render": correlation.render,
        "needs_data": True,
    },
    "regression": {
        "label": "Regressão",
        "icon": "📐",
        "desc": "Entenda como uma ou mais variáveis influenciam uma variável resposta.",
        "render": regression.render,
        "needs_data": True,
    },
    "group_comparison": {
        "label": "Comparação entre Grupos",
        "icon": "⚖️",
        "desc": "Compare se diferentes grupos (turnos, máquinas, fornecedores) se comportam de forma distinta.",
        "render": group_comparison.render,
        "needs_data": True,
    },
    "control_charts": {
        "label": "Controle Estatístico de Processo",
        "icon": "🚦",
        "desc": "Verifique se o processo está estável ao longo do tempo (carta de controle).",
        "render": control_charts.render,
        "needs_data": True,
    },
    "capability": {
        "label": "Capabilidade Cp / Cpk",
        "icon": "🎯",
        "desc": "Avalie se a variação do processo cabe nos limites de especificação e se está centralizado.",
        "render": capability.render,
        "needs_data": True,
    },
    "time_series": {
        "label": "Análise Temporal",
        "icon": "⏱️",
        "desc": "Observe como uma variável evolui no tempo: tendências, oscilações e médias móveis.",
        "render": time_series.render,
        "needs_data": True,
    },
    "lag_analysis": {
        "label": "Análise com Lag",
        "icon": "⏳",
        "desc": "Descubra se uma variável afeta outra após um atraso (defasagem temporal).",
        "render": lag_analysis.render,
        "needs_data": True,
    },
}

_HOME_LABEL = "🏠 Início"


def _label_of(key: str) -> str:
    meta = ANALYSES[key]
    return f"{meta['icon']} {meta['label']}"


def _ensure_base(file_path: str) -> bool:
    """Seletor de aba + carga (cacheada) da base tipada na sessão.

    A base só é recarregada quando o arquivo ou a aba mudam — é isso que
    preserva a "base sem outliers" entre reruns e a descarta na troca.
    """
    names = sheet_names(file_path)
    if len(names) > 1:
        sheet = st.selectbox(
            "Aba da planilha", names, key="ax_sheet",
            help="A base das análises abaixo é a aba selecionada; os Módulos 1 "
                 "e 2 continuam com as próprias seleções.",
        )
    else:
        sheet = names[0]

    sig = (file_path, sheet)
    if st.session_state.get("ax_base_sig") != sig:
        try:
            df, col_types = get_analytics_base(file_path, sheet)
        except Exception as exc:
            st.error(f"Não foi possível ler a aba «{sheet}»: {exc}")
            return False
        uploaded_name = st.session_state.get("uploaded_name", "arquivo")
        st.session_state.update(
            ax_df=df,
            ax_col_types=col_types,
            ax_base_sig=sig,
            ax_file_name=f"{uploaded_name} — {sheet}" if len(names) > 1 else uploaded_name,
        )
        # Mesma limpeza do carregamento original: base nova invalida o
        # tratamento de outliers e os resultados retidos de lag/regressão.
        for k in ("ax_df_original", "ax_col_types_original", "ax_df_no_outliers",
                  "ax_outliers_applied", "ax_outliers_removed_log",
                  "ax_lag_has_run", "ax_lag_arimax", "ax_lag_av_var_prev",
                  "ax_lag_av_lag", "ax_reg_has_run"):
            st.session_state.pop(k, None)
    return True


def _state_view() -> dict:
    """Visão da sessão no formato que as telas portadas esperam.

    As telas só *leem* por aqui; toda escrita delas vai direto ao
    ``st.session_state`` com prefixo ``ax_``.
    """
    ss = st.session_state
    return {
        "df": ss.get("ax_df"),
        "col_types": ss.get("ax_col_types"),
        "file_name": ss.get("ax_file_name"),
        "outliers_applied": ss.get("ax_outliers_applied"),
        "outliers_removed_log": ss.get("ax_outliers_removed_log", []),
    }


def _jump_nav(current_key: str | None) -> None:
    """Selectbox de navegação direta entre as análises (e a home)."""
    labels = [_HOME_LABEL] + [_label_of(k) for k in ANALYSES]

    def _go() -> None:
        chosen = st.session_state["ax_nav"]
        for k in ANALYSES:
            if _label_of(k) == chosen:
                st.session_state["ax_page"] = k
                return
        st.session_state["ax_page"] = "home"

    # Mantém o widget sincronizado com a página atual (inclusive quando a
    # navegação veio dos cards ou do botão "← Início").
    st.session_state["ax_nav"] = _label_of(current_key) if current_key else _HOME_LABEL
    st.selectbox("Ir para análise", labels, key="ax_nav", on_change=_go,
                 label_visibility="collapsed")


def _home() -> None:
    st.markdown("<h1 class='ax-title'>👁️ Analytics</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='ax-sub' style='font-size:1.15rem;'>"
        "Enxergue seus dados por todos os ângulos — análise estatística simples, "
        "visual e guiada para engenharia e processos.</p>",
        unsafe_allow_html=True,
    )

    if has_data():
        df = st.session_state["ax_df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Base", st.session_state.get("ax_file_name") or "—")
        c2.metric("🔢 Linhas", f"{len(df):,}".replace(",", "."))
        c3.metric("📊 Colunas", df.shape[1])
        if st.session_state.get("ax_outliers_applied"):
            st.caption("🧹 Analisando a base **sem outliers** — reverta na tela 🚨 Outliers.")

    st.divider()
    st.markdown("### Escolha uma análise")
    st.caption("Cada card abre uma tela dedicada, com seleção simples de variáveis e explicações.")

    # Grid de cards (4 colunas)
    keys = list(ANALYSES.keys())
    cols_per_row = 4
    for i in range(0, len(keys), cols_per_row):
        row = keys[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, key in zip(cols, row):
            meta = ANALYSES[key]
            disabled = meta["needs_data"] and not has_data()
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {meta['icon']} {meta['label']}")
                    st.caption(meta["desc"])
                    if st.button(
                        "Abrir ▶",
                        key=f"card_{key}",
                        use_container_width=True,
                        disabled=disabled,
                    ):
                        st.session_state["ax_page"] = key
                        st.rerun()
                    if disabled:
                        st.caption("🔒 Requer dados carregados")


def render_analytics(file_path: str | None) -> None:
    st.markdown(
        """
        <style>
          .ax-title { color:#0E4D64; font-weight:800; }
          .ax-sub   { color:#1B98B5; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not file_path:
        st.header("📈 Analytics")
        st.caption(
            "Análises estatísticas guiadas — descritiva, distribuição, "
            "normalidade, correlação, regressão, grupos, CEP, capabilidade, "
            "temporal, lag e outliers."
        )
        st.info("⬅️ Carregue uma planilha (CSV ou Excel) na barra lateral.", icon="📥")
        return

    if not _ensure_base(file_path):
        return

    page = st.session_state.get("ax_page", "home")
    meta = ANALYSES.get(page)

    if page != "home" and meta is None:  # rota desconhecida: volta à home
        st.session_state["ax_page"] = "home"
        st.rerun()
        return

    top = st.columns([0.55, 0.30, 0.15])
    with top[1]:
        _jump_nav(page if meta else None)
    with top[2]:
        if meta and st.button("← Início", use_container_width=True, key="ax_home_btn"):
            st.session_state["ax_page"] = "home"
            st.rerun()

    if meta is None:
        _home()
        return

    with top[0]:
        st.markdown(f"<h2 class='ax-title'>{meta['icon']} {meta['label']}</h2>",
                    unsafe_allow_html=True)
    st.caption(meta["desc"])
    st.divider()

    try:
        meta["render"](_state_view())
    except Exception as exc:  # captura amigável de qualquer erro inesperado
        st.error(
            "⚠️ Ocorreu um erro ao executar esta análise. Verifique se as variáveis "
            "escolhidas são adequadas e tente novamente."
        )
        with st.expander("Detalhes técnicos (para depuração)"):
            st.exception(exc)
