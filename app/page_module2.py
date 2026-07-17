"""Página do Módulo 2 — quem impacta o alvo (análise de influência)."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from causal_analysis import plots_plotly as pp
from causal_analysis.aggregation import reduce_to_scale
from causal_analysis.managerial_report import build_managerial_report
from causal_analysis.pipeline import analyze_dataframe
from shared.io_loader import load_workbook, prepare_analysis_frame
from ui_components import add_to_report_button


@st.cache_data(show_spinner=False)
def _columns_of(file_path: str) -> dict[str, list[str]]:
    frames, _ = load_workbook(file_path)
    return {name: [str(c) for c in df.columns] for name, df in frames.items()}


@st.cache_data(show_spinner=False)
def _row_count(file_path: str, target: str) -> int:
    try:
        df, _, _ = prepare_analysis_frame(file_path, target)
        return len(df)
    except Exception:
        return 0


@st.cache_data(show_spinner="Analisando os dados — isso pode levar alguns instantes...")
def _run(file_path: str, target: str, max_lag: int, alpha: float, max_rows: int):
    df, align, sheet_infos = prepare_analysis_frame(file_path, target)
    df, scale_note = reduce_to_scale(df, max_rows=max_rows)
    result = analyze_dataframe(df, target, max_lag=max_lag, alpha=alpha,
                               verbose=False)
    return df, align, sheet_infos, result, scale_note


def render_module2(file_path: str | None) -> None:
    st.header("🔎 Módulo 2 — O que impacta o alvo")
    st.caption(
        "Combina vários métodos estatísticos (correlações lineares e "
        "não-lineares, efeitos com atraso, precedência temporal e um modelo "
        "preditivo) para **ranquear os indicadores que mais influenciam a "
        "variável alvo** — pensado para dados industriais, que raramente são "
        "lineares ou normais."
    )
    if not file_path:
        st.info("⬅️ Carregue uma planilha (CSV ou Excel) na barra lateral.", icon="📥")
        return

    try:
        cols_by_sheet = _columns_of(file_path)
    except ValueError as e:
        st.error(str(e))
        return

    all_cols = sorted({c for cols in cols_by_sheet.values() for c in cols})
    default_target = st.session_state.get("m2_target")
    idx = all_cols.index(default_target) if default_target in all_cols else 0
    target = st.selectbox(
        "Variável alvo", all_cols, index=idx,
        help="A variável cujo comportamento você quer explicar. Se a "
             "planilha tem várias abas com granularidades diferentes, a aba "
             "que contém o alvo define a grade de tempo; as demais são "
             "alinhadas a ela automaticamente.",
    )
    st.session_state["m2_target"] = target

    n_rows = _row_count(file_path, target)
    with st.expander("⚙️ Opções da análise"):
        max_lag = st.slider(
            "Defasagem máxima testada (lag)", 1, 30, 14,
            help="Até quantos períodos de atraso procurar por efeitos do "
                 "tipo 'a variável mexeu agora e o alvo respondeu depois'.",
        )
        alpha = st.select_slider(
            "Rigor estatístico (α)", options=[0.01, 0.05, 0.10], value=0.05,
            help="Limiar de significância com controle de falsos positivos "
                 "(FDR). Menor = mais rigoroso.",
        )
        max_rows = st.select_slider(
            "Máximo de linhas para a análise (desempenho)",
            options=[5000, 10000, 15000, 30000, 50000, 100000],
            value=10000,
            help="Séries muito longas (ex.: minuto a minuto por meses) deixam "
                 "a análise lenta e podem estourar a memória. Acima deste "
                 "limite, as linhas são agregadas pela média em blocos "
                 "consecutivos (equivale a reamostrar para uma grade mais "
                 "grossa) — os efeitos com atraso são preservados numa escala "
                 "de tempo maior. Menor = mais rápido; maior = mais detalhe.",
        )
    if n_rows > max_rows:
        st.info(
            f"📉 Este conjunto tem **{n_rows:,}** linhas. Para caber no limite "
            f"de **{max_rows:,}**, elas serão **agregadas pela média** em "
            "blocos consecutivos antes da análise (isso preserva os efeitos "
            "com atraso, apenas numa escala de tempo maior). Ajuste o limite "
            "em **⚙️ Opções da análise**.".replace(",", "."),
            icon="⚙️",
        )

    if not st.button("▶️ Analisar", type="primary"):
        if "m2_last" in st.session_state:
            pass  # cai para reexibir a última análise abaixo
        else:
            return

    try:
        df, align, sheet_infos, result, scale_note = _run(
            file_path, target, max_lag, alpha, max_rows)
        st.session_state["m2_last"] = (file_path, target, max_lag, alpha)
    except ValueError as e:
        st.error(str(e))
        return

    if scale_note:
        st.success(f"⚙️ {scale_note}", icon="✅")

    # ---- diagnóstico do alinhamento multi-aba --------------------------
    if len(align.sheets) > 1:
        with st.expander("🗂️ Como as abas foram combinadas"):
            for name, how in align.sheets.items():
                st.markdown(f"- **{name}**: {how}")
            st.caption(
                "Séries mais finas que o alvo geram famílias de métricas por "
                "janela (média, mediana, mínimo, máximo, P10, P90, desvio e "
                "% do tempo em faixa alta/baixa), para capturar picos e "
                "permanências que a média esconderia."
            )
            for note in align.notes:
                st.caption(f"ℹ️ {note}")

    # ---- ranking --------------------------------------------------------
    st.subheader("🏆 Ranking de influência")
    fig_rank = pp.fig_ranking(result.scores)
    st.plotly_chart(fig_rank, use_container_width=True)
    with st.expander("Ver tabela completa com estatísticas"):
        st.dataframe(result.scores, use_container_width=True)

    # ---- relatório gerencial -------------------------------------------
    mgr = build_managerial_report(result)
    st.subheader("📋 Leitura gerencial (sem estatiquês)")
    st.markdown(f"**{mgr.headline}**")
    st.markdown(mgr.summary)
    for f in mgr.findings:
        st.markdown(f"- {f}")
    for c in mgr.cautions:
        st.caption(f"⚠️ {c}")
    if mgr.ranking_table is not None:
        st.dataframe(mgr.ranking_table, use_container_width=True)

    item_id = hashlib.md5(
        f"m2|{target}|{max_lag}|{alpha}|{file_path}".encode()
    ).hexdigest()[:12]
    add_to_report_button(
        {
            "id": item_id,
            "module": "Módulo 2 — Influência",
            "title": f"O que impacta '{target}'",
            "texts": [mgr.headline, mgr.summary] + mgr.findings
                     + [f"Atenção: {c}" for c in mgr.cautions],
            "tables": {
                "Ranking (leitura gerencial)": mgr.ranking_table,
                "Ranking (estatísticas)": result.scores,
            },
            "figures": {"ranking": fig_rank},
        },
        key=f"add_{item_id}",
    )

    # ---- detalhe por indicador -----------------------------------------
    st.divider()
    st.subheader("🔬 Investigar um indicador")
    options = list(result.scores["parametro"])
    chosen = st.selectbox(
        "Indicador para detalhar", options,
        help="Veja como e quando este indicador se relaciona com o alvo.",
    )
    r = result.per_param[chosen]
    y = result.df[result.target]
    x = result.df[chosen]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["⏱️ Quando (lag)", "📈 Forma da relação", "📦 Por faixa", "🕒 No tempo"]
    )
    with tab1:
        st.plotly_chart(
            pp.fig_lag_profile(chosen, r.get("lag_profile", {}),
                               r.get("rolling_profile", {})),
            use_container_width=True,
        )
        lb = r.get("ljungbox")
        gr = r.get("granger")
        c1, c2 = st.columns(2)
        with c1:
            if lb:
                st.metric(
                    "Estrutura temporal (Ljung-Box, p)", f"{lb['p_value']:.3f}",
                    help="p ≤ 0,05 indica que a série tem 'memória' — "
                         "pré-requisito para efeitos com atraso fazerem "
                         "sentido.",
                )
        with c2:
            if gr:
                st.metric(
                    f"Precedência temporal (Granger, lag {gr['best_lag']})",
                    f"p = {gr['p_value']:.3f}",
                    help="p ≤ 0,05 sugere que o PASSADO deste indicador "
                         "ajuda a prever o alvo — evidência de que ele vem "
                         "antes do efeito.",
                )
    with tab2:
        best_feat = r.get("best_feature") or chosen
        label = r.get("best_label", "bruto")
        from causal_analysis.features import derived_features

        fam = derived_features(x, result.max_lag, result.windows)
        xx = fam[best_feat] if best_feat in fam.columns else x
        st.plotly_chart(
            pp.fig_scatter(xx, y, chosen, result.target, label),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(
            pp.fig_quartile_box(x, y, chosen, result.target),
            use_container_width=True,
        )
    with tab4:
        st.plotly_chart(
            pp.fig_timeseries_overlay(y, x, result.target, chosen),
            use_container_width=True,
        )

    # ---- investigação em cadeia ----------------------------------------
    st.divider()
    st.subheader("🧭 Continuar a investigação")
    st.caption(
        "Encontrou um culpado? Investigue **o que impacta ele**: torne-o o "
        "novo alvo e rode a análise de novo — repetindo até chegar à causa "
        "raiz."
    )
    from causal_analysis.aggregation import base_indicator

    next_options = [
        c for c in dict.fromkeys(
            base_indicator(p) for p in result.scores["parametro"]
        )
        if c != target and c in all_cols
    ]
    if next_options:
        c1, c2 = st.columns([3, 1])
        new_target = c1.selectbox("Novo alvo", next_options,
                                  label_visibility="collapsed")
        if c2.button("🎯 Tornar alvo", type="secondary"):
            st.session_state["m2_target"] = new_target
            st.rerun()
