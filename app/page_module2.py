"""Página do Módulo 2 — quem impacta o alvo (análise de influência)."""

from __future__ import annotations

import gc
import hashlib

import pandas as pd
import streamlit as st

from causal_analysis import plots_plotly as pp
from causal_analysis.aggregation import reduce_to_scale
from causal_analysis.managerial_report import build_managerial_report
from causal_analysis.pipeline import analyze_dataframe
from shared.io_loader import load_workbook, prepare_analysis_frame
from ui_components import add_to_report_button


@st.cache_data(show_spinner=False, max_entries=4)
def _sheet_meta(file_path: str) -> dict[str, dict]:
    """Colunas e nº de linhas por aba (uma leitura leve, cacheada)."""
    frames, _ = load_workbook(file_path)
    return {
        name: {"columns": [str(c) for c in df.columns], "n_rows": len(df)}
        for name, df in frames.items()
    }


# cache_resource (e não cache_data): evita re-desserializar por pickle o
# DataFrame + resultado a cada rerun, e max_entries=1 mantém UMA análise em
# memória — essencial para não estourar a RAM do Streamlit Cloud. Os objetos
# retornados são tratados como somente-leitura pela página.
@st.cache_resource(
    max_entries=1,
    show_spinner="Analisando os dados — isso pode levar alguns instantes...",
)
def _run(file_path: str, target: str, max_lag: int, alpha: float, max_rows: int):
    df, align, _sheet_infos = prepare_analysis_frame(file_path, target)
    df, scale_note = reduce_to_scale(df, max_rows=max_rows)
    result = analyze_dataframe(df, target, max_lag=max_lag, alpha=alpha,
                               verbose=False)
    del df  # result.df já contém a versão usada pela página
    gc.collect()
    return align, result, scale_note


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
        sheet_meta = _sheet_meta(file_path)
    except ValueError as e:
        st.error(str(e))
        return

    all_cols = sorted({c for m in sheet_meta.values() for c in m["columns"]})
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

    # nº de linhas da aba do alvo (define a grade da análise) — leitura leve
    n_rows = next(
        (m["n_rows"] for m in sheet_meta.values() if target in m["columns"]), 0
    )
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
        align, result, scale_note = _run(
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

    # ---- abas de exploração --------------------------------------------
    st.divider()
    tab_diag, tab_cf, tab_detalhe = st.tabs(
        ["📅 Diagnóstico do dia", "🎯 Contrafactual e repartição",
         "🔬 Investigar um indicador"]
    )

    with tab_diag:
        _render_day_diagnosis(result)

    with tab_cf:
        _render_counterfactual(result, f"{file_path}|{target}|{max_lag}|{alpha}")

    with tab_detalhe:
        _render_indicator_detail(result)


def _render_indicator_detail(result) -> None:
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


# cache_resource: o modelo contrafactual é caro (comitê de florestas) e não
# deve ser re-serializado a cada rerun; max_entries=1 mantém uma análise por
# vez na memória, como o resto do módulo
@st.cache_resource(
    max_entries=1,
    show_spinner="Ajustando o modelo contrafactual e o comitê de bootstrap...",
)
def _fit_cf(_result, cache_key: str, top: int, n_boot: int):
    from causal_analysis.counterfactual import fit_counterfactual_model

    return fit_counterfactual_model(_result, top=top, n_boot=n_boot)


@st.cache_resource(max_entries=1, show_spinner="Repartindo a variação do alvo...")
def _var_attr(_cfm, cache_key: str):
    from causal_analysis.counterfactual import variance_attribution

    return variance_attribution(_cfm)


@st.cache_resource(max_entries=1, show_spinner="Simulando cenários...")
def _scenarios(_cfm, cache_key: str):
    from causal_analysis.counterfactual import scenario_effects

    return scenario_effects(_cfm)


def _render_counterfactual(result, cache_key: str) -> None:
    """Aba do contrafactual: 'e se X estivesse normal', repartição e IC."""
    from causal_analysis.counterfactual import (
        OTHERS,
        attribute_day,
        response_curve,
    )

    st.caption(
        "O ranking diz **quem** influencia o alvo. Esta aba responde **quanto**, "
        "em unidade do alvo: *o que teria acontecido se este indicador "
        "estivesse no valor típico?* Um modelo é consultado duas vezes — com "
        "os valores que de fato ocorreram e com o indicador forçado ao valor "
        "típico — e a diferença é o efeito atribuído. A repartição usa o "
        "**valor de Shapley**, a única forma de dividir o desvio em que as "
        "fatias somam exatamente o total. Todos os números saem com "
        "**intervalo de confiança** (bootstrap por blocos)."
    )

    with st.expander("⚙️ Opções do contrafactual"):
        top = st.slider(
            "Indicadores decompostos individualmente", 3, 15, 8,
            help="Os demais entram agrupados como 'demais indicadores' — "
                 "assim a repartição continua fechando 100% sem encarecer o "
                 "cálculo.",
        )
        n_boot = st.select_slider(
            "Réplicas de bootstrap (intervalos de confiança)",
            options=[0, 4, 8, 12, 20], value=8,
            help="Cada réplica reajusta um modelo sobre blocos reamostrados "
                 "da série. Mais réplicas = intervalos mais estáveis e "
                 "cálculo mais lento. Zero desliga os intervalos.",
        )

    key = f"{cache_key}|{top}|{n_boot}"
    calcular = st.button("▶️ Calcular contrafactual", type="primary", key="btn_cf")
    # sem clique, só re-exibe o que JÁ foi calculado com estas opções — mudar
    # um controle não pode disparar sozinho um ajuste caro
    if not calcular and st.session_state.get("cf_key") != key:
        st.info(
            "Este cálculo ajusta um modelo preditivo dedicado — rode quando "
            "quiser quantificar o efeito, não só ranqueá-lo.",
            icon="🎯",
        )
        return
    st.session_state["cf_key"] = key

    cfm = _fit_cf(result, key, top, n_boot)
    if not cfm.ok:
        st.warning(f"Contrafactual indisponível: {cfm.skipped_reason}")
        return
    for n in cfm.notes:
        st.caption(f"ℹ️ {n}")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Poder do modelo (R² fora da amostra)",
        f"{cfm.r2_oos:.2f}".replace(".", ",") if pd.notna(cfm.r2_oos) else "—",
        help="Fração da variação do alvo que o modelo acerta em períodos que "
             "ele não viu no treino. Contrafactual só vale a leitura se este "
             "número for razoável.",
    )
    c2.metric(
        "Erro típico do modelo",
        f"±{cfm.resid_sigma:.4g}".replace(".", ",")
        if pd.notna(cfm.resid_sigma) else "—",
        help="Desvio-padrão do erro fora da amostra, na unidade do alvo — a "
             "régua para saber se um desvio é grande.",
    )
    c3.metric(
        "Alvo num período totalmente típico", f"{cfm.y_typical:.4g}".replace(".", ","),
        help="Previsão do modelo com TODOS os indicadores na mediana "
             "histórica — a linha de base de onde parte a repartição.",
    )

    # ---- repartição da variação ----------------------------------------
    st.subheader("📊 De onde vem a variação do alvo")
    var = _var_attr(cfm, key)
    fig_var = pp.fig_variance_shares(var, result.target)
    st.plotly_chart(fig_var, use_container_width=True)
    for f in var.findings:
        st.markdown(f"- {f}")
    with st.expander("Ver tabela da repartição"):
        st.dataframe(var.rows.drop(columns=["grupo"]), use_container_width=True)

    # ---- cenários "e se estivesse típico" -------------------------------
    st.subheader("🎯 E se cada indicador tivesse ficado no valor típico?")
    scen = _scenarios(cfm, key)
    fig_scen = pp.fig_scenarios(scen, conf=cfm.conf)
    st.plotly_chart(fig_scen, use_container_width=True)
    for f in scen.findings:
        st.markdown(f"- {f}")
    with st.expander("Ver tabela dos cenários"):
        st.dataframe(scen.rows.drop(columns=["grupo"]), use_container_width=True)

    # ---- curva de resposta ----------------------------------------------
    st.subheader("📈 Curva de resposta — existe uma faixa boa?")
    alvos = [g for g in cfm.group_order if g != OTHERS]
    escolhido = st.selectbox(
        "Indicador", alvos,
        help="Simula o alvo com este indicador fixado em cada nível que ele "
             "já assumiu no histórico, mantendo os demais como estiveram.",
    )
    curva = response_curve(cfm, escolhido)
    fig_curva = pp.fig_response_curve(curva, conf=cfm.conf)
    st.plotly_chart(fig_curva, use_container_width=True)
    for f in curva.findings:
        st.markdown(f"- {f}")

    # ---- repartição do desvio de um período -----------------------------
    st.subheader("🧾 Repartir o desvio de um período específico")
    labels = list(cfm.index)
    dia = st.selectbox(
        "Período", list(reversed(labels)), format_func=_fmt_day,
        key="cf_dia",
        help="Só aparecem os períodos com todos os indicadores disponíveis "
             "(as versões com defasagem exigem histórico anterior).",
    )
    day = attribute_day(cfm, dia)
    fig_wf = pp.fig_day_waterfall(day, _fmt_day(dia))
    st.plotly_chart(fig_wf, use_container_width=True)
    fig_contrib = pp.fig_day_contributions(day, conf=cfm.conf)
    st.plotly_chart(fig_contrib, use_container_width=True)
    for f in day.findings:
        st.markdown(f"- {f}")
    with st.expander("Ver tabela do período"):
        st.dataframe(day.rows.drop(columns=["grupo"]), use_container_width=True)
    for c in day.cautions:
        st.caption(f"⚠️ {c}")

    # ---- relatório -------------------------------------------------------
    item_id = hashlib.md5(f"m2cf|{key}|{dia}".encode()).hexdigest()[:12]
    add_to_report_button(
        {
            "id": item_id,
            "module": "Módulo 2 — Contrafactual",
            "title": f"Contrafactual e repartição — '{result.target}'",
            "texts": var.findings + scen.findings + curva.findings
                     + day.findings
                     + [f"Atenção: {c}" for c in
                        dict.fromkeys(var.cautions + scen.cautions + day.cautions)],
            "tables": {
                "Repartição da variação": var.rows.drop(columns=["grupo"]),
                "Cenários (e se estivesse típico)": scen.rows.drop(columns=["grupo"]),
                f"Repartição do desvio de {_fmt_day(dia)}":
                    day.rows.drop(columns=["grupo"]),
            },
            "figures": {
                "repartição da variação": fig_var,
                "cenários contrafactuais": fig_scen,
                "curva de resposta": fig_curva,
                "cascata do período": fig_wf,
                "contribuições com IC": fig_contrib,
            },
        },
        key=f"add_{item_id}",
    )


def _fmt_day(label) -> str:
    """Rótulo do dia amigável (datas em pt-BR; sequência como 'linha N')."""
    if isinstance(label, pd.Timestamp):
        if label == label.normalize():
            return label.strftime("%d/%m/%Y")
        return label.strftime("%d/%m/%Y %H:%M")
    return f"linha {label}"


def _render_day_diagnosis(result) -> None:
    from causal_analysis.day_diagnosis import diagnose_day

    st.caption(
        "**O que impactou o alvo num dia específico?** O diagnóstico cruza o "
        "ranking histórico (quem move o alvo) com a **atipicidade de cada "
        "indicador no dia escolhido** (percentil do valor do dia dentro do "
        "próprio histórico). É um indício priorizado para investigação — "
        "não prova causal."
    )
    labels = list(result.df[result.target].dropna().index)
    if len(labels) < 3:
        st.warning("Histórico insuficiente para diagnóstico por dia.")
        return
    dia = st.selectbox(
        "Dia (período da grade do alvo) a diagnosticar",
        list(reversed(labels)), format_func=_fmt_day,
        help="Por padrão, o período mais recente — ex.: o dia de ontem.",
    )
    diag = diagnose_day(result, dia)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Alvo no dia ({result.target})",
              f"{diag.target_value:.4g}".replace(".", ",")
              if pd.notna(diag.target_value) else "—")
    c2.metric("Percentil no histórico",
              f"P{diag.target_pct:.0f}" if pd.notna(diag.target_pct) else "—",
              help="Onde o valor do dia cai dentro de todo o histórico: "
                   "P50 = mediana; P95 = entre os 5% mais altos.")
    c3.metric("Histórico usado", f"{diag.n_history} períodos")

    if diag.rows is not None and not diag.rows.empty:
        fig_diag = pp.fig_day_diagnosis(diag.rows, _fmt_day(dia))
        st.plotly_chart(fig_diag, use_container_width=True)
        tabela = diag.rows.drop(columns=["parametro"])
        st.dataframe(tabela, use_container_width=True)
    else:
        fig_diag, tabela = None, None
        st.info("Sem contribuintes calculáveis para este dia.")

    for f in diag.findings:
        st.markdown(f"- {f}")
    for c in diag.cautions:
        st.caption(f"⚠️ {c}")

    item_id = hashlib.md5(
        f"m2diag|{result.target}|{dia}".encode()
    ).hexdigest()[:12]
    figures = {"contribuintes do dia": fig_diag} if fig_diag else {}
    tables = {"Diagnóstico do dia": tabela} if tabela is not None else {}
    add_to_report_button(
        {
            "id": item_id,
            "module": "Módulo 2 — Diagnóstico do dia",
            "title": f"Diagnóstico de {_fmt_day(dia)} — '{result.target}'",
            "texts": diag.findings + [f"Atenção: {c}" for c in diag.cautions],
            "tables": tables,
            "figures": figures,
        },
        key=f"add_{item_id}",
    )
