"""Página do Módulo 1 — Análise de capabilidade de indicadores."""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

from capability import charts
from capability.data_prep import load_indicator_table
from capability.indices import TOOLTIPS
from capability.pipeline import run_capability
from shared.parsing import read_all_sheets
from ui_components import add_to_report_button, fmt_br, metric_row, parse_br_number

_OUTLIER_OPTIONS = {
    "Não remover (recomendado — a carta de controle cuida dos pontos atípicos)": "nenhum",
    "IQR — remove o que estiver além de 1,5×IQR dos quartis": "iqr",
    "Z-score — remove o que estiver a mais de 3 desvios da média": "zscore",
    "Z-score robusto (MAD) — como o Z-score, mas imune aos próprios outliers": "zscore_mad",
}

_MISSING_OPTIONS = {
    "Manter (os cálculos simplesmente ignoram os vazios)": "nenhum",
    "Interpolar lacunas curtas (até 3 pontos seguidos)": "interpolar",
    "Preencher com a mediana da coluna": "mediana",
    "Preencher com a média da coluna": "media",
}


def render_module1(file_path: str | None) -> None:
    st.header("📊 Módulo 1 — Análise de capabilidade")
    st.caption(
        "Verifica se um indicador é **capaz** de atender aos limites de "
        "atuação definidos para o processo: estabilidade (carta de "
        "controle), normalidade e índices de capabilidade."
    )
    if not file_path:
        st.info("⬅️ Carregue uma planilha (CSV ou Excel) na barra lateral.", icon="📥")
        return

    # ---- seleção de aba e indicador -----------------------------------
    sheets = read_all_sheets(file_path)
    sheet_name = (
        st.selectbox(
            "Aba da planilha", list(sheets),
            help="Cada aba pode ter uma série temporal distinta; escolha a "
                 "que contém o indicador a avaliar.",
        )
        if len(sheets) > 1 else next(iter(sheets))
    )
    try:
        df, diag = load_indicator_table(file_path, sheet=sheet_name)
    except ValueError as e:
        st.error(str(e))
        return

    for note in diag.notes:
        st.caption(f"ℹ️ {note}")

    numeric_cols = [c for c in df.columns if df[c].notna().sum() >= 4]
    if not numeric_cols:
        st.error("Nenhuma coluna numérica com dados suficientes nesta aba.")
        return
    indicator = st.selectbox(
        "Indicador a analisar", numeric_cols,
        help="A análise é feita indicador a indicador — inclusive a coluna "
             "alvo do seu processo pode ser avaliada aqui.",
    )

    # ---- tratamentos --------------------------------------------------
    with st.expander("🧹 Tratamento de dados (outliers e faltantes)", expanded=False):
        st.markdown(
            ":warning: **Atenção:** remover outliers do indicador analisado "
            "apaga justamente os pontos que a carta de controle existe para "
            "sinalizar, e reduz artificialmente a variação (inflando o Cpk). "
            "Prefira tratar os pontos atípicos na etapa da carta, com "
            "justificativa de processo; use a remoção automática apenas para "
            "erros grosseiros de digitação/medição."
        )
        out_label = st.radio(
            "Outliers", list(_OUTLIER_OPTIONS), index=0,
            help="IQR usa a distância entre os quartis Q1 e Q3; Z-score usa "
                 "a distância à média em desvios-padrão.",
        )
        miss_label = st.radio(
            "Dados faltantes", list(_MISSING_OPTIONS), index=0,
            help="Valores preenchidos artificialmente reduzem a variação "
                 "aparente do processo — use com moderação.",
        )
    outlier_method = _OUTLIER_OPTIONS[out_label]
    missing_method = _MISSING_OPTIONS[miss_label]

    # ---- limites de especificação -------------------------------------
    st.subheader("Limites de atuação do processo")
    st.caption(
        "Informe **um ou os dois** limites (use vírgula como separador "
        "decimal, ex.: `12,5`). Limite único é aceito: só inferior "
        "(quanto maior melhor) ou só superior (quanto menor melhor)."
    )
    c1, c2 = st.columns(2)
    lsl_txt = c1.text_input("Limite inferior (LIE)", key=f"lsl_{indicator}",
                            placeholder="ex.: 85,0 — deixe vazio se não houver")
    usl_txt = c2.text_input("Limite superior (LSE)", key=f"usl_{indicator}",
                            placeholder="ex.: 92,5 — deixe vazio se não houver")
    lsl = parse_br_number(lsl_txt)
    usl = parse_br_number(usl_txt)
    if lsl_txt.strip() and lsl is None:
        st.error("Limite inferior inválido — use números como 85,0.")
        return
    if usl_txt.strip() and usl is None:
        st.error("Limite superior inválido — use números como 92,5.")
        return
    if lsl is None and usl is None:
        st.info("Informe ao menos um limite para rodar a análise.", icon="🎯")
        return
    if lsl is not None and usl is not None and lsl >= usl:
        st.error("O limite inferior precisa ser menor que o superior.")
        return

    # ---- execução (1ª passada para exibir a carta) --------------------
    remove_key = f"remove_{sheet_name}_{indicator}"
    removed = st.session_state.get(remove_key, [])
    try:
        rep = run_capability(
            df[indicator], indicator, lsl=lsl, usl=usl,
            outlier_method=outlier_method, missing_method=missing_method,
            remove_labels=removed,
        )
    except ValueError as e:
        st.error(str(e))
        return

    # ---- etapa 1: carta de controle I-AM ------------------------------
    st.subheader("1️⃣ Estabilidade do processo (carta I-AM)", help=TOOLTIPS["iam"])
    fig_imr_spec = charts.fig_imr(rep, show_spec=True)
    fig_imr_ctrl = charts.fig_imr(rep)
    st.plotly_chart(fig_imr_spec, use_container_width=True)
    st.caption(
        "Acima: carta I-AM com os **limites de atuação que você informou** "
        "(LIE/LSE, em vermelho), para comparar o processo com a "
        "especificação. Abaixo: a carta de controle clássica, apenas com os "
        "limites calculados a partir da variação do próprio processo."
    )
    st.plotly_chart(fig_imr_ctrl, use_container_width=True)
    violations = rep.imr.violations if rep.imr else {}
    # só eventos discretos (R1: além de 3σ) são candidatos a exclusão; regras
    # de sequência (R2/R3/R5/R6) indicam instabilidade sistemática, que não
    # se resolve apagando pontos
    r1_points = sorted(
        (lb for lb, rules in violations.items() if 1 in rules), key=str
    )
    systemic = len(violations) - len(r1_points)
    if violations or removed:
        st.warning(
            "Pontos marcados em vermelho são **causas especiais** "
            "(excepcionalidades). Elas podem **deturpar a análise de "
            "capabilidade** — se tiverem justificativa de processo (parada, "
            "erro de medição, evento atípico), selecione-os abaixo para "
            "excluir e recalcular.",
            icon="🔴",
        )
        if systemic > 0:
            st.caption(
                f"⚠️ {systemic} ponto(s) violam regras de *sequência* "
                "(deslocamentos/tendências). Isso indica instabilidade "
                "sistemática do processo — não se corrige excluindo pontos, "
                "e os índices devem ser lidos com cautela. Apenas pontos "
                "além de 3σ (eventos discretos) podem ser excluídos abaixo."
            )
        options = sorted(set(list(r1_points) + list(removed)), key=str)
        if options:
            selected = st.multiselect(
                "Pontos a excluir da análise (a carta é recalculada)",
                options=options, default=removed,
                key=f"sel_{remove_key}",
                help="A exclusão deve ser justificada: remover pontos sem "
                     "causa conhecida embeleza artificialmente os índices.",
            )
            if selected != removed:
                st.session_state[remove_key] = selected
                st.rerun()
    else:
        st.success("Nenhuma causa especial detectada — processo estável.", icon="✅")

    # ---- etapa 2: normalidade -----------------------------------------
    st.subheader("2️⃣ Normalidade dos dados", help=TOOLTIPS["normalidade"])
    nr = rep.normality_raw
    cols = st.columns(3)
    cols[0].metric("Anderson-Darling (p)", fmt_br(nr.ad_p, 3),
                   help=TOOLTIPS["normalidade"])
    cols[1].metric("Shapiro-Wilk (p)", fmt_br(nr.shapiro_p, 3),
                   help="Teste complementar de normalidade (bom para "
                        "amostras menores). p > 0,05 corrobora a normal.")
    cols[2].metric("Resultado",
                   "Normal ✓" if rep.case == 1 else "Não normal",
                   help="Decisão pelo Anderson-Darling com α = 0,05.")

    if rep.case == 2 and rep.transform and rep.transform.best:
        st.info(
            f"Os dados não são normais, mas a transformação "
            f"**{rep.transform.best.label}** os normalizou "
            f"(p = {fmt_br(rep.normality_final.ad_p, 3)} após transformar). "
            "Os índices abaixo foram calculados na escala transformada e os "
            "valores exibidos foram convertidos de volta.",
            icon="🔁",
        )
    elif rep.case == 3:
        st.warning(
            "Os dados **não seguem a distribuição normal**, nem mesmo após "
            "transformações (logaritmo, raiz, Box-Cox, Yeo-Johnson, "
            "Johnson). A análise usa os percentis reais dos dados (box-plot).",
            icon="📦",
        )
    fig_bell = charts.fig_normality_bell(rep)
    st.plotly_chart(fig_bell, use_container_width=True)
    with st.expander("Ver gráfico de probabilidade normal (QQ)"):
        st.plotly_chart(charts.fig_qqplot(rep), use_container_width=True)
        if rep.case == 2:
            st.plotly_chart(charts.fig_normality_bell(rep, transformed=True),
                            use_container_width=True)
            st.plotly_chart(charts.fig_qqplot(rep, transformed=True),
                            use_container_width=True)

    # ---- etapa 3: resultado -------------------------------------------
    st.subheader("3️⃣ Resultado da capabilidade")
    st.markdown(f"**{rep.case_label}**")
    idx = rep.indices
    figures: dict = {}
    if rep.case in (1, 2):
        fig_hist = charts.fig_capability_hist(rep)
        st.plotly_chart(fig_hist, use_container_width=True)
        figures["histograma"] = fig_hist
        pairs = [
            ("Cp", fmt_br(idx.cp), TOOLTIPS["cp"]),
            ("Cpk", fmt_br(idx.cpk), TOOLTIPS["cpk"]),
            ("Pp", fmt_br(idx.pp), TOOLTIPS["pp"]),
            ("Ppk", fmt_br(idx.ppk), TOOLTIPS["ppk"]),
            ("PPM fora (estimado)", fmt_br(idx.ppm_total, 0), TOOLTIPS["ppm"]),
            ("% fora (observado)", fmt_br(idx.obs_pct_out, 2),
             "Percentual de medições do conjunto de dados que ficou fora "
             "dos limites."),
            ("σ curto prazo", fmt_br(idx.sigma_within, 4),
             TOOLTIPS["sigma_within"]),
            ("σ global", fmt_br(idx.sigma_overall, 4),
             TOOLTIPS["sigma_overall"]),
        ]
        metric_row(pairs)
        if np.isfinite(idx.cp) and np.isfinite(idx.cpk) and idx.cp > idx.cpk * 1.3:
            st.info(TOOLTIPS["descentrado"], icon="🎯")
    else:
        fig_box = charts.fig_boxplot(rep)
        st.plotly_chart(fig_box, use_container_width=True)
        figures["boxplot"] = fig_box
        pairs = [
            ("Ppk (percentis)", fmt_br(idx.ppk), TOOLTIPS["percentis"]),
            ("PPM fora (contado)", fmt_br(idx.ppm_total, 0), TOOLTIPS["ppm"]),
            ("% fora (observado)", fmt_br(idx.obs_pct_out, 2),
             "Percentual de medições fora dos limites atuais."),
            ("Mediana", fmt_br(idx.median, 4),
             "Valor central dos dados (50% acima, 50% abaixo)."),
        ]
        metric_row(pairs)
        if rep.suggested and rep.suggested.rationale:
            st.markdown("**💡 Sugestão de limites realistas**")
            st.markdown(rep.suggested.rationale)

    verdict_icon = {"excelente": "🟢", "capaz": "🟢", "marginal": "🟡",
                    "incapaz": "🔴"}.get(idx.verdict, "⚪")
    st.markdown(f"### {verdict_icon} Veredito: processo **{idx.verdict}**")
    st.markdown(rep.narrative)
    for note in rep.notes:
        st.caption(f"ℹ️ {note}")

    # ---- etapa 4: revisão dos limites ---------------------------------
    import pandas as pd

    lr = rep.limit_review
    tabela_limites = None
    if lr is not None and lr.situacao:
        st.subheader(
            "4️⃣ 🎯 Revisão dos limites de atuação",
            help="O Módulo 1 baliza os limites previamente definidos: conclui "
                 "se estão aderentes ao que o processo entrega (Ppk ≥ 1,33) "
                 "ou se merecem revisão, e recomenda novos limites pela regra "
                 "de cada caso.",
        )
        situ_icon = {"aderente": "🟢", "atencao": "🟡",
                     "revisar": "🔴"}[lr.situacao]
        st.markdown(f"**{situ_icon} {lr.situacao_label}**")
        st.markdown(lr.motivo)
        tabela_limites = pd.DataFrame({
            "": ["Limite inferior (LIE)", "Limite superior (LSE)"],
            "atual": [fmt_br(lr.lsl, 4) if lr.lsl is not None else "—",
                      fmt_br(lr.usl, 4) if lr.usl is not None else "—"],
            "recomendado": [
                fmt_br(lr.rec_lsl, 4) if lr.rec_lsl is not None else "—",
                fmt_br(lr.rec_usl, 4) if lr.rec_usl is not None else "—"],
        }).set_index("")
        st.table(tabela_limites)
        st.caption(f"Método da recomendação: {lr.metodo}.")

    # ---- adicionar ao relatório ---------------------------------------
    st.divider()
    item_id = hashlib.md5(
        f"m1|{sheet_name}|{indicator}|{lsl}|{usl}|{outlier_method}|"
        f"{missing_method}|{sorted(map(str, removed))}".encode()
    ).hexdigest()[:12]

    resumo = pd.DataFrame({
        "métrica": ["Caso", "Cp", "Cpk", "Pp", "Ppk", "PPM fora",
                    "% fora observado", "Veredito"],
        "valor": [rep.case_label, fmt_br(idx.cp), fmt_br(idx.cpk),
                  fmt_br(idx.pp), fmt_br(idx.ppk), fmt_br(idx.ppm_total, 0),
                  fmt_br(idx.obs_pct_out, 2), idx.verdict],
    }).set_index("métrica")
    figures["carta I-AM (limites de atuação)"] = fig_imr_spec
    figures["carta I-AM (controle)"] = fig_imr_ctrl
    figures["distribuição vs. sino"] = fig_bell
    tables = {"Resumo dos índices": resumo}
    meta = {}
    if lr is not None and lr.situacao:
        if tabela_limites is not None:
            tables["Revisão de limites"] = tabela_limites
        meta["limit_review"] = {
            "indicador": indicator,
            "situacao": lr.situacao,
            "situacao_label": lr.situacao_label,
            "lsl": lr.lsl, "usl": lr.usl,
            "rec_lsl": lr.rec_lsl, "rec_usl": lr.rec_usl,
            "metodo": lr.metodo,
        }
    add_to_report_button(
        {
            "id": item_id,
            "module": "Módulo 1 — Capabilidade",
            "title": f"Capabilidade de '{indicator}'",
            "texts": [rep.narrative] + rep.notes,
            "tables": tables,
            "figures": figures,
            "meta": meta,
        },
        key=f"add_{item_id}",
    )
