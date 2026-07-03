"""Criador de Modelos Preditivos — aplicação Streamlit.

Executar a partir da raiz do repositório:
    streamlit run Prophet/app.py

Fluxo em abas:
  1. Upload e Dados       → carregar planilha, escolher alvo e tipo de problema
  2. Tratamento           → limpeza automática com relatório do que foi feito
  3. Seleção de Variáveis → score 0-100 com explicação; usuário escolhe as variáveis
  4. Modelo e Treinamento → escolha do modelo (com prós/contras), métricas e download
  5. Simulador            → digitar valores e obter a previsão
  6. Guia de Modelos      → tabelas de prós/contras de todos os modelos
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# garante que os módulos irmãos sejam encontrados mesmo fora do `streamlit run`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_processing as dp
import export
import feature_selection as fs
import simulator
import training
from models_catalog import MODEL_CATALOG, recommend_model

st.set_page_config(page_title="Criador de Modelos Preditivos", page_icon="🔮", layout="wide")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

# chaves de estado por etapa: mudar algo em uma etapa invalida tudo que vem depois
STATE_KEYS_BY_STEP = {
    1: ["raw_df", "file_name", "date_col", "target_col", "problem_type"],
    2: ["treated_df", "feature_meta", "treatment_report"],
    3: ["feature_scores", "selected_features"],
    4: ["trained_pipeline", "train_results", "model_key", "model_params"],
}
ALL_KEYS = [k for chaves in STATE_KEYS_BY_STEP.values() for k in chaves]


def init_state() -> None:
    for k in ALL_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def invalidate_from(step: int) -> None:
    """Zera o estado da etapa `step` em diante (evita modelo/simulador obsoletos)."""
    for s, chaves in STATE_KEYS_BY_STEP.items():
        if s >= step:
            for k in chaves:
                st.session_state[k] = None


def set_and_invalidate(key: str, value, downstream_step: int) -> None:
    """Grava `key` e, se o valor mudou, invalida as etapas seguintes."""
    if st.session_state.get(key) != value:
        invalidate_from(downstream_step)
        st.session_state[key] = value


# ---------------------------------------------------------------- Aba 1
def render_tab_upload() -> None:
    st.subheader("1️⃣ Envie sua planilha")
    st.markdown(
        "A **primeira coluna deve conter datas** (ex.: `31/12/2024`) e as demais colunas os "
        "parâmetros. Formatos aceitos: **.csv, .xlsx, .xls**."
    )

    col_up, col_ex = st.columns([2, 1])
    with col_up:
        arquivo = st.file_uploader("Planilha de dados", type=["csv", "xlsx", "xls"], key="w_upload")
    with col_ex:
        st.markdown("**Ou use um exemplo:**")
        exemplo = None
        if st.button("📈 Exemplo de regressão (vendas)", key="w_ex_reg"):
            exemplo = "exemplo_vendas.csv"
        if st.button("🏷️ Exemplo de classificação (clientes)", key="w_ex_clf"):
            exemplo = "exemplo_clientes.csv"

    novo_df, novo_nome = None, None
    if exemplo:
        caminho = os.path.join(SAMPLE_DIR, exemplo)
        if os.path.exists(caminho):
            with open(caminho, "rb") as f:
                novo_df = dp.load_file(exemplo, f.read())
            novo_nome = exemplo
        else:
            st.error("Arquivo de exemplo não encontrado. Rode `python Prophet/sample_data.py`.")
    elif arquivo is not None and arquivo.name != st.session_state.file_name:
        try:
            novo_df = dp.load_file(arquivo.name, arquivo.getvalue())
            novo_nome = arquivo.name
        except dp.DadosInvalidosError as exc:
            st.error(str(exc))
            return

    if novo_df is not None:
        invalidate_from(1)
        st.session_state.raw_df = novo_df
        st.session_state.file_name = novo_nome
        st.session_state.date_col = novo_df.columns[0]

    raw_df = st.session_state.raw_df
    if raw_df is None:
        st.info("Envie uma planilha ou carregue um exemplo para começar.")
        return

    st.success(
        f"Arquivo **{st.session_state.file_name}** carregado: "
        f"{len(raw_df)} linhas × {raw_df.shape[1]} colunas. "
        f"Coluna de data: **`{st.session_state.date_col}`** (primeira coluna)."
    )
    st.dataframe(raw_df.head(10), width="stretch")

    # escolha da variável alvo (a coluna de data não pode ser alvo)
    opcoes_alvo = [c for c in raw_df.columns if c != st.session_state.date_col]
    idx_atual = (
        opcoes_alvo.index(st.session_state.target_col)
        if st.session_state.target_col in opcoes_alvo
        else len(opcoes_alvo) - 1  # última coluna costuma ser o alvo
    )
    alvo = st.selectbox("🎯 Qual coluna é a variável alvo (o que você quer prever)?",
                        opcoes_alvo, index=idx_atual, key="w_target")
    set_and_invalidate("target_col", alvo, downstream_step=2)

    tipo_detectado = dp.detect_problem_type(raw_df[alvo])
    rotulos = {"regressao": "Regressão (prever um número)",
               "classificacao": "Classificação (prever uma categoria)"}
    st.caption(
        f"Detecção automática: **{rotulos[tipo_detectado]}**. Ajuste abaixo se necessário."
    )
    atual = st.session_state.problem_type or tipo_detectado
    escolha = st.radio(
        "Tipo de problema",
        options=["regressao", "classificacao"],
        format_func=lambda x: rotulos[x],
        index=["regressao", "classificacao"].index(atual),
        horizontal=True,
        key="w_problem_type",
    )
    set_and_invalidate("problem_type", escolha, downstream_step=2)
    st.success("✅ Etapa concluída — avance para a aba **2. Tratamento**.")


# ---------------------------------------------------------------- Aba 2
def render_tab_treatment() -> None:
    st.subheader("2️⃣ Tratamento automático dos dados")
    if st.session_state.raw_df is None or st.session_state.target_col is None:
        st.info("Complete a etapa anterior (**1. Upload e Dados**) primeiro.")
        return

    if st.session_state.treated_df is None:
        try:
            with st.spinner("Tratando os dados..."):
                treated_df, feature_meta, report = dp.treat_data(
                    st.session_state.raw_df,
                    st.session_state.date_col,
                    st.session_state.target_col,
                    st.session_state.problem_type,
                )
        except dp.DadosInvalidosError as exc:
            st.error(str(exc))
            return
        st.session_state.treated_df = treated_df
        st.session_state.feature_meta = feature_meta
        st.session_state.treatment_report = report

    report = st.session_state.treatment_report
    st.markdown("#### 📋 Relatório do tratamento")
    st.markdown(report.to_markdown_pt())

    with st.expander("Ver dados tratados (amostra)", expanded=False):
        st.dataframe(st.session_state.treated_df.head(20), width="stretch")

    st.markdown("#### 📉 Evolução no tempo")
    df = st.session_state.treated_df
    alvo = st.session_state.target_col
    if pd.api.types.is_numeric_dtype(df[alvo]):
        st.line_chart(df.set_index(st.session_state.date_col)[alvo])
    else:
        st.caption("A variável alvo é categórica — gráfico temporal não aplicável.")
    st.success("✅ Tratamento concluído — avance para a aba **3. Seleção de Variáveis**.")


# ---------------------------------------------------------------- Aba 3
def render_tab_features() -> None:
    st.subheader("3️⃣ Seleção de variáveis")
    if st.session_state.treated_df is None:
        st.info("Complete a etapa anterior (**2. Tratamento**) primeiro.")
        return

    if st.session_state.feature_scores is None:
        with st.spinner("Calculando a relevância de cada variável..."):
            st.session_state.feature_scores = fs.score_features(
                st.session_state.treated_df,
                st.session_state.target_col,
                st.session_state.problem_type,
                st.session_state.feature_meta,
            )

    scores = st.session_state.feature_scores
    if scores.empty:
        st.error("Nenhuma variável disponível após o tratamento.")
        return

    st.markdown(
        "Cada variável recebeu um **score de 0 a 100** combinando três análises: "
        "**associação estatística** com o alvo (correlação/ANOVA), **informação mútua** "
        "(captura relações não lineares) e **importância em floresta aleatória**. "
        "A coluna *Motivo* explica a pontuação — use-a para decidir quais variáveis manter."
    )
    exibicao = scores.rename(
        columns={
            "variavel": "Variável",
            "score": "Score (0-100)",
            "assoc": "Associação",
            "mi": "Info. mútua",
            "arvore": "Import. árvore",
            "recomendada": "Recomendada",
            "explicacao": "Motivo",
        }
    )[["Variável", "Score (0-100)", "Recomendada", "Motivo", "Associação", "Info. mútua", "Import. árvore"]]
    st.dataframe(
        exibicao,
        width="stretch",
        hide_index=True,
        column_config={
            "Score (0-100)": st.column_config.ProgressColumn(
                "Score (0-100)", min_value=0, max_value=100, format="%d"
            ),
            "Recomendada": st.column_config.CheckboxColumn("Recomendada"),
        },
    )

    recomendadas = scores.loc[scores["recomendada"], "variavel"].tolist()
    padrao = st.session_state.selected_features or recomendadas
    padrao = [c for c in padrao if c in scores["variavel"].tolist()]
    selecionadas = st.multiselect(
        "✅ Variáveis que serão usadas no modelo (pré-selecionadas as recomendadas):",
        options=scores["variavel"].tolist(),
        default=padrao,
        key="w_features",
    )
    if not selecionadas:
        st.warning("Selecione pelo menos uma variável para continuar.")
        st.session_state.selected_features = None
        invalidate_from(4)
        return
    set_and_invalidate("selected_features", selecionadas, downstream_step=4)
    st.success(
        f"✅ {len(selecionadas)} variável(is) selecionada(s) — avance para a aba "
        "**4. Modelo e Treinamento**."
    )


# ---------------------------------------------------------------- Aba 4
def render_param_widgets(spec: dict, model_key: str) -> dict:
    """Gera os widgets de hiperparâmetros a partir da especificação do catálogo."""
    params = {}
    if not spec:
        st.caption("Este modelo não possui hiperparâmetros ajustáveis.")
        return params
    colunas = st.columns(min(len(spec), 3))
    for i, (nome, cfg) in enumerate(spec.items()):
        with colunas[i % len(colunas)]:
            key = f"w_param_{model_key}_{nome}"
            if cfg["tipo"] == "slider_int":
                params[nome] = st.slider(cfg["rotulo"], cfg["min"], cfg["max"],
                                         cfg["default"], step=cfg.get("step", 1), key=key)
            elif cfg["tipo"] == "slider_float":
                params[nome] = st.slider(cfg["rotulo"], float(cfg["min"]), float(cfg["max"]),
                                         float(cfg["default"]), step=float(cfg.get("step", 0.01)),
                                         key=key)
            elif cfg["tipo"] == "selectbox":
                params[nome] = st.selectbox(cfg["rotulo"], cfg["opcoes"],
                                            index=cfg["opcoes"].index(cfg["default"]), key=key)
    return params


def render_pros_cons_table(spec: dict) -> None:
    n = max(len(spec["pros"]), len(spec["contras"]))
    pros = spec["pros"] + [""] * (n - len(spec["pros"]))
    contras = spec["contras"] + [""] * (n - len(spec["contras"]))
    st.table(pd.DataFrame({"✅ Prós": pros, "❌ Contras": contras}))


def render_tab_model() -> None:
    st.subheader("4️⃣ Modelo e treinamento")
    if not st.session_state.selected_features:
        st.info("Complete a etapa anterior (**3. Seleção de Variáveis**) primeiro.")
        return

    problem_type = st.session_state.problem_type
    catalogo = MODEL_CATALOG[problem_type]
    n_rows = len(st.session_state.treated_df)
    n_features = len(st.session_state.selected_features)

    chave_sugerida, motivo = recommend_model(n_rows, n_features, problem_type)
    st.info(f"💡 **Sugestão:** {catalogo[chave_sugerida]['nome']} — {motivo}")

    chaves = list(catalogo.keys())
    model_key = st.selectbox(
        "Escolha o modelo de machine learning",
        chaves,
        index=chaves.index(chave_sugerida),
        format_func=lambda k: catalogo[k]["nome"],
        key="w_model",
    )
    spec = catalogo[model_key]
    with st.expander(f"ℹ️ Quando usar, prós e contras — {spec['nome']}", expanded=False):
        st.markdown(f"**Quando usar:** {spec['quando_usar']}")
        render_pros_cons_table(spec)
    st.caption("Veja a comparação completa entre modelos na aba **6. Guia de Modelos**.")

    st.markdown("#### ⚙️ Hiperparâmetros")
    params = render_param_widgets(spec["params_ui"], model_key)

    col_a, col_b = st.columns(2)
    with col_a:
        test_size = st.slider("Percentual da base reservado para teste", 10, 40, 20,
                              step=5, key="w_test_size") / 100
    with col_b:
        temporal = st.checkbox(
            "Divisão temporal (usar as últimas datas como teste)",
            value=False,
            key="w_temporal",
            help="Recomendado quando o objetivo é prever o futuro: o modelo é avaliado "
            "nas datas mais recentes, que ele não viu no treino.",
        )

    if st.button("🚀 Treinar modelo", type="primary", key="w_train"):
        with st.spinner("Treinando..."):
            try:
                pipeline, resultados = training.train_model(
                    st.session_state.treated_df,
                    st.session_state.selected_features,
                    st.session_state.target_col,
                    problem_type,
                    model_key,
                    params,
                    st.session_state.feature_meta,
                    test_size=test_size,
                    temporal_split=temporal,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha no treinamento: {exc}")
                return
        st.session_state.trained_pipeline = pipeline
        st.session_state.train_results = resultados
        st.session_state.model_key = model_key
        st.session_state.model_params = params

    resultados = st.session_state.train_results
    if resultados is None:
        return
    if st.session_state.model_key != model_key:
        st.warning("O modelo selecionado mudou — treine novamente para atualizar os resultados.")

    st.markdown("---")
    st.markdown(
        f"#### 📊 Resultados — {catalogo[st.session_state.model_key]['nome']} "
        f"({resultados['n_train']} linhas de treino, {resultados['n_test']} de teste)"
    )
    for aviso in resultados["warnings"]:
        st.warning(aviso)

    colunas_metricas = st.columns(len(resultados["metrics"]))
    for coluna, (nome, valor) in zip(colunas_metricas, resultados["metrics"].items()):
        coluna.metric(nome, valor)

    figuras = resultados["figures"]
    nomes_fig = list(figuras)
    for i in range(0, len(nomes_fig), 2):
        cols = st.columns(2)
        for col_layout, nome_fig in zip(cols, nomes_fig[i : i + 2]):
            with col_layout:
                st.plotly_chart(figuras[nome_fig], width="stretch")

    st.markdown("#### 💾 Baixar o modelo treinado")
    st.markdown(
        "O arquivo `.joblib` contém o **pipeline completo** (tratamento + modelo): basta "
        "carregá-lo em Python e chamar `predict` com um DataFrame das colunas selecionadas. "
        "Baixe também o script de exemplo pronto para rodar."
    )
    artefato = export.serialize_artifact(
        st.session_state.trained_pipeline,
        st.session_state.selected_features,
        st.session_state.feature_meta,
        st.session_state.target_col,
        problem_type,
        catalogo[st.session_state.model_key]["nome"],
        resultados["metrics"],
    )
    script = export.generate_usage_script(
        st.session_state.selected_features,
        st.session_state.feature_meta,
        st.session_state.target_col,
        problem_type,
    )
    col_d1, col_d2 = st.columns(2)
    col_d1.download_button(
        "⬇️ Baixar modelo (modelo_preditivo.joblib)",
        data=artefato,
        file_name="modelo_preditivo.joblib",
        mime="application/octet-stream",
        key="w_dl_model",
    )
    col_d2.download_button(
        "⬇️ Baixar script de exemplo (exemplo_uso.py)",
        data=script,
        file_name="exemplo_uso.py",
        mime="text/x-python",
        key="w_dl_script",
    )
    st.success("✅ Modelo treinado — experimente a aba **5. Simulador**.")


# ---------------------------------------------------------------- Aba 5
def render_tab_simulator() -> None:
    st.subheader("5️⃣ Simulador")
    if st.session_state.trained_pipeline is None:
        st.info("Treine um modelo primeiro (aba **4. Modelo e Treinamento**).")
        return
    simulator.render_simulator(
        st.session_state.trained_pipeline,
        st.session_state.selected_features,
        st.session_state.feature_meta,
        st.session_state.target_col,
        st.session_state.problem_type,
    )


# ---------------------------------------------------------------- Aba 6
def render_tab_guide() -> None:
    st.subheader("6️⃣ Guia de modelos — qual usar em cada situação?")
    st.markdown(
        """
**Como escolher em 30 segundos:**

| Situação | Modelo indicado |
|---|---|
| Poucos dados (< 100 linhas) | Regressão Linear/Ridge ou Logística/Naive Bayes |
| Base média (100 a 5.000 linhas) | Random Forest |
| Base grande (> 5.000 linhas) e busca de máxima precisão | Gradient Boosting |
| Mais colunas do que linhas | Lasso (regressão) ou Logística (classificação) |
| Precisa explicar o peso de cada variável | Regressão Linear / Logística |
| Relações claramente não lineares | Random Forest, Gradient Boosting ou SVM/SVR |
"""
    )
    aba_reg, aba_clf = st.tabs(["📈 Modelos de Regressão", "🏷️ Modelos de Classificação"])
    for aba, tipo in ((aba_reg, "regressao"), (aba_clf, "classificacao")):
        with aba:
            for spec in MODEL_CATALOG[tipo].values():
                st.markdown(f"### {spec['nome']}")
                st.markdown(f"**Quando usar:** {spec['quando_usar']}")
                render_pros_cons_table(spec)
                st.markdown("---")


# ---------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.title("🔮 Criador de Modelos Preditivos")
    st.caption(
        "Envie uma planilha, escolha o alvo, deixe o tratamento automático trabalhar, "
        "selecione as variáveis, treine o modelo, simule cenários e baixe o resultado."
    )
    abas = st.tabs(
        [
            "1. Upload e Dados",
            "2. Tratamento",
            "3. Seleção de Variáveis",
            "4. Modelo e Treinamento",
            "5. Simulador",
            "6. Guia de Modelos",
        ]
    )
    with abas[0]:
        render_tab_upload()
    with abas[1]:
        render_tab_treatment()
    with abas[2]:
        render_tab_features()
    with abas[3]:
        render_tab_model()
    with abas[4]:
        render_tab_simulator()
    with abas[5]:
        render_tab_guide()


if __name__ == "__main__":
    main()
