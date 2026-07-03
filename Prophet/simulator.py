"""Aba Simulador: gera widgets de entrada a partir dos metadados das variáveis
selecionadas e faz a previsão com o pipeline treinado."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processing import DATE_FEATURE_BOUNDS, DATE_FEATURE_LABELS


def render_simulator(pipeline, selected_features: list, feature_meta: dict,
                     target_col: str, problem_type: str) -> None:
    st.markdown(
        "Digite os valores de cada variável e clique em **Prever** para simular o resultado. "
        "Os valores iniciais são a mediana (números) ou o valor mais frequente (categorias) "
        "da sua base de dados."
    )

    entradas: dict = {}
    colunas_layout = st.columns(3)
    for i, col in enumerate(selected_features):
        meta = feature_meta[col]
        destino = colunas_layout[i % 3]
        with destino:
            if meta["tipo"] == "numerica":
                entradas[col] = _numeric_input(col, meta)
            else:
                opcoes = meta["opcoes"]
                idx = opcoes.index(meta["moda"]) if meta["moda"] in opcoes else 0
                entradas[col] = st.selectbox(col, options=opcoes, index=idx, key=f"sim_{col}")

    if st.button("🔮 Prever", type="primary", key="sim_prever"):
        X = pd.DataFrame([entradas])
        try:
            previsao = pipeline.predict(X)[0]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Não foi possível gerar a previsão: {exc}")
            return

        if problem_type == "regressao":
            st.metric(f"Previsão de '{target_col}'", f"{float(previsao):,.4g}")
        else:
            st.metric(f"Previsão de '{target_col}'", str(previsao))
            modelo = pipeline.named_steps["modelo"]
            if hasattr(modelo, "predict_proba"):
                probas = pipeline.predict_proba(X)[0]
                classes = [str(c) for c in modelo.classes_]
                df_proba = pd.DataFrame({"Classe": classes, "Probabilidade": probas})
                fig = px.bar(
                    df_proba, x="Classe", y="Probabilidade",
                    title="Probabilidade de cada classe", range_y=[0, 1],
                )
                st.plotly_chart(fig, width="stretch")


def _numeric_input(col: str, meta: dict):
    rotulo = DATE_FEATURE_LABELS.get(col, col)
    if col in DATE_FEATURE_BOUNDS:
        vmin, vmax = DATE_FEATURE_BOUNDS[col]
        default = int(round(meta["mediana"]))
        default = min(max(default, vmin), vmax)
        return st.number_input(rotulo, min_value=vmin, max_value=vmax, value=default,
                               step=1, key=f"sim_{col}")

    # folga de 50% do intervalo observado para permitir extrapolação controlada
    amplitude = meta["max"] - meta["min"]
    folga = 0.5 * amplitude if amplitude > 0 else max(abs(meta["max"]), 1.0)
    if col == "indice_tendencia":
        # simular "o próximo período" é o caso típico
        default = meta["max"]
        vmin, vmax = meta["min"], meta["max"] + folga
    else:
        default = meta["mediana"]
        vmin, vmax = meta["min"] - folga, meta["max"] + folga

    if meta.get("inteira"):
        return st.number_input(
            rotulo, min_value=int(vmin // 1), max_value=int(-(-vmax // 1)),
            value=int(round(default)), step=1, key=f"sim_{col}",
        )
    return st.number_input(
        rotulo, min_value=float(vmin), max_value=float(vmax), value=float(default),
        format="%.4f", key=f"sim_{col}",
    )
