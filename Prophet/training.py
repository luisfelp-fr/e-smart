"""Construção do pipeline, treinamento, métricas e gráficos de avaliação.

Fase B do pré-processamento acontece aqui, DENTRO do Pipeline sklearn
(imputação + padronização + one-hot), ajustada somente no conjunto de treino
para evitar vazamento de dados. O pipeline completo é o artefato exportado:
ele aceita um DataFrame cru com as colunas selecionadas e devolve a previsão.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models_catalog import MODEL_CATALOG


def build_pipeline(model_key: str, problem_type: str, params: dict, feature_meta: dict,
                   selected_features: list) -> Pipeline:
    """Monta ColumnTransformer (imputação/escala/one-hot) + estimador do catálogo."""
    spec = MODEL_CATALOG[problem_type][model_key]
    kwargs = dict(spec["params_fixos"])
    kwargs.update(params)
    if spec["aceita_random_state"]:
        kwargs["random_state"] = 42
    estimador = spec["classe"](**kwargs)

    numericas = [c for c in selected_features if feature_meta[c]["tipo"] == "numerica"]
    categoricas = [c for c in selected_features if feature_meta[c]["tipo"] == "categorica"]

    transformadores = []
    if numericas:
        transformadores.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                numericas,
            )
        )
    if categoricas:
        transformadores.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            )
        )

    return Pipeline(
        [
            ("preprocessamento", ColumnTransformer(transformadores, remainder="drop")),
            ("modelo", estimador),
        ]
    )


def train_model(
    treated_df: pd.DataFrame,
    selected_features: list,
    target_col: str,
    problem_type: str,
    model_key: str,
    params: dict,
    feature_meta: dict,
    test_size: float = 0.2,
    temporal_split: bool = False,
):
    """Treina o pipeline e retorna (pipeline, resultados).

    resultados = {"metrics": dict, "figures": dict, "warnings": list[str],
                  "n_train": int, "n_test": int}
    """
    warnings: list[str] = []
    X = treated_df[selected_features].copy()
    y = treated_df[target_col].copy()
    if problem_type == "classificacao":
        y = y.astype(str)

    if len(X) < 20:
        warnings.append(
            f"A base tem apenas {len(X)} linhas — as métricas de avaliação serão pouco "
            "confiáveis. Considere coletar mais dados."
        )

    if temporal_split:
        # dados já vêm ordenados por data: usa o final da série como teste
        corte = int(round(len(X) * (1 - test_size)))
        corte = min(max(corte, 1), len(X) - 1)
        X_train, X_test = X.iloc[:corte], X.iloc[corte:]
        y_train, y_test = y.iloc[:corte], y.iloc[corte:]
    else:
        stratify = None
        if problem_type == "classificacao":
            contagens = y.value_counts()
            if contagens.min() >= 2 and len(contagens) * 2 <= len(y) * test_size:
                stratify = y
            elif contagens.min() < 2:
                warnings.append(
                    f"A classe '{contagens.idxmin()}' tem apenas {contagens.min()} exemplo(s) — "
                    "divisão estratificada desativada."
                )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify
        )

    # KNN não pode ter mais vizinhos do que exemplos de treino
    if model_key == "knn":
        k = params.get("n_neighbors", 5)
        k_max = max(len(X_train) - 1, 1)
        if k > k_max:
            params = {**params, "n_neighbors": k_max}
            warnings.append(f"Número de vizinhos reduzido para {k_max} (limite da base de treino).")

    pipeline = build_pipeline(model_key, problem_type, params, feature_meta, selected_features)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if problem_type == "regressao":
        metrics = {
            "R² (quanto da variação o modelo explica)": round(float(r2_score(y_test, y_pred)), 4),
            "MAE (erro absoluto médio)": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "RMSE (raiz do erro quadrático médio)": round(
                float(np.sqrt(mean_squared_error(y_test, y_pred))), 4
            ),
        }
        figures = _regression_figures(y_test, y_pred)
    else:
        media = "binary" if y.nunique() == 2 else "weighted"
        pos_label = sorted(y.unique())[-1] if media == "binary" else None
        kwargs = {"average": media, "zero_division": 0}
        if pos_label is not None:
            kwargs["pos_label"] = pos_label
        metrics = {
            "Acurácia (% de acertos)": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precisão": round(float(precision_score(y_test, y_pred, **kwargs)), 4),
            "Revocação (recall)": round(float(recall_score(y_test, y_pred, **kwargs)), 4),
            "F1 (equilíbrio precisão/revocação)": round(float(f1_score(y_test, y_pred, **kwargs)), 4),
        }
        figures = _classification_figures(y_test, y_pred, sorted(y.unique()))

    fig_importancia = _feature_importance_figure(pipeline, selected_features)
    if fig_importancia is not None:
        figures["importancia"] = fig_importancia

    resultados = {
        "metrics": metrics,
        "figures": figures,
        "warnings": warnings,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, resultados


def _regression_figures(y_test, y_pred) -> dict:
    df_plot = pd.DataFrame({"Real": np.asarray(y_test, dtype=float), "Previsto": y_pred})
    fig_disp = px.scatter(
        df_plot, x="Real", y="Previsto", title="Previsto vs. Real (conjunto de teste)"
    )
    vmin = min(df_plot["Real"].min(), df_plot["Previsto"].min())
    vmax = max(df_plot["Real"].max(), df_plot["Previsto"].max())
    fig_disp.add_trace(
        go.Scatter(
            x=[vmin, vmax], y=[vmin, vmax], mode="lines",
            name="Previsão perfeita", line=dict(dash="dash"),
        )
    )

    residuos = df_plot["Real"] - df_plot["Previsto"]
    fig_res = px.scatter(
        x=df_plot["Previsto"], y=residuos,
        labels={"x": "Previsto", "y": "Resíduo (Real - Previsto)"},
        title="Resíduos — o ideal é ficarem espalhados em torno de zero",
    )
    fig_res.add_hline(y=0, line_dash="dash")
    return {"previsto_vs_real": fig_disp, "residuos": fig_res}


def _classification_figures(y_test, y_pred, classes) -> dict:
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig_cm = px.imshow(
        cm,
        x=[str(c) for c in classes],
        y=[str(c) for c in classes],
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Previsto", "y": "Real", "color": "Quantidade"},
        title="Matriz de confusão (conjunto de teste)",
    )
    return {"matriz_confusao": fig_cm}


def _feature_importance_figure(pipeline, selected_features):
    """Importância por variável de entrada; one-hot é agregado de volta à coluna original."""
    modelo = pipeline.named_steps["modelo"]
    pre = pipeline.named_steps["preprocessamento"]
    try:
        nomes_saida = pre.get_feature_names_out()
    except Exception:  # noqa: BLE001
        return None

    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        coef = np.atleast_2d(modelo.coef_)
        valores = np.abs(coef).mean(axis=0)
    else:
        return None

    agregado: dict[str, float] = {c: 0.0 for c in selected_features}
    for nome, valor in zip(nomes_saida, valores):
        base = nome.split("__", 1)[-1]  # remove prefixo num__/cat__
        original = next(
            (c for c in sorted(selected_features, key=len, reverse=True) if base == c or base.startswith(c + "_")),
            None,
        )
        if original is not None:
            agregado[original] += float(abs(valor))

    df_imp = (
        pd.DataFrame({"Variável": list(agregado), "Importância": list(agregado.values())})
        .sort_values("Importância", ascending=True)
    )
    fig = px.bar(
        df_imp, x="Importância", y="Variável", orientation="h",
        title="Importância das variáveis no modelo treinado",
    )
    return fig
