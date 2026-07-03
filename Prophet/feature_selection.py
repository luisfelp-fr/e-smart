"""Pontuação e explicação da relevância de cada variável em relação ao alvo.

O score (0-100) combina três visões complementares:
  - assoc: associação estatística direta (|Pearson| para regressão,
           ANOVA-F para classificação; mutual information para categóricas)
  - mi:    mutual information (captura relações não lineares)
  - tree:  importância em uma Random Forest rápida

Score = 100 * (0.35*assoc + 0.30*mi + 0.35*tree), com penalidades para
variância quase nula e redundância entre variáveis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import OrdinalEncoder

PESO_ASSOC, PESO_MI, PESO_TREE = 0.35, 0.30, 0.35
SCORE_MINIMO_RECOMENDACAO = 20
LIMIAR_REDUNDANCIA = 0.95


def score_features(
    treated_df: pd.DataFrame, target_col: str, problem_type: str, feature_meta: dict
) -> pd.DataFrame:
    """Retorna DataFrame com colunas: variavel, score, recomendada, explicacao (+sub-scores)."""
    features = list(feature_meta.keys())
    if not features:
        return pd.DataFrame(columns=["variavel", "score", "recomendada", "explicacao"])

    y = treated_df[target_col]
    X_num, discretas = _encode_for_scoring(treated_df, features, feature_meta)

    assoc = _association_scores(X_num, y, features, feature_meta, problem_type)
    mi = _mutual_info_scores(X_num, y, discretas, problem_type)
    tree = _tree_importance_scores(X_num, y, problem_type)

    assoc_n, mi_n, tree_n = _normalize(assoc), _normalize(mi), _normalize(tree)
    scores = np.round(100 * (PESO_ASSOC * assoc_n + PESO_MI * mi_n + PESO_TREE * tree_n))

    # penalidade: variância quase nula
    zero_var = np.zeros(len(features), dtype=bool)
    for i, col in enumerate(features):
        serie = treated_df[col].dropna()
        if feature_meta[col]["tipo"] == "numerica" and len(serie):
            media = abs(serie.mean())
            if serie.nunique() <= 1 or (media > 0 and serie.std() / media < 1e-3) or serie.std() == 0:
                zero_var[i] = True
                scores[i] = 0

    # penalidade: redundância entre variáveis numéricas muito correlacionadas
    redundante_com: dict[str, tuple[str, float]] = {}
    numericas = [c for c in features if feature_meta[c]["tipo"] == "numerica"]
    if len(numericas) >= 2:
        corr = treated_df[numericas].corr().abs()
        for i, a in enumerate(numericas):
            for b in numericas[i + 1 :]:
                r = corr.loc[a, b]
                if pd.notna(r) and r > LIMIAR_REDUNDANCIA:
                    ia, ib = features.index(a), features.index(b)
                    perdedor, vencedor = (a, b) if scores[ia] <= scores[ib] else (b, a)
                    if perdedor not in redundante_com:
                        redundante_com[perdedor] = (vencedor, float(r))
                        scores[features.index(perdedor)] = round(scores[features.index(perdedor)] * 0.5)

    linhas = []
    for i, col in enumerate(features):
        explicacao = _build_explanation(
            col,
            assoc_n[i],
            mi_n[i],
            tree_n[i],
            zero_var[i],
            redundante_com.get(col),
            feature_meta[col],
        )
        linhas.append(
            {
                "variavel": col,
                "score": int(scores[i]),
                "assoc": round(float(assoc_n[i]), 2),
                "mi": round(float(mi_n[i]), 2),
                "arvore": round(float(tree_n[i]), 2),
                "explicacao": explicacao,
            }
        )

    result = pd.DataFrame(linhas).sort_values("score", ascending=False).reset_index(drop=True)

    # recomendação: score mínimo, garantindo pelo menos as 3 melhores não nulas
    result["recomendada"] = (result["score"] >= SCORE_MINIMO_RECOMENDACAO) & (result["score"] > 0)
    top3 = result[result["score"] > 0].head(3).index
    result.loc[top3, "recomendada"] = True
    return result


def _encode_for_scoring(df: pd.DataFrame, features: list, feature_meta: dict):
    """Matriz numérica só para pontuação: ordinal p/ categóricas, imputação simples."""
    X = pd.DataFrame(index=df.index)
    discretas = []
    for col in features:
        if feature_meta[col]["tipo"] == "categorica":
            codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(
                df[[col]].astype(str)
            )
            X[col] = codes[:, 0]
            discretas.append(True)
        else:
            serie = df[col]
            X[col] = serie.fillna(serie.median())
            discretas.append(bool(feature_meta[col].get("inteira")))
    X = X.fillna(0)
    return X, np.array(discretas)


def _association_scores(X, y, features, feature_meta, problem_type):
    """|Pearson| (regressão) ou ANOVA-F (classificação); MI pura para categóricas."""
    assoc = np.zeros(len(features))
    y_arr = y.astype(str) if problem_type == "classificacao" else y.astype(float)
    try:
        if problem_type == "regressao":
            f_vals, _ = f_regression(X.values, y_arr.values)
            # converte F em |correlação| equivalente para ficar em escala interpretável
            n = len(y_arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = f_vals / (f_vals + max(n - 2, 1))
            assoc = np.sqrt(np.nan_to_num(r2))
        else:
            f_vals, _ = f_classif(X.values, y_arr.values)
            f_vals = np.nan_to_num(f_vals)
            assoc = f_vals / f_vals.max() if f_vals.max() > 0 else f_vals
    except Exception:  # noqa: BLE001 - pontuação nunca deve derrubar o app
        pass
    # para categóricas a associação linear não faz sentido; usa MI dedicada
    for i, col in enumerate(features):
        if feature_meta[col]["tipo"] == "categorica":
            assoc[i] = np.nan
    if np.isnan(assoc).any():
        mi_cat = _mutual_info_scores(X, y, np.ones(len(features), dtype=bool), problem_type)
        mi_cat_n = _normalize(mi_cat)
        assoc = np.where(np.isnan(assoc), mi_cat_n, assoc)
    return np.nan_to_num(assoc)


def _mutual_info_scores(X, y, discretas, problem_type):
    try:
        if problem_type == "regressao":
            return mutual_info_regression(
                X.values, y.astype(float).values, discrete_features=discretas, random_state=42
            )
        return mutual_info_classif(
            X.values, y.astype(str).values, discrete_features=discretas, random_state=42
        )
    except Exception:  # noqa: BLE001
        return np.zeros(X.shape[1])


def _tree_importance_scores(X, y, problem_type):
    try:
        if problem_type == "regressao":
            modelo = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            modelo.fit(X.values, y.astype(float).values)
        else:
            modelo = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            modelo.fit(X.values, y.astype(str).values)
        return modelo.feature_importances_
    except Exception:  # noqa: BLE001
        return np.zeros(X.shape[1])


def _normalize(valores: np.ndarray) -> np.ndarray:
    valores = np.nan_to_num(np.asarray(valores, dtype=float))
    vmax = valores.max()
    return valores / vmax if vmax > 0 else valores


def _build_explanation(col, assoc, mi, tree, zero_var, redundancia, meta) -> str:
    if zero_var:
        return "Variância quase nula — praticamente constante, sem valor preditivo."
    partes = []
    if assoc >= 0.7:
        partes.append("forte relação com o alvo (correlação/associação alta)")
    elif assoc >= 0.3:
        partes.append("relação moderada com o alvo")
    else:
        partes.append("relação fraca com o alvo")
    if tree >= 0.5:
        partes.append("importante nos modelos de árvore")
    if mi >= 0.5:
        partes.append("carrega informação não linear sobre o alvo")
    if redundancia:
        outro, r = redundancia
        partes.append(f"redundante com '{outro}' (correlação {r:.2f}) — considere usar apenas uma")
    texto = "; ".join(partes).capitalize() + "."
    if meta.get("derivada_da_data"):
        texto += " (Derivada da coluna de data.)"
    return texto
