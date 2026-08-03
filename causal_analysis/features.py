"""Engenharia de atributos temporais: lags e médias móveis.

Cada parâmetro gera uma família de séries derivadas:

- ``param``            valor bruto (lag 0);
- ``param __lag k``    valor defasado em k períodos (efeito com atraso);
- ``param __mm w``     média móvel de janela w (efeito acumulado/suavizado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_SEP = "__lag"
ROLL_SEP = "__mm"


def derived_features(x: pd.Series, max_lag: int, windows: list[int]) -> pd.DataFrame:
    """Família de transformações temporais de um único parâmetro."""
    name = str(x.name)
    out = {name: x}
    for k in range(1, max_lag + 1):
        out[f"{name}{LAG_SEP}{k}"] = x.shift(k)
    for w in windows:
        if w >= 2:
            out[f"{name}{ROLL_SEP}{w}"] = x.rolling(w, min_periods=max(2, w // 2)).mean()
    return pd.DataFrame(out)


def single_feature(x: pd.Series, feature_name: str) -> pd.Series | None:
    """Recria UMA série derivada pelo nome, sem montar a família inteira.

    Usado por quem já sabe qual transformação quer (ex.: a "melhor versão"
    escolhida pelo ranking). Montar a família toda para descartar 33 das 34
    colunas é desperdício puro — e pior, obriga quem chama a adivinhar o
    ``max_lag`` usado na varredura.

    Devolve ``None`` se o nome não corresponder a uma derivação desta série.
    """
    name = str(x.name)
    if feature_name == name:
        return x
    for sep, build in (
        (LAG_SEP, lambda k: x.shift(k)),
        (ROLL_SEP, lambda w: x.rolling(w, min_periods=max(2, w // 2)).mean()),
    ):
        base, found, arg = feature_name.partition(sep)
        if not found or base != name:
            continue
        try:
            n = int(arg)
        except ValueError:
            return None
        if sep is ROLL_SEP and n < 2:
            return None
        return build(n)
    return None


def build_matrix(
    df: pd.DataFrame, target: str, max_lag: int, windows: list[int]
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Matriz completa (todas as famílias) alinhada ao alvo, para o modelo ML.

    Devolve (X, y, mapa parâmetro-base -> nomes das colunas derivadas).
    """
    params = [c for c in df.columns if c != target]
    frames = []
    groups: dict[str, list[str]] = {}
    for p in params:
        # float32 já na origem: a matriz tem (1 + max_lag + |windows|) colunas
        # por parâmetro e é o pico de memória da análise. Montar em float64
        # para converter depois alocava as duas versões ao mesmo tempo.
        fam = derived_features(df[p], max_lag, windows).astype(np.float32)
        frames.append(fam)
        groups[p] = list(fam.columns)
    X = pd.concat(frames, axis=1)
    y = df[target]
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask], groups


def base_param(feature_name: str) -> str:
    """Nome do parâmetro original a partir do nome da coluna derivada."""
    for sep in (LAG_SEP, ROLL_SEP):
        if sep in feature_name:
            return feature_name.split(sep)[0]
    return feature_name


def feature_label(feature_name: str) -> str:
    """Rótulo humano da transformação ('bruto', 'lag 3', 'média móvel 7')."""
    if LAG_SEP in feature_name:
        return f"lag {feature_name.split(LAG_SEP)[1]}"
    if ROLL_SEP in feature_name:
        return f"média móvel {feature_name.split(ROLL_SEP)[1]}"
    return "bruto (sem defasagem)"
