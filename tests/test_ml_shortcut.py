"""Testes do atalho de R² na importância por permutação.

A permutação custa ~99% do tempo da análise e só entra no score quando o
modelo prevê de fato. ``ml_importance`` mede o R² primeiro e só paga esse
custo se valer.

O teste que importa é o de EQUIVALÊNCIA: quando o R² passa, o resultado tem
de ser idêntico ao de calcular tudo direto. Se divergir, o atalho deixou de
ser otimização e virou mudança de método.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.modeling import ml_importance  # noqa: E402

WINDOWS = [3, 7]
MAX_LAG = 3


def _com_sinal(n: int = 320, seed: int = 5) -> pd.DataFrame:
    """Alvo previsível a partir dos indicadores: o modelo consegue aprender."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n).cumsum()
    x2 = rng.normal(0, 1, n)
    alvo = 3.0 * pd.Series(x1).shift(2).to_numpy() + 1.5 * x2 + rng.normal(0, 0.2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "alvo": alvo}).dropna()


def _sem_sinal(n: int = 320, seed: int = 6) -> pd.DataFrame:
    """Alvo independente dos indicadores: não há o que aprender."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "alvo": rng.normal(0, 1, n),
    })


def test_resultado_equivalente_quando_o_r2_passa():
    """Com o atalho ligado ou desligado, o resultado tem de ser o mesmo.

    O passo 2 reajusta as florestas; com random_state fixo elas são idênticas
    às do passo 1, então nada pode mudar.

    A comparação é por tolerância, e não por igualdade exata, por um motivo
    que NÃO tem a ver com o atalho: quando a floresta roda multithread
    (n_jobs > 1), o sklearn acumula as predições das árvores em ordem que
    depende do escalonamento das threads, e soma de float não é associativa.
    Duas chamadas idênticas já divergiam ~1e-16 antes desta mudança. A
    tolerância aqui é ordens de grandeza menor que qualquer mudança de método
    real, então o teste continua pegando divergência de verdade.
    """
    df = _com_sinal()
    com_atalho = ml_importance(df, "alvo", MAX_LAG, WINDOWS)
    sem_atalho = ml_importance(df, "alvo", MAX_LAG, WINDOWS, min_r2=-np.inf)

    assert com_atalho.skipped_reason is None, com_atalho.skipped_reason
    assert com_atalho.r2_oos == pytest.approx(sem_atalho.r2_oos, rel=1e-9)
    assert com_atalho.importance.keys() == sem_atalho.importance.keys()
    for p in sem_atalho.importance:
        assert com_atalho.importance[p] == pytest.approx(
            sem_atalho.importance[p], rel=1e-9, abs=1e-12), f"divergiu em {p}"
        assert com_atalho.top_feature[p] == sem_atalho.top_feature[p]


def test_pula_a_permutacao_quando_o_modelo_nao_preve():
    """Sem poder preditivo: devolve o R² e o motivo, sem importâncias."""
    res = ml_importance(_sem_sinal(), "alvo", MAX_LAG, WINDOWS)

    assert res.skipped_reason is not None
    assert "R²" in res.skipped_reason
    assert not res.importance, "calculou importância de um modelo que não aprendeu"
    # o R² continua sendo reportado — é ele que justifica a decisão
    assert np.isfinite(res.r2_oos)
    assert res.r2_oos < 0.05


def test_atalho_desligado_calcula_mesmo_sem_sinal():
    """min_r2=-inf força o cálculo, para diagnóstico."""
    res = ml_importance(_sem_sinal(), "alvo", MAX_LAG, WINDOWS, min_r2=-np.inf)

    assert res.skipped_reason is None
    assert res.importance, "deveria ter calculado com o atalho desligado"


def test_r2_sempre_reportado_nos_dois_caminhos():
    """Quem lê o relatório precisa do R² em qualquer cenário."""
    for df in (_com_sinal(), _sem_sinal()):
        res = ml_importance(df, "alvo", MAX_LAG, WINDOWS)
        assert np.isfinite(res.r2_oos)
        assert res.r2_folds
